import os
import streamlit as st
import tempfile
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import numpy as np

# 환경 변수 로드 (로컬 환경에서만)
load_dotenv()

# Supabase 클라이언트 초기화
@st.cache_resource
def init_supabase():
    """Supabase 클라이언트 초기화 (로컬: .env, Cloud: st.secrets)"""
    # Streamlit Cloud의 secrets를 우선 사용, 없으면 환경 변수 사용
    try:
        # Streamlit Cloud의 secrets 사용
        supabase_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
    except:
        # secrets가 없으면 환경 변수만 사용
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        return None
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        return None

supabase: Client = init_supabase()

# 벡터 검색을 위한 커스텀 Retriever 클래스
class SupabaseRetriever:
    """Supabase를 사용한 벡터 검색 Retriever (사용자별 분리)"""
    def __init__(self, supabase_client: Client, user_id: str, session_id: str, embeddings: OpenAIEmbeddings, k: int = 10):
        self.supabase = supabase_client
        self.user_id = user_id
        self.session_id = session_id
        self.embeddings = embeddings
        self.k = k
    
    def invoke(self, query: str) -> List[Any]:
        """쿼리에 대한 유사 문서 검색 (사용자별 필터링)"""
        if self.supabase is None:
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # Supabase에서 벡터 검색 (pgvector 사용, 사용자별 필터링)
            try:
                result = self.supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.7,
                        'match_count': self.k,
                        'user_id': self.user_id,
                        'session_id': self.session_id
                    }
                ).execute()
                
                # 결과를 Document 형식으로 변환
                documents = []
                if result.data:
                    for item in result.data:
                        from langchain.schema import Document
                        doc = Document(
                            page_content=item.get('chunk_text', ''),
                            metadata={
                                'source': item.get('file_name', ''),
                                'chunk_index': item.get('chunk_index', 0),
                                'session_id': item.get('session_id', ''),
                                'user_id': item.get('user_id', '')
                            }
                        )
                        documents.append(doc)
                
                return documents
            except Exception as rpc_error:
                # RPC 함수가 없으면 직접 SQL 쿼리로 검색
                return self._search_with_direct_query(query_embedding)
        except Exception as e:
            return []
    
    def _search_with_direct_query(self, query_embedding: List[float]) -> List[Any]:
        """직접 SQL 쿼리를 사용한 벡터 검색 (사용자별 필터링)"""
        try:
            # Supabase에서 해당 사용자와 세션의 모든 임베딩 가져오기
            result = self.supabase.table("embeddings").select("*").eq("user_id", self.user_id).eq("session_id", self.session_id).execute()
            
            if not result.data:
                return []
            
            # 코사인 유사도 계산
            documents = []
            similarities = []
            
            for item in result.data:
                embedding_raw = item.get('embedding', None)
                
                # embedding이 문자열로 저장되어 있을 수 있으므로 파싱
                if isinstance(embedding_raw, str):
                    try:
                        import ast
                        embedding = ast.literal_eval(embedding_raw)
                    except:
                        continue
                elif isinstance(embedding_raw, list):
                    embedding = embedding_raw
                else:
                    continue
                
                # 임베딩 차원 확인
                if embedding and len(embedding) == len(query_embedding):
                    # 코사인 유사도 계산
                    dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
                    magnitude_a = sum(a * a for a in query_embedding) ** 0.5
                    magnitude_b = sum(b * b for b in embedding) ** 0.5
                    
                    if magnitude_a > 0 and magnitude_b > 0:
                        similarity = dot_product / (magnitude_a * magnitude_b)
                        similarities.append((similarity, item))
            
            # 유사도 순으로 정렬하고 상위 k개 선택
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_items = similarities[:self.k]
            
            # Document 형식으로 변환
            for similarity, item in top_items:
                from langchain.schema import Document
                doc = Document(
                    page_content=item.get('chunk_text', ''),
                    metadata={
                        'source': item.get('file_name', ''),
                        'chunk_index': item.get('chunk_index', 0),
                        'session_id': item.get('session_id', ''),
                        'user_id': item.get('user_id', ''),
                        'similarity': similarity
                    }
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            import traceback
            st.error(f"벡터 검색 오류: {str(e)}")
            return []

# 관련 질문 생성 함수
def generate_followup_questions(prompt: str, response: str, context_text: str, llm_model) -> List[str]:
    """답변 내용을 기반으로 향후 더 필요한 질문 3개 생성"""
    try:
        question_prompt = f"""다음 질문과 답변을 기반으로, 사용자가 더 깊이 있게 알아볼 수 있는 관련 질문 3개를 생성해주세요.

원래 질문: {prompt}

답변 내용:
{response[:1000]}

관련 문서 컨텍스트:
{context_text[:500]}

요구사항:
- 답변 내용과 관련 문서를 바탕으로 더 깊이 있는 질문 생성
- 각 질문은 한 문장으로 작성
- 질문은 구체적이고 실용적이어야 함
- 질문만 출력 (번호나 설명 없이)
- 각 질문은 줄바꿈으로 구분

예시 형식:
질문 1
질문 2
질문 3

관련 질문:"""
        
        questions_text = llm_model.invoke(question_prompt).content.strip()
        
        # 질문들을 리스트로 분리
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            # 번호나 불필요한 접두사 제거
            if line:
                # "1. ", "질문 1: ", "- " 등의 접두사 제거
                for prefix in ['1.', '2.', '3.', '질문 1:', '질문 2:', '질문 3:', '-', '•']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line and len(line) > 5:  # 너무 짧은 질문 제외
                    questions.append(line)
        
        # 최대 3개만 반환
        return questions[:3]
    except Exception as e:
        return []

# 세션 제목 생성 함수 (키워드 기반)
def generate_session_title(chat_history: list, llm_model) -> str:
    """대화 내용을 기반으로 키워드를 추출하여 세션 제목 생성"""
    if not chat_history or len(chat_history) == 0:
        return "새 세션"
    
    # 최근 대화만 사용 (처음 3개 대화)
    recent_chats = chat_history[:6] if len(chat_history) > 6 else chat_history
    
    # 대화 내용 요약
    conversation_text = ""
    for msg in recent_chats:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"사용자: {content[:100]}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content[:200]}\n"
    
    if not conversation_text.strip():
        return "새 세션"
    
    try:
        prompt = f"""다음 대화 내용을 분석하여 주요 키워드 3-5개를 추출하고, 이를 바탕으로 간결한 세션 제목을 생성해주세요.

대화 내용:
{conversation_text}

요구사항:
- 주요 키워드 3-5개를 먼저 추출하세요
- 키워드를 활용하여 20자 이내의 제목을 생성하세요
- 제목은 한글로 작성하세요
- 따옴표나 특수문자 없이 작성하세요
- 제목만 출력하세요 (설명 없이)

제목:"""
        
        title = llm_model.invoke(prompt).content.strip()
        # 따옴표 제거
        title = title.strip('"').strip("'").strip()
        
        # 너무 길면 자르기
        if len(title) > 30:
            title = title[:27] + "..."
        
        return title if title else "새 세션"
    except Exception as e:
        # 오류 발생 시 첫 번째 사용자 메시지 사용
        first_user_msg = next((msg.get("content", "") for msg in chat_history if msg.get("role") == "user"), "")
        if first_user_msg:
            return first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg
        return "새 세션"

# 테이블 존재 확인 및 생성 함수
def check_and_create_embeddings_table():
    """embeddings 테이블이 존재하는지 확인하고 user_id 컬럼이 있는지 확인"""
    if supabase is None:
        return False
    
    try:
        # 테이블 존재 확인 시도
        supabase.table("embeddings").select("id").limit(1).execute()
        
        # user_id 컬럼이 있는지 확인
        try:
            supabase.table("embeddings").select("user_id").limit(1).execute()
            return True
        except Exception as e:
            error_msg = str(e)
            if "column" in error_msg.lower() and "user_id" in error_msg.lower():
                st.error("""
                ⚠️ **embeddings 테이블에 user_id 컬럼이 없습니다!**
                
                **해결 방법:**
                1. Supabase 대시보드에서 SQL Editor를 엽니다
                2. `supabase_multi_user_migration.sql` 파일의 내용을 복사하여 실행합니다
                3. 또는 `supabase_multi_user_setup.sql` 파일을 실행하여 새로 생성합니다
                
                **중요:** 기존 데이터가 있다면 마이그레이션 SQL을 사용하세요.
                """)
                return False
            return False
    except Exception as e:
        error_msg = str(e)
        if "Could not find the table" in error_msg or "PGRST205" in error_msg:
            st.error("""
            ⚠️ **embeddings 테이블이 없습니다!**
            
            Supabase에 `embeddings` 테이블을 생성해야 합니다.
            
            **해결 방법:**
            1. Supabase 대시보드에서 SQL Editor를 엽니다
            2. `supabase_multi_user_setup.sql` 파일의 내용을 복사하여 실행합니다
            """)
            return False
        return False

# 사용자 설정 관리 함수
def save_user_api_keys(user_id: str, api_keys: dict):
    """사용자 API 키를 Supabase에 저장"""
    if supabase is None:
        return False
    
    try:
        # 암호화된 형태로 저장 (실제로는 암호화 권장)
        settings_data = {
            "user_id": user_id,
            "openai_api_key": api_keys.get("openai", ""),
            "claude_api_key": api_keys.get("claude", ""),
            "gemini_api_key": api_keys.get("gemini", ""),
            "updated_at": datetime.now().isoformat()
        }
        
        # 기존 설정이 있는지 확인
        existing = supabase.table("user_settings").select("*").eq("user_id", user_id).execute()
        
        if existing.data:
            # 업데이트
            supabase.table("user_settings").update(settings_data).eq("user_id", user_id).execute()
        else:
            # 새로 생성
            supabase.table("user_settings").insert(settings_data).execute()
        
        return True
    except Exception as e:
        return False

def load_user_api_keys(user_id: str) -> dict:
    """사용자 API 키를 Supabase에서 로드"""
    if supabase is None:
        return {"openai": "", "claude": "", "gemini": ""}
    
    try:
        result = supabase.table("user_settings").select("*").eq("user_id", user_id).execute()
        
        if result.data and len(result.data) > 0:
            settings = result.data[0]
            return {
                "openai": settings.get("openai_api_key", ""),
                "claude": settings.get("claude_api_key", ""),
                "gemini": settings.get("gemini_api_key", "")
            }
        return {"openai": "", "claude": "", "gemini": ""}
    except Exception as e:
        return {"openai": "", "claude": "", "gemini": ""}

# 세션 관리 함수
def save_embeddings_to_supabase(user_id: str, session_id: str, file_name: str, chunks: List[Any], embeddings_model: OpenAIEmbeddings):
    """임베딩을 Supabase에 저장 (이미 존재하면 재사용, 사용자별 분리)"""
    if supabase is None:
        st.error("Supabase 연결이 없습니다.")
        return False
    
    # 테이블 존재 확인
    if not check_and_create_embeddings_table():
        return False
    
    try:
        # 해당 사용자, 세션, 파일의 임베딩이 이미 존재하는지 확인
        existing = supabase.table("embeddings").select("id").eq("user_id", user_id).eq("session_id", session_id).eq("file_name", file_name).limit(1).execute()
        
        if existing.data and len(existing.data) > 0:
            # 이미 임베딩이 존재하면 재사용
            st.info(f"'{file_name}'의 임베딩이 이미 존재합니다. 재사용합니다.")
            return True
        
        # 새 임베딩 생성 및 저장
        batch_size = 30
        total_saved = 0
        total_errors = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            try:
                # 각 청크의 임베딩 생성
                texts = [chunk.page_content for chunk in batch_chunks]
                embeddings_list = embeddings_model.embed_documents(texts)
                
                # Supabase에 배치로 저장
                batch_data = []
                for idx, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings_list)):
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # 텍스트 정제: null 문자 및 제어 문자 제거
                    cleaned_text = chunk.page_content[:50000]
                    # null 문자(\u0000) 제거
                    cleaned_text = cleaned_text.replace('\x00', '')
                    # 다른 제어 문자도 제거 (탭, 줄바꿈, 캐리지 리턴은 유지)
                    cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char in '\n\r\t')
                    
                    embedding_data = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "file_name": file_name,
                        "chunk_index": i + idx,
                        "chunk_text": cleaned_text,
                        "embedding": embedding_str,
                        "metadata": json.dumps(chunk.metadata, ensure_ascii=False) if chunk.metadata else "{}"
                    }
                    batch_data.append(embedding_data)
                
                # 배치 삽입 시도
                if batch_data:
                    try:
                        result = supabase.table("embeddings").insert(batch_data).execute()
                        total_saved += len(batch_data)
                    except Exception as batch_error:
                        # 배치 삽입 실패 시 하나씩 삽입
                        for data in batch_data:
                            try:
                                if isinstance(data["embedding"], list):
                                    data["embedding"] = '[' + ','.join(map(str, data["embedding"])) + ']'
                                
                                # 텍스트 정제 (개별 삽입 시에도 적용)
                                if "chunk_text" in data:
                                    cleaned_text = data["chunk_text"]
                                    cleaned_text = cleaned_text.replace('\x00', '')
                                    cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char in '\n\r\t')
                                    data["chunk_text"] = cleaned_text
                                
                                supabase.table("embeddings").insert(data).execute()
                                total_saved += 1
                            except Exception as single_error:
                                total_errors += 1
                                st.warning(f"청크 저장 실패: {str(single_error)}")
                                continue
                
            except Exception as batch_embed_error:
                st.warning(f"청크 {i}~{i+batch_size} 배치 임베딩 생성 중 오류: {str(batch_embed_error)}")
                total_errors += len(batch_chunks)
                continue
        
        if total_saved > 0:
            st.success(f"'{file_name}': {total_saved}개 청크 임베딩 저장 완료")
            if total_errors > 0:
                st.warning(f"'{file_name}': {total_errors}개 청크 저장 실패")
            return True
        else:
            st.error(f"'{file_name}': 임베딩 저장 실패 (모든 청크 저장 실패)")
            return False
            
    except Exception as e:
        st.error(f"임베딩 저장 중 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def load_embeddings_from_supabase(user_id: str, session_id: str) -> bool:
    """세션의 임베딩을 로드하여 retriever 생성"""
    if supabase is None:
        return False
    
    # 테이블 존재 확인
    if not check_and_create_embeddings_table():
        return False
    
    try:
        # 해당 사용자와 세션의 임베딩이 있는지 확인
        result = supabase.table("embeddings").select("file_name").eq("user_id", user_id).eq("session_id", session_id).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            # 임베딩이 존재하면 retriever 생성
            # API 키는 session_state에서 가져오기
            api_keys = st.session_state.get("api_keys", {"openai": "", "claude": "", "gemini": ""})
            openai_key = api_keys.get("openai", "")
            if not openai_key:
                # 환경 변수에서도 시도
                openai_key = os.getenv("OPENAI_API_KEY", "")
            
            if openai_key:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                st.session_state.retriever = SupabaseRetriever(supabase, user_id, session_id, embeddings, k=10)
                return True
        
        return False
    except Exception as e:
        return False

def save_session_to_supabase(user_id: str, session_id: str, llm_model):
    """현재 세션을 Supabase에 저장 (기존 세션 업데이트)"""
    if supabase is None:
        return False
    
    try:
        # 세션 제목 생성 (대화 내용이 있을 때만)
        title = None
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            title = generate_session_title(st.session_state.chat_history, llm_model)
        
        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "chat_history": json.dumps(st.session_state.chat_history, ensure_ascii=False),
            "conversation_memory": json.dumps(st.session_state.conversation_memory, ensure_ascii=False),
            "processed_files": json.dumps(st.session_state.processed_files, ensure_ascii=False),
            "metadata": json.dumps({
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, ensure_ascii=False)
        }
        
        # 제목이 있으면 추가
        if title:
            session_data["title"] = title
        
        # 기존 세션이 있는지 확인
        existing = supabase.table("sessions").select("*").eq("user_id", user_id).eq("session_id", session_id).execute()
        
        if existing.data:
            # 업데이트
            supabase.table("sessions").update(session_data).eq("user_id", user_id).eq("session_id", session_id).execute()
        else:
            # 새로 생성
            supabase.table("sessions").insert(session_data).execute()
        
        return True
    except Exception as e:
        return False

def save_new_session_to_supabase(user_id: str, llm_model):
    """새로운 세션으로 Supabase에 저장 (항상 INSERT, 첫 질문과 답변으로 제목 생성)"""
    if supabase is None:
        return False, None
    
    try:
        # 첫 질문과 답변으로 세션 제목 생성
        title = "새 세션"
        if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
            # 첫 번째 사용자 질문과 첫 번째 AI 답변 사용
            first_question = ""
            first_answer = ""
            for msg in st.session_state.chat_history[:2]:
                if msg.get("role") == "user" and not first_question:
                    first_question = msg.get("content", "")[:100]
                elif msg.get("role") == "assistant" and not first_answer:
                    first_answer = msg.get("content", "")[:200]
            
            if first_question and first_answer:
                title_prompt = f"""다음 질문과 답변을 기반으로 간결한 세션 제목을 생성해주세요.

질문: {first_question}

답변: {first_answer[:500]}

요구사항:
- 20자 이내의 제목을 생성하세요
- 제목은 한글로 작성하세요
- 따옴표나 특수문자 없이 작성하세요
- 제목만 출력하세요 (설명 없이)

제목:"""
                try:
                    title = llm_model.invoke(title_prompt).content.strip()
                    title = title.strip('"').strip("'").strip()
                    if len(title) > 30:
                        title = title[:27] + "..."
                except:
                    title = first_question[:30] if first_question else "새 세션"
        
        # 새로운 session_id 생성
        new_session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "session_id": new_session_id,
            "title": title,
            "chat_history": json.dumps(st.session_state.chat_history, ensure_ascii=False),
            "conversation_memory": json.dumps(st.session_state.conversation_memory, ensure_ascii=False),
            "processed_files": json.dumps(st.session_state.processed_files, ensure_ascii=False),
            "metadata": json.dumps({
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, ensure_ascii=False)
        }
        
        # 항상 INSERT만 수행
        supabase.table("sessions").insert(session_data).execute()
        
        return True, new_session_id
    except Exception as e:
        return False, None

def load_session_from_supabase(user_id: str, session_id: str):
    """Supabase에서 세션 로드"""
    if supabase is None:
        return False
    
    try:
        result = supabase.table("sessions").select("*").eq("user_id", user_id).eq("session_id", session_id).execute()
        
        if result.data:
            session_data = result.data[0]
            
            # 현재 세션의 상태를 완전히 초기화
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.processed_files = []
            st.session_state.retriever = None
            st.session_state.vectorstore = None
            
            # 로드할 세션의 데이터만 복원
            if session_data.get("chat_history"):
                loaded_history = json.loads(session_data["chat_history"])
                st.session_state.chat_history = loaded_history.copy() if isinstance(loaded_history, list) else []
            else:
                st.session_state.chat_history = []
            
            if session_data.get("conversation_memory"):
                loaded_memory = json.loads(session_data["conversation_memory"])
                st.session_state.conversation_memory = loaded_memory.copy() if isinstance(loaded_memory, list) else []
            else:
                st.session_state.conversation_memory = []
            
            if session_data.get("processed_files"):
                loaded_files = json.loads(session_data["processed_files"])
                st.session_state.processed_files = loaded_files.copy() if isinstance(loaded_files, list) else []
            else:
                st.session_state.processed_files = []
            
            # 임베딩 로드
            load_embeddings_from_supabase(user_id, session_id)
            
            return True
        return False
    except Exception as e:
        # 오류 발생 시에도 상태를 초기화
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.session_state.processed_files = []
        st.session_state.retriever = None
        st.session_state.vectorstore = None
        return False

def list_sessions_from_supabase(user_id: str):
    """Supabase에서 사용자의 모든 세션 목록 가져오기"""
    if supabase is None:
        return []
    
    try:
        result = supabase.table("sessions").select("session_id, title, created_at, updated_at").eq("user_id", user_id).order("updated_at", desc=True).limit(50).execute()
        sessions = result.data if result.data else []
        return sessions
    except Exception as e:
        return []

def delete_session_from_supabase(user_id: str, session_id: str):
    """Supabase에서 세션 삭제"""
    if supabase is None:
        return False
    
    try:
        # 세션 삭제
        supabase.table("sessions").delete().eq("user_id", user_id).eq("session_id", session_id).execute()
        # 관련 임베딩도 삭제
        supabase.table("embeddings").delete().eq("user_id", user_id).eq("session_id", session_id).execute()
        return True
    except Exception as e:
        return False

# 인증 함수
def sign_up(email: str, password: str):
    """회원가입"""
    if supabase is None:
        return None, "Supabase 연결이 없습니다."
    
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return response, None
    except Exception as e:
        return None, str(e)

def sign_in(email: str, password: str):
    """로그인"""
    if supabase is None:
        return None, "Supabase 연결이 없습니다."
    
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return response, None
    except Exception as e:
        return None, str(e)

def sign_out():
    """로그아웃"""
    if supabase is None:
        return False
    
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        return False

def get_current_user():
    """현재 로그인한 사용자 정보 가져오기"""
    if supabase is None:
        return None
    
    try:
        user = supabase.auth.get_user()
        return user.user if user else None
    except Exception as e:
        return None

# LLM 모델 선택 함수
def get_llm_model(model_name: str, api_keys: dict):
    """선택된 모델명에 따라 LLM 인스턴스 반환 (지정한 모델명을 그대로 사용)"""
    # OpenAI 모델 (gpt로 시작)
    if model_name.startswith("gpt") or "openai" in model_name.lower():
        api_key = api_keys.get("openai", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            return None
        return ChatOpenAI(model=model_name, temperature=1, openai_api_key=api_key)
    # Gemini 모델 (gemini로 시작)
    elif model_name.startswith("gemini"):
        api_key = api_keys.get("gemini", os.getenv("GOOGLE_API_KEY"))
        if not api_key:
            return None
        return ChatGoogleGenerativeAI(model=model_name, temperature=1, google_api_key=api_key)
    # Claude 모델 (claude로 시작)
    elif model_name.startswith("claude"):
        api_key = api_keys.get("claude", os.getenv("ANTHROPIC_API_KEY"))
        if not api_key:
            return None
        return ChatAnthropic(model=model_name, temperature=1, anthropic_api_key=api_key)
    else:
        # 기본값으로 OpenAI 사용 (모델명 그대로)
        api_key = api_keys.get("openai", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            return None
        return ChatOpenAI(model=model_name, temperature=1, openai_api_key=api_key)

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 초기 상태 설정
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = False

if "selected_session_index" not in st.session_state:
    st.session_state.selected_session_index = 0

if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = "gpt-5.1"

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "openai": "",
        "claude": "",
        "gemini": ""
    }

if "user" not in st.session_state:
    st.session_state.user = None

if "api_keys_loaded" not in st.session_state:
    st.session_state.api_keys_loaded = False

# CSS 스타일 (ref.py 참고)
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
}
h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
h5 {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 스타일 */
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #ff1493 !important;
}
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown("""
<div style="text-align: center; margin-top: -4rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span> 
        <span style="color: #ffffff; font-size: 0.7em;">기반</span> 
        <span style="color: #9b59b6;">멀티유저</span> 
        <span style="color: #9b59b6;">멀티세션</span> 
        <span style="color: #ffd700;">RAG</span> 
        <span style="color: #d62728; font-size: 0.7em;">챗봇</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# 로그인 상태 확인
current_user = get_current_user()

# 로그인되지 않은 경우 로그인 화면 표시
if not current_user:
    st.markdown("### 로그인이 필요합니다")
    
    tab1, tab2 = st.tabs(["로그인", "회원가입"])
    
    with tab1:
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        
        if st.button("로그인"):
            response, error = sign_in(email, password)
            if error:
                st.error(f"로그인 실패: {error}")
            else:
                st.success("로그인 성공!")
                st.session_state.user = response.user
                st.rerun()
    
    with tab2:
        new_email = st.text_input("이메일 (회원가입)")
        new_password = st.text_input("비밀번호 (회원가입)", type="password")
        confirm_password = st.text_input("비밀번호 확인", type="password")
        
        if st.button("회원가입"):
            if new_password != confirm_password:
                st.error("비밀번호가 일치하지 않습니다.")
            elif len(new_password) < 6:
                st.error("비밀번호는 최소 6자 이상이어야 합니다.")
            else:
                response, error = sign_up(new_email, new_password)
                if error:
                    st.error(f"회원가입 실패: {error}")
                else:
                    st.success("회원가입 성공! 로그인해주세요.")
    
    st.stop()

# 로그인된 경우 메인 화면 표시
st.session_state.user = current_user
user_id = current_user.id

# 로그인 후 API 키 자동 로드 (한 번만 실행)
if not st.session_state.api_keys_loaded:
    loaded_keys = load_user_api_keys(user_id)
    if loaded_keys:
        st.session_state.api_keys = loaded_keys
        st.session_state.api_keys_loaded = True

# 자동 로드 제거 - 수동 로드만 사용

st.markdown(f"**안녕하세요, {current_user.email}님!**")
st.markdown("PDF 파일을 업로드하고 내용에 관해 질문해보세요!")

# 사이드바 설정
with st.sidebar:
    # 로그아웃 버튼
    if st.button("로그아웃"):
        sign_out()
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.session_state.processed_files = []
        st.session_state.retriever = None
        st.session_state.api_keys_loaded = False
        st.rerun()
    
    st.markdown("---")
    
    # API 키 입력 섹션
    st.markdown('<h2 style="color: #1f77b4;">API 키 설정</h2>', unsafe_allow_html=True)
    
    openai_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_keys.get("openai", ""), key="openai_key_input")
    claude_key = st.text_input("Claude API Key", type="password", value=st.session_state.api_keys.get("claude", ""), key="claude_key_input")
    gemini_key = st.text_input("Gemini API Key", type="password", value=st.session_state.api_keys.get("gemini", ""), key="gemini_key_input")
    
    # API 키 변경 감지 및 저장
    api_keys_changed = False
    if openai_key != st.session_state.api_keys.get("openai", ""):
        st.session_state.api_keys["openai"] = openai_key
        api_keys_changed = True
    
    if claude_key != st.session_state.api_keys.get("claude", ""):
        st.session_state.api_keys["claude"] = claude_key
        api_keys_changed = True
    
    if gemini_key != st.session_state.api_keys.get("gemini", ""):
        st.session_state.api_keys["gemini"] = gemini_key
        api_keys_changed = True
    
    # 변경된 경우에만 저장
    if api_keys_changed and supabase:
        save_user_api_keys(user_id, st.session_state.api_keys)
    
    st.markdown("---")
    
    # LLM 모델 선택
    st.markdown('<h2 style="color: #1f77b4;">LLM 모델 선택</h2>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "모델 선택",
        options=["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"],
        index=["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"].index(st.session_state.selected_llm_model) if st.session_state.selected_llm_model in ["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"] else 0,
        key="llm_model_selectbox"
    )
    st.session_state.selected_llm_model = selected_model
    
    model_info = {
        "gpt-5.1": "OpenAI GPT-5.1",
        "gemini-3-pro-preview": "Google Gemini 3 Pro Preview",
        "claude-sonnet-4-5": "Anthropic Claude Sonnet 4.5"
    }
    st.info(f"현재 모델: **{model_info.get(selected_model, selected_model)}**")
    
    st.markdown("---")
    
    # 세션 관리 섹션
    st.markdown('<h2 style="color: #1f77b4;">세션 관리</h2>', unsafe_allow_html=True)
    
    if supabase:
        # 세션 목록 가져오기
        sessions = list_sessions_from_supabase(user_id)
        
        # 세션 옵션 생성 (제목 + 날짜)
        session_options = ["새 세션 생성"]
        for s in sessions:
            title = s.get("title", "제목 없음")
            date = s.get("updated_at", "")[:10] if s.get("updated_at") else ""
            session_options.append(f"{title} ({date})")
        
        # 세션 선택 (자동 로드 제거)
        selected_session = st.selectbox(
            "세션 선택",
            options=session_options,
            index=st.session_state.selected_session_index,
            key="session_selectbox"
        )
        
        # 버튼들을 한 줄에 2개씩 배치
        col1, col2 = st.columns(2)
        
        with col1:
            # 세션 저장 버튼 (새 세션으로 INSERT)
            if st.button("세션 저장", use_container_width=True, type="primary"):
                if not st.session_state.chat_history or len(st.session_state.chat_history) == 0:
                    st.warning("저장할 대화 내용이 없습니다.")
                else:
                    llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
                    if llm_model:
                        success, new_session_id = save_new_session_to_supabase(user_id, llm_model)
                        if success:
                            st.success("세션이 새로 저장되었습니다!")
                            st.session_state.current_session_id = new_session_id
                            st.rerun()
                        else:
                            st.error("세션 저장에 실패했습니다.")
                    else:
                        st.error("LLM 모델을 초기화할 수 없습니다.")
        
        with col2:
            # 세션 로드 버튼
            if st.button("세션 로드", use_container_width=True, type="primary"):
                if selected_session != "새 세션 생성":
                    # 선택된 세션 찾기
                    selected_index = session_options.index(selected_session) - 1
                    if 0 <= selected_index < len(sessions):
                        selected_session_data = sessions[selected_index]
                        full_session_id = selected_session_data['session_id']
                        
                        # 세션 로드
                        if load_session_from_supabase(user_id, full_session_id):
                            st.session_state.current_session_id = full_session_id
                            st.session_state.session_loaded = True
                            st.session_state.selected_session_index = session_options.index(selected_session)
                            st.success(f"세션 '{selected_session_data.get('title', '제목 없음')}'이 로드되었습니다!")
                            st.rerun()
                        else:
                            st.error("세션 로드에 실패했습니다.")
                else:
                    st.warning("로드할 세션을 선택해주세요.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # 세션 삭제 버튼
            if st.button("세션 삭제", use_container_width=True, type="secondary"):
                if selected_session != "새 세션 생성" and sessions:
                    # 선택된 세션 찾기
                    selected_index = session_options.index(selected_session) - 1
                    if 0 <= selected_index < len(sessions):
                        selected_session_data = sessions[selected_index]
                        full_session_id = selected_session_data['session_id']
                        session_title = selected_session_data.get('title', '제목 없음')
                        
                        if delete_session_from_supabase(user_id, full_session_id):
                            st.success(f"세션 '{session_title}'이 삭제되었습니다!")
                            st.session_state.selected_session_index = 0
                            # 현재 세션이 삭제된 세션이면 초기화
                            if st.session_state.current_session_id == full_session_id:
                                st.session_state.current_session_id = str(uuid.uuid4())
                                st.session_state.chat_history = []
                                st.session_state.conversation_memory = []
                                st.session_state.processed_files = []
                                st.session_state.retriever = None
                                st.session_state.vectorstore = None
                            st.rerun()
                        else:
                            st.error("세션 삭제에 실패했습니다.")
                else:
                    st.warning("삭제할 세션을 선택해주세요.")
        
        with col4:
            # 화면 초기화 버튼
            if st.button("화면 초기화", use_container_width=True, type="secondary"):
                # 모든 상태 완전히 초기화
                st.session_state.current_session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.session_state.conversation_memory = []
                st.session_state.processed_files = []
                st.session_state.retriever = None
                st.session_state.vectorstore = None
                st.session_state.session_loaded = False
                st.session_state.session_auto_loaded = False
                st.session_state.selected_session_index = 0
                st.success("화면이 초기화되었습니다!")
                st.rerun()
        
        st.markdown("---")
        # 현재 세션 제목 표시
        current_session_title = "새 세션"
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            try:
                session_data = supabase.table("sessions").select("title").eq("user_id", user_id).eq("session_id", st.session_state.current_session_id).execute()
                if session_data.data and session_data.data[0].get("title"):
                    current_session_title = session_data.data[0]["title"]
                else:
                    # 제목이 없으면 생성
                    llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
                    if llm_model:
                        current_session_title = generate_session_title(st.session_state.chat_history, llm_model)
            except:
                llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
                if llm_model:
                    current_session_title = generate_session_title(st.session_state.chat_history, llm_model)
        
        st.info(f"📌 현재 세션: **{current_session_title}**")
    else:
        st.warning("Supabase가 연결되지 않았습니다. 세션 저장 기능을 사용할 수 없습니다.")
    
    st.markdown("---")
    st.markdown('<h2 style="color: #1f77b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        
        if process_button:
            with st.spinner("PDF 파일을 처리 중입니다..."):
                try:
                    # API 키 확인
                    if not st.session_state.api_keys.get("openai"):
                        st.error("OpenAI API 키를 입력해주세요.")
                        st.stop()
                    
                    # 임시 파일 생성 및 처리
                    temp_dir = tempfile.TemporaryDirectory()
                    
                    all_docs = []
                    new_files = []
                    
                    # 각 파일 처리
                    for uploaded_file in uploaded_files:
                        # 이미 처리된 파일 스킵
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                            
                        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        
                        # 업로드된 파일을 임시 파일로 저장
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # PDF 로더 생성 및 문서 로드
                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        
                        # 메타데이터에 파일 이름 추가
                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name
                        
                        all_docs.extend(documents)
                        new_files.append(uploaded_file.name)
                
                    if not all_docs:
                        st.success("모든 파일이 이미 처리되었습니다.")
                    else:
                        # 텍스트 분할
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        # 모든 청크를 벡터 데이터베이스에 저장
                        total_chunks = len(chunks)
                        st.info(f"총 {total_chunks}개의 청크를 처리합니다.")
                        
                        # 임베딩 및 Supabase 벡터 저장
                        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_keys.get("openai"))
                        
                        # 각 파일별로 임베딩 저장 (이미 존재하면 재사용)
                        saved_count = 0
                        for file_name in new_files:
                            # 해당 파일의 청크만 필터링
                            file_chunks = [chunk for chunk in chunks if chunk.metadata.get("source") == file_name]
                            
                            if file_chunks:
                                # Supabase에 임베딩 저장 (이미 존재하면 재사용)
                                if save_embeddings_to_supabase(
                                    user_id,
                                    st.session_state.current_session_id,
                                    file_name,
                                    file_chunks,
                                    embeddings
                                ):
                                    saved_count += 1
                        
                        # 임베딩 저장 확인
                        if saved_count == 0 and new_files:
                            st.error("임베딩 저장에 실패했습니다. Supabase 설정을 확인해주세요.")
                        
                        # Supabase 기반 retriever 생성
                        st.session_state.retriever = SupabaseRetriever(
                            supabase,
                            user_id,
                            st.session_state.current_session_id,
                            embeddings,
                            k=10
                        )
                        
                        # retriever가 제대로 생성되었는지 확인
                        if st.session_state.retriever and supabase:
                            # 테스트 검색 수행
                            test_result = supabase.table("embeddings").select("id").eq("user_id", user_id).eq("session_id", st.session_state.current_session_id).limit(1).execute()
                            if test_result.data:
                                st.success(f"임베딩 저장 완료: {saved_count}개 파일, 총 {total_chunks}개 청크")
                            else:
                                st.warning("임베딩이 저장되었지만 검색할 수 없습니다. Supabase 설정을 확인해주세요.")
                        
                        # 처리된 파일 목록 업데이트
                        st.session_state.processed_files.extend(new_files)
                        
                        # 파일 처리 후 자동 세션 저장
                        if supabase:
                            llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
                            if llm_model:
                                save_session_to_supabase(user_id, st.session_state.current_session_id, llm_model)
                                st.success("파일이 처리되었고 세션이 자동 저장되었습니다!")
                    
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    st.error("파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")

    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        # 초기화 후 자동 세션 저장
        if supabase:
            llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
            if llm_model:
                save_session_to_supabase(user_id, st.session_state.current_session_id, llm_model)
        st.rerun()
    
    # 메모리 사용량 표시
    if st.session_state.processed_files:
        st.subheader("📊 시스템 상태")
        st.info(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.info(f"대화 기록 수: {len(st.session_state.chat_history)}")

# 대화 내용 표시 (chat_history가 있을 때만)
# chat_history가 비어있거나 None이면 아무것도 표시하지 않음
chat_history = st.session_state.get("chat_history", [])
if chat_history and isinstance(chat_history, list) and len(chat_history) > 0:
    for message in chat_history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 PDF 파일을 업로드하고 처리해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 PDF 파일을 업로드하고 처리해주세요."})
    else:
        try:
            # LLM 모델 가져오기
            llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
            if not llm_model:
                st.error("API 키를 확인해주세요.")
                st.stop()
            
            # RAG 검색 (상위 3개 문서만 사용)
            retrieved_docs = st.session_state.retriever.invoke(prompt)
            
            if not retrieved_docs:
                response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                # 상위 3개 문서만 사용
                top_docs = retrieved_docs[:3]
                
                # 컨텍스트 구성
                context_text = ""
                max_context_length = 8000
                current_length = 0
                
                for i, doc in enumerate(top_docs):
                    doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                    if current_length + len(doc_text) > max_context_length:
                        st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                        break
                    context_text += doc_text
                    current_length += len(doc_text)
                
                # 과거 대화 맥락 구성
                conversation_context = ""
                if st.session_state.conversation_memory:
                    conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                    recent_conversations = st.session_state.conversation_memory[-50:]
                    for conv in recent_conversations:
                        conversation_context += f"{conv}\n"
                    conversation_context += "=== 대화 맥락 끝 ===\n"
                
                # 시스템 프롬프트 구성
                system_prompt = f"""
                질문: {prompt}
                
                관련 문서:
                {context_text}{conversation_context}
                
                위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
                이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.
                
                답변 형식:
                - 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
                - 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
                - 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
                - 답변은 서술형으로 작성하되 존대말을 사용하세요
                - 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
                
                주의사항:
                - 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
                - "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
                - 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
                - 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
                """
                
                # 스트리밍 모드로 답변 생성 및 표시
                with st.chat_message("assistant"):
                    # 스트리밍 응답 수집을 위한 변수
                    full_response = ""
                    
                    # 스트리밍 응답을 표시할 빈 컨테이너 생성
                    message_placeholder = st.empty()
                    
                    # 스트리밍 응답 생성 및 실시간 표시
                    stream = llm_model.stream(system_prompt)
                    for chunk in stream:
                        if hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                        elif hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                        else:
                            chunk_text = str(chunk)
                        
                        if chunk_text:
                            full_response += chunk_text
                            # 실시간으로 응답 업데이트
                            message_placeholder.markdown(full_response + "▌")
                    
                    # 최종 응답 표시 (커서 제거)
                    message_placeholder.markdown(full_response)
                    
                    # 전체 응답을 변수에 저장 (나중에 사용)
                    response = full_response
                
                # 관련 질문 3개 생성 (문서를 찾은 경우에만)
                followup_questions = []
                if retrieved_docs and response:
                    followup_questions = generate_followup_questions(prompt, response, context_text, llm_model)
                
                # 관련 질문이 있으면 추가 표시
                if followup_questions:
                    with st.chat_message("assistant"):
                        st.markdown("---")
                        st.markdown("### 💡 더 알아보기\n")
                        st.markdown("다음 질문들도 도움이 될 수 있습니다:\n")
                        for i, question in enumerate(followup_questions, 1):
                            st.markdown(f"{i}. {question}")
                        
                        # 관련 질문을 전체 응답에 추가
                        response += "\n\n---\n\n"
                        response += "### 💡 더 알아보기\n\n"
                        response += "다음 질문들도 도움이 될 수 있습니다:\n\n"
                        for i, question in enumerate(followup_questions, 1):
                            response += f"{i}. {question}\n"
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # 대화 맥락 메모리에 추가
                if "conversation_memory" not in st.session_state:
                    st.session_state.conversation_memory = []
                
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {response}")
                if len(st.session_state.conversation_memory) > 100:
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                
                # 대화 후 자동으로 Supabase에 세션 저장 (백그라운드, rerun 없이)
                if supabase:
                    try:
                        llm_model = get_llm_model(st.session_state.selected_llm_model, st.session_state.api_keys)
                        if llm_model:
                            save_session_to_supabase(user_id, st.session_state.current_session_id, llm_model)
                    except:
                        pass  # 저장 실패해도 대화는 계속 진행
                
        except Exception as e:
            with st.chat_message("assistant"):
                st.write(f"오류가 발생했습니다: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
