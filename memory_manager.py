# =========================================================
# memory_manager.py
# =========================================================
#
# HYBRID AI MEMORY SYSTEM (SLIDING WINDOW + VECTOR EMBEDDING SEARCH)
#
# 최근 대화 문맥(Sliding Window)과 SentenceTransformer 기반의
# 코사인 유사도(Cosine Similarity) 장기 기억 검색 알고리즘을 통합한 메모리 매니저.
# 5턴 단위로 대화 묶음(Chunk)을 생성하여 인덱싱하며,
# 사용자 질의에 맞춰 연관도가 높은 과거 기억을 자동으로 추출하여 문맥에 결합합니다.
#
# =========================================================

import os
import json
import time
import numpy as np

# 벡터 임베딩 및 유사도 측정 라이브러리 (Lazy Loading 적용)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_VECTOR_LIBS = True
except ImportError:
    HAS_VECTOR_LIBS = False
    print("[WARNING] 'sentence_transformers' 또는 'scikit-learn' 라이브러리가 설치되어 있지 않습니다.")
    print("         기본 단기 슬라이딩 윈도우 기능으로 작동합니다.")

# =========================================================
# CLASS
# =========================================================

class MemoryManager:
    """
    단기 슬라이딩 윈도우 기억과 SentenceTransformer 기반 벡터 장기 기억을
    모두 지원하는 메모리 관리 클래스.
    """

    def __init__(
            self,
            memory_dir="memory",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            similarity_threshold=0.55,
            chunk_size=5,
            max_recent=3,
            use_gpu=False
    ):
        """
        :param memory_dir: 기억 데이터 저장 경로
        :param model_name: SentenceTransformer 모델명
        :param similarity_threshold: 장기 기억 검색 시 반영할 최소 코사인 유사도 (기본값: 0.55)
        :param chunk_size: 장기 기억 블록으로 묶을 대화 턴 수 (chat_bot.py의 5턴 주기 기준)
        :param max_recent: 항상 유지할 최근 대화 턴 수 (Short-term Sliding Window)
        :param use_gpu: GPU 사용 여부
        """
        self.memory_dir = memory_dir
        self.conv_dir = os.path.join(memory_dir, "conversations")
        os.makedirs(self.conv_dir, exist_ok=True)

        self.max_recent = max_recent
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

        # 1. 단기 기억 (Short-term Sliding Window)
        self.recent_memory = []

        # 2. 장기 기억 인덱스 (Long-term Chunks & Embeddings)
        self.long_term_chunks = []      # 각 묶음의 텍스트 리스트
        self.chunk_embeddings = None    # 2D numpy array (num_chunks, embed_dim)
        self.current_chunk_buffer = []  # 현재 5턴을 모으는 임시 버퍼

        # 3. 임베딩 모델 로드
        self.embed_model = None
        if HAS_VECTOR_LIBS:
            try:
                device = "cuda" if use_gpu else "cpu"
                print(f"Loading Embedding Model ({model_name}) on {device}...")
                self.embed_model = SentenceTransformer(model_name, device=device)
                print("Embedding Model Loaded Successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load SentenceTransformer: {e}")

        # 기존 저장이 존재한다면 로드
        self.load_memory_state()

        print("=" * 60)
        print(f"MEMORY SYSTEM READY (Recent: {self.max_recent} turns, Long-term chunk: {self.chunk_size} turns)")
        print("=" * 60)

    def add_memory(
            self,
            user: str,
            assistant: str,
            reasoning: str = None
    ):
        """
        새로운 대화(User-Assistant)와 선택적 CoT(Reasoning)를 추가합니다.
        5턴(chunk_size)이 누적될 때마다 벡터 임베딩을 생성하여 장기 기억 인덱스에 보관합니다.
        """
        memory_entry = {
            "timestamp": time.time(),
            "user": user,
            "assistant": assistant
        }
        if reasoning:
            memory_entry["reasoning"] = reasoning

        # 1. 최근 대화 (Sliding Window) 업데이트
        self.recent_memory.append(memory_entry)
        if len(self.recent_memory) > self.max_recent:
            self.recent_memory = self.recent_memory[-self.max_recent:]

        # 2. 장기 기억 버퍼에 축적 (User, Assistant 문자열 세트)
        qa_pair_text = f"User: {user} | Assistant: {assistant}"
        self.current_chunk_buffer.append(qa_pair_text)

        # 3. chat_bot.py의 5턴(z==5) 주기 기억 매핑 로직
        if len(self.current_chunk_buffer) >= self.chunk_size:
            self._commit_chunk_to_long_term()

        # 4. 상태 자동 저장
        self.save_current_memory_state()

    def _commit_chunk_to_long_term(self):
        """
        버퍼에 모인 5턴의 대화를 하나의 Chunk로 인덱싱하고 벡터 임베딩을 계산합니다.
        """
        if not self.current_chunk_buffer:
            return

        chunk_text = "\n".join(self.current_chunk_buffer)
        self.long_term_chunks.append(chunk_text)

        # SentenceTransformer로 chunk 임베딩 생성
        if self.embed_model is not None:
            new_embed = self.embed_model.encode([chunk_text])  # shape: (1, dim)

            if self.chunk_embeddings is None:
                self.chunk_embeddings = new_embed
            else:
                self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embed])

            print(f"[MEMORY MANAGER] New Long-term Memory Chunk #{len(self.long_term_chunks)} Indexed!")

        # 버퍼 비우기
        self.current_chunk_buffer = []

    def search_relevant_memories(self, query: str, top_k: int = 2) -> list:
        """
        사용자 입력(query)과 과거 장기 기억 블록들 간의 코사인 유사도를 계산하여
        임계값(similarity_threshold) 이상의 연관 대화 내역을 추출합니다.
        """
        if not query or self.embed_model is None or self.chunk_embeddings is None or len(self.long_term_chunks) == 0:
            return []

        # 사용자 입력 임베딩 생성
        query_vec = self.embed_model.encode([query])  # shape: (1, dim)

        # 코사인 유사도 계산 (chat_bot.py 방식)
        similarities = cosine_similarity(query_vec, self.chunk_embeddings).flatten()

        relevant_chunks = []
        # 유사도가 높아 상위 k개에 속하면서 threshold 이상인 항목 필터링
        indexed_sims = list(enumerate(similarities))
        indexed_sims.sort(key=lambda x: x[1], reverse=True)

        for idx, sim in indexed_sims[:top_k]:
            if sim >= self.similarity_threshold:
                relevant_chunks.append({
                    "score": float(sim),
                    "content": self.long_term_chunks[idx]
                })

        return relevant_chunks

    def build_memory_context(
            self,
            current_query: str = None
    ) -> str:
        """
        현재 질의(current_query)를 바탕으로, 검색된 관련 장기 기억과
        최근 3턴의 단기 대화 내역을 하나의 통합 프롬프트 문맥으로 구성합니다.
        """
        print("=" * 60)
        print("BUILDING HYBRID MEMORY CONTEXT FOR CONVERSATION")

        context = ""

        # 1. 벡터 검색을 통한 연관 장기 기억 추출 (RAG)
        if current_query:
            relevant_mems = self.search_relevant_memories(current_query)
            if relevant_mems:
                context += "=== Relevant Past Memories (Retrieved) ===\n"
                for i, mem in enumerate(relevant_mems, 1):
                    context += f"[Memory Block {i} (Similarity: {mem['score']:.2f})]\n"
                    context += f"{mem['content']}\n\n"
                context += "=========================================\n\n"

        # 2. 최근 대화 내역 (Sliding Window)
        context += "=== Recent Conversation Context ===\n"
        if len(self.recent_memory) == 0:
            context += "No prior conversation within this session.\n"
        else:
            for i, mem in enumerate(self.recent_memory, 1):
                context += f"[Turn {i}]\n"
                context += f"User: {mem['user']}\n"

                # CoT 추론 내용 반영
                if "reasoning" in mem and mem["reasoning"]:
                    context += f"Assistant (Thought Process): {mem['reasoning']}\n"

                context += f"Assistant: {mem['assistant']}\n\n"

        return context

    def save_current_memory_state(self):
        """
        단기 및 장기 기억 상태와 버퍼를 JSON 파일로 백업합니다.
        (Numpy 배열은 JSON 호환을 위해 리스트로 변환 저장)
        """
        save_path = os.path.join(self.conv_dir, "memory_state.json")

        embeddings_list = self.chunk_embeddings.tolist() if self.chunk_embeddings is not None else None

        data = {
            "updated_at": time.time(),
            "recent_memory": self.recent_memory,
            "long_term_chunks": self.long_term_chunks,
            "current_chunk_buffer": self.current_chunk_buffer,
            "chunk_embeddings": embeddings_list
        }

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("SAVE FAILED:", e)

    def load_memory_state(self):
        """
        디스크에 저장되어 있는 백업 기억 데이터를 불러옵니다.
        """
        save_path = os.path.join(self.conv_dir, "memory_state.json")
        if not os.path.exists(save_path):
            return

        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.recent_memory = data.get("recent_memory", [])
            self.long_term_chunks = data.get("long_term_chunks", [])
            self.current_chunk_buffer = data.get("current_chunk_buffer", [])

            embed_list = data.get("chunk_embeddings")
            if embed_list is not None:
                self.chunk_embeddings = np.array(embed_list, dtype=np.float32)

            print("[MEMORY MANAGER] Successfully restored previous memory state from disk.")
        except Exception as e:
            print("[MEMORY MANAGER] Failed to load state:", e)

    def print_recent_memory(self):
        """현재 단기 기억 상태 출력"""
        print("=" * 60)
        print(f"RECENT MEMORY STATUS (MAX {self.max_recent})")
        print("=" * 60)
        for i, mem in enumerate(self.recent_memory, 1):
            print(f"[{i} / {len(self.recent_memory)}]")
            print("USER:", mem["user"])
            if "reasoning" in mem and mem["reasoning"]:
                print("REASONING:", mem["reasoning"])
            print("ASSISTANT:", mem["assistant"])
            print("-" * 30)

    def clear_recent_memory(self):
        """기억 초기화"""
        self.recent_memory = []
        self.long_term_chunks = []
        self.current_chunk_buffer = []
        self.chunk_embeddings = None
        self.save_current_memory_state()
        print("=" * 60)
        print("ALL MEMORY CLEARED")