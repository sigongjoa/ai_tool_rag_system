# 2. 검색 및 재정렬: 임베딩, 벡터DB, Rerank

데이터 파이프라인을 통해 정제된 데이터(Nodes)는 사용자의 질문과 가장 관련성 높은 정보를 효율적으로 찾아낼 수 있도록 벡터 형태로 변환되어 데이터베이스에 저장됩니다. 검색(Retrieval) 과정은 1차적으로 유사도가 높은 후보군을 빠르게 선별하고, 2차적으로 재정렬(Re-ranking)을 통해 정확도를 극대화하는 파이프라인으로 구성됩니다.

## 2.1. 임베딩 (Embedding) 전략

임베딩 모델은 텍스트(청크)를 다차원 벡터 공간의 한 점으로 매핑하는 역할을 합니다. 모델 선택은 시스템의 성능(속도, 정확도)과 리소스 요구사항에 직접적인 영향을 미칩니다.

### 가. 추천 모델 및 선택
- **초기(MVP) 모델:** **`GTE-Large (thenlper/gte-large)`**
  - **선택 이유:** MTEB(Massive Text Embedding Benchmark) RAG 평가에서 우수한 성능을 보이며, 상대적으로 적은 리소스(CPU 또는 단일 GPU)로 빠른 추론이 가능해 MVP 단계에 가장 적합합니다.
- **고정밀 요구사항:** `bge-large-en-v1.5`, `Instructor-XL` 등 Instruction-Tuned 모델을 고려할 수 있습니다. 이는 특정 작업(e.g., "문서 요약을 위한 구절 검색")에 더 최적화된 임베딩을 생성할 수 있지만, 더 많은 계산 리소스를 필요로 합니다.
- **멀티모달 확장:** 로드맵에 따라 썸네일, UI 스크린샷 등을 분석해야 할 경우, `CLIP` 또는 `Gemma-Image-2`와 같은 Vision-Language 모델을 추가로 도입합니다.

### 나. 구현 (`embedding.py`)
`LlamaIndex`의 `HuggingFaceEmbedding` 클래스를 사용하여 로컬 또는 Hugging Face Hub의 모델을 쉽게 연동할 수 있습니다.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# GPU 사용 가능 시 'cuda:0', 아닐 경우 'cpu'
Settings.device = 'cuda:0' 
# 전역 설정에 임베딩 모델 지정
Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

# 이제부터 nodes를 생성하거나 벡터 인덱스를 구축할 때
# 별도 지정 없이 Settings.embed_model이 자동으로 사용됩니다.
```

## 2.2. 벡터 데이터베이스 (Vector Database)

임베딩된 벡터와 메타데이터를 저장하고 관리하기 위한 데이터베이스입니다.

### 가. DB 선택: Qdrant
- **선택 이유:**
  - **고성능:** Rust로 작성되어 대규모 데이터셋에서도 낮은 지연 시간(low-latency)과 높은 QPS(Queries Per Second)를 보장합니다.
  - **고급 필터링:** 텍스트 유사도 검색(Vector search)과 메타데이터 기반 필터링(e.g., `price < 10 AND has_api = true`)을 결합한 하이브리드 검색을 효율적으로 지원합니다.
  - **다양한 인덱싱:** HNSW, IVF 등 다양한 벡터 인덱싱 알고리즘을 지원하여 데이터 특성에 맞게 최적화할 수 있습니다.
  - **운영 용이성:** 공식 Docker 이미지를 제공하여 로컬 개발 및 프로덕션 배포가 간편합니다.

### 나. 구현 (`vector_store.py`)
`LlamaIndex`는 `QdrantVectorStore`와 같은 다양한 벡터 DB와의 연동을 지원합니다.

```python
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import qdrant_client

# 1. Qdrant 클라이언트 초기화 (Docker로 실행된 Qdrant 서버에 연결)
client = qdrant_client.QdrantClient(host="localhost", port=6333)

# 2. Vector Store 인스턴스 생성
vector_store = QdrantVectorStore(client=client, collection_name="ai_tools")

# 3. 정규화된 nodes와 vector_store를 사용하여 인덱스 구축
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

## 2.3. Retrieval → Re-rank 파이프라인

정확한 답변 생성을 위해, 관련성 높은 정보를 빠짐없이, 그리고 순서대로 LLM에 전달하는 것이 중요합니다.

### 가. 1차 검색 (Retrieval)
- **방식:** HNSW(Hierarchical Navigable Small World) 알고리즘을 사용한 벡터 유사도 검색.
- **설정:** `top_k = 20`. 사용자의 질문과 의미적으로 유사한 상위 20개의 청크를 후보군으로 빠르게 필터링합니다. 이는 재현율(recall)을 높이기 위한 전략입니다.

### 나. 2차 재정렬 (Re-ranking)
- **필요성:** 1차 검색은 의미적 유사성에만 기반하므로, 때로는 질문의 핵심 의도와 미묘하게 다른 문서를 상위에 노출시킬 수 있습니다. Cross-Encoder를 사용한 재정렬은 이러한 '노이즈'를 제거하고 정확도(precision)를 높입니다.
- **모델 선택:** **`Cohere/rerank-english-v3.0`** 또는 `bge-reranker-large`
  - `Cohere` 모델은 상용 API로 제공되어 사용이 간편하고 성능이 뛰어나며, `bge-reranker`는 로컬 GPU에서 직접 호스팅하여 비용을 절감할 수 있습니다.
- **구현 (`retriever.py`):**
  `LlamaIndex`의 `VectorIndexRetriever`와 `CohereRerank`를 조합하여 파이프라인을 구성합니다.

```python
from llama_index.core.vector_stores import VectorStoreInfo
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 1. Retriever 설정 (1차 검색)
retriever = VectorIndexRetriever(index=index, similarity_top_k=20)

# 2. Reranker 설정 (2차 재정렬)
# COHERE_API_KEY 환경변수 설정 필요
reranker = CohereRerank(top_n=5) # 최종 5개 문서만 선택

# 3. QueryEngine 또는 Retriever에 Postprocessor로 연결
query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker]
)
```

### 다. 후처리 필터링 (Post-filtering)
- 재정렬된 결과에 대해, 사용자가 명시한 메타데이터 조건(가격, 라이선스 등)을 AND 연산으로 최종 필터링합니다. Qdrant의 필터링 기능을 활용하면 이 과정을 검색 단계에서 효율적으로 처리할 수 있습니다.

---
> **다음 문서:** [03_Generation_and_API.md](./03_Generation_and_API.md) 에서는 재정렬된 최종 문서를 바탕으로 LLM이 어떻게 답변을 생성하고, 이를 API로 제공하는지에 대해 다룹니다. 