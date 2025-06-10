import os
import json
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode, Document
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def index_data():
    processed_data_dir = "processed_data_storage" # parser에서 저장한 청크 파일 경로
    processed_chunks_filepath = os.path.join(processed_data_dir, "processed_chunks.json")

    if not os.path.exists(processed_chunks_filepath):
        print(f"오류: 처리된 청크 파일이 없습니다. {processed_chunks_filepath}를 확인해주세요.")
        return

    # 1. 임베딩 모델 설정 (GTE-Large)
    Settings.device = 'cpu' # 로컬 CPU 환경을 가정
    Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large", device="cpu")
    print(f"임베딩 모델 로드 완료: {Settings.embed_model.model_name}")

    # 2. SimpleVectorStore 초기화 및 로드
    vector_storage_path = "vector_storage"
    os.makedirs(vector_storage_path, exist_ok=True)

    # default__vector_store.json 파일이 존재하지 않을 경우 새로 생성하도록 변경
    vector_store_file = os.path.join(vector_storage_path, "default__vector_store.json")
    if os.path.exists(vector_store_file):
        vector_store = SimpleVectorStore.from_persist_dir(persist_dir=vector_storage_path)
        print("기존 SimpleVectorStore 로드 완료")
    else:
        vector_store = SimpleVectorStore()
        print("새 SimpleVectorStore 생성 완료")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("SimpleVectorStore 설정 완료")

    # 3. 청크 파일 로드 및 인덱싱
    documents = [] # nodes 대신 documents 리스트를 사용
    with open(processed_chunks_filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f) # JSON 파일 전체를 로드
        for data in json_data: # 배열의 각 항목을 순회
            # TextNode 대신 Document 객체 생성
            doc = Document(
                id_=data["id"],
                text=data["text"],
                metadata=data["metadata"]
            )
            documents.append(doc)

    if not documents: # nodes 대신 documents 사용
        print("로드할 문서(청크)가 없습니다. 인덱싱을 건너뜝니다.")
        return

    print(f"총 {len(documents)}개의 문서(청크) 로드 완료. SimpleVectorStore에 인덱싱 시작...") # 메시지 수정
    
    try:
        # VectorStoreIndex.from_documents에 Document 객체 리스트 전달
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        storage_context.persist(persist_dir=vector_storage_path)
        print("SimpleVectorStore 인덱싱 및 저장 완료!")
    except Exception as e:
        print(f"SimpleVectorStore 인덱싱 중 오류 발생: {e}")

def build_faiss_index(df: pd.DataFrame):
    """
    DataFrame의 'description' 컬럼을 임베딩하고 FAISS 인덱스를 구축합니다.
    """
    print("FAISS 인덱스 구축을 시작합니다...")
    # 모델 로드 (캐시 경로 설정)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("문장 임베딩 모델을 로드했습니다.")

    # 'description' 컬럼이 문자열이 아닐 경우 빈 문자열로 처리
    descriptions = df["description"].fillna("").tolist()

    embs = model.encode(descriptions, show_progress_bar=True)
    print(f"총 {len(embs)}개의 임베딩을 생성했습니다. 임베딩 차원: {embs.shape[1]}")

    d = embs.shape[1]  # 임베딩 차원
    idx = faiss.IndexFlatIP(d)  # Inner Product (내적) 기반 인덱스 생성
    idx.add(np.array(embs).astype('float32'))  # numpy 배열로 변환하여 인덱스에 추가 (float32 타입으로)
    print("FAISS 인덱스 구축을 완료했습니다.")
    return idx

def save_faiss_index(index, path: str):
    """
    FAISS 인덱스를 지정된 경로에 저장합니다.
    """
    faiss.write_index(index, path)
    print(f"FAISS 인덱스를 {path}에 저장했습니다.")

def load_faiss_index(path: str):
    """
    지정된 경로에서 FAISS 인덱스를 로드합니다.
    """
    index = faiss.read_index(path)
    print(f"FAISS 인덱스를 {path}에서 로드했습니다.")
    return index

if __name__ == "__main__":
    # 테스트를 위한 더미 데이터 (processor.py의 출력 형태를 모방)
    dummy_processed_data = [
        {
            "name": "ai tool a",
            "description": "this is a fantastic ai tool for natural language processing.",
            "url": "http://example.com/ai-tool-a",
            "category": "natural language processing",
            "tags": ["nlp", "ai", "language"],
            "source": "product hunt",
            "collected_at": "2024-05-20T10:00:00Z"
        },
        {
            "name": "ai image generator x",
            "description": "generate amazing images from text descriptions with this ai tool.",
            "url": "http://example.com/ai-image-generator-x",
            "category": "image generation",
            "tags": ["image", "ai", "art"],
            "source": "product hunt",
            "collected_at": "2024-05-20T10:05:00Z"
        },
        {
            "name": "ai assistant y",
            "description": "a smart ai assistant to help you with your daily tasks.",
            "url": "http://example.com/ai-assistant-y",
            "category": "productivity",
            "tags": ["assistant", "ai", "task management"],
            "source": "product hunt",
            "collected_at": "2024-05-20T10:10:00Z"
        },
        {
            "name": "ai code helper z",
            "description": "this ai tool helps developers write cleaner and more efficient code.",
            "url": "http://example.com/ai-code-helper-z",
            "category": "developer tools",
            "tags": ["coding", "ai", "development"],
            "source": "product hunt",
            "collected_at": "2024-05-20T10:15:00Z"
        },
        {
            "name": "ai music composer m",
            "description": "compose unique musical pieces with the power of ai.",
            "url": "http://example.com/ai-music-composer-m",
            "category": "music & audio",
            "tags": ["music", "ai", "composition"],
            "source": "product hunt",
            "collected_at": "2024-05-20T10:20:00Z"
        }
    ]

    dummy_df = pd.DataFrame(dummy_processed_data)
    faiss_index = build_faiss_index(dummy_df)
    print(f"FAISS 인덱스에 추가된 벡터 수: {faiss_index.ntotal}")

    # 인덱스 저장 및 로드 테스트
    index_path = "test_faiss_index.bin"
    save_faiss_index(faiss_index, index_path)
    loaded_index = load_faiss_index(index_path)
    print(f"로드된 FAISS 인덱스 벡터 수: {loaded_index.ntotal}")

    # 테스트 후 파일 정리
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"{index_path} 파일이 삭제되었습니다.") 