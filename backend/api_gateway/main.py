from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import re
from contextlib import asynccontextmanager
from llama_index.core.prompts import PromptTemplate
import httpx
from ..parser.parser import parse_and_chunk_single_url

# CORS 미들웨어 추가 (FastAPI 인스턴스 초기화 후로 이동)
origins = [
    "http://localhost:3000",  # 프런트엔드 서버의 주소
    "http://127.0.0.1:3000", # 프런트엔드 서버의 주소 (IP 직접 명시)
    "http://localhost:3001", # 프런트엔드 서버의 새 포트
    "http://127.0.0.1:3001", # 프런트엔드 서버의 새 포트 (IP 직접 명시)
    "http://172.30.1.26:3001", # 사용자 환경의 IP 주소 추가
    "null"  # file:// 로컬 파일 열기 시 출처가 null로 인식될 수 있음
]

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "open-solar-ko-10.7b")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:1234/v1")

# 데이터 및 인덱스 로드 경로
DATA_DIR = os.getenv("DATA_DIR", "./processed_data_storage")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_ai_tools_metadata.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "ai_tools_faiss_index.bin")

# 전역 변수로 인덱스와 메타데이터를 저장
faiss_index = None
processed_metadata = None
embedding_model = None

# 메타데이터 추출 도우미 함수 (parser.py에서 복사 및 수정)
def _extract_metadata_from_text(text):
    metadata = {
        "price": None,
        "has_api": False,
        "category": [],
        "tags": []
    }

    price_match = re.search(r'\$([0-9.]+)(/mo|/month|/yr|/year| per month| per year)?', text, re.IGNORECASE)
    if price_match:
        metadata["price"] = float(price_match.group(1))

    if re.search(r'API|developer sdk|rest api', text, re.IGNORECASE):
        metadata["has_api"] = True
        metadata["tags"].append("api_enabled")

    if re.search(r'image generation|art generator|image creator', text, re.IGNORECASE):
        metadata["category"].append("Image Generation")
        metadata["tags"].append("image")
        metadata["tags"].append("art")
    if re.search(r'text generation|chatbot|writing assistant|content creation', text, re.IGNORECASE):
        metadata["category"].append("Text Generation")
        metadata["tags"].append("text")
        metadata["tags"].append("writing")
    if re.search(r'video editing|video creation', text, re.IGNORECASE):
        metadata["category"].append("Video Editing")
        metadata["tags"].append("video")
    if re.search(r'audio generation|speech to text|text to speech', text, re.IGNORECASE):
        metadata["category"].append("Audio")
        metadata["tags"].append("audio")
    if re.search(r'developer tools|code generation|sdk', text, re.IGNORECASE):
        metadata["category"].append("Developer Tools")
        metadata["tags"].append("dev")
        metadata["tags"].append("code")
    if re.search(r'productivity|workflow automation|task management', text, re.IGNORECASE):
        metadata["category"].append("Productivity")
        metadata["tags"].append("productivity")
        metadata["tags"].append("management")
    if re.search(r'business intelligence|analytics|data visualization', text, re.IGNORECASE):
        metadata["category"].append("Business Intelligence")
        metadata["tags"].append("data")
        metadata["tags"].append("analytics")

    metadata["tags"] = list(set(metadata["tags"])) # 중복 제거

    return metadata

@asynccontextmanager
async def lifespan(app: FastAPI):
    global faiss_index, processed_metadata, embedding_model
    
    logger.info("API Gateway 시작: 임베딩 모델 로드 및 기존 FAISS 인덱스 로드 시도 중...")
    
    # 임베딩 모델 로드 (FAISS 검색에 필요)
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer 임베딩 모델 로드 완료.")
    except Exception as e:
        logger.error(f"SentenceTransformer 모델 로드 중 오류 발생: {e}")
        embedding_model = None
        # 모델 로드 실패 시, 서비스가 정상 작동하지 않을 수 있으므로 에러를 발생시킬 수도 있음
        # raise RuntimeError("임베딩 모델 로드 실패") from e

    # FAISS 인덱스 로드
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"기존 FAISS 인덱스 로드 완료. 벡터 수: {faiss_index.ntotal}")
        except Exception as e:
            logger.error(f"기존 FAISS 인덱스 로드 중 오류 발생: {e}")
            faiss_index = None # 로드 실패 시 None으로 설정
    else:
        logger.warning(f"기존 FAISS 인덱스 파일이 없습니다: {FAISS_INDEX_PATH}. 새로운 URL 추가 시 인덱스가 생성됩니다.")

    # 메타데이터 로드
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            processed_metadata = pd.read_json(PROCESSED_DATA_PATH, orient="records")
            logger.info(f"기존 전처리된 메타데이터 로드 완료. 항목 수: {len(processed_metadata)}")
        except Exception as e:
            logger.error(f"기존 메타데이터 로드 중 오류 발생: {e}")
            processed_metadata = None # 로드 실패 시 None으로 설정
    else:
        logger.warning(f"기존 전처리된 메타데이터 파일이 없습니다: {PROCESSED_DATA_PATH}. 새로운 URL 추가 시 메타데이터가 생성됩니다.")

    if embedding_model is None:
        logger.error("임베딩 모델 로드에 실패하여 검색 기능이 작동하지 않습니다.")
    
    # 초기 로드에 실패했더라도, process_url에서 동적으로 생성/업데이트 가능하도록 None 허용
    if faiss_index is None:
        logger.info("FAISS 인덱스를 초기화합니다. 첫 번째 URL 추가 시 생성됩니다.")
        # 임베딩 모델이 로드되었다면, 임베딩 차원을 사용하여 빈 인덱스 초기화
        if embedding_model:
            # 임시로 더미 임베딩을 생성하여 차원 확인
            dummy_embedding = embedding_model.encode(["dummy"]).flatten()
            faiss_index = faiss.IndexFlatL2(dummy_embedding.shape[0])
            logger.info(f"빈 FAISS 인덱스 초기화 완료 (차원: {dummy_embedding.shape[0]}).")
        else:
            logger.error("임베딩 모델이 없어서 FAISS 인덱스를 초기화할 수 없습니다.")

    if processed_metadata is None:
        logger.info("메타데이터 DataFrame을 초기화합니다. 첫 번째 URL 추가 시 채워집니다.")
        processed_metadata = pd.DataFrame()
    
    yield # 어플리케이션이 실행되는 동안 유지
    logger.info("API Gateway 종료: 리소스 해제 중...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 모든 일반적인 HTTP 메서드 및 OPTIONS 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# LLM 설정 (LM Studio 로컬 서버 사용) - httpx.AsyncClient 직접 사용
llm_http_client = httpx.AsyncClient(base_url=LLM_API_BASE, trust_env=False, timeout=httpx.Timeout(60.0))

# Prompt Templates
SUMMARY_PROMPT_TMPL = """당신은 AI 도구 전문 분석가입니다. 아래에 제공된 검색 결과를 바탕으로, 사용자의 질문에 가장 적합한 AI 도구를 요약하고 추천해 주세요. 관련 없는 정보는 포함하지 마세요. 답변은 명확하고 간결하게 작성해주세요.

<검색 결과>
{context_str}

사용자 질문: {query_str}

답변: 
"""
SUMMARY_PROMPT = PromptTemplate(SUMMARY_PROMPT_TMPL)

COMPARISON_PROMPT_TMPL = """당신은 AI 도구 전문 분석가입니다. 아래에 제공된 검색 결과를 바탕으로, "{tool_A}"와 "{tool_B}"를 다음 기준에 따라 표 형식으로 비교 분석해 주세요.

- 주요 기능
- 가격 정책 (월간 구독료 기준)
- API 제공 여부
- 핵심 타겟 사용자

<검색 결과>
{context_str}

사용자 질문: {query_str}

<분석 결과 (Markdown 표 형식)>
"""
COMPARISON_PROMPT = PromptTemplate(COMPARISON_PROMPT_TMPL)

# 버전별 카테고리/태그 매핑
VERSION_MAP = {
    "ai": {"categories": ["Natural Language Processing", "Image Generation", "AI & ML", "Audio", "Video Editing"], "tags": ["ai", "ml", "generative", "nlp", "image", "art", "speech", "transcription"]},
    "dev": {"categories": ["Developer Tools", "Code Generation", "DevOps Tools", "Business Intelligence", "Project Management"], "tags": ["coding", "developer", "devops", "api", "testing", "management", "python", "infrastructure"]},
    "office": {"categories": ["Productivity", "Business", "Communication", "Project Management", "Design Tools", "Writing Tools"], "tags": ["office", "productivity", "collaboration", "meeting", "management", "design", "notes", "organization", "writing", "documents"]}
}

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    response_format: str = "summary"
    tool_A: Optional[str] = None
    tool_B: Optional[str] = None
    version: str = "ai" # 새 버전 파라미터 추가 (기본값: ai)

class SearchResponse(BaseModel):
    request_id: str
    data: Dict[str, Any]
    retrieved_sources: List[Dict[str, str]]

class ProcessURLRequest(BaseModel):
    url: str

@app.post("/api/v1/search", response_model=SearchResponse)
async def handle_search(request: SearchRequest):
    from uuid import uuid4
    request_id = str(uuid4())

    if faiss_index is None or processed_metadata is None or embedding_model is None:
        logger.error("검색 기능에 필요한 FAISS 인덱스, 메타데이터 또는 임베딩 모델이 로드되지 않았습니다.")
        raise HTTPException(status_code=500, detail="서버가 아직 준비되지 않았거나 데이터 로드에 실패했습니다. 관리자에게 문의하세요.")

    # 버전별 필터링 조건 가져오기
    selected_version_filters = VERSION_MAP.get(request.version, VERSION_MAP["ai"]) # 기본값은 'ai'
    allowed_categories = selected_version_filters["categories"]
    allowed_tags = selected_version_filters["tags"]

    # 쿼리 임베딩
    query_embedding = embedding_model.encode([request.query])
    
    # FAISS를 사용하여 유사한 벡터 검색 (전체 인덱스에서 검색)
    # k는 전체 인덱스 크기로 설정하여 모든 후보군을 가져옵니다.
    # FAISS 문서에 따르면, IndexFlatL2는 ntotal이 max k가 될 수 있음
    k_search = faiss_index.ntotal if faiss_index.ntotal > 0 else 1 # 인덱스에 벡터가 없을 경우 1로 설정
    D, I = faiss_index.search(query_embedding.astype('float32'), k=k_search)
    
    # 검색된 문서의 인덱스를 사용하여 원본 메타데이터 가져오기
    candidate_items = []
    for doc_idx in I[0]:
        if doc_idx != -1 and doc_idx < len(processed_metadata): # 유효한 인덱스 확인
            candidate_items.append(processed_metadata.iloc[doc_idx].to_dict())

    # 버전별 필터링 적용
    filtered_items = []
    for item in candidate_items:
        # item.get("category")가 리스트 또는 문자열일 수 있으므로 유연하게 처리
        item_categories = []
        raw_categories = item.get("category")
        if isinstance(raw_categories, list):
            item_categories = [c.lower() for c in raw_categories]
        elif isinstance(raw_categories, str):
            item_categories = [c.lower() for c in raw_categories.split(", ")]
        
        item_tags = [t.lower() for t in item.get("tags", [])]

        category_match = any(cat.lower() in [ac.lower() for ac in allowed_categories] for cat in item_categories)
        tag_match = any(tag.lower() in [at.lower() for at in allowed_tags] for tag in item_tags)
        
        # 카테고리 또는 태그 중 하나라도 일치하면 포함
        if category_match or tag_match:
            filtered_items.append(item)
            
    # 필터링된 항목 모두 사용 (더 이상 제한 없음)
    retrieved_items = filtered_items

    if not retrieved_items:
        logger.info(f"질의 '{request.query}'에 대한 검색 결과가 없습니다.")
        # 검색 결과가 없는 경우, LLM 호출 없이 응답 반환
        return SearchResponse(
            request_id=request_id,
            data={"summary": "요청하신 내용에 대한 검색 결과를 찾을 수 없습니다.", "comparison": None},
            retrieved_sources=[]
        )

    # LLM에 전달할 context_str 생성
    context_str = "\n---\n".join([
        f"도구 이름: {item.get('name', 'N/A')}\n설명: {item.get('description', 'N/A')}\nURL: {item.get('url', 'N/A')}\n카테고리: {item.get('category', [])}\n태그: {item.get('tags', [])}"
        for item in retrieved_items
    ])

    llm_response_content = ""
    if request.response_format == "summary":
        prompt = SUMMARY_PROMPT.format(context_str=context_str, query_str=request.query)
    elif request.response_format == "comparison" and request.tool_A and request.tool_B:
        prompt = COMPARISON_PROMPT.format(context_str=context_str, query_str=request.query, tool_A=request.tool_A, tool_B=request.tool_B)
    else:
        raise HTTPException(status_code=400, detail="잘못된 응답 형식 또는 비교 도구 이름이 누락되었습니다.")

    try:
        llm_response = await llm_http_client.post(
            "/chat/completions",
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
        )
        llm_response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        llm_response_content = llm_response.json()["choices"][0]["message"]["content"]
    except httpx.RequestError as e:
        logger.error(f"LLM API 요청 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 서비스에 연결할 수 없습니다: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API 응답 오류: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail=f"LLM 서비스 응답 오류: {e.response.status_code}")
    except Exception as e:
        logger.error(f"LLM 응답 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 응답 처리 중 알 수 없는 오류 발생: {e}")

    retrieved_sources_list = [
        {"name": item.get("name", "N/A"), "url": item.get("url", "N/A"), "description": item.get("description", "N/A")}
        for item in retrieved_items
    ]

    return SearchResponse(
        request_id=request_id,
        data={
            "summary": llm_response_content if request.response_format == "summary" else None,
            "comparison": llm_response_content if request.response_format == "comparison" else None
        },
        retrieved_sources=retrieved_sources_list
    )

@app.post("/api/v1/process_url")
async def process_url_endpoint(request: ProcessURLRequest):
    global faiss_index, processed_metadata, embedding_model
    from uuid import uuid4
    request_id = str(uuid4())

    if embedding_model is None:
        logger.error("임베딩 모델이 로드되지 않아 URL 처리를 할 수 없습니다.")
        raise HTTPException(status_code=500, detail="서버가 아직 준비되지 않았거나 임베딩 모델 로드에 실패했습니다. 관리자에게 문의하세요.")

    try:
        # URL에서 HTML 콘텐츠 가져오기
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request.url)
            response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
            html_content = response.text

        # parse_and_chunk_single_url 함수를 사용하여 파싱 및 청크 생성
        new_metadata_records, new_faiss_vectors = parse_and_chunk_single_url(request.url, html_content, embedding_model)

        if not new_metadata_records:
            logger.warning(f"URL {request.url}에서 파싱된 데이터가 없습니다.")
            raise HTTPException(status_code=400, detail="제공된 URL에서 유효한 콘텐츠를 파싱할 수 없습니다.")

        # 기존 메타데이터에 새 레코드 추가
        new_df = pd.DataFrame(new_metadata_records)
        if processed_metadata.empty:
            processed_metadata = new_df
        else:
            processed_metadata = pd.concat([processed_metadata, new_df], ignore_index=True)
        logger.info(f"새로운 메타데이터 {len(new_metadata_records)}개 추가 완료. 총 메타데이터 수: {len(processed_metadata)}")

        # FAISS 인덱스 업데이트
        if new_faiss_vectors:
            if faiss_index.ntotal == 0: # 인덱스가 비어있는 경우 초기화
                d = new_faiss_vectors[0].shape[0]
                faiss_index = faiss.IndexFlatL2(d)
            faiss_index.add(np.array(new_faiss_vectors).astype('float32'))
            logger.info(f"새로운 FAISS 벡터 {len(new_faiss_vectors)}개 추가 완료. 총 벡터 수: {faiss_index.ntotal}")

        # 업데이트된 FAISS 인덱스와 메타데이터를 파일로 저장 (선택 사항이지만 영속성을 위해 권장)
        os.makedirs(DATA_DIR, exist_ok=True)
        if faiss_index and faiss_index.ntotal > 0:
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            logger.info(f"FAISS 인덱스가 {FAISS_INDEX_PATH}에 저장되었습니다.")
        if not processed_metadata.empty:
            processed_metadata.to_json(PROCESSED_DATA_PATH, orient="records", indent=4, force_ascii=False)
            logger.info(f"메타데이터가 {PROCESSED_DATA_PATH}에 저장되었습니다.")

        return {"request_id": request_id, "message": f"URL '{request.url}'이(가) 성공적으로 처리되고 인덱싱되었습니다.", "parsed_items_count": len(new_metadata_records)}

    except httpx.RequestError as e:
        logger.error(f"URL '{request.url}' 가져오기 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"URL을 가져올 수 없습니다: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"URL '{request.url}' 응답 오류: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"URL 응답 오류: {e.response.status_code}")
    except Exception as e:
        logger.error(f"URL '{request.url}' 처리 중 알 수 없는 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"URL 처리 중 알 수 없는 오류 발생: {e}")

@app.post("/api/v1/parse_and_ingest")
async def parse_and_ingest_data_endpoint():
    return {"message": "이 엔드포인트는 더 이상 사용되지 않습니다. /api/v1/process_url을 사용해주세요.", "status": "deprecated"}

@app.get("/health")
async def health_check():
    health_status = {
        "status": "ok",
        "faiss_index_loaded": faiss_index is not None and faiss_index.ntotal > 0,
        "processed_metadata_loaded": processed_metadata is not None and not processed_metadata.empty,
        "embedding_model_loaded": embedding_model is not None
    }
    if not health_status["faiss_index_loaded"] or not health_status["processed_metadata_loaded"] or not health_status["embedding_model_loaded"]:
        health_status["status"] = "degraded"
        health_status["message"] = "일부 컴포넌트가 로드되지 않아 검색 기능이 제한될 수 있습니다."
    return health_status

# 개발 편의를 위한 메인 실행 블록
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True, log_level="info") # log_level을 "info"로 설정하여 상세 로그 확인 