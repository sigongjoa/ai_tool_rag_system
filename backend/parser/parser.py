import os
from bs4 import BeautifulSoup
# from llama_index.core import SimpleDirectoryReader # SimpleDirectoryReader 제거
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
import json
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 전역 변수 정의
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_html_dir = os.path.join(BASE_DIR, "..", "raw_html_storage")
processed_data_dir = os.path.join(BASE_DIR, "..", "processed_data_storage")

# 임베딩 모델 전역 로드 (최초 1회만 로드, main.py에서 전달받도록 변경)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_metadata(text: str) -> Dict[str, Any]:
    metadata = {
        "price": None,
        "has_api": False,
        "category": [],
        "tags": []
    }

    # 가격 정보 추출 (간단한 예시, 복잡한 패턴은 추가 필요)
    price_match = re.search(r'\$([0-9.]+)(/mo|/month|/yr|/year| per month| per year)?', text, re.IGNORECASE)
    if price_match:
        metadata["price"] = float(price_match.group(1))

    # API 제공 여부
    if re.search(r'API|developer sdk|rest api', text, re.IGNORECASE):
        metadata["has_api"] = True
        metadata["tags"].append("api_enabled")

    # 카테고리 및 태그 (키워드 기반)
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

    # 중복 태그 제거
    metadata["tags"] = list(set(metadata["tags"]))

    return metadata

def parse_and_chunk_single_url(url: str, html_content: str, embedding_model: SentenceTransformer) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    processed_metadata_records = []
    faiss_vectors = []

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        full_text = soup.get_text(separator=' ', strip=True)
        
        name = soup.title.string if soup.title else url # 웹 페이지 제목이 없으면 URL 사용
        
        # 메타 설명 추출 또는 텍스트의 첫 부분 사용
        site_description = None
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and meta_description.get('content'):
            site_description = meta_description['content'].strip()
        elif len(full_text) > 200: # 충분한 텍스트가 있을 경우 첫 200자 사용
            site_description = full_text[:200].strip() + "..."
        else:
            site_description = full_text.strip() # 텍스트가 짧으면 전체 사용

        doc = Document(
            text=full_text,
            extra_info={
                "name": name,
                "url": url,
                "source_path": url, # 원본 URL을 source_path로 사용
                "collected_at": datetime.now().isoformat() + "Z",
                "description": site_description # 추출된 설명 추가
            }
        )

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents([doc])
        
        for node in nodes:
            extracted_metadata = extract_metadata(node.text) # 노드 텍스트 기반 메타데이터 추출
            
            node.metadata.update({
                "name": doc.extra_info.get("name", "N/A"),
                "url": doc.extra_info.get("url", "N/A"),
                "description": doc.extra_info.get("description", "N/A"), # doc의 설명 사용
                "category": extracted_metadata.get("category", []),
                "tags": extracted_metadata.get("tags", []), # 노드 텍스트 기반 태그 사용
                "source": doc.extra_info.get("url", "N/A"),
                "collected_at": doc.extra_info.get("collected_at", "N/A")
            })
            
            record = {
                "name": node.metadata.get("name", "N/A"),
                "description": node.metadata.get("description", "N/A"), # 노드의 설명 사용
                "url": node.metadata.get("url", "N/A"),
                "category": node.metadata.get("category", []),
                "tags": node.metadata.get("tags", []),
                "source": node.metadata.get("source", "N/A"),
                "collected_at": node.metadata.get("collected_at", "N/A")
            }
            processed_metadata_records.append(record)

            embedding = embedding_model.encode([node.text]).flatten()
            faiss_vectors.append(embedding)

    except Exception as e:
        print(f"URL 파싱 및 청크 생성 중 오류 발생 {url}: {e}")
        return [], [] # 오류 발생 시 빈 리스트 반환

    return processed_metadata_records, faiss_vectors

# 이 부분은 더 이상 직접 실행되지 않으므로 제거합니다.
# if __name__ == "__main__":
#     parse_and_chunk() 