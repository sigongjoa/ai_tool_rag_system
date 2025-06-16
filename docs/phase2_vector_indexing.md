# Phase 2: 벡터 인덱스 및 검색 전략 고도화

## 목표
FAISS 인덱스 구조를 개선하고, URL 정보를 활용한 Reranking을 도입하여 검색 정확도를 향상시킵니다.

## 2-1. 인코딩 대상 변경

### 목표
FAISS에 벡터를 저장할 때, 웹페이지 원문 대신 Phase 1에서 LLM으로 정제한 `description` 및 `usage_examples` 문자열을 임베딩하여 저장합니다. 이를 통해 노이즈를 줄이고 의미적으로 중요한 정보에 집중할 수 있습니다.

### 구현 상세
`backend/parser/parser.py`의 `parse_and_chunk_single_url` 함수 내에서 `faiss_vectors`를 생성하는 부분을 수정합니다.

**`backend/parser/parser.py` 수정 예정 사항:**
- 현재 `chunk_text`를 임베딩하고 있습니다. 여기서 LLM 추출 메타데이터에 포함된 `description`과 `usage_examples`를 결합하여 임베딩하는 로직을 고려합니다. 각 청크는 특정 도구의 설명과 사용 예시 문맥 내에 존재하므로, `chunk_text` 자체에 이미 이 정보가 잘 포함되어 있습니다. 따라서, `chunk_text`를 임베딩하는 현재 로직은 유효합니다.
- 만약, `usage_examples`가 별도의 필드로 분리되어 있고, 이를 각 청크와 함께 임베딩해야 한다면, 청크 텍스트 생성 시 이 정보를 포함하거나, 별도의 벡터를 생성하여 관리하는 방안을 고려할 수 있습니다. 현재 `chunk_text`에 HTML 본문 내용이 담기므로, 본문 내 `description`이나 `usage_examples` 관련 내용이 잘 포함된다면 추가 수정이 필요 없을 수 있습니다. (검토 후 결정)

## 2-2. URL 임베딩 및 메타데이터 확장

### 목표
도구의 URL 자체 또는 도메인명을 별도로 임베딩하여 `processed_metadata`에 `url_vector`와 같은 필드로 저장합니다. 이는 특정 사이트 관련 질문에 대한 가중치를 부여하는 데 사용됩니다.

### 구현 상세
`backend/parser/parser.py`의 `parse_and_chunk_single_url` 함수와 `backend/api_gateway/main.py`에서 `processed_metadata`를 처리하는 부분을 수정합니다.

**`backend/parser/parser.py` 수정 예정 사항:**
- `parse_and_chunk_single_url` 함수 내에서 각 `record`를 생성할 때, `url` 또는 `url`에서 파싱한 도메인(domain)에 대한 임베딩 벡터를 생성하여 `url_vector` 필드에 추가합니다.
  - `url_vector`는 `embedding_model.encode([record["url"]]).flatten()` 또는 `embedding_model.encode([domain_name]).flatten()` 형태로 생성될 수 있습니다.
  - 이렇게 생성된 `url_vector`는 `processed_metadata_records`의 각 레코드에 포함됩니다.

**`backend/api_gateway/main.py` 수정 예정 사항:**
- `SearchResponse` 모델에 `url_vector` 필드를 추가할 필요는 없지만, 검색 로직(`handle_search`)에서 `processed_metadata`를 로드하고 `url_vector`를 사용할 수 있도록 준비합니다.

## 2-3. Reranking 로직 도입

### 목표
`/api/v1/search` 엔드포인트에서 FAISS로 1차 검색을 수행한 후, 검색된 결과에 대해 Reranking을 수행합니다. Reranking 점수는 `(기존 유사도 점수) + (사용자 질문 벡터와 URL 벡터의 코사인 유사도 * 가중치)`와 같은 방식으로 계산하여, 질문과 연관성이 높은 도메인의 순위를 높여줍니다.

### 구현 상세
`backend/api_gateway/main.py`의 `handle_search` 함수를 수정합니다.

**`backend/api_gateway/main.py` 수정 예정 사항:**
- **URL 벡터 로드:** `handle_search` 함수 내에서 `processed_metadata`에서 각 도구의 `url_vector`를 가져올 수 있도록 합니다. (만약 `url_vector`를 별도의 FAISS 인덱스로 관리한다면 해당 인덱스도 로드)
- **사용자 질문 URL 임베딩:** 사용자 쿼리(`request.query`)를 임베딩할 때, 쿼리에서 도메인 정보나 특정 URL을 언급하는 부분이 있다면 이를 파싱하여 별도로 임베딩합니다. (초기에는 질문 전체를 임베딩한 벡터를 사용)
- **Reranking 계산:**
  - FAISS `search` 결과로 얻은 `D` (거리)와 `I` (인덱스)를 사용하여 `candidate_items`를 구성합니다.
  - 각 `candidate_item`에 대해 해당 도구의 `url_vector`와 사용자 질문의 임베딩 벡터 간의 코사인 유사도를 계산합니다.
  - 기존 FAISS 거리(`D`)를 유사도(`1 - D/max_distance` 또는 `-D`)로 변환하고, 여기에 URL 유사도에 가중치(`URL_SIM_WEIGHT`)를 곱하여 더한 새로운 점수를 계산합니다.
  - 계산된 새 점수를 기준으로 `filtered_items`를 다시 정렬합니다.
- **가중치 정의:** `URL_SIM_WEIGHT`와 같은 상수를 `main.py` 상단에 정의하여 가중치를 조절할 수 있도록 합니다. 