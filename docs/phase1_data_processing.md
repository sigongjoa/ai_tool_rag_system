# Phase 1: 데이터 수집 및 전처리 파이프라인 강화

## 목표
현재의 정규식 기반 메타데이터 추출 방식을 LLM을 활용한 방식으로 전환하여, 데이터의 품질과 정밀도를 극대화합니다.

## 1-1. LLM 기반 정보 추출기 구현

### JSON 스키마 확정
LLM이 추출해야 할 AI 도구의 핵심 정보를 명확히 정의한 JSON 스키마입니다. 이 스키마는 LLM의 함수 호출(Function Calling) 기능을 활용하여 안정적인 파싱을 목표로 합니다.

```json
{
  "type": "function",
  "function": {
    "name": "extract_tool_info",
    "description": "Extracts key information about an AI tool from a given webpage text.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "The official name of the AI tool."
        },
        "url": {
          "type": "string",
          "description": "The URL of the AI tool's official page."
        },
        "tags": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Descriptive tags for the AI tool (e.g., #RAG, #ImageGeneration, #DeveloperTools)."
        },
        "description": {
          "type": "string",
          "description": "A concise summary of the AI tool's purpose, key features, and benefits."
        },
        "usage_examples": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Specific examples of how the tool can be used, demonstrating its functionality."
        },
        "price": {
          "type": "number",
          "description": "Estimated monthly price in USD, if available. Null if free or not specified."
        },
        "has_api": {
          "type": "boolean",
          "description": "True if the tool provides an API or SDK for programmatic access, false otherwise."
        },
        "category": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Broad categories the tool belongs to (e.g., Image Generation, Text Generation, Developer Tools)."
        }
      },
      "required": ["name", "url", "description"]
    }
  }
}
```

### 프롬프트 설계
위 JSON 스키마에 맞춰 정보를 추출하도록 LLM에 지시하는 프롬프트 예시입니다. LM Studio와 같은 로컬 LLM을 고려하여 간결하게 작성합니다.

```
You are an expert AI tool analyst. Extract the following information from the provided text about an AI tool. Format the output as a JSON object strictly following the given schema. If a field is not found, omit it unless it's a required field.

<tool_schema>
{
  "type": "function",
  "function": {
    "name": "extract_tool_info",
    "description": "Extracts key information about an AI tool from a given webpage text.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "The official name of the AI tool."
        },
        "url": {
          "type": "string",
          "description": "The URL of the AI tool's official page."
        },
        "tags": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Descriptive tags for the AI tool (e.g., #RAG, #ImageGeneration, #DeveloperTools)."
        },
        "description": {
          "type": "string",
          "description": "A concise summary of the AI tool's purpose and key features."
        },
        "usage_examples": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Specific examples of how the tool can be used, demonstrating its functionality."
        },
        "price": {
          "type": "number",
          "description": "Estimated monthly price in USD, if available. Null if free or not specified."
        },
        "has_api": {
          "type": "boolean",
          "description": "True if the tool provides an API or SDK for programmatic access, false otherwise."
        },
        "category": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Broad categories the tool belongs to (e.g., Image Generation, Text Generation, Developer Tools)."
        }
      },
      "required": ["name", "url", "description"]
    }
  }
}
</tool_schema>

<text_to_analyze>
{html_content_or_plain_text}
</text_to_analyze>

Provide only the JSON output without any additional text or explanations.
```

### 구현 상세
`backend/api_gateway/main.py` 파일 내에 `llm_parse_html` 또는 이와 유사한 함수를 새로 정의합니다. 이 함수는 웹 크롤링으로 얻은 HTML 콘텐츠를 입력받아 LLM을 호출하고, LLM의 응답을 파싱하여 정제된 JSON 데이터를 반환합니다.

기존 `_extract_metadata_from_text` 함수는 LLM 기반 추출 로직으로 대체되거나, LLM 추출 실패 시 폴백(fallback)으로 활용될 수 있습니다.

**`backend/api_gateway/main.py` 수정 예정 사항:**
- `llm_parse_html` (가칭) 함수 추가:
  - `httpx.AsyncClient`를 사용하여 LM Studio (또는 지정된 LLM API)에 `chat/completions` 요청을 보냅니다.
  - 요청 시 위에서 정의한 프롬프트와 `tools` 파라미터를 사용하여 함수 호출 모드로 유도합니다.
  - LLM 응답에서 `tool_calls` 필드를 파싱하여 필요한 정보를 추출합니다.
  - 추출된 데이터를 pandas DataFrame의 레코드 형식으로 변환하여 반환합니다.
- `process_url_endpoint` 함수 내에서 `_extract_metadata_from_text` 대신 `llm_parse_html`을 호출하도록 변경합니다.

## 1-2. 지능적인 Chunking 전략 적용

### 목표
LLM이 추출한 `description`과 `usage_examples` 텍스트를 문맥을 유지하면서 500~1,000 토큰 단위로 분할(Chunking)합니다. 각 Chunk는 원본 URL, LLM이 추출한 태그 등 메타데이터와 연결되어야 합니다.

### 전략
- **문단 기반 Chunking:** 텍스트를 우선 문단 단위로 분리합니다. 각 문단은 하나의 독립적인 Chunk가 될 수 있습니다.
- **헤더 기반 Chunking:** H1, H2 등의 HTML 헤더 태그를 기준으로 섹션을 나누고, 각 섹션 내의 텍스트를 Chunk로 처리합니다.
- **오버랩(Overlap) 적용:** Chunk 간에 약간의 중복(예: 10~20%)을 허용하여 문맥 손실을 최소화합니다.
- **토큰 길이 제한:** 각 Chunk의 토큰 수가 500~1,000 범위 내에 들도록 조절합니다. (임베딩 모델의 최대 입력 토큰 수를 고려)

### 구현 상세
`backend/parser/parser.py` 파일 내의 `parse_and_chunk_single_url` 함수를 고도화합니다.

**`backend/parser/parser.py` 수정 예정 사항:**
- `html_content`에서 `BeautifulSoup`를 사용하여 문단(`p`), 리스트(`li`), 헤더(`h1`~`h6`) 등의 구조를 파악합니다.
- 추출된 `description` 및 `usage_examples` 텍스트를 위 전략에 따라 여러 개의 Chunk로 분할합니다.
- 각 Chunk마다 `source_url`, `tool_name`, `category`, `tags` 등 LLM이 추출한 메타데이터를 함께 저장합니다. 이는 FAISS 인덱싱 시 각 벡터에 대한 풍부한 정보를 제공하게 됩니다.
- Chunk 분할 시 토큰 길이를 측정하기 위해 `tiktoken`과 같은 라이브러리를 활용합니다.

## 1-3. URL 처리 엔드포인트 업데이트

### 목표
`/api/v1/process_url` 엔드포인트가 Phase 1-1의 `llm_parse_html`과 Phase 1-2의 새로운 Chunking 로직을 사용하도록 수정합니다.

### 구현 상세
`backend/api_gateway/main.py`의 `process_url_endpoint` 함수를 수정합니다.

**`backend/api_gateway/main.py` 수정 예정 사항:**
- `parse_and_chunk_single_url` 함수 호출 전, 먼저 LLM 기반 정보 추출을 시도합니다.
- LLM 추출 결과와 원래 HTML 콘텐츠를 `parse_and_chunk_single_url` 함수에 전달하여, 정제된 정보와 구조화된 청크를 생성하도록 합니다.
- LLM 호출 실패 시, 적절한 에러 핸들링 또는 기존 `_extract_metadata_from_text`와 같은 폴백 로직을 고려합니다.
- 최종적으로 `new_metadata_records`와 `new_faiss_vectors`가 LLM과 개선된 청크 전략을 통해 생성되도록 합니다. 