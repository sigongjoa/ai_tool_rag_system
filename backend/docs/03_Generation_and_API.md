# 3. 답변 생성 및 API 제공

검색 및 재정렬(Retrieval & Re-rank) 파이프라인을 통해 확보된 관련성 높은 문서(Top-k documents)는 이제 LLM(Large Language Model)에 전달되어 사용자의 최종 답변을 생성하는 데 사용됩니다. 이 과정의 핵심은 LLM이 주어진 컨텍스트(검색된 문서)를 충실하게 활용하여, 질문의 의도에 맞는 형태로 답변을 가공하도록 유도하는 것입니다.

## 3.1. 답변 생성 (Generation/Answering)

LLM 서버는 vLLM과 같이 처리량(throughput)이 높은 서빙 엔진을 기반으로 구축하여, 동시 다발적인 요청에도 빠른 응답 속도를 유지하는 것을 목표로 합니다.

### 가. 사용 시나리오별 LLM 선택

| 시나리오 | 추천 LLM | 선택 이유 |
|---|---|---|
| **요약/리뷰 분석** | `Mixtral-8x22B`, `GPT-4o`, `Claude 3.5 Sonnet` | 여러 문서의 핵심 내용을 종합하고, 장단점을 비교 분석하는 등 복잡한 추론 능력과 긴 컨텍스트 처리 능력이 뛰어납니다. |
| **단순 Q&A 챗봇** | `Nous-Hermes-2-Mixtral-8x7B-SFT`, `Llama-3-8B-Instruct` | 특정 질문에 대한 사실 기반의 짧은 답변 생성에 최적화되어 있으며, 13B 이하 모델은 vLLM 환경에서 매우 높은 QPS를 달성할 수 있습니다. |
| **멀티턴 대화** | `GPT-4o`, `Claude 3.5 Sonnet` | 이전 대화의 맥락을 기억하고 Function Calling과 같은 외부 도구를 호출하는 능력이 뛰어나, 복잡한 사용자 요구사항을 대화 형태로 해결하는 데 유리합니다. |

**MVP 전략:** 초기에는 범용성이 가장 뛰어난 `Mixtral` 또는 `Claude 3.5 Sonnet`을 단일 모델로 사용하여 모든 시나리오에 대응하고, 추후 사용 패턴이 분석되면 각 시나리오에 최적화된 모델을 추가 도입하는 것을 고려합니다.

### 나. 프롬프트 템플릿 (Prompt Template)

LLM이 일관성 있고 정확한 결과물을 생성하도록 지시하는 '레시피'입니다. LangChain의 `PromptTemplate`이나 LlamaIndex의 `PromptTemplate` 클래스를 사용하여 동적으로 프롬프트를 구성합니다.

**비교 분석 시나리오 프롬프트 예시:**

```jinja2
당신은 AI 도구 전문 분석가입니다. 아래에 제공된 검색 결과를 바탕으로, "{{ tool_A }}"와 "{{ tool_B }}"를 다음 기준에 따라 표 형식으로 비교 분석해 주세요.

- 주요 기능
- 가격 정책 (월간 구독료 기준)
- API 제공 여부
- 핵심 타겟 사용자

<검색 결과>
{% for doc in documents %}
---
문서 {{ loop.index }}:
{{ doc.text }}
---
{% endfor %}

<분석 결과 (Markdown 표 형식)>
```

이러한 템플릿 체인화를 통해 "검색 결과 N개 JSON → 툴 요약 카드 생성 → 비교표 형식 출력"과 같은 복잡한 작업 흐름을 자동화할 수 있습니다.

## 3.2. API 서버 (FastAPI / Next.js)

생성된 답변은 표준화된 API 엔드포인트를 통해 웹 프론트엔드(Next.js)나 외부 애플리케이션에 제공됩니다.

### 가. API 엔드포인트 설계

- **`POST /api/v1/search`**
  - **설명:** 사용자의 주된 검색 요청을 처리하는 엔드포인트입니다.
  - **Request Body:**
    ```json
    {
      "query": "텍스트를 이미지로 바꿔주는 AI 툴 알려줘",
      "filters": {
        "price_max_usd": 20,
        "has_api": true,
        "category": ["Image Generation", "Art"]
      },
      "response_format": "summary" // "summary", "comparison", "raw_docs"
    }
    ```
  - **Response Body (Success):**
    ```json
    {
      "request_id": "uuid-...",
      "data": {
        "type": "summary",
        "content": "귀하의 요청에 맞는 AI 툴은 다음과 같습니다: [1] Midjourney [2] ... (LLM이 생성한 요약 텍스트)" 
      },
      "retrieved_sources": [
        {"source_url": "...", "title": "..."},
        ...
      ]
    }
    ```

### 나. 구현 (`main.py`)

`FastAPI`를 사용하여 비동기(async) 방식으로 각 컴포넌트(Retriever, Reranker, LLM)를 호출합니다.

```python
from fastapi import FastAPI
from pydantic import BaseModel
# ... (내부 모듈 import)

app = FastAPI()

# Pydantic 모델로 Request Body 타입 정의
class SearchRequest(BaseModel):
    query: str
    # ...

@app.post("/api/v1/search")
async def handle_search(request: SearchRequest):
    # 1. Retriever로 1차 검색
    retrieved_nodes = await retriever.retrieve(request.query, filters=request.filters)

    # 2. Reranker로 재정렬
    reranked_nodes = await reranker.rerank(retrieved_nodes, request.query)

    # 3. LLM에 전달하여 답변 생성
    prompt = build_prompt(template_name=request.response_format, context=reranked_nodes)
    final_answer = await llm_client.generate(prompt)

    # 4. 최종 결과 반환
    return {"data": {"content": final_answer}, "sources": ...}
```

---
> **다음 문서:** [04_System_Architecture_and_Deployment.md](./04_System_Architecture_and_Deployment.md) 에서는 지금까지 설계한 컴포넌트들을 컨테이너화하고, 실제 운영 환경에 배포 및 모니터링하는 전략을 다룹니다. 