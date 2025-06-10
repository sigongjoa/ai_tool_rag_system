# AI Tool RAG System: 구현 계획서

## 1. 프로젝트 개요

본 문서는 "AI 툴 정보 과부하" 문제를 해결하기 위한 RAG(Retrieval-Augmented Generation) 시스템의 상세 구현 계획을 기술합니다. 사용자는 최신 AI 툴 정보를 자연어 질문, 특정 조건 필터링, 비교 분석 등 다양한 방식으로 탐색하고 요약/추천받을 수 있습니다.

### 주요 목표 (MVP)
- **데이터 수집:** 매일 1,000개 이상의 AI 툴 관련 웹 페이지(제품 홈페이지, 블로그, 리뷰 등) 정보를 수집합니다.
- **응답 속도:** 사용자 질의에 대해 2초 내에 답변을 생성하는 것을 목표로 합니다.
- **핵심 기능:**
    1.  **최신 AI 툴 요약 및 추천:** 새로 출시되거나 주목받는 툴의 정보를 정리하여 제공합니다.
    2.  **조건 기반 검색:** 가격, 지원 모델, API 제공 여부 등 구조화된 조건으로 필터링합니다.
    3.  **비교 분석:** 여러 툴의 벤치마크, 주요 기능, 가격 정책 등을 표 형태로 비교하여 제공합니다.

## 2. 시스템 아키텍처

본 시스템은 데이터 수집부터 답변 생성까지의 과정을 여러 개의 독립적인 모듈로 구성하여, Docker Compose를 통해 컨테이너 환경에서 운영됩니다. 이를 통해 각 컴포넌트의 독립적인 개발, 배포, 확장이 용이해집니다.

### 아키텍처 다이어그램

```mermaid
graph TD;
    subgraph "사용자 인터페이스 (Next.js)"
        A[FastAPI 기반 API Gateway]
    end

    subgraph "응답 생성 (LLM Server)"
        B[LLM Server (vLLM)]
    end

    subgraph "검색 및 재정렬"
        C[Reranker (GPU)]
        D[Vector DB (Qdrant)]
    end

    subgraph "데이터 파이프라인"
        E[Pre-ETL Parser]
        F[Ingestion Worker]
    end

    F -- "Raw HTML" --> E;
    E -- "Chunk + Metadata" --> D;
    D -- "KNN IDs" --> C;
    C -- "Top-k Docs" --> B;
    B -- "Answer" --> A;

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style B fill:#cfc,stroke:#333,stroke-width:2px
    style A fill:#cfc,stroke:#333,stroke-width:2px

```

### 컴포넌트 설명

| 컴포넌트 | 역할 | 주요 기술 |
| --- | --- | --- |
| **Ingestion Worker** | 주기적으로 대상 웹사이트(AI 툴 디렉터리, 블로그 등)를 크롤링하여 원본 HTML 데이터를 수집합니다. | `Python`, `Playwright`, `CronJob` |
| **Pre-ETL Parser** | 수집된 HTML에서 본문, 메타데이터(가격, 카테고리 등)를 추출하고, 의미 단위로 텍스트를 분할(Chunking) 및 임베딩합니다. | `BeautifulSoup`, `LlamaIndex`, `GTE-Large` |
| **Vector DB** | 임베딩된 텍스트 벡터와 메타데이터를 저장하고, 유사도 기반의 빠른 검색을 지원합니다. | `Qdrant` |
| **Reranker** | Vector DB에서 1차 검색된 결과들을 Cross-Encoder를 사용해 질문과의 관련성을 기준으로 재정렬하여 정확도를 높입니다. | `Cohere Rerank`, `bge-reranker` |
| **LLM Server** | 재정렬된 최종 문서를 바탕으로 사용자의 질문에 대한 답변(요약, 비교, 추천)을 생성합니다. | `vLLM`, `Mixtral`, `GPT-4o` |
| **API Gateway** | 외부 사용자와 시스템 내부 로직을 연결하는 인터페이스 역할을 하며, HTTP 요청을 처리합니다. | `FastAPI`, `Next.js` |

---

> **다음 문서:** [01_Data_Pipeline.md](./01_Data_Pipeline.md) 에서는 데이터 수집 및 정규화 과정에 대해 더 자세히 다룹니다. 