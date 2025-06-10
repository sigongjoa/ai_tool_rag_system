# 1. 데이터 파이프라인: 수집 및 정규화

RAG 시스템의 품질은 입력 데이터의 품질에 따라 결정됩니다. 따라서 본 파이프라인은 신뢰도 높은 데이터를 효율적으로 수집하고, 검색에 용이한 형태로 가공(정규화)하는 것을 목표로 합니다.

## 1.1. 개념 스코프 및 대상 데이터

### 가. 수집 대상
초기(MVP)에는 아래 소스를 중심으로 데이터를 수집합니다.
- **AI 툴 디렉터리:** Futurepedia, There's An AI For That 등
- **SaaS 제품 정보:** 각 AI 툴의 공식 홈페이지, 가격 정책 페이지, 주요 기능 소개
- **기술 콘텐츠:** 공식 블로그, GitHub README, 기술 문서(Documentation), 백서(Whitepaper)

### 나. 주요 사용 시나리오
- **"새로 나온 AI 툴 요약·추천"**: 최신성(recency)이 중요한 동향 파악
- **"이미지 생성 AI 중 월 10$ 이하, API 지원 툴 찾아줘"**: 다중 조건 필터링
- **"Midjourney와 Stable Diffusion 비교"**: 특정 대상에 대한 심층 비교 분석

## 1.2. 데이터 수집 (Ingestion)

`Ingestion Worker`는 `Playwright`와 `Python`을 사용하여 구현하며, 동적 콘텐츠(JavaScript 렌더링)가 많은 최신 웹사이트에 대응합니다.

### 가. 스크래핑 설정
사이트별 스크래핑 규칙은 중앙에서 관리하기 용이하도록 `YAML` 형식으로 설정 파일을 분리합니다.

**`scraper_config.yaml` 예시:**
```yaml
- site_name: "Futurepedia"
  url: "https://www.futurepedia.io/"
  target_selector: ".tool-card" # 툴 목록을 감싸는 CSS 선택자
  metadata_selectors:
    title: ".tool-title"
    category: ".tool-category"
    description: ".tool-description"
  # ... 기타 사이트 설정
```

### 나. 구현 로직 (`worker.py`)
1. `scraper_config.yaml` 파일을 로드합니다.
2. 설정된 사이트를 순회하며 `Playwright`로 페이지에 접속합니다.
3. `target_selector`에 해당하는 모든 HTML 조각을 추출하여 원본(raw) 데이터 저장소(예: S3, 로컬 파일시스템)에 저장합니다.
4. 이 작업은 Docker 컨테이너 내에서 `Cron`을 통해 매일 자정(00:00)에 실행되도록 스케줄링합니다.

## 1.3. 데이터 정규화 (Pre-ETL)

`Pre-ETL Parser`는 수집된 원본 HTML을 LLM이 이해하고 검색하기 좋은 형식으로 가공합니다. 이 과정은 LlamaIndex 프레임워크를 적극 활용하여 코드 복잡도를 줄입니다.

### 가. 본문 및 메타데이터 추출
- **본문 추출:** `BeautifulSoup` 또는 `readability-lxml` 라이브러리를 사용해 광고, 네비게이션 바 등 불필요한 HTML 태그를 제거하고 순수 텍스트 본문만 추출합니다.
- **메타데이터 파싱:** 정규표현식(Regex)과 키워드 매칭을 통해 가격(e.g., "$10/mo", "Free Plan"), 지원 모델(e.g., "GPT-4", "Claude 3.5"), API 제공 여부 등의 정형 데이터를 추출합니다.

### 나. 문서 분할 (Chunking)
- **LlamaIndex 활용:** `LlamaIndex`의 `SimpleWebPageReader`를 사용하여 URL 또는 HTML 파일로부터 `Document` 객체를 직접 생성할 수 있습니다.
    ```python
    from llama_index.core import SimpleDirectoryReader
    
    # HTML 파일이 저장된 디렉토리로부터 Document 로드
    loader = SimpleDirectoryReader("./raw_html_storage")
    documents = loader.load_data()
    ```
- `SentenceSplitter`를 사용해 `Document`를 의미적으로 유사한 문장 단위로 분할(chunking)합니다. 각 청크는 약 2~3KB 크기를 유지하여 임베딩 시 컨텍스트 손실을 최소화합니다.
    ```python
    from llama_index.core.node_parser import SentenceSplitter

    # 512 토큰 단위로 분할, 20 토큰은 청크 간 중첩
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    ```
- 분할된 각 `Node`(청크)에는 추출된 메타데이터(가격, 카테고리 등)가 함께 저장됩니다. 이는 후처리 필터링 단계에서 매우 중요하게 사용됩니다.

> **팁:** `firecrawl.dev`와 같은 외부 서비스를 활용하면 스크래핑 및 정규화 파이프라인을 더욱 간소화할 수 있습니다. 초기에는 직접 구현하며 구조를 잡고, 추후 확장 시 서비스 도입을 검토할 수 있습니다.

---
> **다음 문서:** [02_Search_and_Retrieval.md](./02_Search_and_Retrieval.md) 에서는 정규화된 데이터를 임베딩하고 Vector DB에 저장하여 검색하는 과정을 다룹니다. 