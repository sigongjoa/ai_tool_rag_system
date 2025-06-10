# AI Tool RAG System: 구현 계획 문서

본 디렉터리는 "AI 툴 추천 및 분석 RAG 시스템"의 기획부터 배포, 확장까지의 전 과정을 담은 상세 구현 계획 문서를 포함합니다. 각 문서는 시스템 개발의 특정 단계를 다루고 있으며, 아래 목차를 통해 원하는 문서로 빠르게 이동할 수 있습니다.

## 문서 목차

### 📄 [00_Project_Overview.md](./00_Project_Overview.md)
- **내용:** 프로젝트의 목표, 핵심 기능, 그리고 전체 시스템의 컴포넌트와 데이터 흐름을 한눈에 파악할 수 있는 아키텍처를 정의합니다. 프로젝트를 처음 접하는 분들이 가장 먼저 읽어야 할 문서입니다.

### 📄 [01_Data_Pipeline.md](./01_Data_Pipeline.md)
- **내용:** RAG 시스템의 근간이 되는 데이터 수집(Crawling) 및 정규화(Parsing, Chunking) 과정을 상세히 다룹니다. 어떤 데이터를 어디서 수집하고, 어떻게 가공하여 저장할지에 대한 구체적인 계획을 포함합니다.

### 📄 [02_Search_and_Retrieval.md](./02_Search_and_Retrieval.md)
- **내용:** 가공된 데이터를 검색 가능한 벡터로 변환하는 임베딩 전략, 고성능 벡터 DB(Qdrant)의 선택 및 활용법, 그리고 검색 정확도를 높이기 위한 1차 검색(Retrieval) 및 2차 재정렬(Re-rank) 파이프라인 구축 방법을 설명합니다.

### 📄 [03_Generation_and_API.md](./03_Generation_and_API.md)
- **내용:** 검색된 정보를 바탕으로 사용자의 질문에 대한 최종 답변을 생성하는 LLM의 역할과 프롬프트 엔지니어링 전략을 기술합니다. 또한, 이 모든 기능을 외부에서 사용할 수 있도록 FastAPI 기반의 API 서버를 설계하는 방법을 다룹니다.

### 📄 [04_System_Architecture_and_Deployment.md](./04_System_Architecture_and_Deployment.md)
- **내용:** 개발된 시스템을 실제 운영 환경에 배포하고 관리하기 위한 구체적인 방법을 제시합니다. Docker-Compose를 이용한 컨테이너화, GitHub Actions 기반의 CI/CD, Prometheus/Grafana를 활용한 모니터링 및 품질 평가 전략을 포함합니다.

### 📄 [05_Roadmap.md](./05_Roadmap.md)
- **내용:** MVP(최소 기능 제품) 출시 이후, 서비스를 더욱 발전시키기 위한 장기적인 확장 계획을 다룹니다. 멀티모달 검색, 개인화, 플러그인 API 제공 등 미래 비전을 제시합니다.

---

**핵심 성공 조건:** 신뢰도 높은 메타데이터 파싱 + 정확한 검색/재정렬 + LLM 템플릿 품질. 