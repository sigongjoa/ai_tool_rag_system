# AI Tool RAG System

## 프로젝트 개요

이 프로젝트는 AI 관련 도구 정보를 검색하고 사용자 질의에 답변하는 RAG(Retrieval Augmented Generation) 시스템의 Proof-of-Concept 구현입니다. 사용자는 웹 인터페이스를 통해 질문을 입력하고, 시스템은 내부 지식 기반을 활용하여 관련 정보를 검색한 후 LLM을 통해 답변을 생성합니다.

## 아키텍처

```mermaid
graph TD;
    User[사용자] --> Frontend[프론트엔드: React/HTML/CSS/JS];
    Frontend --> Backend[백엔드: FastAPI];
    Backend --> FAISS[FAISS 벡터 인덱스];
    Backend --> LLM[로컬 LLM (LM Studio)];
    HTMLParser[HTML 파서] --> ProcessedData[전처리된 데이터: processed_ai_tools_metadata.json];
    ProcessedData --> FAISS;

    subgraph Data Preparation
        HTMLParser
        ProcessedData
    end

    subgraph RAG Pipeline
        Frontend
        Backend
        FAISS
        LLM
    end
```

## 설치 및 실행 가이드

### 사전 요구사항

*   **Python 3.9+**: 백엔드 및 데이터 전처리 스크립트 실행.
*   **Node.js & npm (or yarn)**: 프론트엔드 빌드 및 실행.
*   **LM Studio**: 로컬에서 LLM을 실행하기 위한 애플리케이션. `http://127.0.0.1:1234` 포트에서 LLM이 실행 중이어야 합니다.

### 환경 변수 설정

백엔드 폴더(`.backend`) 내에 `.env` 파일을 생성하고 다음 내용을 추가합니다.

```
LLM_BASE_URL=http://127.0.0.1:1234/v1
DATA_DIR=./processed_data_storage
PROCESSED_DATA_PATH=./processed_data_storage/processed_ai_tools_metadata.json
FAISS_INDEX_PATH=./processed_data_storage/ai_tools_faiss_index.bin
```

### 1. 백엔드 설정 및 실행

```bash
# backend 디렉토리로 이동
cd backend

# 가상 환경 생성 및 활성화 (선택 사항이지만 권장)
C:\Users\YourUser\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 데이터 전처리 (최초 1회 또는 데이터 업데이트 시)
# HTML 파일들이 'data/html_data' 경로에 있는지 확인하세요.
python parser/parser.py

# 백엔드 서버 실행
uvicorn api_gateway.main:app --host 0.0.0.0 --port 8081 --reload
```

### 2. 프론트엔드 설정 및 실행

```bash
# frontend 디렉토리로 이동
cd frontend

# 의존성 설치
npm install

# 프론트엔드 서버 실행
npm start
```

### 3. LM Studio 실행

LM Studio를 실행하고 원하는 LLM 모델을 로드한 후, API 서버가 `http://127.0.0.1:1234` 에서 동작하도록 설정해야 합니다.

## 사용법

1.  위 지침에 따라 백엔드, 프론트엔드, LM Studio를 모두 실행합니다.
2.  브라우저에서 `http://localhost:3000` (프론트엔드 기본 포트)에 접속합니다.
3.  질문 입력창에 AI 도구 관련 질문을 입력하고 검색 버튼을 클릭합니다.
4.  백엔드를 통해 검색된 결과와 LLM이 생성한 답변을 확인할 수 있습니다. 