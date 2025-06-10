# 테스트 가이드

이 문서는 프로젝트의 백엔드 및 프론트엔드 코드베이스에 대한 테스트 목록과 실행 방법을 제공합니다.

## 1. 백엔드 테스트

백엔드는 주로 Python으로 작성되었으며, `pytest` 프레임워크를 사용하여 테스트를 실행할 수 있습니다.

### 테스트 위치

일반적으로 각 백엔드 컴포넌트 디렉토리 내에 테스트 파일이 존재합니다. 예를 들어:
- `backend/indexer/tests/`
- `backend/processor/tests/`
- `backend/collectors/tests/`
- `backend/parser/tests/`
- `backend/worker/tests/`
- `backend/api_gateway/tests/` (현재는 `main.py`만 있지만, 필요에 따라 `tests` 디렉토리를 생성할 수 있습니다.)

각 컴포넌트의 테스트 파일은 `test_*.py`와 같은 명명 규칙을 따를 수 있습니다.

### 테스트 실행 방법

1.  **가상 환경 활성화:**
    백엔드 루트 디렉토리(`backend/`)에서 가상 환경을 활성화합니다.
    ```bash
    cd backend
    ./venv/Scripts/activate # Windows
    # source venv/bin/activate # Linux/macOS
    ```

2.  **의존성 설치 (필요시):**
    `requirements.txt`에 명시된 모든 의존성이 설치되어 있는지 확인합니다.
    ```bash
    pip install -r requirements.txt
    ```
    `pytest`가 설치되어 있지 않다면 설치합니다.
    ```bash
    pip install pytest
    ```

3.  **모든 테스트 실행:**
    백엔드 루트 디렉토리에서 모든 테스트를 실행합니다.
    ```bash
    pytest
    ```

4.  **특정 디렉토리 또는 파일 테스트 실행:**
    특정 컴포넌트의 테스트만 실행하려면 해당 디렉토리로 이동하거나 경로를 지정합니다.
    ```bash
    pytest backend/indexer/tests/
    # 또는
    pytest backend/indexer/tests/test_indexing.py
    ```

### 새 테스트 작성 가이드라인

-   테스트 파일은 `test_*.py` 또는 `*_test.py`로 명명합니다.
-   테스트 함수는 `test_`로 시작해야 합니다.
-   `pytest`의 [문서](https://docs.pytest.org/en/stable/)를 참조하여 더 자세한 테스트 작성 방법을 확인하십시오.

### 1.1. `backend/api_gateway` 테스트

`backend/api_gateway`는 FastAPI를 기반으로 하는 API 게이트웨이 역할을 하며, 검색, URL 처리, 데이터 수집 및 상태 확인과 같은 핵심 기능을 제공합니다.

#### 주요 기능 및 검증 방법

1.  **시작 이벤트 (`startup_event`)**
    *   **기능**: 애플리케이션 시작 시 FAISS 인덱스, 전처리된 메타데이터, SentenceTransformer 임베딩 모델을 로드합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 필요한 모든 데이터와 모델이 성공적으로 로드되는지 확인합니다. 파일이 없거나 로드에 실패했을 때 적절한 경고/오류 로깅이 발생하는지 확인합니다.
        *   **테스트 시나리오**:
            *   모든 파일(FAISS 인덱스, 메타데이터 JSON)이 정상적으로 존재할 때 로드가 성공하는지 확인.
            *   특정 파일이 없을 때 애플리케이션이 시작되지만 검색 기능이 제한될 수 있다는 경고가 발생하는지 확인 (FAISS 인덱스, 메타데이터 파일 부재 시).
            *   `SentenceTransformer` 모델 로드 실패 시 에러가 발생하는지 확인.
        *   **`pytest`를 사용한 접근**: `pytest-mock` 또는 `unittest.mock`을 사용하여 `os.path.exists`, `faiss.read_index`, `pd.read_json`, `SentenceTransformer` 등을 모킹(mocking)하여 다양한 로드 시나리오를 시뮬레이션할 수 있습니다. 실제 애플리케이션 시작 없이 `startup_event` 함수를 직접 호출하여 테스트할 수 있습니다.

2.  **상태 확인 엔드포인트 (`GET /health`)**
    *   **기능**: API 게이트웨이의 동작 상태를 확인합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 엔드포인트가 성공적으로 `HTTP 200 OK` 응답과 함께 상태 메시지를 반환하는지 확인합니다.
        *   **테스트 시나리오**: `GET /health` 요청을 보내고 응답의 상태 코드와 내용을 확인합니다.
        *   **`pytest`를 사용한 접근**: `FastAPI`의 `TestClient`를 사용하여 요청을 보내고 응답을 검증합니다.
            ```python
            # test_api_gateway.py 예시
            from fastapi.testclient import TestClient
            from backend.api_gateway.main import app

            client = TestClient(app)

            def test_health_check():
                response = client.get("/health")
                assert response.status_code == 200
                assert response.json() == {"status": "ok", "message": "API Gateway is running"}
            ```

3.  **검색 엔드포인트 (`POST /api/v1/search`)**
    *   **기능**: 사용자 쿼리를 기반으로 AI 도구를 검색하고, 요약 또는 비교 분석 결과를 반환합니다. 필터링, 응답 형식 지정, 버전(AI/Dev/Office) 선택 기능을 지원합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 다양한 쿼리, 필터, 응답 형식(요약, 비교) 및 버전에 대해 올바른 검색 결과와 응답 구조가 반환되는지 확인합니다. FAISS 인덱스나 임베딩 모델이 로드되지 않았을 때의 동작도 확인합니다.
        *   **테스트 시나리오**:
            *   **기본 검색**: 일반적인 쿼리에 대한 요약 응답 확인.
            *   **필터링**: `filters` 매개변수를 사용하여 가격, API 제공 여부, 카테고리, 태그 등을 필터링하여 검색 결과가 올바르게 제한되는지 확인.
            *   **비교 분석**: `response_format="comparison"`과 `tool_A`, `tool_B` 매개변수를 사용하여 두 도구 간의 비교 분석 결과가 Markdown 표 형식으로 올바르게 생성되는지 확인.
            *   **버전별 검색**: `version` 매개변수를 사용하여 AI, Dev, Office 등 특정 카테고리/태그에 해당하는 도구만 검색되는지 확인.
            *   **의존성 부재**: `faiss_index`, `processed_metadata`, `embedding_model`이 로드되지 않은 상태에서 검색 요청 시 적절한 오류 또는 경고 응답이 반환되는지 확인 (모킹 필요).
            *   **잘못된 입력**: 유효하지 않은 `response_format` 또는 기타 잘못된 입력에 대한 유효성 검사 및 에러 처리 확인.
        *   **`pytest`를 사용한 접근**: `TestClient`를 사용하여 `POST` 요청을 보내고 응답을 검증합니다. `pytest-mock`을 사용하여 `embedding_model.encode`, FAISS 검색 결과, LLM 응답 등을 모킹하여 외부 의존성 없이 로직을 테스트합니다.

4.  **URL 처리 엔드포인트 (`POST /api/v1/process_url`)**
    *   **기능**: 주어진 URL에서 정보를 처리하는 역할을 합니다. (현재 `main.py`에는 구현 세부 정보가 없지만, `collector` 또는 `parser` 모듈과 연동될 것으로 예상됩니다.)
    *   **검증 방법**:
        *   **테스트 목표**: 유효한 URL과 유효하지 않은 URL에 대한 응답을 확인합니다.
        *   **테스트 시나리오**:
            *   유효한 URL을 POST 요청으로 보냈을 때 예상되는 성공 응답 확인.
            *   유효하지 않은 URL (예: 잘못된 형식, 존재하지 않는 도메인)을 보냈을 때 적절한 에러 응답 확인.
            *   매우 큰 페이지나 복잡한 페이지에 대한 처리 성능 및 안정성 (이는 통합 테스트에 더 가깝지만, 로컬 테스트에서도 간단히 확인 가능).
        *   **`pytest`를 사용한 접근**: `TestClient`를 사용하여 `POST` 요청을 보내고 응답을 검증합니다. 실제 웹 요청을 보내지 않도록 `requests` 라이브러리 호출을 모킹할 수 있습니다.

5.  **데이터 수집/파싱 엔드포인트 (`POST /api/v1/parse_and_ingest`)**
    *   **기능**: `parser.py`의 기능을 통합하거나 호출하여 데이터를 파싱하고 수집하는 역할을 합니다. 현재는 더미 응답을 제공합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 엔드포인트가 호출되었을 때 예상되는 더미 응답이 반환되는지 확인하고, 향후 실제 기능이 구현될 경우 해당 기능의 동작을 검증합니다.
        *   **테스트 시나리오**: POST 요청을 보내고 반환되는 더미 응답의 상태 코드와 메시지를 확인합니다.
        *   **`pytest`를 사용한 접근**: `TestClient`를 사용하여 `POST` 요청을 보내고 응답을 검증합니다.

6.  **메타데이터 추출 도우미 함수 (`_extract_metadata_from_text`)**
    *   **기능**: 주어진 텍스트에서 가격, API 유무, 카테고리, 태그와 같은 AI 도구 메타데이터를 정규 표현식을 사용하여 추출합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 다양한 텍스트 입력에 대해 함수가 올바른 메타데이터를 추출하는지 확인합니다.
        *   **테스트 시나리오**:
            *   가격 정보가 포함된 텍스트에서 정확한 가격 추출.
            *   API 관련 키워드(API, developer sdk 등) 유무에 따른 `has_api` 및 `tags` 값 확인.
            *   다양한 카테고리(Image Generation, Text Generation 등) 키워드에 따른 `category` 및 `tags` 값 확인.
            *   중복 태그가 올바르게 제거되는지 확인.
            *   아무 정보도 없는 텍스트에 대한 기본값 반환 확인.
        *   **`pytest`를 사용한 접근**: `main.py` 파일 내의 함수이므로 직접 임포트하여 다양한 입력값으로 함수를 호출하고 반환 값을 어설션(assert)합니다.
            ```python
            # test_api_gateway_helpers.py (또는 test_api_gateway.py에 포함) 예시
            from backend.api_gateway.main import _extract_metadata_from_text

            def test_extract_metadata_price():
                text = "This tool costs $9.99/month."
                metadata = _extract_metadata_from_text(text)
                assert metadata["price"] == 9.99

            def test_extract_metadata_api_and_tags():
                text = "Supports a REST API for developers."
                metadata = _extract_metadata_from_text(text)
                assert metadata["has_api"] == True
                assert "api_enabled" in metadata["tags"]
                assert "dev" in metadata["tags"]
                assert "code" in metadata["tags"]
            ```

#### `backend/api_gateway` 테스트 실행

`backend/api_gateway` 디렉토리 내에 `tests` 디렉토리를 생성하고, 그 안에 `test_api_gateway.py`와 같은 테스트 파일을 작성합니다.

```bash
# 백엔드 루트 디렉토리에서
cd backend
pytest api_gateway/tests/
```

## 1.2. 다른 백엔드 모듈 테스트 (Indexer, Processor, Collectors, Parser, Worker)

각 백엔드 모듈은 특정 데이터 처리 또는 수집 기능을 담당합니다. 이 문서에서는 `api_gateway`에 대한 상세한 가이드를 제공했으며, 다른 모듈들도 유사하게 상세한 테스트 가이드를 포함해야 합니다.

**각 모듈별로 다음 내용을 명확히 해야 합니다:**
-   **핵심 기능**: 해당 모듈이 수행하는 주요 작업 (예: `indexer`는 데이터 인덱싱, `processor`는 데이터 전처리 등).
-   **입력 및 출력**: 모듈이 어떤 데이터를 입력받고 어떤 데이터를 출력하는지.
-   **의존성**: 외부 서비스, 데이터베이스, 파일 시스템 등 어떤 것에 의존하는지.
-   **검증 방법**:
    -   **단위 테스트 (Unit Tests)**: 각 함수나 클래스 메서드가 고립된 환경에서 예상대로 동작하는지 확인.
    -   **통합 테스트 (Integration Tests)**: 모듈 내 여러 컴포넌트 또는 다른 모듈과의 상호작용이 올바른지 확인.
    -   **데이터 유효성 검사**: 처리된 데이터의 정확성과 무결성 확인.

각 모듈에 대한 상세한 테스트 가이드는 해당 모듈의 코드 분석이 완료된 후 이 문서에 추가될 예정입니다.

### 1.2.1. `backend/indexer` 테스트

`backend/indexer` 모듈은 벡터 스토어에 데이터 청크를 인덱싱하고 FAISS 인덱스를 관리하는 기능을 제공합니다. 핵심 파일은 `indexer.py`입니다.

#### 주요 기능 및 검증 방법

1.  **데이터 인덱싱 (`index_data()`)**
    *   **기능**: `processed_chunks.json` 파일에서 처리된 데이터 청크를 로드하고, `SimpleVectorStore`를 사용하여 문서(Document)를 인덱싱한 다음, 이를 `vector_storage` 디렉토리에 저장합니다. `HuggingFaceEmbedding` 모델을 사용하여 임베딩을 생성합니다.
    *   **검증 방법**:
        *   **테스트 목표**: `processed_chunks.json` 파일이 존재할 때 데이터가 성공적으로 로드되고 `SimpleVectorStore`에 인덱싱되는지 확인합니다. 파일이 없거나 데이터가 비어 있을 때의 예외 처리 및 경고 메시지를 확인합니다.
        *   **테스트 시나리오**:
            *   **정상 작동**: 유효한 `processed_chunks.json` 파일이 주어졌을 때, `SimpleVectorStore`에 문서가 올바르게 추가되고 저장되는지 확인합니다.
            *   **파일 부재**: `processed_chunks.json` 파일이 존재하지 않을 때 함수가 적절한 오류 메시지를 출력하고 종료되는지 확인합니다.
            *   **빈 데이터**: `processed_chunks.json` 파일이 비어 있거나 유효한 문서가 없을 때 인덱싱이 건너뛰어지는지 확인합니다.
            *   **기존 벡터 스토어 로드**: `vector_storage/default__vector_store.json` 파일이 존재할 때 기존 스토어가 성공적으로 로드되는지 확인합니다.
        *   **`pytest`를 사용한 접근**: `pytest-mock` 또는 `unittest.mock`을 사용하여 `os.path.exists`, `json.load`, `SimpleVectorStore.from_persist_dir`, `StorageContext.persist`, `HuggingFaceEmbedding` 등을 모킹(mocking)하여 파일 시스템 및 외부 라이브러리 의존성 없이 로직을 테스트합니다. 임시 파일을 생성하여 실제 파일 시스템 상호작용을 테스트할 수도 있습니다.
            ```python
            # backend/indexer/tests/test_indexer.py 예시
            import pytest
            from unittest.mock import patch, MagicMock
            import os
            import json
            from backend.indexer.indexer import index_data
            from llama_index.core.vector_stores.simple import SimpleVectorStore
            from llama_index.core.schema import Document

            @patch('os.path.exists')
            @patch('builtins.open', new_callable=MagicMock)
            @patch('json.load')
            @patch('llama_index.core.vector_stores.simple.SimpleVectorStore')
            @patch('llama_index.core.storage_context.StorageContext.from_defaults')
            @patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding')
            @patch('llama_index.core.vector_store_index.VectorStoreIndex.from_documents')
            def test_index_data_success(mock_from_documents, mock_embedding, mock_storage_context_from_defaults,
                                        mock_simple_vector_store, mock_json_load, mock_open, mock_exists):
                # Setup mocks for successful scenario
                mock_exists.return_value = True # processed_chunks.json exists
                mock_json_load.return_value = [
                    {"id": "1", "text": "test chunk 1", "metadata": {"source": "test"}}
                ]
                mock_vector_store_instance = MagicMock(spec=SimpleVectorStore)
                mock_simple_vector_store.return_value = mock_vector_store_instance
                mock_storage_context_instance = MagicMock()
                mock_storage_context_from_defaults.return_value = mock_storage_context_instance

                index_data()

                # Assertions
                mock_exists.assert_called_with('processed_data_storage\processed_chunks.json')
                mock_json_load.assert_called_once()
                mock_embedding.assert_called_once_with(model_name="thenlper/gte-large", device="cpu")
                mock_from_documents.assert_called_once()
                mock_storage_context_instance.persist.assert_called_once()

            @patch('os.path.exists')
            @patch('builtins.print')
            def test_index_data_no_chunks_file(mock_print, mock_exists):
                mock_exists.return_value = False # processed_chunks.json does not exist
                index_data()
                mock_print.assert_any_call('오류: 처리된 청크 파일이 없습니다. processed_data_storage\processed_chunks.json를 확인해주세요.')

            # 추가 시나리오: 빈 청크 파일, 기존 벡터 스토어 로드 등
            ```

2.  **FAISS 인덱스 구축 (`build_faiss_index(df: pd.DataFrame)`)**
    *   **기능**: 입력 `DataFrame`의 `description` 컬럼에 있는 텍스트를 `SentenceTransformer`를 사용하여 임베딩하고, 이 임베딩을 기반으로 FAISS `IndexFlatIP` 인덱스를 구축합니다.
    *   **검증 방법**:
        *   **테스트 목표**: DataFrame의 텍스트가 정확하게 임베딩되고, FAISS 인덱스가 올바른 차원과 함께 생성되는지 확인합니다. `description` 컬럼이 없는 경우 또는 비어 있는 경우를 처리하는지 확인합니다.
        *   **테스트 시나리오**:
            *   **정상 작동**: 유효한 description이 포함된 DataFrame으로 FAISS 인덱스 구축.
            *   **결측값 처리**: `description` 컬럼에 `NaN` 값이 있을 때 해당 값이 빈 문자열로 처리되는지 확인.
            *   **빈 DataFrame**: 빈 DataFrame이 주어졌을 때 FAISS 인덱스가 예상대로 생성되는지 (예: 벡터 수가 0) 확인.
        *   **`pytest`를 사용한 접근**: `pandas.DataFrame` 객체를 생성하여 함수를 직접 호출하고, 반환된 FAISS 인덱스 객체의 속성(예: `ntotal`, `d`)을 검증합니다. `SentenceTransformer.encode`를 모킹하여 실제 임베딩 모델 로드 및 연산을 건너뛸 수 있습니다.
            ```python
            # backend/indexer/tests/test_indexer.py 예시
            import pandas as pd
            from backend.indexer.indexer import build_faiss_index
            from unittest.mock import patch, MagicMock
            import faiss
            import numpy as np

            @patch('sentence_transformers.SentenceTransformer')
            def test_build_faiss_index_success(mock_sentence_transformer):
                mock_model_instance = MagicMock()
                mock_model_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype='float32')
                mock_sentence_transformer.return_value = mock_model_instance

                dummy_df = pd.DataFrame({
"                'description': ['text one', 'text two'],
"                'other_col': ['a', 'b']
"                })
                index = build_faiss_index(dummy_df)

                assert isinstance(index, faiss.IndexFlatIP)
                assert index.ntotal == 2
                assert index.d == 2
                mock_model_instance.encode.assert_called_once()
            ```

3.  **FAISS 인덱스 저장 및 로드 (`save_faiss_index()` 및 `load_faiss_index()`)**
    *   **기능**: `save_faiss_index`는 FAISS 인덱스를 지정된 경로에 이진 파일로 저장하고, `load_faiss_index`는 해당 경로에서 인덱스를 로드합니다.
    *   **검증 방법**:
        *   **테스트 목표**: FAISS 인덱스가 손상 없이 성공적으로 저장되고 로드되는지 확인합니다.
        *   **테스트 시나리오**:
            *   **저장 후 로드**: FAISS 인덱스를 저장한 후 다시 로드했을 때 원본 인덱스와 동일한 속성(예: `ntotal`, `d`)을 가지는지 확인합니다.
            *   **파일 존재 여부**: `save_faiss_index` 호출 후 파일이 생성되고, `load_faiss_index` 호출 시 파일이 없으면 오류가 발생하는지 확인합니다.
        *   **`pytest`를 사용한 접근**: `tempfile` 모듈을 사용하여 임시 파일을 생성하고 삭제하여 실제 파일 시스템 상호작용을 테스트할 수 있습니다. `faiss.write_index`와 `faiss.read_index`를 직접 호출하여 통합적으로 테스트합니다.
            ```python
            # backend/indexer/tests/test_indexer.py 예시
            import faiss
            import numpy as np
            import os
            import tempfile
            from backend.indexer.indexer import save_faiss_index, load_faiss_index

            def test_save_and_load_faiss_index():
                # Create a dummy FAISS index
                d = 64
                nb = 10
                xb = np.random.rand(nb, d).astype('float32')
                original_index = faiss.IndexFlatIP(d)
                original_index.add(xb)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
                    test_path = tmp_file.name

                try:
                    save_faiss_index(original_index, test_path)
                    assert os.path.exists(test_path)

                    loaded_index = load_faiss_index(test_path)
                    assert loaded_index.ntotal == original_index.ntotal
                    assert loaded_index.d == original_index.d
                    # 더 깊은 검증을 위해 벡터를 비교할 수도 있습니다.
                    # D, I = loaded_index.search(xb[:1], 1)
                    # assert I[0][0] == 0

                finally:
                    if os.path.exists(test_path):
                        os.remove(test_path)
            ```

#### `backend/indexer` 테스트 실행

`backend/indexer` 디렉토리 내에 `tests` 디렉토리를 생성하고, 그 안에 `test_indexer.py`와 같은 테스트 파일을 작성합니다.

```bash
# 백엔드 루트 디렉토리에서
cd backend
pytest indexer/tests/
```

### 1.2.2. `backend/processor` 테스트

`backend/processor` 모듈은 수집된 데이터를 통합 스키마에 맞게 전처리하는 역할을 담당합니다. 핵심 파일은 `processor.py`입니다.

#### 주요 기능 및 검증 방법

1.  **텍스트 클리닝 (`clean_text(text: str)`)**
    *   **기능**: 입력 텍스트에서 HTML 태그를 제거하고, 여러 공백 및 줄바꿈을 단일 공백으로 정리한 후 소문자로 변환합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 다양한 HTML 태그와 공백 패턴을 가진 텍스트가 올바르게 정리되는지 확인합니다. 문자열이 아닌 입력에 대한 처리도 확인합니다.
        *   **테스트 시나리오**:
            *   **HTML 태그 제거**: `<p>`, `<b>`, `<br/>` 등의 HTML 태그가 포함된 텍스트가 올바르게 제거되는지 확인.
            *   **공백 및 줄바꿈 정리**: 여러 개의 공백, 탭, 줄바꿈이 단일 공백으로 대체되고 선행/후행 공백이 제거되는지 확인.
            *   **대소문자 변환**: 텍스트가 모두 소문자로 변환되는지 확인.
            *   **비문자열 입력**: `None`, 숫자 등 문자열이 아닌 입력이 주어졌을 때 빈 문자열이 반환되는지 확인.
        *   **`pytest`를 사용한 접근**: `processor.py`의 `clean_text` 함수를 직접 임포트하여 다양한 입력값을 넣어 반환 값을 어설션(assert)합니다.
            ```python
            # backend/processor/tests/test_processor.py 예시
            from backend.processor.processor import clean_text

            def test_clean_text_html_tags():
                text = "<p>Hello <b>World</b>!<br/>How are you?</p>"
                expected = "hello world! how are you?"
                assert clean_text(text) == expected

            def test_clean_text_whitespace():
                text = "  This   is\n a \t test.  "
                expected = "this is a test."
                assert clean_text(text) == expected

            def test_clean_text_non_string_input():
                assert clean_text(None) == ""
                assert clean_text(123) == ""
            ```

2.  **데이터 전처리 (`preprocess(records: list[dict])`)**
    *   **기능**: 수집된 레코드 목록을 받아 Pandas DataFrame으로 변환하고, 다음을 수행합니다:
        *   필수 필드(name, description, url, category, tags, source, collected_at)를 확인하고 없는 경우 `None`으로 채웁니다.
        *   `description` 필드에 `clean_text` 함수를 적용하여 텍스트를 정리합니다.
        *   `name`과 `url` 조합을 기준으로 중복된 레코드를 제거합니다.
        *   `name`, `description`, `url` 중 하나라도 누락된 레코드를 제거합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 다양한 레코드 입력에 대해 데이터가 올바르게 전처리되고, 스키마 통일, 텍스트 클리닝, 중복 제거, 필수 필드 누락 처리가 정확하게 이루어지는지 확인합니다.
        *   **테스트 시나리오**:
            *   **정상 작동**: 완전하고 유효한 레코드 목록에 대해 `DataFrame`이 올바르게 생성되고 `description`이 클리닝되는지 확인.
            *   **HTML 및 공백 처리**: HTML 태그와 불필요한 공백이 포함된 `description`이 `clean_text`를 통해 올바르게 처리되는지 확인.
            *   **중복 제거**: `name`과 `url`이 동일한 중복 레코드가 올바르게 제거되는지 확인.
            *   **필수 필드 누락**: `name`, `description`, `url` 중 하나라도 `None`인 레코드가 제거되는지 확인.
            *   **필드 추가/누락**: 입력 레코드에 필수 필드 외의 필드가 포함되거나 일부 필수 필드가 없는 경우 `DataFrame` 스키마가 올바르게 통일되는지 확인.
            *   **빈 레코드 목록**: 빈 목록이 주어졌을 때 빈 `DataFrame`이 반환되는지 확인.
        *   **`pytest`를 사용한 접근**: Pandas DataFrame을 생성하여 `preprocess` 함수를 호출하고, 반환된 `DataFrame`의 행 수, 컬럼, 특정 셀의 값 등을 어설션합니다. `clean_text` 함수가 `preprocess` 내에서 올바르게 호출되는지 확인할 수 있습니다.
            ```python
            # backend/processor/tests/test_processor.py 예시
            import pandas as pd
            from backend.processor.processor import preprocess

            def test_preprocess_basic_cleaning_and_deduplication():
                records = [
                    {
                        "name": "Tool A",
                        "description": "<p>Desc <b>one</b>.</p>",
                        "url": "http://example.com/a",
                        "source": "test",
                        "category": "Utils", "tags": ["t1"], "collected_at": "2024-01-01"
                    },
                    {
                        "name": "Tool B",
                        "description": " Desc Two  ",
                        "url": "http://example.com/b",
                        "source": "test",
                        "category": "Dev", "tags": ["t2"], "collected_at": "2024-01-02"
                    },
                    {
                        "name": "Tool A", # Duplicate
                        "description": "Another desc one.",
                        "url": "http://example.com/a",
                        "source": "test",
                        "category": "Utils", "tags": ["t1"], "collected_at": "2024-01-03"
                    },
                    {
                        "name": "Tool C",
                        "description": None, # Missing required field
                        "url": "http://example.com/c",
                        "source": "test",
                        "category": "Other", "tags": ["t3"], "collected_at": "2024-01-04"
                    }
                ]
                df = preprocess(records)

                # Check deduplication and missing fields removal
                assert len(df) == 2
                assert "Tool A" in df["name"].tolist()
                assert "Tool B" in df["name"].tolist()
                assert "Tool C" not in df["name"].tolist()

                # Check text cleaning
                assert df[df["name"] == "Tool A"]["description"].iloc[0] == "desc one."
                assert df[df["name"] == "Tool B"]["description"].iloc[0] == "desc two"

                # Check required fields presence
                expected_cols = ["name", "description", "url", "category", "tags", "source", "collected_at"]
                assert all(col in df.columns for col in expected_cols)

            def test_preprocess_empty_records():
                df = preprocess([])
                assert df.empty
                expected_cols = ["name", "description", "url", "category", "tags", "source", "collected_at"]
                assert all(col in df.columns for col in expected_cols) # Even for empty, schema should be consistent
            ```

#### `backend/processor` 테스트 실행

`backend/processor` 디렉토리 내에 `tests` 디렉토리를 생성하고, 그 안에 `test_processor.py`와 같은 테스트 파일을 작성합니다.

```bash
# 백엔드 루트 디렉토리에서
cd backend
pytest processor/tests/
```

### 1.2.3. `backend/collectors` 테스트

`backend/collectors` 모듈은 외부 소스에서 데이터를 수집하는 역할을 담당합니다. 현재는 Product Hunt 데이터를 수집하는 `producthunt.py` 파일이 핵심입니다.

#### 주요 기능 및 검증 방법

1.  **Product Hunt 데이터 수집 (`fetch_producthunt(api_key)`)**
    *   **기능**: Product Hunt API에서 AI 툴 데이터를 가져옵니다. 현재 구현은 실제 API 호출 대신 미리 정의된 더미 데이터를 반환합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 함수가 예상된 형식(딕셔너리 리스트)의 데이터를 반환하는지 확인합니다. 각 데이터 항목이 필수 필드를 포함하고 올바른 데이터 타입과 내용을 가지는지 검증합니다. (향후 실제 API 연동 시) API 키 유효성, 네트워크 오류, 응답 형식 변경 등에 대한 견고성을 확인해야 합니다.
        *   **테스트 시나리오**:
            *   **더미 데이터 구조 검증**: 반환된 리스트가 비어 있지 않은지, 각 딕셔너리가 `name`, `description`, `url`, `category`, `tags`, `source`, `collected_at`과 같은 필수 키를 포함하는지 확인합니다.
            *   **데이터 타입 및 내용 검증**: `name`, `description`, `url`, `source`가 문자열이고, `category`와 `tags`가 문자열 리스트이며, `collected_at`이 ISO 형식의 날짜 문자열인지 확인합니다.
            *   **API 키 전달 확인 (현재는 더미이므로 로깅/모킹으로 확인)**: `api_key` 매개변수가 함수에 올바르게 전달되는지 확인합니다 (실제 API 호출 시 중요).
            *   **(향후 실제 API 연동 시)**: 유효하지 않은 API 키 또는 네트워크 문제 발생 시 적절한 예외 처리 또는 오류 반환을 검증합니다.
            *   **(향후 실제 API 연동 시)**: API 응답의 양이 많을 경우 페이징 처리 및 모든 데이터가 올바르게 수집되는지 확인합니다.
        *   **`pytest`를 사용한 접근**: `collectors/producthunt.py`의 `fetch_producthunt` 함수를 직접 임포트하여 호출하고 반환 값을 어설션합니다. 실제 API 호출이 없는 현재 구현에서는 모킹이 크게 필요하지 않지만, 향후 실제 API 연동 시 `requests.get` 또는 `requests.post`와 같은 HTTP 요청 라이브러리를 `unittest.mock.patch`를 사용하여 모킹해야 합니다.
            ```python
            # backend/collectors/tests/test_producthunt.py 예시
            import pytest
            from backend.collectors.producthunt import fetch_producthunt

            def test_fetch_producthunt_returns_expected_structure():
                api_key = "dummy_api_key"
                data = fetch_producthunt(api_key)

                assert isinstance(data, list)
                assert len(data) > 0 # Ensure dummy data is not empty

                # Check structure of the first item
                first_item = data[0]
                assert "name" in first_item
                assert "description" in first_item
                assert "url" in first_item
                assert "category" in first_item
                assert "tags" in first_item
                assert "source" in first_item
                assert "collected_at" in first_item

                assert isinstance(first_item["name"], str)
                assert isinstance(first_item["description"], str)
                assert isinstance(first_item["url"], str)
                assert isinstance(first_item["category"], str)
                assert isinstance(first_item["tags"], list)
                assert all(isinstance(tag, str) for tag in first_item["tags"])
                assert isinstance(first_item["source"], str)
                
                # Check collected_at format (simple check for ISO format with Z)
                assert isinstance(first_item["collected_at"], str)
                assert first_item["collected_at"].endswith('Z')
                # You could add more robust datetime parsing and validation here if needed
            
            # 향후 실제 API 연동 시, requests 모킹 예시:
            # from unittest.mock import patch
            # def test_fetch_producthunt_with_mocked_api_call():
            #     with patch('requests.post') as mock_post:
            #         mock_response = MagicMock()
            #         mock_response.status_code = 200
            #         mock_response.json.return_value = {"data": {"products": {"edges": []}}}
            #         mock_post.return_value = mock_response
            #         data = fetch_producthunt("real_api_key")
            #         assert len(data) == 0
            #         mock_post.assert_called_once()
            ```

#### `backend/collectors` 테스트 실행

`backend/collectors` 디렉토리 내에 `tests` 디렉토리를 생성하고, 그 안에 `test_producthunt.py`와 같은 테스트 파일을 작성합니다.

```bash
# 백엔드 루트 디렉토리에서
cd backend
pytest collectors/tests/
```

### 1.2.4. `backend/parser` 테스트

`backend/parser` 모듈은 원본 HTML 파일에서 텍스트를 파싱하고, 메타데이터를 추출하며, 텍스트를 청크로 분할하고, 임베딩을 생성하여 FAISS 인덱스를 구축하는 역할을 담당합니다. 핵심 파일은 `parser.py`입니다.

#### 주요 기능 및 검증 방법

1.  **메타데이터 추출 (`extract_metadata(text)`)**
    *   **기능**: 주어진 텍스트에서 정규 표현식을 사용하여 가격, API 제공 여부, 카테고리, 태그와 같은 AI 도구 관련 메타데이터를 추출합니다.
    *   **검증 방법**:
        *   **테스트 목표**: 다양한 텍스트 입력에 대해 함수가 올바른 메타데이터를 추출하는지 확인합니다. 특히 가격, API 키워드, 카테고리/태그 키워드 인식에 중점을 둡니다.
        *   **테스트 시나리오**:
            *   **가격 추출**: `"$9.99/month"`, `"$100 per year"` 등 다양한 형식의 가격 정보에서 정확한 가격 값 추출.
            *   **API 유무**: `"API support"`, `"developer SDK"`, `"REST API"` 등의 키워드가 있을 때 `has_api`가 `True`로 설정되고 `api_enabled` 태그가 추가되는지 확인.
            *   **카테고리 및 태그**: `"image generation"`, `"text generation"`, `"developer tools"` 등 다양한 카테고리/태그 키워드에 대해 올바른 카테고리 및 태그가 추가되는지 확인.
            *   **중복 태그 제거**: 동일한 태그가 여러 번 추출될 때 최종 목록에서 중복이 제거되는지 확인.
            *   **정보 부재**: 아무런 관련 정보가 없는 텍스트에 대해 기본 메타데이터(빈 카테고리/태그, `has_api=False`, `price=None`)가 반환되는지 확인.
        *   **`pytest`를 사용한 접근**: `parser.py`의 `extract_metadata` 함수를 직접 임포트하여 다양한 입력 텍스트로 함수를 호출하고 반환된 딕셔너리의 내용을 어설션합니다.
            ```python
            # backend/parser/tests/test_parser.py 예시
            from backend.parser.parser import extract_metadata

            def test_extract_metadata_price():
                text = "Our plan starts at $19.99/month. Includes API."
                metadata = extract_metadata(text)
                assert metadata["price"] == 19.99
                assert metadata["has_api"] == True

            def test_extract_metadata_categories_and_tags():
                text = "Generate stunning images with our AI art generator. Also great for video editing."
                metadata = extract_metadata(text)
                assert "Image Generation" in metadata["category"]
                assert "Video Editing" in metadata["category"]
                assert "image" in metadata["tags"]
                assert "art" in metadata["tags"]
                assert "video" in metadata["tags"]
                assert len(set(metadata["tags"])) == len(metadata["tags"]) # Check for no duplicates

            def test_extract_metadata_no_match():
                text = "A generic tool for daily tasks."
                metadata = extract_metadata(text)
                assert metadata["price"] is None
                assert metadata["has_api"] == False
                assert metadata["category"] == []
                assert metadata["tags"] == []
            ```

2.  **HTML 파싱 및 청킹 (`parse_and_chunk()`)**
    *   **기능**: `raw_html_storage` 디렉토리의 HTML 파일을 읽어 파싱하고, 텍스트를 추출하며, `SentenceSplitter`를 사용하여 텍스트를 청크(노드)로 분할합니다. 각 노드에 메타데이터를 추가하고, 임베딩을 생성하며, 최종적으로 FAISS 인덱스를 구축하고 처리된 메타데이터를 JSON 파일로 저장합니다.
    *   **검증 방법**:
        *   **테스트 목표**: HTML 파일이 올바르게 파싱되고, 텍스트가 추출되며, 청크가 생성되고, 메타데이터가 정확하게 할당되며, FAISS 인덱스와 메타데이터 파일이 성공적으로 생성되는지 확인합니다.
        *   **테스트 시나리오**:
            *   **정상 작동**: 유효한 HTML 파일이 있는 `raw_html_storage` 디렉토리에 대해 함수가 성공적으로 실행되고, `processed_data_storage`에 `ai_tools_faiss_index.bin` 및 `processed_ai_tools_metadata.json` 파일이 생성되는지 확인.
            *   **HTML 콘텐츠 파싱**: 복잡한 HTML 구조(중첩 태그, 다양한 공백)에서 순수 텍스트가 올바르게 추출되고 정리되는지 확인. `soup.get_text(separator=' ', strip=True)`의 작동 검증.
            *   **메타데이터 통합**: `extract_metadata` 함수로 추출된 메타데이터와 파일명 기반 태그가 노드의 메타데이터에 올바르게 결합되는지 확인.
            *   **청크 분할**: `SentenceSplitter`가 텍스트를 적절한 크기로 분할하는지, 오버랩이 올바르게 적용되는지 확인. 각 청크가 독립적인 의미 단위를 가지는지 확인.
            *   **임베딩 생성**: 각 청크에 대해 임베딩이 성공적으로 생성되는지 확인 (실제 임베딩 값보다는 생성이 성공했는지 여부).
            *   **FAISS 인덱스 생성 및 저장**: 임베딩이 생성되었을 때 FAISS 인덱스가 올바르게 구축되고 지정된 경로에 저장되는지 확인.
            *   **메타데이터 JSON 저장**: 모든 처리된 메타데이터 레코드가 `processed_ai_tools_metadata.json` 파일에 올바른 형식으로 저장되는지 확인.
            *   **빈 `raw_html_storage`**: `raw_html_storage` 디렉토리가 비어 있거나 존재하지 않을 때 함수가 적절한 오류 메시지를 출력하고 종료되는지 확인.
        *   **`pytest`를 사용한 접근**: `pytest-mock` 또는 `unittest.mock`을 사용하여 파일 시스템(예: `os.listdir`, `builtins.open`, `os.makedirs`) 및 외부 라이브러리(예: `BeautifulSoup`, `SentenceSplitter`, `SentenceTransformer`, `faiss.write_index`)의 상호작용을 모킹하여 `parse_and_chunk` 로직을 테스트합니다. 임시 디렉토리를 생성하여 실제 파일 시스템과 유사한 환경에서 테스트할 수도 있습니다.
            ```python
            # backend/parser/tests/test_parser.py 예시
            import pytest
            from unittest.mock import patch, MagicMock
            import os
            import json
            import tempfile
            from backend.parser.parser import parse_and_chunk, embedding_model
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core.schema import Document, TextNode

            @pytest.fixture
            def setup_temp_dirs():
                with tempfile.TemporaryDirectory() as raw_dir, \
                     tempfile.TemporaryDirectory() as processed_dir:
                    # Mock os.path.dirname(__file__) to point to a test-controlled path
                    with patch('os.path.dirname', return_value=os.path.join(os.getcwd(), "backend/parser")):
                        # Mock the paths used in parse_and_chunk to use temp dirs
                        with patch('backend.parser.parser.raw_html_dir', raw_dir), \
                             patch('backend.parser.parser.processed_data_dir', processed_dir):
                            yield raw_dir, processed_dir

            @patch('backend.parser.parser.SentenceTransformer') # Mock the SentenceTransformer loading
            @patch('backend.parser.parser.faiss.write_index')
            def test_parse_and_chunk_success(mock_write_index, mock_sentence_transformer, setup_temp_dirs):
                raw_dir, processed_dir = setup_temp_dirs
                
                # Create a dummy HTML file
                html_content = "<html><head><title>Test AI Tool</title></head><body><h1>Welcome</h1><p>This is an AI tool for <b>image generation</b>.</p></body></html>"
                with open(os.path.join(raw_dir, "test_ai_tool.html"), "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                # Mock embedding model to return dummy embeddings
                mock_embed_instance = MagicMock()
                mock_embed_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype='float32')
                mock_sentence_transformer.return_value = mock_embed_instance

                parse_and_chunk()

                # Assertions
                assert os.path.exists(os.path.join(processed_dir, "ai_tools_faiss_index.bin"))
                assert os.path.exists(os.path.join(processed_dir, "processed_ai_tools_metadata.json"))
                mock_write_index.assert_called_once()

                with open(os.path.join(processed_dir, "processed_ai_tools_metadata.json"), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                assert len(metadata) > 0
                assert metadata[0]["name"] == "Test AI Tool"
                assert "image generation" in metadata[0]["description"].lower()
                assert "Image Generation" in metadata[0]["category"]
                assert "image" in metadata[0]["tags"]
                assert "ai" in metadata[0]["tags"]

            @patch('backend.parser.parser.faiss.write_index')
            @patch('builtins.print')
            def test_parse_and_chunk_no_html_files(mock_print, mock_write_index, setup_temp_dirs):
                raw_dir, processed_dir = setup_temp_dirs
                # No HTML files created in raw_dir

                parse_and_chunk()

                mock_print.assert_any_call(f'오류: 원본 HTML 파일이 없습니다. {raw_dir} 디렉토리를 확인해주세요.')
                assert not os.path.exists(os.path.join(processed_dir, "ai_tools_faiss_index.bin"))
                assert not os.path.exists(os.path.join(processed_dir, "processed_ai_tools_metadata.json"))
                mock_write_index.assert_not_called()
            ```

#### `backend/parser` 테스트 실행

`backend/parser` 디렉토리 내에 `tests` 디렉토리를 생성하고, 그 안에 `test_parser.py`와 같은 테스트 파일을 작성합니다.

```bash
# 백엔드 루트 디렉토리에서
cd backend
pytest parser/tests/
```

### 1.2.5. `backend/worker` 테스트

`backend/worker` 모듈은 웹사이트를 스크래핑하여 HTML 콘텐츠를 수집하고 저장하는 역할을 담당합니다. 핵심 파일은 `worker.py`이며, `scraper_config.yaml` 파일을 통해 스크래핑 대상을 설정합니다.

#### 주요 기능 및 검증 방법

1.  **웹사이트 스크래핑 (`scrape_site(config)`)**
    *   **기능**: 주어진 설정(`config`)에 따라 Playwright를 사용하여 웹사이트로 이동하고, HTML 콘텐츠를 추출하여 `raw_html_storage` 디렉토리에 저장합니다.
    *   **검증 방법**:
        *   **테스트 목표**: `scrape_site` 함수가 성공적으로 웹 페이지에 접속하고, HTML을 가져와, 지정된 디렉토리에 올바른 파일명으로 저장하는지 확인합니다. 네트워크 오류나 페이지 로딩 실패 시 예외 처리가 올바르게 동작하는지 확인합니다.
        *   **테스트 시나리오**:
            *   **정상 작동**: 유효한 URL과 사이트 이름이 포함된 `config`가 주어졌을 때, Playwright 브라우저가 실행되고, 페이지가 로드되며, HTML 콘텐츠가 `raw_html_storage`에 파일로 저장되는지 확인합니다. 저장된 파일의 내용이 예상되는 HTML의 일부를 포함하는지 검증합니다.
            *   **잘못된 URL/네트워크 오류**: 존재하지 않거나 접근할 수 없는 URL이 주어졌을 때, Playwright가 예외를 발생시키고 `scrape_site` 함수가 이를 처리하며 적절한 오류 메시지를 출력하는지 확인합니다.
            *   **타임아웃 처리**: 페이지 로딩이 `timeout`을 초과했을 때 예외가 발생하고 처리되는지 확인합니다.
            *   **파일명 규칙**: 저장되는 HTML 파일의 이름이 `site_name`과 현재 시간을 포함하여 올바르게 생성되는지 확인합니다.
            *   **`raw_html_storage` 디렉토리 생성**: `raw_html_storage` 디렉토리가 없을 때 자동으로 생성되는지 확인합니다.
        *   **`pytest`를 사용한 접근**: `pytest-asyncio`를 사용하여 비동기 함수를 테스트합니다. `unittest.mock.patch`를 사용하여 Playwright의 `async_playwright`, `browser.new_page`, `page.goto`, `page.content` 등을 모킹하여 실제 브라우저 실행 및 네트워크 요청 없이 로직을 테스트합니다. `os.makedirs` 및 `builtins.open`도 모킹하여 파일 시스템 상호작용을 제어할 수 있습니다.
            ```python
            # backend/worker/tests/test_worker.py 예시
            import pytest
            from unittest.mock import patch, AsyncMock, MagicMock
            import os
            import asyncio
            from backend.worker.worker import scrape_site

            @pytest.mark.asyncio
            @patch('playwright.async_api.async_playwright')
            @patch('os.makedirs')
            @patch('builtins.open', new_callable=MagicMock)
            @patch('os.path.join', side_effect=os.path.join) # Keep original join behavior
            def test_scrape_site_success(mock_join, mock_open, mock_makedirs, mock_async_playwright):
                # Setup mocks for Playwright
                mock_p = AsyncMock()
                mock_browser = AsyncMock()
                mock_page = AsyncMock()
                mock_async_playwright.return_value.__aenter__.return_value = mock_p
                mock_p.chromium.launch.return_value = mock_browser
                mock_browser.new_page.return_value = mock_page
                mock_page.goto.return_value = None
                mock_page.content.return_value = "<html><body>Test Content</body></html>"

                config = {"site_name": "Test Site", "url": "http://test.com"}

                asyncio.get_event_loop().time = MagicMock(return_value=123456789.0) # Mock time for filename

                await scrape_site(config)

                mock_async_playwright.assert_called_once()
                mock_p.chromium.launch.assert_called_once()
                mock_browser.new_page.assert_called_once()
                mock_page.goto.assert_called_once_with(config['url'], timeout=60000)
                mock_page.wait_for_load_state.assert_called_once_with("networkidle")
                mock_page.content.assert_called_once()
                
                mock_makedirs.assert_called_once_with("raw_html_storage", exist_ok=True)
                mock_open.assert_called_once()
                mock_open.return_value.__enter__.return_value.write.assert_called_once_with("<html><body>Test Content</body></html>")
                mock_browser.close.assert_called_once()

            @pytest.mark.asyncio
            @patch('playwright.async_api.async_playwright')
            @patch('builtins.print')
            async def test_scrape_site_error_handling(mock_print, mock_async_playwright):
                mock_p = AsyncMock()
                mock_browser = AsyncMock()
                mock_page = AsyncMock()
                mock_async_playwright.return_value.__aenter__.return_value = mock_p
                mock_p.chromium.launch.return_value = mock_browser
                mock_browser.new_page.return_value = mock_page
                mock_page.goto.side_effect = Exception("Network error") # Simulate error

                config = {"site_name": "Error Site", "url": "http://error.com"}

                await scrape_site(config)

                mock_browser.close.assert_called_once()
                mock_print.assert_any_call('스크래핑 중 오류 발생 Error Site: Network error')
            ```

2.  **메인 실행 (`main()`)**
    *   **기능**: `scraper_config.yaml` 파일에서 스크래핑 설정을 로드하고, 각 설정에 대해 `scrape_site` 함수를 비동기적으로 실행합니다.
    *   **검증 방법**:
        *   **테스트 목표**: `main` 함수가 `scraper_config.yaml`을 올바르게 로드하고, 파일이 없을 때 적절하게 실패하며, 여러 `scrape_site` 호출을 병렬로 트리거하는지 확인합니다.
        *   **테스트 시나리오**:
            *   **정상 작동**: 유효한 `scraper_config.yaml` 파일이 있을 때, 파일이 올바르게 로드되고 각 설정에 대해 `scrape_site`가 호출되는지 확인합니다.
            *   **설정 파일 부재**: `scraper_config.yaml` 파일이 없을 때 함수가 적절한 오류 메시지를 출력하고 종료되는지 확인합니다.
            *   **여러 설정 처리**: `scraper_config.yaml`에 여러 사이트 설정이 포함되어 있을 때 모든 `scrape_site` 호출이 성공적으로 트리거되는지 확인합니다.
        *   **`pytest`를 사용한 접근**: `pytest-asyncio`를 사용합니다. `unittest.mock.patch`를 사용하여 `os.path.exists`, `builtins.open`, `yaml.safe_load`, `asyncio.gather` 및 `scrape_site` 함수를 모킹합니다. 이를 통해 실제 파일 시스템 상호작용 및 웹 스크래핑 없이 `main` 함수의 로직 흐름을 테스트합니다.
            ```python
            # backend/worker/tests/test_worker.py 예시
            import pytest
            from unittest.mock import patch, AsyncMock
            import os
            from backend.worker.worker import main, scrape_site

            @pytest.mark.asyncio
            @patch('os.path.exists')
            @patch('builtins.open', new_callable=MagicMock)
            @patch('yaml.safe_load')
            @patch('asyncio.gather', new_callable=AsyncMock) # Mock asyncio.gather
            @patch('backend.worker.worker.scrape_site', new_callable=AsyncMock) # Mock scrape_site
            async def test_main_success(mock_scrape_site, mock_gather, mock_yaml_load, mock_open, mock_exists):
                mock_exists.return_value = True
                mock_yaml_load.return_value = [
                    {"site_name": "Site1", "url": "http://site1.com"},
                    {"site_name": "Site2", "url": "http://site2.com"}
                ]

                await main()

                mock_open.assert_called_once()
                mock_yaml_load.assert_called_once()
                assert mock_scrape_site.call_count == 2 # Called for each config
                # Ensure gather was called with the correct tasks
                mock_gather.assert_called_once_with(
                    mock_scrape_site.return_value, mock_scrape_site.return_value
                )
            
            @pytest.mark.asyncio
            @patch('os.path.exists')
            @patch('builtins.print')
            async def test_main_no_config_file(mock_print, mock_exists):
                mock_exists.return_value = False
                await main()
                mock_print.assert_any_call('오류: scraper_config.yaml 파일을 찾을 수 없습니다: ' + os.path.join(os.path.dirname(__file__), 'scraper_config.yaml'))
            ```

#### `backend/worker` 테스트 실행

`backend/worker` 디렉토리 내에 `tests` 디렉토리를 생성하고, 그 안에 `test_worker.py`와 같은 테스트 파일을 작성합니다.

```bash
# 백엔드 루트 디렉토리에서
cd backend
pytest worker/tests/
```

## 2. 프론트엔드 테스트

프론트엔드는 HTML, CSS, JavaScript로 구성된 웹 애플리케이션입니다. 현재 프로젝트 구조에서는 별도의 단위 테스트 프레임워크가 설정되어 있지 않습니다. 주로 수동 테스트 또는 브라우저 기반의 통합 테스트가 필요합니다.

### 2.1. 수동 테스트 및 브라우저 개발자 도구 활용

프론트엔드 기능은 주로 사용자 인터랙션과 백엔드 API와의 통신을 포함합니다. 다음 단계를 통해 수동으로 기능을 검증할 수 있습니다.

1.  **웹 페이지 열기:**
    `frontend/index.html` 파일을 웹 브라우저에서 직접 엽니다.
    ```bash
    start frontend/index.html # Windows
    # open frontend/index.html # macOS
    # xdg-open frontend/index.html # Linux
    ```

2.  **주요 기능 시나리오 확인:**
    *   **페이지 로드 및 초기 상태**: 페이지가 로드될 때 모든 UI 요소(검색창, 버튼, 결과 표시 영역 등)가 올바른 초기 상태로 표시되는지 확인합니다.
    *   **검색 기능**: 검색 입력란에 쿼리를 입력하고 검색 버튼을 클릭했을 때:
        *   백엔드 `/api/v1/search` 엔드포인트로 요청이 올바르게 전송되는지 확인합니다. (브라우저 개발자 도구의 `Network` 탭 활용)
        *   검색 결과가 페이지에 올바르게 표시되는지 확인합니다. (HTML 구조, 데이터 일치 여부)
        *   필터링(예: 카테고리, 가격)이 적용되었을 때 결과가 정확하게 반영되는지 확인합니다.
    *   **기타 인터랙션**: 페이지 내의 모든 버튼, 링크, 입력 필드 등 사용자 인터랙션 요소가 예상대로 작동하는지 확인합니다.

3.  **브라우저 개발자 도구 활용 (F12)**
    *   **Console 탭**: JavaScript 오류, 경고 또는 디버깅 메시지가 없는지 확인합니다. 콘솔에 출력되는 메시지를 통해 애플리케이션의 내부 동작을 이해할 수 있습니다.
    *   **Network 탭**: 백엔드 API 호출이 성공적으로 이루어지는지(HTTP 상태 코드 200), 요청 페이로드와 응답 데이터가 예상과 일치하는지 확인합니다. 특히 `/api/v1/search` 호출에 대한 응답 데이터를 주의 깊게 검사합니다.
    *   **Elements 탭**: 동적으로 생성되는 HTML 요소나 CSS 스타일이 올바르게 적용되는지 확인합니다. 검색 결과가 표시될 때 해당 HTML 요소의 구조를 검사합니다.
    *   **Sources 탭**: `script.js` 코드에 중단점(breakpoint)을 설정하여 JavaScript 실행 흐름을 단계별로 추적하고 변수 값을 검사할 수 있습니다. 백엔드 응답을 처리하는 로직을 디버깅하는 데 유용합니다.

### 2.2. 자동화된 프론트엔드 테스트 설정 제안

프로젝트가 성장하고 프론트엔드 복잡성이 증가함에 따라, 수동 테스트만으로는 한계가 있습니다. 다음은 자동화된 테스트 환경 구축을 위한 제안입니다.

-   **단위 테스트 (Unit Testing for `script.js`):**
    `script.js` 파일 내의 독립적인 함수나 유틸리티 로직에 대한 단위 테스트를 위해 [Jest](https://jestjs.io/)와 같은 JavaScript 테스트 프레임워크를 설정할 수 있습니다.
    ```bash
    # frontend 디렉토리에서
    cd frontend
    npm init -y # package.json이 없는 경우
    npm install --save-dev jest
    ```
    이후 `package.json`의 `scripts` 섹션에 테스트 명령을 추가하고, `frontend/tests/` 디렉토리에 `test_*.js` 파일을 작성할 수 있습니다.

-   **통합/E2E 테스트 (Integration/End-to-End Testing):**
    전체 사용자 흐름을 시뮬레이션하고 백엔드와의 통합을 검증하기 위해 [Cypress](https://www.cypress.io/) 또는 [Playwright](https://playwright.dev/)와 같은 브라우저 자동화 도구를 사용할 수 있습니다. 이 도구들은 실제 브라우저 환경에서 사용자 인터랙션을 모방하여 테스트를 실행할 수 있습니다.
    ```bash
    # frontend 디렉토리에서 (예시: Playwright 설치)
    cd frontend
    npm init playwright@latest # Playwright 설치 및 초기 설정
    # 설치 시 JavaScript/TypeScript, 브라우저 선택 등 프롬프트에 응답
    ```
    이러한 도구를 사용하면 UI 컴포넌트, 페이지 탐색, 폼 제출, API 호출 결과 반영 등 복합적인 시나리오를 자동화하여 테스트할 수 있습니다.

---

이 문서는 프로젝트의 테스트 현황과 기본적인 실행 방법을 다룹니다. 코드베이스가 발전함에 따라 이 문서도 함께 업데이트되어야 합니다. 각 모듈의 상세 기능 분석을 통해 보다 구체적인 테스트 가이드를 제공할 예정입니다. 