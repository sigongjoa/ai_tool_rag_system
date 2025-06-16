# Phase 3: LM Studio 기반 답변 모델 파인튜닝

## 목표
실제 사용자 질의 데이터를 활용하여 로컬 LLM을 파인튜닝하고, 도메인에 특화된 고품질 답변을 생성하는 모델을 개발합니다.

## 3-1. 학습 데이터셋 구축 파이프라인 생성

### 목표
`/api/v1/search` 엔드포인트에서 발생하는 사용자 질문, 검색된 청크 내용, 최종 답변 등을 수집하여 모델 학습에 필요한 (input, label) 쌍의 데이터셋을 자동으로 또는 반자동으로 생성합니다.

### 구현 상세

#### 3-1-1. 로그 수집 및 저장
- `/api/v1/search` 엔드포인트에서 요청 처리 후, 다음 정보를 로그 또는 별도의 DB/파일에 저장합니다:
  - `request_id`: 요청 고유 ID
  - `timestamp`: 요청 시각
  - `user_query`: 사용자의 원본 질문
  - `retrieved_items`: FAISS 검색 및 Reranking 후 최종적으로 LLM에 전달된 도구 메타데이터 (이름, URL, 설명 등)
  - `llm_response_content`: LLM이 생성한 최종 답변
- 이 데이터는 모델 튜닝을 위한 원시 데이터로 사용됩니다.

#### 3-1-2. 데이터 정제 및 가공 (스크립트 개발)
- 수집된 원시 데이터를 사용하여 모델 학습을 위한 `(input, label)` 쌍을 생성하는 Python 스크립트를 별도로 개발합니다. (`scripts/generate_finetuning_data.py`)
- **Input (`prompt` 필드):** `사용자 질문 + FAISS에서 검색된 Top-k Chunk 요약` 형태로 구성합니다.
  - `retrieved_items`의 `description` 및 `chunk_text`를 요약하거나 직접 활용하여 `context_str`을 재구성합니다.
  - 예를 들어: `"질문: {user_query}\n\n관련 정보:\n{summarized_context}\n\n답변: "`
- **Label (`completion` 필드):** `llm_response_content`를 초기 `label`로 사용합니다. 향후 수동 검토나 GPT-4와 같은 고성능 LLM을 이용하여 "Golden Answer"로 대체/보강하는 파이프라인을 고려합니다.
- `DataFrame` 형태로 데이터를 처리하고, 최종적으로 JSON Lines (`.jsonl`) 형식으로 저장하여 `transformers` 라이브러리에서 쉽게 로드할 수 있도록 합니다.
  ```json
  {"prompt": "...", "completion": "..."}
  {"prompt": "...", ""completion": "..."}
  ```

#### 3-1-3. Synthetic QA 생성
- **목표:** 학습 데이터의 다양성을 확보하고, 모델이 특정 질문 유형이나 경계선 케이스에 더 강건하게 반응하도록 합니다.
- **구현:** 별도의 스크립트 (`scripts/generate_synthetic_data.py`)를 개발합니다.
  - `processed_metadata`의 도구 정보(설명, 태그, 카테고리 등)를 활용하여 다양한 질문과 예상 답변을 LLM(LM Studio 또는 외부 API)에게 생성하도록 요청합니다.
  - 예를 들어, 특정 도구의 `description`을 주고 "이 도구에 대해 궁금해할 만한 질문 5가지와 답변을 만들어줘" 와 같이 프롬프팅합니다.
  - 생성된 Synthetic QA 쌍도 정제 과정을 거쳐 기존 데이터셋과 병합합니다.

## 3-2. 파인튜닝 스크립트 작성

### 목표
Hugging Face의 `transformers`, `peft`, `trl` 라이브러리를 사용하여 LM Studio에서 서빙 중인 모델을 효율적으로 파인튜닝합니다.

### 구현 상세
- `scripts/finetune_model.py` 파일을 생성합니다.
- **라이브러리 사용:**
  - `transformers.AutoTokenizer`와 `AutoModelForCausalLM`를 사용하여 LM Studio 모델 (`open-solar-ko-10.7b` 등)을 로드합니다.
  - `peft.LoraConfig` 및 `get_peft_model`을 사용하여 LoRA(Low-Rank Adaptation) 설정을 정의하고 모델에 적용합니다. 4-bit 양자화를 통해 GPU 메모리 사용량을 절감합니다.
  - `trl.SFTTrainer`를 사용하여 학습 데이터셋으로 모델을 파인튜닝합니다.
- **학습 설정:**
  - `TrainingArguments`를 사용하여 `num_train_epochs` (3~5), `learning_rate` (예: 2e-5), `per_device_train_batch_size` 등의 하이퍼파라미터를 설정합니다.
  - 평가 셋은 전체 데이터의 10%를 홀드아웃하여 모델 성능을 검증합니다.
- **결과물 저장:** 파인튜닝된 LoRA 어댑터를 특정 디렉토리(`models/finetuned_adapter`)에 저장합니다.

## 3-3. 파인튜닝 모델 서빙

### 목표
FastAPI 애플리케이션 시작 시 학습된 LoRA 어댑터를 기본 LLM에 병합하여, `/api/v1/search` 엔드포인트에서 이 파인튜닝된 모델을 사용하도록 합니다.

### 구현 상세
`backend/api_gateway/main.py`의 `lifespan` 함수를 수정합니다.

**`backend/api_gateway/main.py` 수정 예정 사항:**
- `lifespan` 함수 내에서 `embedding_model` 로드 후, 다음 단계를 추가합니다:
  - `transformers.AutoTokenizer`와 `AutoModelForCausalLM`를 사용하여 LM Studio의 기본 LLM (`LLM_MODEL_NAME`)을 로드합니다.
  - `peft.PeftModel.from_pretrained`를 사용하여 저장된 LoRA 어댑터(`models/finetuned_adapter` 경로)를 로드합니다.
  - `model.merge_and_unload()` 메서드를 호출하여 LoRA 어댑터를 기본 모델에 병합합니다.
  - 병합된 모델을 `global llm_model`과 같은 전역 변수에 할당합니다. (LLM은 이미 `httpx.AsyncClient`를 통해 외부 LM Studio 서버에 연결되어 있으므로, 이 부분은 로컬에서 직접 모델을 로드하여 서빙하는 경우에 해당합니다. LM Studio를 계속 외부 서버로 사용하는 경우 이 단계는 불필요하며, LM Studio 서버 자체를 파인튜닝된 모델로 교체하거나, 다른 API 엔드포인트를 사용해야 합니다.)

**LM Studio를 외부 서버로 계속 사용하는 경우의 대안 (더 현실적):**
- LM Studio 서버 자체를 파인튜닝된 모델로 변경하는 것이 더 현실적인 접근 방식입니다.
- 또는, 파인튜닝된 모델을 별도의 FastAPI 서비스로 서빙하고, `llm_http_client`가 이 새로운 엔드포인트를 바라보도록 설정합니다.
- 여기서는 `main.py`에서 LLM을 직접 로드하는 방식이 아닌, LM Studio 서버를 사용하는 것을 전제로 `llm_http_client`가 특정 API_BASE를 바라보게 설정하는 것으로 간주하겠습니다. 따라서, `3-3 파인튜닝 모델 서빙`은 LM Studio 서버에 파인튜닝된 모델이 배포되었다는 가정하에 진행됩니다. 즉, 코드를 직접 수정하는 부분은 없을 수 있습니다. 문서에서는 LM Studio 서버를 파인튜닝된 모델로 구성하는 방법을 설명합니다.

**따라서, `backend/api_gateway/main.py`의 `lifespan` 함수 수정은 LLM 모델 자체를 로드하는 것이 아니라, `LLM_API_BASE` 환경 변수를 파인튜닝된 LM Studio 모델을 서빙하는 URL로 변경하여 사용한다는 내용으로 대체됩니다. LM Studio에서 파인튜닝된 모델을 로드하고 서버를 시작하는 것이 필요합니다.** 