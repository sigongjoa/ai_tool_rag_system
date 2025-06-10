# 4. 시스템 아키텍처 및 배포

지금까지 설계한 개별 컴포넌트(Worker, Parser, DB, API 등)들은 안정적이고 확장 가능한 방식으로 통합 운영되어야 합니다. 본 문서에서는 `Docker-Compose`를 활용한 컨테이너화, CI/CD 파이프라인 구축, 그리고 MLOps(모니터링, 품질 평가) 전략에 대해 기술합니다.

## 4.1. 시스템 아키텍처 (MVP)

단일 고성능 머신(예: RTX 4090/3090 1-GPU, 32GB RAM) 환경에서 모든 컴포넌트를 효율적으로 운영하는 것을 목표로 합니다.

### 가. 컨테이너화 (Docker-Compose)

`docker-compose.yml` 파일을 사용하여 전체 시스템을 한 번에 실행하고 관리합니다. 각 서비스는 독립적인 `Dockerfile`을 가집니다.

**`docker-compose.yml` 예시:**
```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.9.2
    container_name: qdrant_db
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    networks:
      - rag_network

  vllm:
    image: vllm/vllm-openai:latest
    container_name: llm_server
    # GPUs 사용을 위한 설정
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      # 사용할 LLM 모델 지정 (예: Mixtral)
      - MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
    ports:
      - "8000:8000"
    networks:
      - rag_network

  api-gateway:
    build: ./api_gateway # FastAPI 애플리케이션 경로
    container_name: api_gateway
    ports:
      - "8080:80"
    depends_on:
      - qdrant
      - vllm
    networks:
      - rag_network
  
  worker:
    build: ./worker # 데이터 수집 Worker 경로
    container_name: ingestion_worker
    # '0 0 * * *' (매일 자정) 스케줄 실행을 위한 entrypoint 설정
    command: ["cron", "-f"] 
    depends_on:
      - qdrant
    networks:
      - rag_network

networks:
  rag_network:
    driver: bridge
```

## 4.2. 배포 및 MLOps (CI/CD)

코드 변경 사항을 안정적으로 프로덕션에 배포하고, 운영 중인 시스템의 상태를 지속적으로 관찰 및 개선합니다.

### 가. CI (Continuous Integration)

- **도구:** `GitHub Actions`
- **워크플로우 (`.github/workflows/ci.yml`):**
  1. **On Push/Pull Request to `main`:** main 브랜치에 코드가 푸시되거나 PR이 생성될 때 워크플로우를 트리거합니다.
  2. **Lint & Test:** `flake8`, `pytest` 등을 사용하여 코드 스타일과 유닛 테스트를 실행합니다.
  3. **Docker Image Build:** `docker buildx`를 사용하여 멀티-아키텍처 이미지를 빌드합니다.
  4. **Security Scan:** `Trivy` 또는 `Snyk`을 사용하여 빌드된 Docker 이미지의 취약점을 스캔합니다.
  5. **Push to Registry:** 모든 검사를 통과한 이미지를 Docker Hub 또는 ECR(AWS)과 같은 컨테이너 레지스트리에 푸시합니다.

### 나. CD (Continuous Deployment)

- **도구:** `Argo Rollouts` (Kubernetes 환경) 또는 `docker-compose pull && docker-compose up -d` (단일 서버 환경)
- **전략:** Blue-Green 배포를 통해 무중단으로 새 버전을 릴리즈합니다. 새 버전(Green)을 먼저 배포하고, 정상 동작이 확인되면 트래픽을 이전 버전(Blue)에서 Green으로 전환합니다. 문제가 발생하면 즉시 Blue로 롤백합니다.

### 다. 모니터링

- **도구:** `Prometheus` + `Grafana`
- **주요 모니터링 지표:**
  - **Vector DB:** QPS(초당 쿼리 수), Latency(응답 지연 시간), 디스크 사용량
  - **GPU:** GPU 사용률(Utilization), 메모리 사용량, 온도
  - **API Server:** HTTP 요청 수, 응답 상태 코드(2xx, 4xx, 5xx), 엔드포인트별 평균 지연 시간
- `Grafana` 대시보드를 구축하여 위 지표들을 시각화하고, 특정 임계치(예: GPU 온도 > 85°C)를 초과할 경우 Slack 또는 이메일로 알림을 받도록 설정합니다.

## 4.3. 품질 평가 및 자동 벤치마크

- **Retrieval 평가:**
  - **Metrics:** `R@5`(Recall@5), `nDCG@10`, `MRR`(Mean Reciprocal Rank)
  - **방법:** 수작업으로 "질문-정답 문서 ID" 쌍으로 구성된 평가 데이터셋(Gold Set)을 구축하고, Retriever가 정답 문서를 얼마나 높은 순위에 가져오는지 주기적으로 측정합니다.

- **Generation 평가:**
  - **Metrics:** `BLEU`, `ROUGE-L` (vs. Human Summary), LLM-as-a-Judge
  - **방법:** 정답 요약본과 LLM이 생성한 답변 간의 유사도를 측정하거나, 더 강력한 LLM(예: GPT-4o)을 심판으로 사용하여 생성된 답변의 논리성, 명료성 등을 점수화합니다.

- **A/B 테스트:**
  - 새로운 임베딩 모델이나 프롬프트 템플릿을 도입할 때, 기존 버전과 새 버전에 트래픽을 50:50으로 분배합니다. 이후 실제 사용자의 클릭률, 재방문율, 만족도 피드백 등을 측정하여 어느 버전이 더 우수한지 객관적으로 평가합니다.

- **피드백 루프:**
  - 사용자가 입력한 쿼리와 클릭한 검색 결과 로그를 수집하여, "어떤 질문에 어떤 문서가 가장 유용했는가"에 대한 데이터를 축적합니다. 이 데이터는 주기적으로 fine-tuning 데이터셋을 생성하는 데 사용되어, 모델 성능을 지속적으로 개선하는 선순환 구조를 만듭니다.

---
> **다음 문서:** [05_Roadmap.md](./05_Roadmap.md) 에서는 MVP 이후 시스템을 확장하기 위한 장기적인 로드맵을 제시합니다. 