# rag-optimization-experiment

RAG 파라미터(TopK, Chunk Size, Overlap) 최적화 실험

## Contents

- [`experiment.md`](experiments/experiment.md) - 실험 결과 및 분석 문서
- [`preprocessing_pipeline.py`](experiments/preprocessing_pipeline.py) - 문서 청킹 및 임베딩 파이프라인
- [`experiment.py`](experiments/experiment.py) - RAG 실험 실행 스크립트
- [`evaluate.py`](experiments/evaluate.py) - RAGAS 메트릭 기반 성능 평가
- [`questions.json`](experiments/questions.json) - 평가용 질문 셋
- [`evaluation_results.json`](experiments/evaluation_results.json) - 평가 결과 데이터

## Setup

```bash
cd experiments
pip install -r requirements.txt
```

## Download Dataset

```bash
git clone https://github.com/kubernetes/website
cp -r  ./website/content/ko ./ko
```

## Pinecone Setup

Pinecone에 `rag-notes` 인덱스와 아래 9개의 네임스페이스 생성이 필요합니다.

| Namespace | Chunk Size | Overlap |
|-----------|------------|---------|
| cs256-ov0 | 256 | 0% |
| cs256-ov15 | 256 | 15% |
| cs256-ov30 | 256 | 30% |
| cs512-ov0 | 512 | 0% |
| cs512-ov15 | 512 | 15% |
| cs512-ov30 | 512 | 30% |
| cs1024-ov0 | 1024 | 0% |
| cs1024-ov15 | 1024 | 15% |
| cs1024-ov30 | 1024 | 30% |

## Usage

모든 스크립트는 `experiments/` 디렉토리에서 실행합니다.

### 1. 문서 전처리 및 임베딩

```bash
python preprocessing_pipeline.py
```

`./ko` 폴더의 마크다운 문서를 청킹하여 Pinecone에 업로드합니다. 위 9개 네임스페이스에 각각 다른 파라미터로 임베딩된 데이터가 저장됩니다.

### 2. RAG 실험 실행

```bash
python experiment.py
```

`questions.json`의 질문들에 대해 다양한 파라미터 조합으로 RAG 실험을 수행하고, 결과를 `experiment_results.json`에 저장합니다.

### 3. RAGAS 평가

```bash
python evaluate.py
```

`experiment_results.json`의 실험 결과를 RAGAS 메트릭으로 평가하고, 결과를 `evaluation_results.json`에 저장합니다.