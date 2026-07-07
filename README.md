# 논술 원고지 OCR 비교/Repair 로컬 웹앱

로컬 답안 이미지에서 하단 20x20 원고지 영역을 crop하고, 빨간펜 채점 흔적을 제거한 뒤 Chandra OCR과 GPT OAuth OCR 결과를 비교/repair하는 실험용 앱입니다.

## 실행

```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8767
```

브라우저:

```text
http://127.0.0.1:8767
```

## 환경변수

앱은 프로젝트 루트의 `.env`를 자동으로 읽습니다. 먼저 예시 파일을 복사한 뒤 키를 채우면 됩니다.

```bash
cp .env.example .env
```

```bash
export DOGOK_PROXY_API_KEY="dogok proxy key"
export GPT_OAUTH_BASE_URL="http://192.168.0.16:31835"
export PUBLIC_BASE_URL="http://192.168.0.127:8767"
export VLM_PRIMARY_BASE_URL="http://classday.iptime.org:8979/v1"
export VLM_FALLBACK_BASE_URL="http://210.115.224.151:31882/v1"
export VLM_MODEL_ID="chandra"
export VLLM_API_KEY="vllm api key"
```

`DOGOK_PROXY_API_KEY`가 없으면 GPT 실행은 비활성화됩니다. `PUBLIC_BASE_URL`은 dogok GPT proxy가 crop 이미지를 가져갈 수 있는 이 앱의 LAN URL이어야 합니다. Crop, red-mask preview, Chandra 설정 확인, 수동 편집/저장은 그대로 사용할 수 있습니다.

## 산출물

각 처리 run은 `runs/<run_id>/` 아래에 저장됩니다.

- `crop.png`
- `masked.png`
- `ocr_chandra.json`, `ocr_gpt.json`
- `render_chandra.html`, `render_gpt.html`
- `repair.json`, `render_repair.html`
- `final.json`, `final.html`

## OCR 실험 매트릭스

기본 OCR/repair 흐름과 별도로 같은 crop에 대해 여러 입력 전략을 비교할 수 있습니다.

- `full_grid`: 빨간펜 제거된 20x20 전체 crop을 한 번에 OCR합니다.
- `row_1`: 20개 행을 한 줄씩 잘라 각 행을 OCR한 뒤 20x20으로 합칩니다.
- `row_2`: 두 줄씩 잘라 10개 chunk를 OCR한 뒤 합칩니다.
- `row_5`: 다섯 줄씩 잘라 4개 chunk를 OCR한 뒤 합칩니다.
- `vote_full_row_1_2`: `full_grid`, `row_1`, `row_2` 결과를 cell 단위로 voting합니다.

웹 UI의 `실험 매트릭스` 패널에서 전략을 선택해 실행하면 `runs/<run_id>/experiments/` 아래에 전략별 입력 이미지, JSON 결과, HTML render, summary가 저장됩니다. 각 전략 카드는 클릭해서 20x20 editor와 `Input + Prompt = Output` trace에서 바로 비교할 수 있습니다.

## 공개/협업용 문서

- [Hermes Kanban 운영 메모](docs/HERMES_KANBAN.md)
- [OCR 방법론 및 테스트 계획](docs/METHODOLOGY_AND_TEST_PLAN.md)
- [GitHub 공개 준비](docs/GITHUB_PUBLISHING.md)

## 샘플 데이터

원본 학생 이미지는 GitHub에 올리지 않는 것을 기본으로 합니다. 대신 공개 repo에서도 재현 가능한 synthetic 샘플을 `data/samples_synthetic/`에 둡니다.

샘플을 다시 생성하려면:

```bash
python3 scripts/create_synthetic_samples.py
```
