# Hermes Kanban 운영 메모

Hermes에 전용 보드를 만들었습니다.

- Board: `논술 원고지 OCR 비교/Repair`
- Slug: `essay-grid-ocr`
- URL: `https://hermes-agent-suvi.srv1765895.hstgr.cloud/kanban`
- 현재 상태: Triage lane에 8개 카드 생성

## 넣어둔 카드

1. Project brief
2. GitHub publishing hygiene
3. Methodology implementation review
4. Baseline OCR comparison test
5. Prompt and correction policy test
6. Creative experiment: visual self-check loop
7. Creative experiment: robustness stress testing
8. Human review workflow

## 권장 흐름

1. GitHub에 올릴 repo를 준비한다.
2. `.env`, `data/` 원본, `runs/`, PDF 원본이 commit 대상에서 빠지는지 확인한다.
3. `data/samples_synthetic/` 샘플로 앱이 재현 가능한지 확인한다.
4. Hermes Kanban에서 필요한 카드를 Ready로 옮기거나 `Nudge dispatcher`를 눌러 작업을 시작한다.
5. Hermes가 만든 결과물은 카드별로 산출물 링크, 테스트 로그, 실패 이미지를 남기게 한다.

## 주의

- 원본 학생 답안 이미지는 공개 repo에 올리지 않는다.
- 실제 학생 샘플을 써야 한다면 익명화 여부를 먼저 결정한다.
- API 키와 내부망 주소는 `.env.example`에는 기본 구조만 두고 실제 값은 `.env`에만 둔다.
