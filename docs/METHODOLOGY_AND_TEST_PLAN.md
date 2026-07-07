# OCR 방법론 및 테스트 계획

## 목표

20x20 원고지에 학생이 실제로 쓴 내용을 의미 보정 없이 cell 단위로 복원한다. 틀린 표현, 띄어쓰기, 쉼표/마침표 위치, 빈칸은 이미지에 보이는 그대로 유지한다.

## 처리 흐름

1. 폴더와 이미지를 선택한다.
2. 파란 20x20 격자를 검출해 crop bbox를 만든다.
3. bbox는 UI에서 수동 보정할 수 있어야 한다.
4. `crop.png`와 빨간펜 제거용 `masked.png`를 만든다.
5. Chandra OCR과 GPT-5.5 OCR을 같은 조건으로 실행한다.
6. 모델별 입력 이미지, prompt, raw response, parsed 20x20 matrix를 저장한다.
7. matrix가 정확히 20행 x 20칸인지 검증한다.
8. HTML grid로 렌더링하고 사용자가 matrix를 직접 편집할 수 있게 한다.
9. Repair는 crop 이미지와 OCR 결과를 비교해 cell-level 수정안만 제안한다.
10. 사용자가 승인한 결과만 `final.html`로 저장한다.

## OCR 원칙

- 빨간펜 주석과 채점 흔적은 답안 텍스트로 읽지 않는다.
- 학생이 틀리게 쓴 표현도 고치지 않는다.
- 고유명사는 조건문에서 제공된 철자를 OCR ambiguity 해소에만 사용한다.
- 쉼표나 마침표가 다음 칸에 보이면 다음 칸에 둔다.
- 한 칸에 실제로 두 글자가 보이는 경우만 같은 cell에 둔다.
- 빈칸과 줄 앞 공백은 의미 있는 답안 상태이므로 보존한다.

## 기본 검증 항목

- crop bbox가 20x20 답안 영역을 정확히 잡는가.
- red-mask 후 빨간 글씨/동그라미/점수만 제거되는가.
- OCR JSON이 parse 가능한가.
- matrix가 정확히 20행 x 20칸인가.
- render HTML이 원본 crop의 cell 위치와 맞는가.
- GPT repair가 학생 답을 문법적으로 고쳐 쓰지 않는가.
- 다른 페이지로 이동해도 진행 중 job과 완료 run history가 유지되는가.

## Synthetic sample test

`data/samples_synthetic/`에는 공개 repo에 넣어도 되는 재현용 샘플을 둔다.

- `clean_grid.png`: 기본 20x20 grid와 일반 답안
- `red_marked_grid.png`: 빨간펜 주석 제거 테스트
- `spacing_punctuation_grid.png`: 빈칸, 쉼표, 마침표, 오타 보존 테스트

각 이미지의 기대 matrix는 같은 폴더의 `truth_*.json`에 저장한다.

## 실험 아이디어

- Chandra OCR, GPT-5.5 OCR, GPT repair를 같은 crop으로 비교한다.
- Chandra/GPT가 disagree한 cell만 repair 후보로 보낸다.
- rendered grid screenshot과 crop을 LLM vision으로 비교해 row/col mismatch만 반환하게 한다.
- 회전, 기울어짐, 연필 농도 저하, 빨간펜 겹침을 synthetic 변형으로 생성해 robustness를 측정한다.
- 사람이 승인한 final matrix를 regression fixture로 축적한다.
