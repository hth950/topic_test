# GitHub 공개 준비

이 프로젝트 폴더는 현재 원본 이미지와 run 결과가 커서 그대로 push하면 안 됩니다.

## 포함할 것

- `server.py`
- `static/`
- `tests/`
- `scripts/`
- `requirements.txt`
- `.env.example`
- `README.md`
- `docs/`
- `data/samples_synthetic/`

## 제외할 것

- `.env`
- `data/` 아래 원본 학생 이미지 폴더
- `runs/`
- `work/`
- `outputs/`
- PDF 원본
- Python cache와 OS metadata

## 최초 push 순서

```bash
git init
git add .gitignore README.md requirements.txt server.py static tests scripts docs data/samples_synthetic .env.example
git status --short
git commit -m "Add essay grid OCR comparison app"
git branch -M main
git remote add origin <github-repo-url>
git push -u origin main
```

`git status --short`에서 `.env`, `runs/`, 원본 `data/` 폴더가 보이면 push하지 말고 `.gitignore`를 먼저 고칩니다.

## 실제 학생 샘플을 올리고 싶을 때

공개 repo라면 권장하지 않습니다. 꼭 필요하면 파일명을 익명화하고, 학생 개인정보/학원 내부 식별자가 포함되지 않는지 확인한 뒤 별도 private repo나 object storage를 쓰는 편이 낫습니다.
