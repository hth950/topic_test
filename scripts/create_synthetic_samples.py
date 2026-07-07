from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "samples_synthetic"
CELL = 48
MARGIN = 40
GRID = 20
WIDTH = MARGIN * 2 + CELL * GRID
HEIGHT = MARGIN * 2 + CELL * GRID


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT = find_font(28)
SMALL_FONT = find_font(22)


def blank_matrix() -> list[list[str]]:
    return [[" " for _ in range(GRID)] for _ in range(GRID)]


def put(row: list[str], start: int, cells: Iterable[str]) -> None:
    for idx, text in enumerate(cells):
        if 0 <= start + idx < GRID:
            row[start + idx] = text


def draw_sample(matrix: list[list[str]], path: Path, red_marks: bool = False) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    blue = (91, 159, 211)
    pencil = (72, 72, 72)
    red = (214, 52, 59)

    for i in range(GRID + 1):
        x = MARGIN + i * CELL
        y = MARGIN + i * CELL
        draw.line((x, MARGIN, x, MARGIN + GRID * CELL), fill=blue, width=2)
        draw.line((MARGIN, y, MARGIN + GRID * CELL, y), fill=blue, width=2)

    for r, row in enumerate(matrix):
        for c, text in enumerate(row):
            if text == " ":
                continue
            x0 = MARGIN + c * CELL
            y0 = MARGIN + r * CELL
            font = SMALL_FONT if len(text) >= 2 else FONT
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            jitter_x = ((r * 7 + c * 3) % 7) - 3
            jitter_y = ((r * 5 + c * 11) % 7) - 2
            draw.text(
                (x0 + (CELL - tw) / 2 + jitter_x, y0 + (CELL - th) / 2 - 3 + jitter_y),
                text,
                fill=pencil,
                font=font,
            )

    if red_marks:
        draw.ellipse((MARGIN + CELL * 13, MARGIN + CELL * 1, MARGIN + CELL * 18, MARGIN + CELL * 4), outline=red, width=8)
        draw.text((MARGIN + CELL * 12, MARGIN + CELL * 5), "표현 확인", fill=red, font=SMALL_FONT)
        draw.line((MARGIN + CELL * 2, MARGIN + CELL * 9, MARGIN + CELL * 10, MARGIN + CELL * 9), fill=red, width=6)

    image.save(path)


def save_truth(name: str, matrix: list[list[str]]) -> None:
    with (OUT_DIR / f"truth_{name}.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": matrix}, f, ensure_ascii=False, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = blank_matrix()
    put(clean[0], 0, ["곤", "이", "의", " ", "행", "동", "은", " ", "문", "제", "를", " ", "드", "러", "낸", "다", ".", " ", " ", " "])
    put(clean[1], 0, ["그", "러", "나", " ", "그", " ", "행", "동", "은", " ", "새", "로", "운", " ", "의", "미", "도", " ", "있", "다"])
    put(clean[2], 0, ["근", "거", "는", " ", "두", " ", "가", "지", "이", "며", ",", " ", "첫", "째", "는", " ", "용", "기", "이", "다"])
    draw_sample(clean, OUT_DIR / "clean_grid.png")
    save_truth("clean_grid", clean)

    red = blank_matrix()
    put(red[0], 1, ["발", "생", "한", " ", "새", "로", "운", " ", "종", "이", "라", "는", " ", "관", "점", "으", "로", " ", "아"])
    put(red[1], 0, ["프", "리", "카", "의", " ", "상", "황", "을", " ", "이", "해", "할", " ", "수", " ", "있", "다", "그", " "])
    put(red[2], 0, ["근", "거", "로", " ", "생", "태", "계", "와", " ", "인", "간", "의", " ", "관", "계", "를", " ", "본", "다", "."])
    draw_sample(red, OUT_DIR / "red_marked_grid.png", red_marks=True)
    save_truth("red_marked_grid", red)

    stress = blank_matrix()
    put(stress[0], 0, [" ", " ", "첫", " ", "칸", "은", " ", "비", "워", "야", " ", "한", "다", ".", " ", " ", " ", " ", " ", " "])
    put(stress[1], 0, ["2", "0", "0만", "년", "전", "에", " ", "인", "류", "는", " ", "이", "동", "했", "다", ".", " ", " ", " ", " "])
    put(stress[2], 0, ["이", "는", " ", "사", "실", "이", "며", ",", " ", "고", "쳐", "쓰", "면", " ", "안", "된", "다", ".", " ", " "])
    put(stress[3], 0, ["있", "다", "그", " ", "근", "거", "로", " ", "제", "시", "한", "다", ".", " ", " ", " ", " ", " ", " ", " "])
    draw_sample(stress, OUT_DIR / "spacing_punctuation_grid.png", red_marks=True)
    save_truth("spacing_punctuation_grid", stress)


if __name__ == "__main__":
    main()
