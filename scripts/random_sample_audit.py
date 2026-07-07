from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = selectable_images(args)
    selected = random.Random(args.seed).sample(records, min(args.count, len(records)))
    items = []
    for index, rec in enumerate(selected, start=1):
        items.append(audit_record(index, rec, out_dir, args))

    summary = summarize(items)
    report = {
        "seed": args.seed,
        "count": len(items),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ideas": experiment_ideas(),
        "summary": summary,
        "items": items,
    }
    write_json(out_dir / "report.json", report)
    write_markdown(out_dir / "REPORT.md", report, out_dir)
    write_contact_sheet(items, out_dir / "contact_sheet.jpg")
    print(json.dumps({"report": str(out_dir / "REPORT.md"), "contact_sheet": str(out_dir / "contact_sheet.jpg"), **summary}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample answer images and audit crop/red-mask readiness.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--out-dir", default="work/random10_audit")
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    parser.add_argument("--create-runs", action="store_true", help="Also create /api/crop runs so results are visible in the local UI.")
    parser.add_argument("--include-synthetic", action="store_true")
    return parser.parse_args()


def selectable_images(args: argparse.Namespace) -> list[server.ImageRecord]:
    records: list[server.ImageRecord] = []
    for folder in server.list_image_folders():
        if not args.include_synthetic and folder.name == "samples_synthetic":
            continue
        records.extend(server.list_images(folder.id))
    if not records:
        raise RuntimeError("No images found")
    return records


def audit_record(index: int, rec: server.ImageRecord, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    img = Image.open(rec.path).convert("RGB")
    bbox = server.clamp_bbox(server.detect_grid_bbox(img), img.width, img.height)
    crop = img.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))
    masked = server.remove_red_marks(crop)
    occupancy = server.detect_cell_occupancy(masked)
    preview = draw_bbox_preview(img, bbox)

    safe_stem = f"{index:02d}_{safe_name(Path(rec.name).stem)}"
    original_preview = out_dir / f"{safe_stem}_original_bbox.jpg"
    crop_path = out_dir / f"{safe_stem}_crop.png"
    masked_path = out_dir / f"{safe_stem}_masked.png"
    preview.save(original_preview, quality=88)
    crop.save(crop_path)
    masked.save(masked_path)

    before_red = red_pixel_count(crop)
    after_red = red_pixel_count(masked)
    occupied_rows = [row for row in occupancy if row.get("first_col") is not None]
    occupied_cells = sum(len(server.occupied_columns(row)) for row in occupancy)
    full_width_rows = [
        row["row"] + 1
        for row in occupancy
        if row.get("first_col") is not None and row.get("last_col") is not None and int(row["last_col"]) - int(row["first_col"]) >= 18
    ]
    run_id = create_crop_run(rec.id, args.base_url) if args.create_runs else None
    flags = quality_flags(bbox, img.size, before_red, after_red, len(occupied_rows), occupied_cells)
    return {
        "index": index,
        "image_id": rec.id,
        "image_name": rec.name,
        "folder_id": rec.folder_id,
        "folder_name": rec.folder_name,
        "size": [rec.width, rec.height],
        "bbox": bbox,
        "bbox_ratio": round(bbox["width"] / max(1, bbox["height"]), 3),
        "red_pixels_before": before_red,
        "red_pixels_after": after_red,
        "red_removed_pct": round((1 - after_red / before_red) * 100, 1) if before_red else 100.0,
        "occupied_rows": len(occupied_rows),
        "occupied_cells": occupied_cells,
        "full_width_rows": full_width_rows,
        "flags": flags,
        "run_id": run_id,
        "preview": original_preview.name,
        "crop": crop_path.name,
        "masked": masked_path.name,
    }


def create_crop_run(image_id: str, base_url: str) -> str | None:
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/crop", json={"image_id": image_id}, timeout=60)
        response.raise_for_status()
        return response.json().get("run_id")
    except Exception as exc:
        return f"crop_failed:{exc}"


def quality_flags(
    bbox: dict[str, int],
    image_size: tuple[int, int],
    before_red: int,
    after_red: int,
    occupied_rows: int,
    occupied_cells: int,
) -> list[str]:
    width, height = image_size
    flags: list[str] = []
    ratio = bbox["width"] / max(1, bbox["height"])
    if not 0.82 <= ratio <= 1.18:
        flags.append("bbox_aspect_check")
    if bbox["width"] < width * 0.25 or bbox["height"] < height * 0.20:
        flags.append("bbox_too_small")
    if bbox["x"] > width * 0.35:
        flags.append("bbox_not_left_side")
    if before_red > 0 and after_red > max(30, int(before_red * 0.18)):
        flags.append("red_mask_residual")
    if occupied_rows < 3:
        flags.append("few_handwriting_rows")
    if occupied_cells < 20:
        flags.append("low_occupancy")
    return flags or ["ok"]


def draw_bbox_preview(img: Image.Image, bbox: dict[str, int]) -> Image.Image:
    preview = ImageOps.contain(img, (800, 1100)).convert("RGB")
    scale_x = preview.width / img.width
    scale_y = preview.height / img.height
    rect = [
        int(bbox["x"] * scale_x),
        int(bbox["y"] * scale_y),
        int((bbox["x"] + bbox["width"]) * scale_x),
        int((bbox["y"] + bbox["height"]) * scale_y),
    ]
    draw = ImageDraw.Draw(preview)
    draw.rectangle(rect, outline=(0, 170, 210), width=5)
    return preview


def red_pixel_count(img: Image.Image) -> int:
    arr = np.asarray(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    red = (r > 135) & ((r - g) > 35) & ((r - b) > 35) & (g < 190) & (b < 190)
    return int(red.sum())


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [item for item in items if item["flags"] != ["ok"]]
    return {
        "images": len(items),
        "ok": len(items) - len(flagged),
        "needs_review": len(flagged),
        "avg_occupied_rows": round(sum(item["occupied_rows"] for item in items) / max(1, len(items)), 2),
        "avg_occupied_cells": round(sum(item["occupied_cells"] for item in items) / max(1, len(items)), 2),
        "avg_red_removed_pct": round(sum(item["red_removed_pct"] for item in items) / max(1, len(items)), 2),
    }


def experiment_ideas() -> list[dict[str, str]]:
    return [
        {
            "name": "cell-occupancy constrained decoding",
            "description": "OCR 결과를 이미지의 실제 occupied cell 패턴에 맞춰 DP로 재정렬한다. 이미 일부 적용되어 있지만, 모델 호출 전 prompt에도 occupied mask를 넣어 강제할 수 있다.",
        },
        {
            "name": "disagreement-only repair",
            "description": "full/row_1/row_2 또는 Chandra/GPT가 다른 cell만 작은 crop과 함께 repair에 보낸다. 비용과 과보정을 줄이는 방식이다.",
        },
        {
            "name": "line image + neighbor context",
            "description": "1줄 crop만 보내지 말고 위아래 줄은 흐리게 context로 붙이고, 대상 줄만 진하게 표시한다. 위치 정확도와 문맥을 같이 얻는다.",
        },
        {
            "name": "cell zoom strip OCR",
            "description": "한 줄을 20개 cell 썸네일 strip으로 재배열해 번호를 붙여 보낸다. 쉼표/마침표가 어느 칸인지 모델이 더 명시적으로 판단한다.",
        },
        {
            "name": "two-pass literal guard",
            "description": "1차 OCR 후 2차 모델에게 '맞춤법 교정 금지 위반 후보'만 찾게 한다. 있다그→있다고 같은 의미 보정을 탐지한다.",
        },
        {
            "name": "preprocessing sweep",
            "description": "red-mask 강도, 대비, 샤프닝, padding, 해상도 scale을 작은 grid로 바꿔 같은 이미지에 돌리고 best-of-N을 고른다.",
        },
        {
            "name": "human approval memory",
            "description": "사용자가 승인한 final matrix를 같은 학생/스캔 유형의 regression fixture로 저장해서 다음 prompt와 validator에 반영한다.",
        },
    ]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# 랜덤 10개 원고지 샘플 Audit",
        "",
        f"- seed: `{report['seed']}`",
        f"- 생성: `{report['created_at']}`",
        f"- 요약: `{report['summary']['ok']}` OK / `{report['summary']['needs_review']}` review",
        f"- 평균 occupied rows: `{report['summary']['avg_occupied_rows']}`",
        f"- 평균 occupied cells: `{report['summary']['avg_occupied_cells']}`",
        f"- 평균 red removal: `{report['summary']['avg_red_removed_pct']}%`",
        "",
        "## 색다른 실험 아이디어",
        "",
    ]
    for idea in report["ideas"]:
        lines.append(f"- **{idea['name']}**: {idea['description']}")
    lines.extend(["", "## 랜덤 10개 결과", ""])
    lines.append("| # | image | folder | bbox | occupied | red before→after | flags | run | previews |")
    lines.append("|---:|---|---|---|---:|---:|---|---|---|")
    for item in report["items"]:
        bbox = item["bbox"]
        run = item["run_id"] or ""
        if run and not str(run).startswith("crop_failed"):
            run = f"[{run}](http://127.0.0.1:8767/api/runs/{run})"
        lines.append(
            "| {index} | {image} | {folder} | x={x}, y={y}, w={w}, h={h} | {occ_rows} rows / {occ_cells} cells | {before}→{after} ({pct}%) | {flags} | {run} | [bbox]({preview}) / [crop]({crop}) / [masked]({masked}) |".format(
                index=item["index"],
                image=item["image_name"],
                folder=item["folder_name"],
                x=bbox["x"],
                y=bbox["y"],
                w=bbox["width"],
                h=bbox["height"],
                occ_rows=item["occupied_rows"],
                occ_cells=item["occupied_cells"],
                before=item["red_pixels_before"],
                after=item["red_pixels_after"],
                pct=item["red_removed_pct"],
                flags=", ".join(item["flags"]),
                run=run,
                preview=item["preview"],
                crop=item["crop"],
                masked=item["masked"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(items: list[dict[str, Any]], path: Path) -> None:
    thumbs = []
    base_dir = path.parent
    for item in items:
        crop = Image.open(base_dir / item["crop"]).convert("RGB")
        masked = Image.open(base_dir / item["masked"]).convert("RGB")
        crop.thumbnail((240, 240))
        masked.thumbnail((240, 240))
        tile = Image.new("RGB", (520, 300), "white")
        tile.paste(crop, (10, 30))
        tile.paste(masked, (270, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 8), f"{item['index']:02d} {item['image_name'][:46]}", fill=(20, 20, 20))
        draw.text((10, 274), f"flags: {', '.join(item['flags'])}", fill=(120, 20, 20) if item["flags"] != ["ok"] else (20, 110, 60))
        thumbs.append(tile)
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 520, rows * 300), "white")
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % columns) * 520, (idx // columns) * 300))
    sheet.save(path, quality=90)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)[:80]


if __name__ == "__main__":
    main()
