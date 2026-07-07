from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


RESULT_FILES = {
    "ocr_chandra.json": ("render_chandra.html", "chandra OCR", True),
    "ocr_gpt.json": ("render_gpt.html", "gpt OCR", True),
    "repair.json": ("render_repair.html", "Repair 제안", True),
    "final.json": ("final.html", "최종 답안", False),
}


def main() -> None:
    args = parse_args()
    report = run_audit(args.passes, args.repair_stored, args.repair_final)
    write_reports(report)
    if args.promote_best:
        promote_best_results()
    if args.visual_samples:
        write_visual_samples()
    print_summary(report)
    if report["summary"]["failures"] and args.fail_on_issue:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated QA checks for 20x20 essay-grid OCR outputs.")
    parser.add_argument("--passes", type=int, default=3, help="Number of audit passes per image.")
    parser.add_argument("--repair-stored", action="store_true", help="Rewrite OCR/repair JSON+HTML to canonical cells.")
    parser.add_argument("--repair-final", action="store_true", help="Also rewrite final.json/final.html when canonical cells differ.")
    parser.add_argument("--visual-samples", action="store_true", help="Write crop/render comparison PNGs for latest runs.")
    parser.add_argument("--promote-best", action="store_true", help="Write final.json/final.html from the best available result per image.")
    parser.add_argument("--fail-on-issue", action="store_true", help="Exit non-zero if unresolved failures remain.")
    return parser.parse_args()


def run_audit(passes: int, repair_stored: bool, repair_final: bool) -> dict[str, Any]:
    images = server.list_images()
    runs_by_image = collect_runs_by_image()
    pass_reports: list[dict[str, Any]] = []
    started = time.time()

    for pass_index in range(max(1, passes)):
        image_reports: list[dict[str, Any]] = []
        for rec in images:
            image_reports.append(audit_image(rec, runs_by_image.get(rec.id, []), pass_index, repair_stored, repair_final))
        pass_reports.append(
            {
                "pass": pass_index + 1,
                "images": image_reports,
                "failures": count_failures(image_reports),
                "canonical_updates": sum_image_field(image_reports, "canonical_updates"),
                "result_files_checked": sum_image_field(image_reports, "result_files_checked"),
            }
        )

    summary = summarize(images, runs_by_image, pass_reports, started)
    return {
        "summary": summary,
        "passes": pass_reports,
    }


def collect_runs_by_image() -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for metadata_path in sorted(server.RUNS_DIR.glob("*/metadata.json")):
        try:
            metadata = server.read_json(metadata_path)
        except Exception:
            continue
        image_id = metadata.get("image_id")
        if image_id:
            grouped[str(image_id)].append(metadata_path.parent)
    return grouped


def audit_image(
    rec: server.ImageRecord,
    run_dirs: list[Path],
    pass_index: int,
    repair_stored: bool,
    repair_final: bool,
) -> dict[str, Any]:
    img = Image.open(rec.path).convert("RGB")
    bbox = server.clamp_bbox(server.detect_grid_bbox(img), img.width, img.height)
    crop = img.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))
    masked = server.remove_red_marks(crop)
    occupancy = server.detect_cell_occupancy(masked)
    crop_report = audit_crop(crop, masked, bbox, occupancy)

    result_reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result_reports.extend(audit_run_results(run_dir, repair_stored, repair_final))

    return {
        "image_id": rec.id,
        "image_name": rec.name,
        "pass": pass_index + 1,
        "bbox": bbox,
        "crop": crop_report,
        "runs_checked": len(run_dirs),
        "result_files_checked": len(result_reports),
        "canonical_updates": sum(1 for item in result_reports if item["canonical_updated"]),
        "failures": crop_report["failures"] + [failure for item in result_reports for failure in item["failures"]],
        "reviews": crop_report["reviews"] + [review for item in result_reports for review in item["reviews"]],
        "results": result_reports,
    }


def audit_crop(
    crop: Image.Image,
    masked: Image.Image,
    bbox: dict[str, int],
    occupancy: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    reviews: list[str] = []
    if len(occupancy) != 20:
        failures.append(f"occupancy row count is {len(occupancy)}, expected 20")

    aspect = bbox["width"] / max(1, bbox["height"])
    if not 0.75 <= aspect <= 1.25:
        reviews.append(f"bbox aspect ratio {aspect:.3f} is outside loose square-grid range")

    before_red = red_pixel_count(crop)
    after_red = red_pixel_count(masked)
    if before_red > 500 and after_red > before_red * 0.15:
        failures.append(f"red-mask left too many red pixels: before={before_red}, after={after_red}")

    occupied_rows = [row for row in occupancy if row.get("first_col") is not None]
    if len(occupied_rows) < 3:
        reviews.append(f"only {len(occupied_rows)} rows detected with handwriting")

    wide_rows = []
    for row in occupied_rows:
        first_col = row.get("first_col")
        last_col = row.get("last_col")
        if first_col is not None and last_col is not None and int(last_col) - int(first_col) >= 19:
            wide_rows.append(row.get("row"))

    return {
        "failures": failures,
        "reviews": reviews,
        "red_pixels_before": before_red,
        "red_pixels_after": after_red,
        "occupied_rows": len(occupied_rows),
        "full_width_rows": wide_rows,
    }


def audit_run_results(run_dir: Path, repair_stored: bool, repair_final: bool) -> list[dict[str, Any]]:
    try:
        metadata = server.read_json(run_dir / "metadata.json")
    except Exception as exc:
        return [
            {
                "run_id": run_dir.name,
                "file": "metadata.json",
                "canonical_updated": False,
                "failures": [f"metadata read failed: {exc}"],
                "reviews": [],
            }
        ]
    occupancy = metadata.get("cell_occupancy") or []
    reports = []
    for filename, (render_name, title, repair_default) in RESULT_FILES.items():
        path = run_dir / filename
        if not path.exists():
            continue
        should_repair = repair_stored and (repair_default or repair_final)
        reports.append(audit_result_file(path, occupancy, render_name, title, should_repair))
    return reports


def audit_result_file(
    path: Path,
    occupancy: list[dict[str, Any]],
    render_name: str,
    title: str,
    repair: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    reviews: list[str] = []
    canonical_updated = False
    try:
        data = server.read_json(path)
    except Exception as exc:
        return {
            "run_id": path.parent.name,
            "file": path.name,
            "canonical_updated": False,
            "failures": [f"{path.name} read failed: {exc}"],
            "reviews": [],
        }

    source = data.get("cells") or data.get("rows") or []
    cells, validation = server.normalize_cells(source, occupancy)
    stored_cells = data.get("cells")
    if not is_matrix(stored_cells):
        failures.append(f"{path.name} has no valid cells matrix")
    else:
        failures.extend(cell_failures(stored_cells, path.name))
        reviews.extend(cell_reviews(stored_cells, path.name))

    canonical_failures = cell_failures(cells, f"{path.name} canonical")
    failures.extend(canonical_failures)
    reviews.extend(cell_reviews(cells, f"{path.name} canonical"))

    differs = not matrices_equal(stored_cells, cells)
    if differs:
        reviews.append(f"{path.name} differs from current canonical normalization")
    if differs and repair:
        data["cells"] = cells
        data["rows"] = server.cells_to_text_rows(cells)
        data["validation"] = validation
        server.write_json(path, data)
        (path.parent / render_name).write_text(server.render_grid_html(cells, title), encoding="utf-8")
        canonical_updated = True

    return {
        "run_id": path.parent.name,
        "file": path.name,
        "canonical_updated": canonical_updated,
        "canonical_differs": differs,
        "failures": failures,
        "reviews": reviews,
    }


def is_matrix(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 20 and all(isinstance(row, list) and len(row) == 20 for row in value)


def matrices_equal(left: Any, right: Any) -> bool:
    return is_matrix(left) and is_matrix(right) and left == right


def cell_failures(cells: Any, label: str) -> list[str]:
    failures: list[str] = []
    if not is_matrix(cells):
        return [f"{label}: matrix is not 20x20"]
    for row_index, row in enumerate(cells):
        for col_index, cell in enumerate(row):
            text = str(cell).strip()
            chars = list(text)
            if len(chars) > 2:
                failures.append(f"{label}: row {row_index + 1} col {col_index + 1} has >2 chars: {text!r}")
            if len(chars) > 1 and chars[-1] in server.TRAILING_CELL_PUNCTUATION and col_index != 19:
                failures.append(f"{label}: row {row_index + 1} col {col_index + 1} attaches mid-row punctuation: {text!r}")
            if len(chars) > 1 and all("가" <= char <= "힣" for char in chars):
                failures.append(f"{label}: row {row_index + 1} col {col_index + 1} joins Hangul syllables: {text!r}")
    return failures


def cell_reviews(cells: Any, label: str) -> list[str]:
    reviews: list[str] = []
    if not is_matrix(cells):
        return reviews
    for row_index, row in enumerate(cells):
        for col_index, cell in enumerate(row):
            text = str(cell).strip()
            chars = list(text)
            if len(chars) == 2 and not text.isdigit() and not (
                col_index == 19 and chars[-1] in server.TRAILING_CELL_PUNCTUATION
            ):
                if any(char.isdigit() for char in chars) or all("a" <= char <= "z" for char in chars):
                    reviews.append(f"{label}: row {row_index + 1} col {col_index + 1} has review-needed 2-char cell: {text!r}")
    return reviews


def red_pixel_count(img: Image.Image) -> int:
    arr = np.asarray(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    red = (r > 135) & ((r - g) > 35) & ((r - b) > 35) & (g < 185) & (b < 185)
    return int(red.sum())


def count_failures(image_reports: list[dict[str, Any]]) -> int:
    return sum(len(item["failures"]) for item in image_reports)


def sum_image_field(image_reports: list[dict[str, Any]], field: str) -> int:
    return sum(int(item.get(field, 0)) for item in image_reports)


def summarize(
    images: list[server.ImageRecord],
    runs_by_image: dict[str, list[Path]],
    pass_reports: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    final_pass = pass_reports[-1]
    return {
        "image_count": len(images),
        "passes_per_image": len(pass_reports),
        "total_image_passes": len(images) * len(pass_reports),
        "runs_seen": sum(len(items) for items in runs_by_image.values()),
        "result_files_checked_final_pass": final_pass["result_files_checked"],
        "canonical_updates_total": sum(item["canonical_updates"] for item in pass_reports),
        "failures": final_pass["failures"],
        "duration_seconds": round(time.time() - started, 2),
    }


def write_reports(report: dict[str, Any]) -> None:
    work_dir = ROOT / "work"
    work_dir.mkdir(exist_ok=True)
    json_path = work_dir / "grid_qa_loop_report.json"
    md_path = work_dir / "grid_qa_loop_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Grid QA Loop Report",
        "",
        f"- Images: {summary['image_count']}",
        f"- Passes per image: {summary['passes_per_image']}",
        f"- Total image passes: {summary['total_image_passes']}",
        f"- Runs seen: {summary['runs_seen']}",
        f"- Result files checked in final pass: {summary['result_files_checked_final_pass']}",
        f"- Canonical updates total: {summary['canonical_updates_total']}",
        f"- Final-pass failures: {summary['failures']}",
        f"- Duration seconds: {summary['duration_seconds']}",
        "",
        "## Final Pass Images",
        "",
    ]
    for image in report["passes"][-1]["images"]:
        lines.append(f"### {image['image_name']}")
        lines.append(f"- bbox: `{image['bbox']}`")
        lines.append(f"- runs checked: {image['runs_checked']}")
        lines.append(f"- result files checked: {image['result_files_checked']}")
        lines.append(f"- failures: {len(image['failures'])}")
        if image["failures"]:
            for failure in image["failures"][:10]:
                lines.append(f"  - {failure}")
        lines.append(f"- reviews: {len(image['reviews'])}")
        if image["reviews"]:
            for review in image["reviews"][:10]:
                lines.append(f"  - {review}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {ROOT / 'work' / 'grid_qa_loop_report.md'}")
    print(f"Visual samples: {ROOT / 'work' / 'grid_qa_visual_samples'}")


def write_visual_samples() -> None:
    output_dir = ROOT / "work" / "grid_qa_visual_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in output_dir.glob("*.png"):
        old_sample.unlink()
    for rec in server.list_images():
        selected = select_latest_result(rec.id)
        if not selected:
            sample = render_crop_only_sample(rec)
            safe_name = rec.name.replace("/", "_").replace(" ", "_")
            sample.save(output_dir / f"{safe_name}_crop_occupancy.png")
        else:
            run_dir, source_name, cells = selected
            crop_path = run_dir / "masked.png"
            if not crop_path.exists():
                crop_path = run_dir / "crop.png"
            if not crop_path.exists():
                continue
            sample = render_visual_sample(Image.open(crop_path).convert("RGB"), cells, rec.name, run_dir.name, source_name)
            safe_name = rec.name.replace("/", "_").replace(" ", "_")
            sample.save(output_dir / f"{safe_name}_{run_dir.name}_{source_name}.png")


def select_latest_result(image_id: str) -> tuple[Path, str, list[list[str]]] | None:
    candidates: list[tuple[int, int, float, float, Path, str, list[list[str]]]] = []
    for metadata_path in server.RUNS_DIR.glob("*/metadata.json"):
        try:
            metadata = server.read_json(metadata_path)
        except Exception:
            continue
        if metadata.get("image_id") != image_id:
            continue
        run_dir = metadata_path.parent
        created_at = float(metadata.get("created_at") or 0)
        for filename, source_name in (
            ("final.json", "final"),
            ("repair.json", "repair"),
            ("ocr_gpt.json", "gpt"),
            ("ocr_chandra.json", "chandra"),
        ):
            path = run_dir / filename
            if not path.exists():
                continue
            try:
                data = server.read_json(path)
            except Exception:
                continue
            if source_name == "final" and data.get("source") != "manual":
                continue
            cells = data.get("cells")
            if is_matrix(cells):
                rank = result_rank(source_name, data)
                label = resolved_source_name(source_name, data)
                candidates.append((rank, count_nonempty_cells(cells), path.stat().st_mtime, created_at, run_dir, label, cells))
    if candidates:
        _, _, _, _, run_dir, source_name, cells = max(candidates)
        return run_dir, source_name, cells
    return None


def count_nonempty_cells(cells: list[list[str]]) -> int:
    return sum(1 for row in cells for cell in row if str(cell).strip())


def result_rank(source_name: str, data: dict[str, Any]) -> int:
    if source_name == "final":
        source = str(data.get("source") or "")
        if source == "manual":
            return 50
        return result_rank(source, {})
    ranks = {
        "repair": 40,
        "gpt": 30,
        "chandra": 20,
    }
    return ranks.get(source_name, 0)


def resolved_source_name(source_name: str, data: dict[str, Any]) -> str:
    if source_name != "final":
        return source_name
    source = str(data.get("source") or "")
    if source in {"manual", "repair", "gpt", "chandra"}:
        return source
    return "final"


def iter_best_results() -> list[tuple[server.ImageRecord, Path, str, list[list[str]]]]:
    results: list[tuple[server.ImageRecord, Path, str, list[list[str]]]] = []
    for rec in server.list_images():
        selected = select_latest_result(rec.id)
        if selected:
            run_dir, source_name, cells = selected
            results.append((rec, run_dir, source_name, cells))
    return results


def promote_best_results() -> None:
    for rec, run_dir, source_name, cells in iter_best_results():
        final = {
            "source": source_name,
            "rows": server.cells_to_text_rows(cells),
            "cells": cells,
            "validation": {
                "normalized_shape": [20, 20],
                "promoted_by": "scripts/grid_qa_loop.py",
                "image_name": rec.name,
            },
            "saved_at": time.time(),
        }
        server.write_json(run_dir / "final.json", final)
        (run_dir / "final.html").write_text(server.render_grid_html(cells, f"Best final: {rec.name}"), encoding="utf-8")


def _legacy_select_latest_result(image_id: str) -> tuple[Path, str, list[list[str]]] | None:
    candidates: list[tuple[float, Path]] = []
    for metadata_path in server.RUNS_DIR.glob("*/metadata.json"):
        try:
            metadata = server.read_json(metadata_path)
        except Exception:
            continue
        if metadata.get("image_id") == image_id:
            candidates.append((float(metadata.get("created_at") or 0), metadata_path.parent))
    for _, run_dir in sorted(candidates, reverse=True):
        for filename, source_name in (
            ("repair.json", "repair"),
            ("final.json", "final"),
            ("ocr_gpt.json", "gpt"),
            ("ocr_chandra.json", "chandra"),
        ):
            path = run_dir / filename
            if not path.exists():
                continue
            try:
                data = server.read_json(path)
            except Exception:
                continue
            cells = data.get("cells")
            if is_matrix(cells):
                return run_dir, source_name, cells
    return None


def render_visual_sample(
    crop: Image.Image,
    cells: list[list[str]],
    image_name: str,
    run_id: str,
    source_name: str,
) -> Image.Image:
    crop_width = 560
    crop_height = int(crop.height * (crop_width / crop.width))
    crop_preview = crop.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    grid_size = min(760, max(560, crop_height))
    grid = draw_cells_grid(cells, grid_size)
    title_height = 64
    canvas = Image.new(
        "RGB",
        (crop_preview.width + grid.width + 28, max(crop_preview.height, grid.height) + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    small = load_font(16)
    draw.text((12, 10), image_name, fill=(20, 20, 20), font=font)
    draw.text((12, 38), f"run={run_id} source={source_name}", fill=(80, 80, 80), font=small)
    canvas.paste(crop_preview, (12, title_height))
    canvas.paste(grid, (crop_preview.width + 24, title_height))
    return canvas


def render_crop_only_sample(rec: server.ImageRecord) -> Image.Image:
    img = Image.open(rec.path).convert("RGB")
    bbox = server.clamp_bbox(server.detect_grid_bbox(img), img.width, img.height)
    crop = img.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))
    masked = server.remove_red_marks(crop)
    occupancy = server.detect_cell_occupancy(masked)
    crop_width = 560
    crop_height = int(masked.height * (crop_width / masked.width))
    crop_preview = masked.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    heatmap = draw_occupancy_grid(occupancy, min(760, max(560, crop_height)))
    title_height = 64
    canvas = Image.new(
        "RGB",
        (crop_preview.width + heatmap.width + 28, max(crop_preview.height, heatmap.height) + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    small = load_font(16)
    draw.text((12, 10), rec.name, fill=(20, 20, 20), font=font)
    draw.text((12, 38), f"crop/occupancy only bbox={bbox}", fill=(80, 80, 80), font=small)
    canvas.paste(crop_preview, (12, title_height))
    canvas.paste(heatmap, (crop_preview.width + 24, title_height))
    return canvas


def draw_occupancy_grid(occupancy: list[dict[str, Any]], size: int) -> Image.Image:
    cell = size // 20
    grid_size = cell * 20
    img = Image.new("RGB", (grid_size + 1, grid_size + 1), "white")
    draw = ImageDraw.Draw(img)
    for row_index, row in enumerate(occupancy[:20]):
        counts = row.get("counts") if isinstance(row, dict) else None
        threshold = int(row.get("threshold") or 1) if isinstance(row, dict) else 1
        if not isinstance(counts, list):
            continue
        max_count = max([int(count) for count in counts[:20]] or [1])
        for col_index, count in enumerate(counts[:20]):
            count = int(count)
            if count <= 0:
                continue
            ratio = min(1.0, count / max(1, max_count))
            if count >= threshold:
                fill = (210 - int(80 * ratio), 228 - int(90 * ratio), 214 - int(60 * ratio))
            else:
                fill = (245, 238, 198)
            x0 = col_index * cell + 1
            y0 = row_index * cell + 1
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=fill)
    grid_color = (82, 155, 185)
    for index in range(21):
        pos = index * cell
        width = 2 if index in {0, 20} else 1
        draw.line((0, pos, grid_size, pos), fill=grid_color, width=width)
        draw.line((pos, 0, pos, grid_size), fill=grid_color, width=width)
    return img


def draw_cells_grid(cells: list[list[str]], size: int) -> Image.Image:
    cell = size // 20
    grid_size = cell * 20
    img = Image.new("RGB", (grid_size + 1, grid_size + 1), "white")
    draw = ImageDraw.Draw(img)
    grid_color = (82, 155, 185)
    text_color = (20, 20, 20)
    font = load_font(max(13, int(cell * 0.46)))
    small_font = load_font(max(11, int(cell * 0.36)))
    for index in range(21):
        pos = index * cell
        width = 2 if index in {0, 20} else 1
        draw.line((0, pos, grid_size, pos), fill=grid_color, width=width)
        draw.line((pos, 0, pos, grid_size), fill=grid_color, width=width)
    for row_index, row in enumerate(cells):
        for col_index, value in enumerate(row):
            text = str(value)
            if not text.strip():
                continue
            font_for_cell = small_font if len(list(text.strip())) > 1 else font
            bbox = draw.textbbox((0, 0), text, font=font_for_cell)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = col_index * cell + (cell - text_width) / 2
            y = row_index * cell + (cell - text_height) / 2 - 1
            draw.text((x, y), text, fill=text_color, font=font_for_cell)
    return img


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
