#!/usr/bin/env python3
"""Split the scanned manuscript PDF into one PNG image per page."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path


DEFAULT_PDF_NAME = "중1_3과정 1주차_원고지_스캔 파일.pdf"
DEFAULT_DPI = 200


def normalize_name(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_pdf_path(root: Path, requested: str) -> Path:
    raw = Path(requested).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(root / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    requested_name = normalize_name(raw.name)
    for pdf_path in root.glob("*.pdf"):
        if normalize_name(pdf_path.name) == requested_name:
            return pdf_path.resolve()

    available = ", ".join(sorted(normalize_name(path.name) for path in root.glob("*.pdf")))
    raise FileNotFoundError(
        f"PDF not found: {requested}. Available PDFs in {root}: {available or 'none'}"
    )


def default_output_dir(root: Path, pdf_path: Path) -> Path:
    safe_stem = normalize_name(pdf_path.stem)
    safe_stem = re.sub(r"\s+", "_", safe_stem)
    safe_stem = re.sub(r"[^\w가-힣.-]+", "_", safe_stem)
    safe_stem = safe_stem.strip("._") or "pdf_pages"
    return root / "data" / f"{safe_stem}_pages"


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise RuntimeError(
        f"'{name}' was not found on PATH. Install Poppler, then run this script again."
    )


def get_pdf_page_count(pdf_path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None

    completed = subprocess.run(
        [pdfinfo, str(pdf_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def page_number(path: Path) -> int:
    match = re.fullmatch(r"page-(\d+)\.png", path.name)
    if not match:
        raise ValueError(f"Unexpected pdftoppm output filename: {path.name}")
    return int(match.group(1))


def render_pages(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    pdftoppm = find_tool("pdftoppm")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_pages = get_pdf_page_count(pdf_path)

    with tempfile.TemporaryDirectory(prefix="pdf_pages_", dir=output_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        prefix = tmp_dir / "page"
        subprocess.run(
            [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True,
        )

        rendered = sorted(tmp_dir.glob("page-*.png"), key=page_number)
        if not rendered:
            raise RuntimeError("pdftoppm completed but did not produce any PNG files.")
        if expected_pages is not None and len(rendered) != expected_pages:
            raise RuntimeError(
                f"Expected {expected_pages} page images, but rendered {len(rendered)}."
            )

        for old_page in output_dir.glob("page_*.png"):
            old_page.unlink()

        width = max(3, len(str(page_number(rendered[-1]))))
        final_paths: list[Path] = []
        for source in rendered:
            destination = output_dir / f"page_{page_number(source):0{width}d}.png"
            source.replace(destination)
            final_paths.append(destination)

    return final_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render each page of a PDF as a separate PNG image."
    )
    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF_NAME,
        help=f"PDF path or filename. Default: {DEFAULT_PDF_NAME}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for rendered PNG files. Default: data/<pdf_stem>_pages",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG render resolution. Default: {DEFAULT_DPI}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    pdf_path = resolve_pdf_path(root, args.pdf)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else default_output_dir(root, pdf_path)
    )
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    final_paths = render_pages(pdf_path, output_dir.resolve(), args.dpi)
    print(f"PDF: {pdf_path}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Rendered pages: {len(final_paths)}")
    print(f"First image: {final_paths[0]}")
    print(f"Last image: {final_paths[-1]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
