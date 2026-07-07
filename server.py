from __future__ import annotations

import base64
import hashlib
import heapq
import html
import ipaddress
import json
import os
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
STATIC_DIR = ROOT / "static"
RUNS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_DATA_DIR_NAME = "논술원고지_답안"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")

GPT_OAUTH_BASE_URL = os.getenv("GPT_OAUTH_BASE_URL", "http://192.168.0.16:31835").rstrip("/")
DOGOK_PROXY_API_KEY = os.getenv("DOGOK_PROXY_API_KEY", "")
GPT_OAUTH_DEFAULT_MODEL = "gpt-5.5"
GPT_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
TRAILING_CELL_PUNCTUATION = {".", ","}
PROPER_NOUN_CONTEXT_NOUNS = {
    "행동",
    "말",
    "태도",
    "선택",
    "입장",
    "관점",
    "생각",
    "가치",
    "의미",
    "역할",
    "성격",
    "특징",
    "모습",
}
PROPER_NOUN_STOPWORDS = {
    "내용",
    "문장",
    "근거",
    "측면",
    "주안점",
    "요약",
    "정리",
    "재진술",
    "접속표현",
}
VLM_PRIMARY_BASE_URL = os.getenv("VLM_PRIMARY_BASE_URL", "http://classday.iptime.org:8979/v1").rstrip("/")
VLM_FALLBACK_BASE_URL = os.getenv("VLM_FALLBACK_BASE_URL", "http://210.115.224.151:31882/v1").rstrip("/")
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "chandra")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

OCR_PROMPT = """You are extracting a Korean student's handwritten essay from a 20 by 20 grid answer sheet.

Return only valid JSON. Do not use markdown fences.

Rules:
- Ignore every red mark, red circle, red score, teacher correction, and comment.
- Read only gray/black pencil handwriting inside the blue 20x20 grid.
- This is literal transcription, not essay correction. Never correct grammar, spelling, particles, endings, word choice, awkward wording, or factual mistakes.
- Preserve the student's mistakes exactly. If the handwriting shows an unnatural or wrong phrase such as "있다그", output "있다그"; do not change it to "있다고".
- Do not infer the intended word from context. The image's visible handwriting is the only source of truth.
- Preserve the student's spacing exactly. Empty cells must be a single space.
- Output exactly 20 rows and exactly 20 cells per row.
- Prefer this schema: {"cells":[[" ","글","자", "... 20 cells"], "... 20 rows"]}.
- Each JSON cell must match one physical grid square in the image.
- Do not tokenize numbers or words by semantic rules. Read the physical square boundaries.
- If two visible characters are written inside one square, keep both in that one JSON cell, even when they mix digit/Hangul/English, for example a square containing "0만" must be output as "0만".
- If the image shows "20" in one square and "0만" in the next square, output ["20","0만"], not ["20","0","만"].
- Put commas and periods in their own physical grid cell when the image shows the mark in the next square, for example ["으","며",","] or ["고",","].
- Only attach a comma or period to the previous cell when the punctuation mark would overflow at the end of a row, for example ["했","다."].
- Do not join lines into paragraphs.
"""

CHANDRA_OCR_PROMPT = """OCR this image to HTML.

Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', 'href', 'alt', 'align'].

Guidelines:
* Transcribe only gray/black student handwriting inside the blue 20 by 20 grid.
* Ignore every red mark, red circle, red score, teacher correction, and comment.
* This is literal transcription, not essay correction. Never correct grammar, spelling, particles, endings, word choice, awkward wording, or factual mistakes.
* Preserve the student's mistakes exactly. If the handwriting shows an unnatural or wrong phrase such as "있다그", output "있다그"; do not change it to "있다고".
* Do not infer the intended word from context. The image's visible handwriting is the only source of truth.
* Preserve the written row order. Use <br> at original grid row breaks.
* Preserve visible spaces where the student left blank cells.
* Follow the physical grid squares, not semantic word or number boundaries.
* If two visible characters are written in one square, keep them together, for example "20" or "0만".
* Keep commas and periods in their own physical grid cell when the mark is written in the next square.
* Do not output layout-only labels or bounding boxes without text content.
* Use the simplest possible HTML that accurately represents the handwriting.
"""

REPAIR_PROMPT = """You are repairing OCR for a Korean 20 by 20 handwritten grid answer sheet.

Compare the image with the current OCR rows. Ignore every red mark, red circle, red score, teacher correction, and comment.
Return only valid JSON using this schema: {"cells":[[" ","글","자", "... 20 cells"], "... 20 rows"], "notes":"short Korean explanation"}.
The rows must match the student's gray/black handwriting and preserve empty cells as spaces.
This is literal OCR repair, not essay correction. Never correct grammar, spelling, particles, endings, word choice, awkward wording, or factual mistakes.
Preserve the student's mistakes exactly. If the handwriting shows an unnatural or wrong phrase such as "있다그", output "있다그"; do not change it to "있다고".
Do not infer the intended word from context. The image's visible handwriting is the only source of truth, even when the sentence is wrong Korean.
Each JSON cell must match one physical grid square in the image. Do not split numbers or words by semantic rules.
If two visible characters are written inside one square, keep both in that one JSON cell, for example "20" or "0만".
Keep commas and periods in their own physical grid cell when the mark is written in the next square, for example ["으","며",","].
Only attach a comma or period to the previous cell when the punctuation mark would overflow past the row end.

Current OCR data:
{rows}
"""

CHANDRA_REPAIR_PROMPT = CHANDRA_OCR_PROMPT + """

Compare the image with these current OCR cells and return image-faithful HTML text only. Do not fix student grammar, spelling, particles, endings, word choice, awkward wording, or factual mistakes:
{rows}
"""


@dataclass(frozen=True)
class ImageFolder:
    id: str
    name: str
    path: Path
    relative_path: str
    image_count: int


@dataclass(frozen=True)
class ImageRecord:
    id: str
    name: str
    path: Path
    width: int
    height: int
    folder_id: str
    folder_name: str
    relative_path: str


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class CropRequest(BaseModel):
    image_id: str
    bbox: BBox | None = None


class ModelSettings(BaseModel):
    model: str | None = None
    temperature: float = Field(default=0.3, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    max_tokens: int = Field(default=4096, ge=128, le=20000)
    reasoning_effort: str = "low"
    n: int = Field(default=1, ge=1, le=5)


class OcrRequest(BaseModel):
    run_id: str
    provider: Literal["chandra", "gpt"]
    settings: ModelSettings = Field(default_factory=ModelSettings)
    user_conditions: str = Field(default="", max_length=4000)


class RepairRequest(BaseModel):
    run_id: str
    source_provider: Literal["chandra", "gpt"]
    provider: Literal["gpt", "chandra"] = "gpt"
    settings: ModelSettings = Field(default_factory=ModelSettings)
    user_conditions: str = Field(default="", max_length=4000)


ExperimentStrategy = Literal["full_grid", "row_1", "row_2", "row_5", "vote_full_row_1_2"]


class ExperimentRequest(BaseModel):
    run_id: str
    provider: Literal["chandra", "gpt"]
    strategies: list[ExperimentStrategy] = Field(default_factory=lambda: ["full_grid", "row_1", "row_2", "vote_full_row_1_2"])
    settings: ModelSettings = Field(default_factory=ModelSettings)
    user_conditions: str = Field(default="", max_length=4000)


class FinalizeRequest(BaseModel):
    run_id: str
    source: Literal["chandra", "gpt", "repair", "manual"]
    rows: list[Any] | None = None
    cells: list[list[str]] | None = None


jobs_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}

app = FastAPI(title="논술 원고지 OCR 비교/Repair", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    response = FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/config")
def config() -> dict[str, Any]:
    public_base_url = get_public_base_url()
    gpt_image_warning = gpt_image_url_warning(public_base_url)
    return {
        "gpt_oauth": {
            "base_url": GPT_OAUTH_BASE_URL,
            "configured": bool(DOGOK_PROXY_API_KEY),
            "default_model": GPT_OAUTH_DEFAULT_MODEL,
            "public_base_url": public_base_url,
            "image_warning": gpt_image_warning,
        },
        "chandra": {
            "primary_base_url": VLM_PRIMARY_BASE_URL,
            "fallback_base_url": VLM_FALLBACK_BASE_URL,
            "configured": bool(VLM_PRIMARY_BASE_URL and VLLM_API_KEY),
            "default_model": VLM_MODEL_ID,
        },
    }


@app.get("/api/image-folders")
def api_image_folders() -> dict[str, Any]:
    folders = list_image_folders()
    default_folder = default_image_folder(folders)
    return {
        "default_folder_id": default_folder.id if default_folder else None,
        "folders": [
            {
                "id": folder.id,
                "name": folder.name,
                "relative_path": folder.relative_path,
                "image_count": folder.image_count,
            }
            for folder in folders
        ],
    }


def get_public_base_url() -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return f"http://{detect_lan_ip()}:8767"


def detect_lan_ip() -> str:
    import socket

    candidates: list[str] = []
    for host in (socket.gethostname(), socket.getfqdn()):
        try:
            candidates.extend(socket.gethostbyname_ex(host)[2])
        except OSError:
            pass
    for ip in candidates:
        if ip and not ip.startswith("127."):
            return ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def run_public_url(run_id: str, filename: str) -> str:
    return f"{get_public_base_url()}/runs/{run_id}/{filename}"


def gpt_image_url_warning(public_base_url: str) -> str | None:
    parsed = urlparse(public_base_url)
    host = parsed.hostname or ""
    if not host:
        return "PUBLIC_BASE_URL is empty or invalid; GPT image OCR cannot fetch crop images."
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return "GPT image OCR needs a public image URL; localhost URLs are not fetchable by the upstream model."
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return "GPT image OCR needs a public image URL; current PUBLIC_BASE_URL is private/LAN."
    except ValueError:
        pass
    return None


@app.get("/api/images")
def api_images(folder_id: str | None = None) -> dict[str, Any]:
    folder = get_image_folder(folder_id) if folder_id else default_image_folder_or_404()
    records = list_images(folder.id)
    return {
        "folder": {
            "id": folder.id,
            "name": folder.name,
            "relative_path": folder.relative_path,
            "image_count": folder.image_count,
        },
        "images": [
            {
                "id": rec.id,
                "name": rec.name,
                "folder_id": rec.folder_id,
                "folder_name": rec.folder_name,
                "relative_path": rec.relative_path,
                "width": rec.width,
                "height": rec.height,
                "url": f"/api/images/{rec.id}/file",
            }
            for rec in records
        ]
    }


@app.get("/api/images/{image_id}/file")
def image_file(image_id: str) -> FileResponse:
    rec = get_image_record(image_id)
    return FileResponse(rec.path)


@app.get("/api/runs")
def api_runs(folder_id: str | None = None, image_id: str | None = None) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for metadata_path in RUNS_DIR.glob("*/metadata.json"):
        try:
            metadata = read_json(metadata_path)
        except Exception:
            continue
        if image_id and metadata.get("image_id") != image_id:
            continue
        metadata_folder_id = metadata.get("folder_id")
        if not metadata_folder_id:
            metadata_folder_id = image_folder_id_for_image_id(metadata.get("image_id"))
        if folder_id and metadata_folder_id != folder_id:
            continue
        run_dir = metadata_path.parent
        runs.append(
            {
                "run_id": metadata.get("run_id") or run_dir.name,
                "image_id": metadata.get("image_id"),
                "image_name": metadata.get("image_name"),
                "folder_id": metadata_folder_id,
                "folder_name": metadata.get("folder_name") or folder_name_for_id(metadata_folder_id),
                "created_at": metadata.get("created_at", 0),
                "has_chandra": (run_dir / "ocr_chandra.json").exists(),
                "has_gpt": (run_dir / "ocr_gpt.json").exists(),
                "has_repair": (run_dir / "repair.json").exists(),
                "has_experiments": any((run_dir / "experiments").glob("summary_*.json")) if (run_dir / "experiments").exists() else False,
                "has_final": (run_dir / "final.json").exists(),
            }
        )
    runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"runs": runs[:80]}


@app.post("/api/crop")
def api_crop(req: CropRequest) -> dict[str, Any]:
    rec = get_image_record(req.image_id)
    img = Image.open(rec.path).convert("RGB")
    bbox = req.bbox.model_dump() if req.bbox else detect_grid_bbox(img)
    bbox = clamp_bbox(bbox, img.width, img.height)
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    crop = img.crop((bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))
    masked = remove_red_marks(crop)
    occupancy = detect_cell_occupancy(masked)
    crop.save(run_dir / "crop.png")
    masked.save(run_dir / "masked.png")

    metadata = {
        "run_id": run_id,
        "image_id": rec.id,
        "image_name": rec.name,
        "folder_id": rec.folder_id,
        "folder_name": rec.folder_name,
        "folder_path": rec.relative_path.rsplit("/", 1)[0] if "/" in rec.relative_path else "",
        "image_url": f"/api/images/{rec.id}/file",
        "bbox": bbox,
        "grid": {
            "rows": 20,
            "columns": 20,
            "cell_width": bbox["width"] / 20,
            "cell_height": bbox["height"] / 20,
        },
        "cell_occupancy": occupancy,
        "created_at": time.time(),
    }
    write_json(run_dir / "metadata.json", metadata)
    return run_payload(run_id)


@app.post("/api/ocr")
def api_ocr(req: OcrRequest) -> dict[str, Any]:
    ensure_run(req.run_id)
    if req.provider == "gpt" and not DOGOK_PROXY_API_KEY:
        raise HTTPException(status_code=400, detail="DOGOK_PROXY_API_KEY is not set")
    if req.provider == "chandra" and not (VLM_PRIMARY_BASE_URL and VLLM_API_KEY):
        raise HTTPException(status_code=400, detail="VLM endpoint is not configured")

    job_id = create_job("ocr", req.model_dump())
    thread = threading.Thread(target=run_ocr_job, args=(job_id, req), daemon=True)
    thread.start()
    return jobs[job_id]


@app.post("/api/repair")
def api_repair(req: RepairRequest) -> dict[str, Any]:
    ensure_run(req.run_id)
    run_dir = RUNS_DIR / req.run_id
    if not (run_dir / f"ocr_{req.source_provider}.json").exists():
        raise HTTPException(status_code=400, detail=f"No OCR result for {req.source_provider}")
    if req.provider == "gpt" and not DOGOK_PROXY_API_KEY:
        raise HTTPException(status_code=400, detail="DOGOK_PROXY_API_KEY is not set")

    job_id = create_job("repair", req.model_dump())
    thread = threading.Thread(target=run_repair_job, args=(job_id, req), daemon=True)
    thread.start()
    return jobs[job_id]


@app.post("/api/experiments")
def api_experiments(req: ExperimentRequest) -> dict[str, Any]:
    ensure_run(req.run_id)
    if req.provider == "gpt" and not DOGOK_PROXY_API_KEY:
        raise HTTPException(status_code=400, detail="DOGOK_PROXY_API_KEY is not set")
    if req.provider == "chandra" and not (VLM_PRIMARY_BASE_URL and VLLM_API_KEY):
        raise HTTPException(status_code=400, detail="VLM endpoint is not configured")
    if not req.strategies:
        raise HTTPException(status_code=400, detail="At least one experiment strategy is required")

    job_id = create_job("experiment", req.model_dump())
    thread = threading.Thread(target=run_experiment_job, args=(job_id, req), daemon=True)
    thread.start()
    return jobs[job_id]


@app.post("/api/finalize")
def api_finalize(req: FinalizeRequest) -> dict[str, Any]:
    run_dir = ensure_run(req.run_id)
    if req.cells is not None or req.rows is not None:
        cells, validation = normalize_cells(req.cells if req.cells is not None else req.rows)
    else:
        cells, validation = cells_from_source(run_dir, req.source)
    final = {
        "source": req.source,
        "rows": cells_to_text_rows(cells),
        "cells": cells,
        "validation": validation,
        "saved_at": time.time(),
    }
    write_json(run_dir / "final.json", final)
    (run_dir / "final.html").write_text(render_grid_html(cells, "최종 답안"), encoding="utf-8")
    return run_payload(req.run_id)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(job)


@app.get("/api/jobs")
def api_jobs(folder_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    with jobs_lock:
        items = [dict(job) for job in jobs.values()]
    filtered: list[dict[str, Any]] = []
    for job in items:
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        job_run_id = payload.get("run_id")
        if run_id and job_run_id != run_id:
            continue
        if folder_id and job_run_id:
            try:
                metadata = read_json(RUNS_DIR / job_run_id / "metadata.json")
            except Exception:
                continue
            metadata_folder_id = metadata.get("folder_id") or image_folder_id_for_image_id(metadata.get("image_id"))
            if metadata_folder_id != folder_id:
                continue
        filtered.append(job)
    filtered.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"jobs": filtered}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict[str, Any]:
    ensure_run(run_id)
    return run_payload(run_id)


@app.get("/runs/{run_id}/{filename}")
def run_file(run_id: str, filename: str) -> FileResponse:
    run_dir = ensure_run(run_id)
    allowed = {
        "crop.png",
        "masked.png",
        "render_chandra.html",
        "render_gpt.html",
        "render_repair.html",
        "final.html",
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="file not found")
    path = run_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


@app.get("/runs/{run_id}/experiments/{filename}")
def experiment_file(run_id: str, filename: str) -> FileResponse:
    run_dir = ensure_run(run_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
        raise HTTPException(status_code=404, detail="file not found")
    path = run_dir / "experiments" / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


def list_image_folders() -> list[ImageFolder]:
    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="data directory not found")

    folders: list[ImageFolder] = []
    for child in sorted(DATA_DIR.iterdir(), key=lambda path: natural_sort_key(unicodedata.normalize("NFC", path.name))):
        if not child.is_dir():
            continue
        image_count = len(iter_image_paths(child))
        if image_count <= 0:
            continue
        rel = child.relative_to(ROOT).as_posix()
        folders.append(
            ImageFolder(
                id=folder_id_for_path(child),
                name=unicodedata.normalize("NFC", child.name),
                path=child,
                relative_path=rel,
                image_count=image_count,
            )
        )

    if not folders:
        raise HTTPException(status_code=404, detail="answer image directory not found")
    return sorted(folders, key=folder_sort_key)


def folder_id_for_path(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]


def folder_sort_key(folder: ImageFolder) -> tuple[int, str]:
    normalized = unicodedata.normalize("NFC", folder.name)
    if normalized == TARGET_DATA_DIR_NAME:
        return (0, normalized)
    if "논술" in normalized and "답안" in normalized:
        return (1, normalized)
    return (2, normalized)


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def default_image_folder(folders: list[ImageFolder] | None = None) -> ImageFolder | None:
    candidates = folders if folders is not None else list_image_folders()
    return candidates[0] if candidates else None


def default_image_folder_or_404() -> ImageFolder:
    folder = default_image_folder()
    if not folder:
        raise HTTPException(status_code=404, detail="answer image directory not found")
    return folder


def get_image_folder(folder_id: str) -> ImageFolder:
    for folder in list_image_folders():
        if folder.id == folder_id:
            return folder
    raise HTTPException(status_code=404, detail="image folder not found")


def folder_name_for_id(folder_id: str | None) -> str | None:
    if not folder_id:
        return None
    try:
        return get_image_folder(folder_id).name
    except HTTPException:
        return None


def list_images(folder_id: str | None = None) -> list[ImageRecord]:
    folder = get_image_folder(folder_id) if folder_id else default_image_folder_or_404()
    paths = iter_image_paths(folder.path)
    records: list[ImageRecord] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            continue
        rel = path.relative_to(ROOT).as_posix()
        image_id = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
        records.append(
            ImageRecord(
                id=image_id,
                name=unicodedata.normalize("NFC", path.name),
                path=path,
                width=width,
                height=height,
                folder_id=folder.id,
                folder_name=folder.name,
                relative_path=path.relative_to(ROOT).as_posix(),
            )
        )
    return records


def find_answer_dir() -> Path:
    return default_image_folder_or_404().path


def get_image_record(image_id: str) -> ImageRecord:
    for folder in list_image_folders():
        for rec in list_images(folder.id):
            if rec.id == image_id:
                return rec
    raise HTTPException(status_code=404, detail="image not found")


def image_folder_id_for_image_id(image_id: str | None) -> str | None:
    if not image_id:
        return None
    try:
        folders = list_image_folders()
    except HTTPException:
        return None
    for folder in folders:
        for path in iter_image_paths(folder.path):
            rel = path.relative_to(ROOT).as_posix()
            if hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12] == image_id:
                return folder.id
    return None


def iter_image_paths(folder_path: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder_path.rglob("*"), key=lambda item: natural_sort_key(item.relative_to(folder_path).as_posix()))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def detect_grid_bbox(img: Image.Image) -> dict[str, int]:
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    yy, xx = np.indices((h, w))
    lower_left = (yy > int(h * 0.36)) & (xx < int(w * 0.79))
    blue_grid = (
        lower_left
        & (g > 105)
        & (b > 105)
        & ((g - r) > 10)
        & ((b - r) > 10)
        & (np.abs(g - b) < 95)
    )
    ys, xs = np.where(blue_grid)
    if xs.size < 100:
        return default_bbox(w, h)

    x1 = int(np.percentile(xs, 0.3))
    x2 = int(np.percentile(xs, 99.7))
    y1 = int(np.percentile(ys, 0.3))
    y2 = int(np.percentile(ys, 99.7))

    # Expand slightly so outer grid lines are not clipped.
    return clamp_bbox(
        {"x": x1 - 6, "y": y1 - 6, "width": (x2 - x1) + 12, "height": (y2 - y1) + 12},
        w,
        h,
    )


def detect_cell_occupancy(img: Image.Image) -> list[dict[str, Any]]:
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    blue_grid = (
        (g > 105)
        & (b > 105)
        & ((g - r) > 10)
        & ((b - r) > 10)
        & (np.abs(g - b) < 95)
    )
    dark_pencil = (r < 185) & (g < 185) & (b < 185) & (~blue_grid)
    rows: list[dict[str, Any]] = []
    for row_index in range(20):
        y0 = int(row_index * h / 20)
        y1 = int((row_index + 1) * h / 20)
        counts: list[int] = []
        for col_index in range(20):
            x0 = int(col_index * w / 20)
            x1 = int((col_index + 1) * w / 20)
            counts.append(int(dark_pencil[y0:y1, x0:x1].sum()))
        threshold = max(20, int(max(counts, default=0) * 0.10))
        occupied = [i for i, count in enumerate(counts) if count >= threshold]
        rows.append(
            {
                "row": row_index,
                "first_col": occupied[0] if occupied else None,
                "last_col": occupied[-1] if occupied else None,
                "counts": counts,
                "threshold": threshold,
            }
        )
    return rows


def align_rows_to_occupancy(rows: list[str], occupancy: list[dict[str, Any]]) -> list[str]:
    cells, _ = normalize_cells(rows, occupancy)
    return cells_to_text_rows(cells)


def align_cells_to_occupancy(cells: list[list[str]], occupancy: list[dict[str, Any]]) -> list[list[str]]:
    aligned: list[list[str]] = []
    for row_index in range(20):
        row = fit_cells(cells[row_index] if row_index < len(cells) else [])
        occ = occupancy[row_index] if row_index < len(occupancy) else {}
        if not row_has_text(row):
            aligned.append([" "] * 20)
            continue

        occupied = occupied_columns(occ)
        if occupied and occ.get("counts"):
            aligned.append(align_row_to_occupancy_pattern(row, occ, occupied))
            continue

        first_col = occ.get("first_col")
        if first_col is None:
            aligned.append(row)
            continue
        target = [" "] * 20
        for offset, cell in enumerate(trim_empty_cells(row)[: 20 - int(first_col)]):
            target[int(first_col) + offset] = normalize_cell_value(cell)
        aligned.append(target)
    return aligned


def align_row_to_occupancy_pattern(row: list[str], occ: dict[str, Any], occupied: list[int]) -> list[str]:
    first_col = occupied[0]
    source = trim_empty_cells(row)
    last_col = max(occupied[-1], min(19, first_col + len(source) - 1))
    target_states = occupancy_states(occ, first_col, last_col)
    source = expand_cells_for_occupancy(source)
    if not has_internal_blank(row):
        compact_text = "".join(cell for cell in row if cell.strip())
        compact_units = pack_text_units(compact_text, preserve_spaces=False)
        if compact_units:
            source = compact_units

    span = align_source_to_occupancy(source, target_states)
    target = [" "] * 20
    for offset, cell in enumerate(span[: len(target_states)]):
        target[first_col + offset] = normalize_cell_value(cell)
    return target


def occupancy_states(occ: dict[str, Any], first_col: int, last_col: int) -> list[bool | None]:
    counts = occ.get("counts")
    if not isinstance(counts, list):
        occupied = set(occupied_columns(occ))
        return [col in occupied for col in range(first_col, last_col + 1)]
    numeric_counts = [safe_int(count, 0) for count in counts[:20]]
    threshold = safe_int(occ.get("threshold"), max(20, int(max(numeric_counts, default=0) * 0.10)))
    states: list[bool | None] = []
    for col in range(first_col, last_col + 1):
        if col >= len(numeric_counts):
            states.append(True)
        elif numeric_counts[col] >= threshold:
            states.append(True)
        elif numeric_counts[col] == 0:
            states.append(False)
        else:
            states.append(None)
    return states


def align_source_to_occupancy(source: list[str], target_states: list[bool | None]) -> list[str]:
    n = len(source)
    m = len(target_states)
    queue: list[tuple[float, int, int, list[str]]] = [(0.0, 0, 0, [])]
    best: dict[tuple[int, int], float] = {(0, 0): 0.0}
    candidates: list[tuple[float, list[str]]] = []

    def push(cost: float, i: int, j: int, cells: list[str]) -> None:
        key = (i, j)
        if cost >= best.get(key, float("inf")):
            return
        best[key] = cost
        heapq.heappush(queue, (cost, i, j, cells))

    while queue:
        cost, i, j, cells = heapq.heappop(queue)
        if cost > best.get((i, j), float("inf")):
            continue
        if j == m:
            leftover = source[i:]
            leftover_cost = sum(0.2 if not normalize_cell_value(cell).strip() else 2.0 for cell in leftover)
            candidates.append((cost + leftover_cost, cells))
            if cells and leftover:
                leftover_units = [normalize_cell_value(cell) for cell in leftover if normalize_cell_value(cell).strip()]
                if (
                    len(leftover_units) == 1
                    and leftover_units[0] in TRAILING_CELL_PUNCTUATION
                    and normalize_cell_value(cells[-1]).strip()
                    and len(list(normalize_cell_value(cells[-1]).strip())) < 2
                ):
                    merged_cells = cells[:-1] + [normalize_cell_value(cells[-1] + leftover_units[0])]
                    candidates.append((cost + 0.08, merged_cells))
            continue

        source_cell = normalize_cell_value(source[i]) if i < n else None
        if source_cell is not None and not source_cell.strip():
            push(cost + 0.12, i + 1, j, cells)

        state = target_states[j]
        if state is True:
            if source_cell is not None and source_cell.strip():
                push(cost, i + 1, j + 1, cells + [source_cell])
            if i + 1 < n and source[i].strip() and source[i + 1].strip():
                merged = merge_adjacent_cells(source[i], source[i + 1])
                if merged:
                    push(cost + merge_cost(source[i], source[i + 1]), i + 2, j + 1, cells + [merged])
            push(cost + 2.0, i, j + 1, cells + [" "])
        elif state is False:
            if source_cell is not None and not source_cell.strip():
                push(cost, i + 1, j + 1, cells + [" "])
            push(cost + 0.35, i, j + 1, cells + [" "])
            if source_cell is not None:
                if source_cell in TRAILING_CELL_PUNCTUATION:
                    push(cost + 0.1, i + 1, j + 1, cells + [source_cell])
                elif source_cell.strip():
                    push(cost + 3.0, i + 1, j + 1, cells + [" "])
        else:
            if source_cell is not None:
                if source_cell.strip():
                    punctuation_bonus = -0.25 if source_cell in TRAILING_CELL_PUNCTUATION else 0.0
                    push(cost + 0.05 + punctuation_bonus, i + 1, j + 1, cells + [source_cell])
                else:
                    push(cost, i + 1, j + 1, cells + [" "])
            push(cost + 0.2, i, j + 1, cells + [" "])
            if i + 1 < n and source[i].strip() and source[i + 1].strip():
                merged = merge_adjacent_cells(source[i], source[i + 1])
                if merged:
                    push(cost + merge_cost(source[i], source[i + 1]) + 0.1, i + 2, j + 1, cells + [merged])

    if not candidates:
        return fit_cells(source, m)
    return min(candidates, key=lambda item: item[0])[1]


def merge_adjacent_cells(left: str, right: str) -> str | None:
    left_text = normalize_cell_value(left).strip()
    right_text = normalize_cell_value(right).strip()
    if left_text in TRAILING_CELL_PUNCTUATION or right_text in TRAILING_CELL_PUNCTUATION:
        return None
    if not (
        (left_text.isdigit() and is_hangul(right_text))
        or (left_text.isdigit() and right_text.isdigit())
        or (is_ascii_lower(left_text) and is_ascii_lower(right_text))
    ):
        return None
    merged = left_text + right_text
    if not merged or len(list(merged)) > 2:
        return None
    return merged


def merge_cost(left: str, right: str) -> float:
    left_text = normalize_cell_value(left).strip()
    right_text = normalize_cell_value(right).strip()
    if left_text.isdigit() and is_hangul(right_text):
        return 0.05
    if left_text.isdigit() and right_text.isdigit():
        return 0.15
    if is_ascii_lower(left_text) and is_ascii_lower(right_text):
        return 0.2
    return 0.7


def is_hangul(value: str) -> bool:
    return bool(value) and all("가" <= char <= "힣" for char in value)


def is_ascii_lower(value: str) -> bool:
    return bool(value) and all("a" <= char <= "z" for char in value)


def occupied_columns(occ: dict[str, Any]) -> list[int]:
    counts = occ.get("counts")
    if isinstance(counts, list) and counts:
        numeric_counts = [safe_int(count, 0) for count in counts[:20]]
        threshold = safe_int(occ.get("threshold"), max(20, int(max(numeric_counts, default=0) * 0.10)))
        return [index for index, count in enumerate(numeric_counts) if count >= threshold]
    first_col = occ.get("first_col")
    last_col = occ.get("last_col")
    if first_col is None or last_col is None:
        return []
    return list(range(max(0, safe_int(first_col, 0)), min(19, safe_int(last_col, 19)) + 1))


def row_has_text(row: list[str]) -> bool:
    return any(cell.strip() for cell in row)


def has_internal_blank(row: list[str]) -> bool:
    text_positions = [index for index, cell in enumerate(row) if cell.strip()]
    if len(text_positions) < 2:
        return False
    return any(not row[index].strip() for index in range(text_positions[0] + 1, text_positions[-1]))


def trim_empty_cells(row: list[str]) -> list[str]:
    start = 0
    end = len(row)
    while start < end and not row[start].strip():
        start += 1
    while end > start and not row[end - 1].strip():
        end -= 1
    return row[start:end]


def default_bbox(width: int, height: int) -> dict[str, int]:
    return {
        "x": int(width * 0.045),
        "y": int(height * 0.39),
        "width": int(width * 0.71),
        "height": int(height * 0.55),
    }


def clamp_bbox(bbox: dict[str, int | float], width: int, height: int) -> dict[str, int]:
    x = max(0, min(int(round(bbox["x"])), width - 1))
    y = max(0, min(int(round(bbox["y"])), height - 1))
    w = max(1, min(int(round(bbox["width"])), width - x))
    h = max(1, min(int(round(bbox["height"])), height - y))
    return {"x": x, "y": y, "width": w, "height": h}


def remove_red_marks(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).copy()
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    red = (r > 135) & ((r - g) > 35) & ((r - b) > 35) & (g < 185) & (b < 185)
    red_img = Image.fromarray(red.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))
    mask = np.asarray(red_img) > 0
    arr[mask] = [255, 255, 255]
    return Image.fromarray(arr)


def create_job(kind: str, payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0,
            "payload": payload,
            "created_at": time.time(),
        }
    return job_id


def update_job(job_id: str, **values: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(values)
            jobs[job_id]["updated_at"] = time.time()


def run_experiment_job(job_id: str, req: ExperimentRequest) -> None:
    update_job(job_id, status="running", progress=0.05)
    try:
        run_dir = ensure_run(req.run_id)
        metadata = read_json(run_dir / "metadata.json")
        occupancy = metadata.get("cell_occupancy", [])
        exp_dir = run_dir / "experiments"
        exp_dir.mkdir(exist_ok=True)
        strategies = expand_experiment_strategies(req.strategies)
        model_strategies = [strategy for strategy in strategies if strategy != "vote_full_row_1_2"]
        results: dict[str, dict[str, Any]] = {}

        for index, strategy in enumerate(model_strategies):
            update_job(job_id, progress=0.08 + (index / max(len(model_strategies), 1)) * 0.72, active_strategy=strategy)
            result = run_model_experiment_strategy(req, run_dir, exp_dir, strategy, occupancy)
            results[strategy] = result
            write_json(exp_dir / f"{strategy}_{req.provider}.json", result)
            (exp_dir / f"render_{strategy}_{req.provider}.html").write_text(
                render_grid_html(result["cells"], f"{req.provider} {strategy}"),
                encoding="utf-8",
            )

        if "vote_full_row_1_2" in strategies:
            update_job(job_id, progress=0.87, active_strategy="vote_full_row_1_2")
            vote = build_vote_experiment_result(req, req.run_id, results)
            results["vote_full_row_1_2"] = vote
            write_json(exp_dir / f"vote_full_row_1_2_{req.provider}.json", vote)
            (exp_dir / f"render_vote_full_row_1_2_{req.provider}.html").write_text(
                render_grid_html(vote["cells"], f"{req.provider} vote"),
                encoding="utf-8",
            )

        summary = {
            "run_id": req.run_id,
            "provider": req.provider,
            "settings": req.settings.model_dump(),
            "strategies": strategies,
            "variants": summarize_experiment_variants(req.run_id, req.provider, results),
            "completed_at": time.time(),
        }
        write_json(exp_dir / f"summary_{req.provider}.json", summary)
        update_job(job_id, status="completed", progress=1, result=summary, active_strategy=None)
    except Exception as exc:
        update_job(job_id, status="failed", progress=1, error=str(exc))


def expand_experiment_strategies(strategies: list[ExperimentStrategy]) -> list[ExperimentStrategy]:
    expanded: list[ExperimentStrategy] = []
    for strategy in strategies:
        if strategy == "vote_full_row_1_2":
            for dependency in ("full_grid", "row_1", "row_2"):
                if dependency not in expanded:
                    expanded.append(dependency)  # type: ignore[arg-type]
        if strategy not in expanded:
            expanded.append(strategy)
    return expanded


def run_model_experiment_strategy(
    req: ExperimentRequest,
    run_dir: Path,
    exp_dir: Path,
    strategy: ExperimentStrategy,
    occupancy: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks = build_experiment_chunks(run_dir, exp_dir, strategy)
    all_cells: list[list[str]] = []
    raw_outputs: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for chunk in chunks:
        prompt = build_experiment_prompt(req.provider, strategy, chunk["row_start"], chunk["row_count"], req.user_conditions)
        if req.provider == "chandra":
            image_uri = image_to_data_uri(prepare_model_image(Image.open(chunk["path"]).convert("RGB")))
            raw = call_chandra(image_uri, prompt, req.settings)
        else:
            raw = call_gpt_responses(
                run_public_experiment_url(req.run_id, Path(chunk["path"]).name),
                prompt,
                req.settings,
            )
        parsed = parse_model_rows(raw)
        chunk_cells, chunk_validation = normalize_chunk_cells(parsed, chunk["row_count"], occupancy[chunk["row_start"] : chunk["row_start"] + chunk["row_count"]])
        all_cells.extend(chunk_cells)
        raw_outputs.append(
            {
                "row_start": chunk["row_start"],
                "row_count": chunk["row_count"],
                "image_url": f"/runs/{req.run_id}/experiments/{Path(chunk['path']).name}",
                "raw_output": raw,
                "validation": chunk_validation,
            }
        )
        prompts.append({"row_start": chunk["row_start"], "row_count": chunk["row_count"], "prompt": prompt})
    cells, validation = normalize_cells(all_cells, occupancy)
    return {
        "kind": "experiment",
        "provider": req.provider,
        "strategy": strategy,
        "settings": req.settings.model_dump(),
        "trace": build_experiment_trace(req, strategy, chunks, prompts),
        "raw_output": json.dumps(raw_outputs, ensure_ascii=False, indent=2),
        "rows": cells_to_text_rows(cells),
        "cells": cells,
        "validation": validation,
        "completed_at": time.time(),
    }


def build_experiment_chunks(run_dir: Path, exp_dir: Path, strategy: ExperimentStrategy) -> list[dict[str, Any]]:
    masked = Image.open(run_dir / "masked.png").convert("RGB")
    if strategy == "full_grid":
        path = exp_dir / "input_full_grid.png"
        masked.save(path)
        return [{"row_start": 0, "row_count": 20, "path": path}]

    rows_per_chunk = {"row_1": 1, "row_2": 2, "row_5": 5}.get(strategy)
    if rows_per_chunk is None:
        raise ValueError(f"Unsupported model experiment strategy: {strategy}")

    chunks: list[dict[str, Any]] = []
    for row_start in range(0, 20, rows_per_chunk):
        row_count = min(rows_per_chunk, 20 - row_start)
        y0 = int(masked.height * row_start / 20)
        y1 = int(masked.height * (row_start + row_count) / 20)
        pad = max(2, int(masked.height / 20 * 0.08))
        crop = masked.crop((0, max(0, y0 - pad), masked.width, min(masked.height, y1 + pad)))
        path = exp_dir / f"input_{strategy}_{row_start + 1:02d}_{row_start + row_count:02d}.png"
        crop.save(path)
        chunks.append({"row_start": row_start, "row_count": row_count, "path": path})
    return chunks


def normalize_chunk_cells(parsed: list[str] | list[list[str]], expected_rows: int, occupancy: list[dict[str, Any]]) -> tuple[list[list[str]], dict[str, Any]]:
    cells, validation = normalize_cells(parsed, occupancy)
    return cells[:expected_rows], validation


def build_experiment_prompt(
    provider: Literal["chandra", "gpt"],
    strategy: ExperimentStrategy,
    row_start: int,
    row_count: int,
    user_conditions: str,
) -> str:
    base = CHANDRA_OCR_PROMPT if provider == "chandra" else OCR_PROMPT
    if strategy == "full_grid":
        strategy_text = (
            "\nExperiment strategy: full_grid. The image contains the whole 20x20 grid. "
            "Transcribe every visible row and cell."
        )
    else:
        strategy_text = (
            f"\nExperiment strategy: {strategy}. The image is a horizontal crop containing only original grid "
            f"row {row_start + 1} through row {row_start + row_count}. "
            f"Return exactly {row_count} row(s), each with exactly 20 physical cells. "
            "Do not include missing rows above or below this crop. Preserve original cell positions within each visible row."
        )
    return apply_user_conditions(base.rstrip() + "\n" + strategy_text + "\n", user_conditions)


def build_experiment_trace(
    req: ExperimentRequest,
    strategy: ExperimentStrategy,
    chunks: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "experiment",
        "provider": req.provider,
        "strategy": strategy,
        "input": {
            "chunks": [
                {
                    "row_start": chunk["row_start"],
                    "row_count": chunk["row_count"],
                    "image": f"/runs/{req.run_id}/experiments/{Path(chunk['path']).name}",
                    "public_url": run_public_experiment_url(req.run_id, Path(chunk["path"]).name) if req.provider == "gpt" else None,
                }
                for chunk in chunks
            ],
            "user_conditions": req.user_conditions,
        },
        "prompt": "\n\n--- chunk prompt ---\n\n".join(item["prompt"] for item in prompts),
        "settings": req.settings.model_dump(),
        "request_shape": {
            "strategy": strategy,
            "provider": req.provider,
            "calls": len(chunks),
            "transport": "public image_url" if req.provider == "gpt" else "data URI from experiment chunk",
        },
    }


def build_vote_experiment_result(req: ExperimentRequest, run_id: str, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cells, vote_stats = vote_experiment_cells(results)
    validation = {
        "row_count": 20,
        "original_lengths": [20] * 20,
        "valid_original_shape": True,
        "normalized_shape": [20, 20],
        "aligned_to_occupancy": False,
        "vote_stats": vote_stats,
    }
    return {
        "kind": "experiment",
        "provider": req.provider,
        "strategy": "vote_full_row_1_2",
        "settings": req.settings.model_dump(),
        "trace": {
            "kind": "experiment_vote",
            "provider": req.provider,
            "strategy": "vote_full_row_1_2",
            "input": {
                "source_strategies": [key for key in ("full_grid", "row_1", "row_2") if key in results],
                "rule": "Per-cell majority wins; if there is no majority, prefer full_grid, then row_2, then row_1.",
                "user_conditions": req.user_conditions,
            },
            "prompt": "No model prompt. This is deterministic cell voting over completed experiment variants.",
            "settings": req.settings.model_dump(),
            "request_shape": {"strategy": "vote_full_row_1_2", "model_calls": 0},
        },
        "raw_output": json.dumps(vote_stats, ensure_ascii=False, indent=2),
        "rows": cells_to_text_rows(cells),
        "cells": cells,
        "validation": validation,
        "completed_at": time.time(),
    }


def vote_experiment_cells(results: dict[str, dict[str, Any]]) -> tuple[list[list[str]], dict[str, Any]]:
    order = ["full_grid", "row_2", "row_1"]
    matrices = {
        strategy: fit_cells_matrix(result.get("cells") or result.get("rows") or [])
        for strategy, result in results.items()
        if strategy in order
    }
    voted: list[list[str]] = []
    disagreements = 0
    majority_cells = 0
    fallback_cells = 0
    for row_index in range(20):
        row: list[str] = []
        for col_index in range(20):
            values = [normalize_cell_value(matrices[strategy][row_index][col_index]) for strategy in order if strategy in matrices]
            nonblank_values = [value for value in values if value.strip()]
            unique_values = set(values)
            if len(unique_values) > 1:
                disagreements += 1
            chosen = " "
            for value in nonblank_values:
                if nonblank_values.count(value) >= 2:
                    chosen = value
                    majority_cells += 1
                    break
            else:
                for strategy in order:
                    if strategy in matrices:
                        candidate = normalize_cell_value(matrices[strategy][row_index][col_index])
                        if candidate.strip():
                            chosen = candidate
                            fallback_cells += 1
                            break
            row.append(chosen)
        voted.append(row)
    return voted, {
        "source_strategies": [strategy for strategy in order if strategy in matrices],
        "disagreement_cells": disagreements,
        "majority_cells": majority_cells,
        "fallback_cells": fallback_cells,
    }


def summarize_experiment_variants(run_id: str, provider: str, results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for strategy, result in results.items():
        validation = result.get("validation") or {}
        variants.append(
            {
                "strategy": strategy,
                "provider": provider,
                "row_count": len(result.get("cells") or []),
                "valid_original_shape": validation.get("valid_original_shape"),
                "vote_stats": validation.get("vote_stats"),
                "result_path": f"/runs/{run_id}/experiments/{strategy}_{provider}.json",
                "render_url": f"/runs/{run_id}/experiments/render_{strategy}_{provider}.html",
                "completed_at": result.get("completed_at"),
            }
        )
    return variants


def run_public_experiment_url(run_id: str, filename: str) -> str:
    return f"{get_public_base_url()}/runs/{run_id}/experiments/{filename}"


def run_ocr_job(job_id: str, req: OcrRequest) -> None:
    update_job(job_id, status="running", progress=0.1)
    try:
        run_dir = ensure_run(req.run_id)
        image_uri = image_to_data_uri(prepare_model_image(Image.open(run_dir / "masked.png").convert("RGB")))
        update_job(job_id, progress=0.25)
        prompt = apply_user_conditions(CHANDRA_OCR_PROMPT if req.provider == "chandra" else OCR_PROMPT, req.user_conditions)
        if req.provider == "chandra":
            raw = call_chandra(image_uri, prompt, req.settings)
        else:
            raw = call_gpt_responses(run_public_url(req.run_id, "masked.png"), prompt, req.settings, job_id)
        update_job(job_id, progress=0.8)
        parsed = parse_model_rows(raw)
        occupancy = read_json(run_dir / "metadata.json").get("cell_occupancy", [])
        cells, validation = normalize_cells(parsed, occupancy)
        result = {
            "provider": req.provider,
            "settings": req.settings.model_dump(),
            "trace": build_ocr_trace(req.provider, req.run_id, prompt, req.settings, req.user_conditions),
            "raw_output": raw,
            "rows": cells_to_text_rows(cells),
            "cells": cells,
            "validation": validation,
            "completed_at": time.time(),
        }
        write_json(run_dir / f"ocr_{req.provider}.json", result)
        (run_dir / f"render_{req.provider}.html").write_text(
            render_grid_html(cells, f"{req.provider} OCR"),
            encoding="utf-8",
        )
        update_job(job_id, status="completed", progress=1, result=result)
    except Exception as exc:
        update_job(job_id, status="failed", progress=1, error=str(exc))


def run_repair_job(job_id: str, req: RepairRequest) -> None:
    update_job(job_id, status="running", progress=0.1)
    try:
        run_dir = ensure_run(req.run_id)
        ocr = read_json(run_dir / f"ocr_{req.source_provider}.json")
        rows_text = build_repair_context(run_dir, req.source_provider, ocr)
        prompt = build_repair_prompt(req.provider, rows_text, req.user_conditions)
        image_uri = image_to_data_uri(prepare_model_image(Image.open(run_dir / "masked.png").convert("RGB")))
        update_job(job_id, progress=0.3)
        if req.provider == "chandra":
            raw = call_chandra(image_uri, prompt, req.settings)
        else:
            raw = call_gpt_responses(run_public_url(req.run_id, "masked.png"), prompt, req.settings, job_id)
        update_job(job_id, progress=0.8)
        parsed = parse_model_rows(raw)
        occupancy = read_json(run_dir / "metadata.json").get("cell_occupancy", [])
        cells, validation = normalize_cells(parsed, occupancy)
        result = {
            "provider": req.provider,
            "source_provider": req.source_provider,
            "settings": req.settings.model_dump(),
            "trace": build_repair_trace(req.provider, req.source_provider, req.run_id, rows_text, prompt, req.settings, req.user_conditions),
            "raw_output": raw,
            "rows": cells_to_text_rows(cells),
            "cells": cells,
            "validation": validation,
            "completed_at": time.time(),
        }
        write_json(run_dir / "repair.json", result)
        (run_dir / "render_repair.html").write_text(render_grid_html(cells, "Repair 제안"), encoding="utf-8")
        update_job(job_id, status="completed", progress=1, result=result)
    except Exception as exc:
        update_job(job_id, status="failed", progress=1, error=str(exc))


def apply_user_conditions(prompt: str, user_conditions: str | None) -> str:
    conditions = (user_conditions or "").strip()
    if not conditions:
        return prompt
    proper_nouns = extract_expected_proper_nouns(conditions)
    full_prompt = (
        prompt.rstrip()
        + "\n\nAdditional user conditions for this run:\n"
        + conditions
        + "\n"
    )
    if proper_nouns:
        full_prompt += (
            "\nExpected proper nouns/name spellings inferred from the conditions:\n"
            + "\n".join(f"- {name}" for name in proper_nouns)
            + "\n\nProper-noun correction rule:\n"
            "- For the expected names above only, normalize visually uncertain OCR to the exact spelling above when the handwriting clearly refers to that name.\n"
            "- Preserve physical 20x20 grid cell boundaries while doing this. If the name is written one syllable per square, keep one syllable per JSON cell.\n"
            "- Do not use this rule to correct ordinary words, grammar, endings, particles, spacing, factual mistakes, or sentence naturalness.\n"
        )
    return full_prompt


def extract_expected_proper_nouns(user_conditions: str | None) -> list[str]:
    text = unicodedata.normalize("NFC", user_conditions or "")
    candidates: list[str] = []

    quote_pattern = r"[\"'“‘`「『]([가-힣A-Za-z][가-힣A-Za-z0-9]{1,12})[\"'”’`」』]"
    candidates.extend(re.findall(quote_pattern, text))

    contexts = "|".join(sorted(PROPER_NOUN_CONTEXT_NOUNS))
    candidates.extend(re.findall(rf"([가-힣]{{2,8}})의\s+(?:{contexts})", text))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = strip_name_particle(candidate)
        if not (2 <= len(value) <= 8):
            continue
        if value in PROPER_NOUN_STOPWORDS:
            continue
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def strip_name_particle(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "").strip()
    for particle in ("에게서", "에게", "으로", "로서", "로써", "라는", "이란", "와", "과", "은", "는", "이", "가", "을", "를", "의", "도", "만", "로", "에"):
        if len(text) > len(particle) + 1 and text.endswith(particle):
            return text[: -len(particle)]
    return text


def build_repair_prompt(provider: Literal["gpt", "chandra"], rows_text: str, user_conditions: str | None = None) -> str:
    template = CHANDRA_REPAIR_PROMPT if provider == "chandra" else REPAIR_PROMPT
    return apply_user_conditions(template.replace("{rows}", rows_text), user_conditions)


def build_repair_context(run_dir: Path, source_provider: str, source_ocr: dict[str, Any]) -> str:
    context: dict[str, Any] = {
        "primary_provider": source_provider,
        "primary_cells": source_ocr.get("cells") or source_ocr.get("rows") or [],
    }
    alternates: dict[str, Any] = {}
    for provider in ("chandra", "gpt"):
        if provider == source_provider:
            continue
        path = run_dir / f"ocr_{provider}.json"
        if path.exists():
            try:
                data = read_json(path)
            except Exception:
                continue
            alternates[provider] = data.get("cells") or data.get("rows") or []
    if alternates:
        context["alternate_ocr_candidates"] = alternates
        context["instruction"] = (
            "Use alternate_ocr_candidates only as hints when they match the image better; "
            "the image is the source of truth. Do not prefer a candidate merely because it is "
            "more grammatical or natural; preserve visible student mistakes exactly."
        )
    return json.dumps(context, ensure_ascii=False)


def build_image_trace(provider: Literal["gpt", "chandra"], run_id: str) -> dict[str, Any]:
    trace = {
        "kind": "red-masked crop",
        "local_url": f"/runs/{run_id}/masked.png",
        "crop_url": f"/runs/{run_id}/crop.png",
        "transport": "public image_url" if provider == "gpt" else "data URI from masked crop",
    }
    if provider == "gpt":
        trace["public_url"] = run_public_url(run_id, "masked.png")
    return trace


def build_request_shape(provider: Literal["gpt", "chandra"], prompt: str, settings: ModelSettings | dict[str, Any], run_id: str) -> dict[str, Any]:
    settings_dict = settings.model_dump() if isinstance(settings, ModelSettings) else dict(settings or {})
    if provider == "gpt":
        return {
            "method": "POST",
            "path": "/v1/responses",
            "body": {
                "model": GPT_OAUTH_DEFAULT_MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": run_public_url(run_id, "masked.png"), "detail": "high"},
                        ],
                    }
                ],
                "reasoning_effort": normalize_gpt_reasoning_effort(settings_dict.get("reasoning_effort")),
                "temperature": settings_dict.get("temperature"),
                "top_p": settings_dict.get("top_p"),
                "max_tokens": settings_dict.get("max_tokens"),
            },
        }
    return {
        "method": "POST",
        "path": "/chat/completions",
        "body": {
            "model": settings_dict.get("model") or VLM_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": settings_dict.get("temperature"),
            "top_p": settings_dict.get("top_p"),
            "max_tokens": settings_dict.get("max_tokens"),
            "n": settings_dict.get("n"),
        },
    }


def build_ocr_trace(
    provider: Literal["gpt", "chandra"],
    run_id: str,
    prompt: str,
    settings: ModelSettings | dict[str, Any],
    user_conditions: str | None = None,
) -> dict[str, Any]:
    settings_dict = settings.model_dump() if isinstance(settings, ModelSettings) else dict(settings or {})
    return {
        "kind": "ocr",
        "provider": provider,
        "input": {
            "image": build_image_trace(provider, run_id),
            "user_conditions": user_conditions or "",
        },
        "prompt": prompt,
        "settings": settings_dict,
        "request_shape": build_request_shape(provider, prompt, settings_dict, run_id),
    }


def build_repair_trace(
    provider: Literal["gpt", "chandra"],
    source_provider: Literal["gpt", "chandra"],
    run_id: str,
    rows_text: str,
    prompt: str,
    settings: ModelSettings | dict[str, Any],
    user_conditions: str | None = None,
) -> dict[str, Any]:
    settings_dict = settings.model_dump() if isinstance(settings, ModelSettings) else dict(settings or {})
    try:
        source_cells = json.loads(rows_text)
    except json.JSONDecodeError:
        source_cells = rows_text
    return {
        "kind": "repair",
        "provider": provider,
        "source_provider": source_provider,
        "input": {
            "image": build_image_trace(provider, run_id),
            "source_provider": source_provider,
            "source_cells": source_cells,
            "source_cells_json": rows_text,
            "user_conditions": user_conditions or "",
        },
        "prompt": prompt,
        "settings": settings_dict,
        "request_shape": build_request_shape(provider, prompt, settings_dict, run_id),
    }


def hydrate_result_trace(result: dict[str, Any], run_id: str, source: str, run_dir: Path) -> dict[str, Any]:
    if result.get("trace"):
        return result
    provider = result.get("provider") or ("gpt" if source == "gpt" else "chandra")
    settings = result.get("settings") or {}
    if source == "repair":
        source_provider = result.get("source_provider") or "gpt"
        source_path = run_dir / f"ocr_{source_provider}.json"
        source_cells: Any = []
        if source_path.exists():
            source_data = read_json(source_path)
            source_cells = source_data.get("cells") or source_data.get("rows") or []
        rows_text = json.dumps(source_cells, ensure_ascii=False)
        prompt = build_repair_prompt(provider, rows_text)
        result["trace"] = build_repair_trace(provider, source_provider, run_id, rows_text, prompt, settings)
    else:
        prompt = CHANDRA_OCR_PROMPT if provider == "chandra" else OCR_PROMPT
        result["trace"] = build_ocr_trace(provider, run_id, prompt, settings)
    return result


def call_chandra(image_uri: str, prompt: str, settings: ModelSettings) -> str:
    body = {
        "model": settings.model or VLM_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "n": settings.n,
        "presence_penalty": 2,
        "extra_body": {"top_k": 40},
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"}
    errors: list[str] = []
    for base_url in [VLM_PRIMARY_BASE_URL, VLM_FALLBACK_BASE_URL]:
        if not base_url:
            continue
        try:
            resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=900)
            if resp.status_code >= 400:
                errors.append(f"{base_url}: HTTP {resp.status_code} {resp.text[:300]}")
                continue
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"{base_url}: no choices in response")
            return choices[0].get("message", {}).get("content", "")
        except Exception as exc:
            errors.append(f"{base_url}: {exc}")
    raise RuntimeError("; ".join(errors) or "No Chandra endpoint configured")


def normalize_gpt_reasoning_effort(value: str | None) -> str:
    effort = (value or "low").strip().lower()
    return effort if effort in GPT_REASONING_EFFORTS else "low"


def call_gpt_responses(image_url: str, prompt: str, settings: ModelSettings, job_id: str | None = None) -> str:
    warning = gpt_image_url_warning(get_public_base_url())
    if warning:
        raise RuntimeError(
            f"{warning} The GPT OAuth /v1/responses path works for text-only requests, "
            "and now preserves image inputs, but the image URL still must be fetchable by the upstream model. "
            "Use a public HTTPS PUBLIC_BASE_URL."
        )
    headers = {"Authorization": f"Bearer {DOGOK_PROXY_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GPT_OAUTH_DEFAULT_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            }
        ],
        "reasoning_effort": normalize_gpt_reasoning_effort(settings.reasoning_effort),
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
    }
    created = requests.post(f"{GPT_OAUTH_BASE_URL}/v1/responses", headers=headers, json=body, timeout=30)
    if created.status_code >= 400:
        raise RuntimeError(f"GPT OAuth responses create failed: HTTP {created.status_code} {created.text[:500]}")
    response_id = created.json()["id"]
    if job_id:
        update_job(job_id, progress=0.35, remote_response_id=response_id, remote_status="queued")
    deadline = time.time() + 900
    poll_started_at = time.time()
    while time.time() < deadline:
        polled = requests.get(f"{GPT_OAUTH_BASE_URL}/v1/responses/{response_id}", headers=headers, timeout=30)
        if polled.status_code >= 400:
            raise RuntimeError(f"GPT OAuth responses poll failed: HTTP {polled.status_code} {polled.text[:500]}")
        data = polled.json()
        status = data.get("status")
        if job_id:
            elapsed = time.time() - poll_started_at
            progress = min(0.75, 0.35 + (elapsed / 300) * 0.4)
            update_job(job_id, progress=progress, remote_response_id=response_id, remote_status=status or "unknown")
        if status == "completed":
            return data.get("output_text") or extract_output_text(data)
        if status in {"failed", "cancelled"}:
            error_text = json.dumps(data.get("error"), ensure_ascii=False)
            if "407" in error_text or "downloading file" in error_text:
                error_text += (
                    " | GPT OAuth could not download the crop image URL. "
                    "Set PUBLIC_BASE_URL to a URL the GPT upstream can fetch."
                )
            raise RuntimeError(f"GPT OAuth response {status}: {error_text}")
        time.sleep(2)
    raise RuntimeError("GPT OAuth polling timed out")


def extract_output_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def parse_model_rows(raw: str) -> list[str] | list[list[str]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    html_rows = html_to_text_rows(text)
    if html_rows:
        return html_rows
    try:
        parsed = json.loads(text)
        if is_bbox_only_layout(parsed):
            raise ValueError("Model returned layout bounding boxes without OCR text content")
        collected = collect_rows(parsed)
        if collected:
            return collected
    except json.JSONDecodeError:
        pass
    positioned_rows = collect_position_fields(text)
    if positioned_rows:
        return positioned_rows
    text_rows = collect_text_fields(text)
    if text_rows:
        return text_rows
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if is_bbox_only_layout(data):
                raise ValueError("Model returned layout bounding boxes without OCR text content")
            collected = collect_rows(data)
            if collected:
                return collected
        except json.JSONDecodeError:
            pass
    return [line for line in text.splitlines() if line.strip()]


def html_to_text_rows(text: str) -> list[str]:
    if not re.search(r"</?(div|p|br|span|pre|table|tr|td)\b", text, re.I):
        return []
    normalized = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    normalized = re.sub(r"</(p|div|tr|pre)>", "\n", normalized, flags=re.I)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = html.unescape(normalized)
    rows = [collapse_soft_spaces(line).strip() for line in normalized.splitlines()]
    return [row for row in rows if row]


def collapse_soft_spaces(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text)


def is_bbox_only_layout(value: Any) -> bool:
    if isinstance(value, list) and value:
        return all(is_bbox_only_layout(item) for item in value)
    if not isinstance(value, dict):
        return False
    has_bbox = "bbox" in value or "data-bbox" in value
    has_text = any(str(value.get(key) or "").strip() for key in ("text", "content", "html", "markdown"))
    return has_bbox and not has_text


def collect_text_fields(text: str) -> list[str]:
    rows: list[str] = []
    for match in re.finditer(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', text):
        try:
            rows.append(json.loads(f'"{match.group(1)}"'))
        except json.JSONDecodeError:
            rows.append(match.group(1))
    return rows


def collect_position_fields(text: str) -> list[str]:
    items: list[dict[str, Any]] = []
    pattern = re.compile(
        r'"x"\s*:\s*(-?\d+)\s*,\s*"y"\s*:\s*(-?\d+)\s*,\s*"text"\s*:\s*"((?:\\.|[^"\\])*)"',
        re.S,
    )
    for match in pattern.finditer(text):
        try:
            value = json.loads(f'"{match.group(3)}"')
        except json.JSONDecodeError:
            value = match.group(3)
        items.append({"x": int(match.group(1)), "y": int(match.group(2)), "text": value})
    return rows_from_position_items(items)


def collect_rows(value: Any) -> list[Any]:
    if isinstance(value, dict):
        for key in ("cells", "rows", "matrix", "grid"):
            if key in value:
                return collect_rows(value[key])
        for key in ("text", "row", "content"):
            if isinstance(value.get(key), str):
                return [value[key]]
        return []
    if isinstance(value, list):
        positioned = rows_from_position_items(value)
        if positioned:
            return positioned
        if all(not isinstance(item, (dict, list)) for item in value):
            return [value]
        if all(
            isinstance(item, list) and all(not isinstance(cell, (dict, list)) for cell in item)
            for item in value
        ):
            return value
        rows: list[Any] = []
        for item in value:
            rows.extend(collect_rows(item))
        return rows
    if isinstance(value, str):
        return [value]
    return []


def rows_from_position_items(items: list[Any]) -> list[str]:
    positioned = [
        item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("y") is not None
    ]
    if len(positioned) < 2:
        return []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in positioned:
        try:
            y = int(item.get("y", 0))
        except (TypeError, ValueError):
            y = 0
        grouped.setdefault(y, []).append(item)
    rows: list[str] = []
    for y in sorted(grouped):
        pieces = sorted(grouped[y], key=lambda item: safe_int(item.get("x"), 0))
        compact = "".join(compact_ocr_segment(piece.get("text", "")) for piece in pieces)
        rows.append(compact)
    return rows


def compact_ocr_segment(value: str) -> str:
    # Position-style VLM output often inserts spaces between syllables inside a
    # chunk. Those are not reliable cell blanks, so occupancy alignment handles
    # leading blanks later.
    return "".join(str(value).split())


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_rows(value: list[Any]) -> tuple[list[str], dict[str, Any]]:
    cells, validation = normalize_cells(value)
    rows = cells_to_legacy_rows(cells)
    validation["original_lengths"] = [len(list(row)) for row in rows]
    validation["valid_original_shape"] = len(value) == 20 and all(length == 20 for length in validation["original_lengths"][:20])
    return rows, validation


def normalize_cells(value: Any, occupancy: list[dict[str, Any]] | None = None) -> tuple[list[list[str]], dict[str, Any]]:
    source_rows = value[:20] if isinstance(value, list) else []
    cells: list[list[str]] = []
    original_lengths: list[int] = []
    for raw_row in source_rows:
        row_cells = row_value_to_cells(raw_row)
        original_lengths.append(len(row_cells))
        cells.append(fit_cells(row_cells))
    while len(cells) < 20:
        cells.append([" "] * 20)
        original_lengths.append(0)
    cells = cells[:20]
    aligned = False
    if occupancy:
        cells = align_cells_to_occupancy(cells, occupancy)
        aligned = True
    validation = {
        "row_count": len(value) if isinstance(value, list) else 0,
        "original_lengths": original_lengths[:20],
        "valid_original_shape": len(value) == 20 and all(length == 20 for length in original_lengths[:20]),
        "normalized_shape": [20, 20],
        "aligned_to_occupancy": aligned,
        "cell_rules": {
            "max_visible_chars_per_cell": 2,
            "mixed_digit_hangul_cell_allowed": True,
            "row_final_period_pair": True,
            "row_final_punctuation_pair": True,
        },
    }
    return cells, validation


def row_value_to_cells(raw_row: Any) -> list[str]:
    if isinstance(raw_row, list):
        return [normalize_cell_value(cell) for cell in raw_row]
    return pack_text_units(str(raw_row), preserve_spaces=True)


def fit_cells(cells: list[str], width: int = 20) -> list[str]:
    normalized = [normalize_cell_value(cell) for cell in cells]
    normalized = expand_inline_punctuation_cells(normalized)
    normalized = squeeze_punctuation_for_capacity(normalized, width)
    return (normalized + [" "] * width)[:width]


def normalize_cell_value(value: Any) -> str:
    text = str(value)
    if not text.strip():
        return " "
    return "".join(list(text.strip())[:2])


def pack_text_units(text: str, preserve_spaces: bool) -> list[str]:
    chars = list(text)
    units: list[str] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        if char.isspace():
            if preserve_spaces:
                units.append(" ")
            index += 1
            continue
        units.append(char)
        index += 1
    return units


def expand_inline_punctuation_cells(units: list[str]) -> list[str]:
    expanded: list[str] = []
    for unit in units:
        cell = normalize_cell_value(unit)
        chars = list(cell.strip())
        if len(chars) == 2 and chars[1] in TRAILING_CELL_PUNCTUATION:
            expanded.append(chars[0])
            expanded.append(chars[1])
        else:
            expanded.append(cell)
    return expanded


def expand_cells_for_occupancy(units: list[str]) -> list[str]:
    expanded: list[str] = []
    for unit in units:
        cell = normalize_cell_value(unit)
        chars = list(cell.strip())
        if len(chars) > 1 and all(char.isdigit() for char in chars):
            expanded.append(cell)
            continue
        if len(chars) > 1:
            expanded.extend(chars)
        else:
            expanded.append(cell)
    return expanded


def squeeze_punctuation_for_capacity(units: list[str], capacity: int) -> list[str]:
    if capacity <= 0:
        return []
    squeezed = [normalize_cell_value(unit) for unit in units]
    while len(squeezed) > capacity:
        punctuation_blank_index = None
        for index in range(len(squeezed) - 1, 0, -1):
            if not squeezed[index].strip() and squeezed[index - 1] in TRAILING_CELL_PUNCTUATION:
                punctuation_blank_index = index
                break
        if punctuation_blank_index is not None:
            del squeezed[punctuation_blank_index]
            continue
        if not squeezed[-1].strip():
            squeezed.pop()
            continue
        punctuation_index = None
        for index in range(len(squeezed) - 1, 0, -1):
            if squeezed[index] in TRAILING_CELL_PUNCTUATION and len(list(squeezed[index - 1])) < 2:
                punctuation_index = index
                break
        if punctuation_index is None:
            break
        squeezed[punctuation_index - 1] = normalize_cell_value(
            squeezed[punctuation_index - 1] + squeezed[punctuation_index]
        )
        del squeezed[punctuation_index]
    return squeezed


def cells_to_text_rows(cells: list[list[str]]) -> list[str]:
    return ["".join(fit_cells(row)) for row in fit_cells_matrix(cells)]


def cells_to_legacy_rows(cells: list[list[str]]) -> list[str]:
    rows: list[str] = []
    for row in fit_cells_matrix(cells):
        legacy_chars = [(cell if len(list(cell)) == 1 else cell[0]) for cell in row]
        rows.append("".join(legacy_chars))
    return rows


def fit_cells_matrix(cells: list[list[str]]) -> list[list[str]]:
    matrix = [fit_cells(row) for row in cells[:20]]
    while len(matrix) < 20:
        matrix.append([" "] * 20)
    return matrix


def cells_from_source(run_dir: Path, source: str) -> tuple[list[list[str]], dict[str, Any]]:
    if source == "repair":
        data = read_json(run_dir / "repair.json")
    else:
        data = read_json(run_dir / f"ocr_{source}.json")
    return normalize_cells(data.get("cells") or data.get("rows") or [])


def render_grid_html(rows: list[str] | list[list[str]], title: str) -> str:
    matrix, _ = normalize_cells(rows)
    cells = []
    for row in matrix:
        for cell_text in row:
            classes = "cell multi" if len(list(cell_text.strip())) > 1 else "cell"
            cells.append(f'<span class="{classes}">{html.escape(cell_text)}</span>')
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --cell: 24px; --paper-bg: #fdfcf0; --grid-line: #9dccd3; --ink: #252525; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; background: #fff; font-family: Apple SD Gothic Neo, Noto Sans KR, Malgun Gothic, sans-serif; }}
    .paper {{ width: calc(var(--cell) * 20); height: calc(var(--cell) * 20); background-color: var(--paper-bg);
      background-image: linear-gradient(var(--grid-line) 1px, transparent 1px), linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
      background-size: var(--cell) var(--cell); border: 1px solid #9dccd3; overflow: hidden; }}
    .paper-text {{ display: grid; grid-template-columns: repeat(20, var(--cell)); grid-template-rows: repeat(20, var(--cell)); color: var(--ink); font-size: 16px; font-weight: 500; }}
    .cell {{ display: flex; align-items: center; justify-content: center; width: var(--cell); height: var(--cell); white-space: pre; }}
    .cell.multi {{ font-size: 12px; }}
  </style>
</head>
<body>
  <main class="paper"><div class="paper-text">{''.join(cells)}</div></main>
</body>
</html>
"""


def prepare_model_image(img: Image.Image) -> Image.Image:
    padded = Image.new("RGB", (img.width + 40, img.height + 40), "white")
    padded.paste(img, (20, 20))
    if padded.width < 2600:
        scale = 2600 / padded.width
        padded = padded.resize((int(padded.width * scale), int(padded.height * scale)), Image.Resampling.LANCZOS)
    max_pixels = 3072 * 2048
    pixels = padded.width * padded.height
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        padded = padded.resize((int(padded.width * scale), int(padded.height * scale)), Image.Resampling.LANCZOS)
    return padded


def image_to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ensure_run(run_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", run_id):
        raise HTTPException(status_code=404, detail="run not found")
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return run_dir


def run_payload(run_id: str) -> dict[str, Any]:
    run_dir = ensure_run(run_id)
    payload = read_json(run_dir / "metadata.json")
    payload["crop_url"] = f"/runs/{run_id}/crop.png"
    payload["masked_url"] = f"/runs/{run_id}/masked.png"
    payload["ocr"] = {}
    for provider in ("chandra", "gpt"):
        result_path = run_dir / f"ocr_{provider}.json"
        if result_path.exists():
            payload["ocr"][provider] = hydrate_result_trace(read_json(result_path), run_id, provider, run_dir)
            payload["ocr"][provider]["render_url"] = f"/runs/{run_id}/render_{provider}.html"
    if (run_dir / "repair.json").exists():
        payload["repair"] = hydrate_result_trace(read_json(run_dir / "repair.json"), run_id, "repair", run_dir)
        payload["repair"]["render_url"] = f"/runs/{run_id}/render_repair.html"
    payload["experiments"] = load_experiment_payload(run_id, run_dir)
    if (run_dir / "final.json").exists():
        payload["final"] = read_json(run_dir / "final.json")
        payload["final"]["render_url"] = f"/runs/{run_id}/final.html"
    return payload


def load_experiment_payload(run_id: str, run_dir: Path) -> dict[str, Any]:
    exp_dir = run_dir / "experiments"
    payload: dict[str, Any] = {}
    if not exp_dir.exists():
        return payload
    for summary_path in sorted(exp_dir.glob("summary_*.json")):
        provider = summary_path.stem.replace("summary_", "", 1)
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        variants: list[dict[str, Any]] = []
        for item in summary.get("variants") or []:
            strategy = item.get("strategy")
            result_path = exp_dir / f"{strategy}_{provider}.json"
            if not strategy or not result_path.exists():
                continue
            result = read_json(result_path)
            result["render_url"] = f"/runs/{run_id}/experiments/render_{strategy}_{provider}.html"
            result["result_url"] = f"/runs/{run_id}/experiments/{strategy}_{provider}.json"
            variants.append(result)
        summary["variants"] = variants
        payload[provider] = summary
    return payload


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
