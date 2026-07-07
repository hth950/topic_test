from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "work" / "model_batch_report.json"


def main() -> None:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    images = requests.get(f"{args.base_url}/api/images", timeout=20).json()["images"]
    selected = select_images(images, args)
    report = {
        "started_at": time.time(),
        "base_url": args.base_url,
        "provider": args.provider,
        "repair_provider": args.repair_provider,
        "source_provider": args.source_provider,
        "reasoning_effort": args.reasoning_effort,
        "items": [],
    }

    for image in selected:
        item = run_image(image, args)
        report["items"].append(item)
        write_report(report)
        if item.get("fatal"):
            break
    report["completed_at"] = time.time()
    write_report(report)
    print(json.dumps(summarize(report), ensure_ascii=False, indent=2))
    print(f"Report: {REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR/repair jobs through the local FastAPI app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    parser.add_argument("--provider", choices=["chandra", "gpt", "none"], default="chandra")
    parser.add_argument("--repair-provider", choices=["chandra", "gpt", "none"], default="none")
    parser.add_argument("--source-provider", choices=["chandra", "gpt"], default="gpt")
    parser.add_argument("--image-contains", default="", help="Only process images whose filename contains this text.")
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-latest-crop", action="store_true")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--job-timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=int, default=5)
    return parser.parse_args()


def select_images(images: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [image for image in images if not args.image_contains or args.image_contains in image["name"]]
    if args.max_images > 0:
        selected = selected[: args.max_images]
    return selected


def run_image(image: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    print(f"== {image['name']}")
    item: dict[str, Any] = {
        "image_id": image["id"],
        "image_name": image["name"],
        "started_at": time.time(),
        "jobs": [],
    }
    try:
        run = get_or_create_run(image, args)
        item["run_id"] = run["run_id"]
        if args.provider != "none":
            if args.skip_existing and run_has_result(args.base_url, run["run_id"], f"ocr.{args.provider}"):
                item["jobs"].append({"kind": "ocr", "provider": args.provider, "status": "skipped_existing"})
            else:
                item["jobs"].append(run_job(args, "/api/ocr", {"run_id": run["run_id"], "provider": args.provider}))
                run = requests.get(f"{args.base_url}/api/runs/{run['run_id']}", timeout=20).json()
        if args.repair_provider != "none":
            source_provider = args.source_provider
            if args.provider in {"chandra", "gpt"}:
                source_provider = args.provider
            if not run_has_result(args.base_url, run["run_id"], f"ocr.{source_provider}"):
                item["jobs"].append(
                    {
                        "kind": "repair",
                        "provider": args.repair_provider,
                        "status": "skipped_missing_source",
                        "source_provider": source_provider,
                    }
                )
            elif args.skip_existing and run_has_result(args.base_url, run["run_id"], "repair"):
                item["jobs"].append({"kind": "repair", "provider": args.repair_provider, "status": "skipped_existing"})
            else:
                item["jobs"].append(
                    run_job(
                        args,
                        "/api/repair",
                        {
                            "run_id": run["run_id"],
                            "source_provider": source_provider,
                            "provider": args.repair_provider,
                        },
                    )
                )
    except Exception as exc:
        item["fatal"] = str(exc)
        print("ERROR", exc)
    item["completed_at"] = time.time()
    return item


def get_or_create_run(image: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.reuse_latest_crop:
        latest = latest_run_for_image(args.base_url, image["id"])
        if latest:
            print("reuse", latest["run_id"])
            return requests.get(f"{args.base_url}/api/runs/{latest['run_id']}", timeout=20).json()
    run = requests.post(f"{args.base_url}/api/crop", json={"image_id": image["id"]}, timeout=60).json()
    print("crop", run["run_id"])
    return run


def latest_run_for_image(base_url: str, image_id: str) -> dict[str, Any] | None:
    runs = requests.get(f"{base_url}/api/runs", timeout=20).json().get("runs", [])
    matching = [run for run in runs if run.get("image_id") == image_id]
    return matching[0] if matching else None


def run_job(args: argparse.Namespace, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["settings"] = {
        "model": "gpt-5.5" if payload.get("provider") == "gpt" else "chandra",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "n": 1,
    }
    created = requests.post(f"{args.base_url}{path}", json=payload, timeout=30).json()
    job_id = created["id"]
    print(path, job_id)
    started = time.time()
    last_progress = None
    while True:
        job = requests.get(f"{args.base_url}/api/jobs/{job_id}", timeout=30).json()
        progress = round(float(job.get("progress") or 0), 3)
        if progress != last_progress:
            print(" ", job_id, job.get("status"), progress)
            last_progress = progress
        if job.get("status") in {"completed", "failed", "cancelled"}:
            job["elapsed_seconds"] = round(time.time() - started, 2)
            return compact_job(job)
        if time.time() - started > args.job_timeout:
            return {
                "id": job_id,
                "kind": job.get("kind"),
                "status": "timeout",
                "progress": job.get("progress"),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        time.sleep(args.poll_interval)


def compact_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    result = {
        "id": job.get("id"),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "progress": job.get("progress"),
        "elapsed_seconds": job.get("elapsed_seconds"),
    }
    if payload:
        result["provider"] = payload.get("provider")
        result["source_provider"] = payload.get("source_provider")
        result["run_id"] = payload.get("run_id")
    if job.get("error"):
        result["error"] = job.get("error")
    return result


def run_has_result(base_url: str, run_id: str, dotted: str) -> bool:
    run = requests.get(f"{base_url}/api/runs/{run_id}", timeout=20).json()
    value: Any = run
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return bool(value)


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    jobs = [job for item in report["items"] for job in item.get("jobs", [])]
    return {
        "images": len(report["items"]),
        "jobs": len(jobs),
        "completed": sum(1 for job in jobs if job.get("status") == "completed"),
        "failed": sum(1 for job in jobs if job.get("status") == "failed"),
        "timeout": sum(1 for job in jobs if job.get("status") == "timeout"),
        "skipped": sum(1 for job in jobs if str(job.get("status", "")).startswith("skipped")),
    }


if __name__ == "__main__":
    main()
