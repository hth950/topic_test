from __future__ import annotations

from pathlib import Path
import pytest

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import server


client = TestClient(server.app)


def sample_folder() -> server.ImageFolder:
    return next(folder for folder in server.list_image_folders() if folder.name == "samples_synthetic")


def sample_image(name: str = "clean_grid.png") -> server.ImageRecord:
    folder = sample_folder()
    return next(image for image in server.list_images(folder.id) if image.name == name)


def test_images_are_discoverable() -> None:
    response = client.get("/api/images")
    assert response.status_code == 200
    images = response.json()["images"]
    assert len(images) >= 3
    assert images[0]["width"] > 0
    assert images[0]["height"] > 0


def test_image_folders_are_discoverable() -> None:
    response = client.get("/api/image-folders")
    assert response.status_code == 200

    data = response.json()
    folders = data["folders"]
    folder_ids = {folder["id"] for folder in folders}

    assert data["default_folder_id"] in folder_ids
    assert any(folder["name"] == "samples_synthetic" for folder in folders)
    assert all(folder["image_count"] > 0 for folder in folders)


def test_images_can_be_filtered_by_folder() -> None:
    folders = client.get("/api/image-folders").json()["folders"]
    folder = folders[-1]

    response = client.get("/api/images", params={"folder_id": folder["id"]})
    assert response.status_code == 200

    data = response.json()
    assert data["folder"]["id"] == folder["id"]
    assert len(data["images"]) == folder["image_count"]
    assert {image["folder_id"] for image in data["images"]} == {folder["id"]}


def test_unknown_image_folder_returns_404() -> None:
    response = client.get("/api/images", params={"folder_id": "missing-folder"})
    assert response.status_code == 404


def test_jobs_are_listed_without_secrets() -> None:
    job_id = server.create_job("ocr", {"run_id": "abc123abc123", "provider": "gpt"})
    try:
        response = client.get("/api/jobs")
        assert response.status_code == 200
        jobs = response.json()["jobs"]
        job = next(item for item in jobs if item["id"] == job_id)
        assert job["payload"]["provider"] == "gpt"
        assert "api_key" not in job
    finally:
        with server.jobs_lock:
            server.jobs.pop(job_id, None)


def test_detect_grid_bbox_on_sample_image() -> None:
    image = sample_image()
    img = Image.open(image.path).convert("RGB")
    bbox = server.detect_grid_bbox(img)
    assert 65 <= bbox["x"] <= 95
    assert 665 <= bbox["y"] <= 705
    assert 850 <= bbox["width"] <= 910
    assert 850 <= bbox["height"] <= 910


def test_red_mask_removes_red_and_keeps_gray() -> None:
    img = Image.new("RGB", (80, 40), "white")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "가", fill=(90, 90, 90))
    draw.line((5, 30, 70, 30), fill=(230, 30, 40), width=4)
    masked = server.remove_red_marks(img)
    assert masked.getpixel((30, 30)) == (255, 255, 255)
    assert masked.getpixel((8, 10)) != (255, 255, 255)


def test_rows_are_normalized_to_20_by_20() -> None:
    rows, validation = server.normalize_rows(["안녕", list("abcdefghijklmnopqrstuv")])
    assert len(rows) == 20
    assert all(len(row) == 20 for row in rows)
    assert rows[0].startswith("안녕")
    assert rows[1] == "abcdefghijklmnopqrst"
    assert validation["normalized_shape"] == [20, 20]


def test_align_rows_to_occupancy_restores_leading_blank_cell() -> None:
    rows = ["현생 인류".ljust(20)] + [" " * 20 for _ in range(19)]
    occupancy = [{"first_col": 1, "last_col": 5}] + [{"first_col": None, "last_col": None} for _ in range(19)]
    aligned = server.align_rows_to_occupancy(rows, occupancy)
    assert aligned[0][0] == " "
    assert aligned[0][1:].startswith("현생 인류")


def test_cell_rules_preserve_mixed_cells_and_row_final_period() -> None:
    cells, validation = server.normalize_cells([["20", "0만", "년"]])
    assert cells[0][:3] == ["20", "0만", "년"]
    overflow, _ = server.normalize_cells(["가" * 18 + "했다."])
    assert overflow[0][19] == "다."
    assert validation["cell_rules"]["mixed_digit_hangul_cell_allowed"] is True
    assert validation["cell_rules"]["max_visible_chars_per_cell"] == 2


def test_prompt_does_not_force_numeric_tokenization() -> None:
    assert '["10","0"]' not in server.OCR_PROMPT
    assert '["20","0만"]' in server.OCR_PROMPT
    assert '"0만"' in server.REPAIR_PROMPT
    assert '["으","며",","]' in server.OCR_PROMPT


def test_prompts_require_literal_student_mistakes() -> None:
    for prompt in (server.OCR_PROMPT, server.CHANDRA_OCR_PROMPT, server.REPAIR_PROMPT, server.CHANDRA_REPAIR_PROMPT):
        assert "literal" in prompt
        assert "Never correct grammar" in prompt or "Do not fix student grammar" in prompt
        assert "있다그" in prompt
        assert "있다고" in prompt


def test_user_conditions_are_appended_to_prompts_and_trace() -> None:
    prompt = server.apply_user_conditions(server.OCR_PROMPT, "첫 칸이 비면 공백 유지")
    assert "Additional user conditions" in prompt
    assert "첫 칸이 비면 공백 유지" in prompt

    repair_prompt = server.build_repair_prompt("gpt", '{"cells":[]}', "쉼표는 보이는 칸 기준")
    assert "쉼표는 보이는 칸 기준" in repair_prompt

    trace = server.build_ocr_trace("gpt", "abc123abc123", prompt, {}, "조건 테스트")
    assert trace["input"]["user_conditions"] == "조건 테스트"


def test_proper_nouns_are_inferred_from_user_conditions() -> None:
    conditions = (
        "1. 곤이의 행동이 갖는 부정적 측면을 밝히되 그 행동의 긍정적 의미에 주안점을 두어 서술할 것\n"
        "2. 앞의 내용을 요약 정리 및 재진술하는 문장으로 마무리할 것"
    )
    assert server.extract_expected_proper_nouns(conditions) == ["곤이"]

    prompt = server.apply_user_conditions(server.OCR_PROMPT, conditions)
    assert "Expected proper nouns/name spellings" in prompt
    assert "- 곤이" in prompt
    assert "Do not use this rule to correct ordinary words" in prompt


def test_string_fallback_does_not_apply_numeric_tokenization() -> None:
    cells, validation = server.normalize_cells(["100asd"])
    assert cells[0][:6] == ["1", "0", "0", "a", "s", "d"]
    assert validation["cell_rules"]["row_final_period_pair"] is True


def test_occupancy_alignment_repairs_split_mixed_numeric_hangul_cell() -> None:
    row = [[" ", "호", "모", " ", "사", "피", "엔", "스", "는", " ", "20", "0", "만", "년", "전", "에", "탄", "생", "하", "였"]]
    occupied = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19]
    counts = [100 if index in occupied else 0 for index in range(20)]
    occupancy = [{"first_col": 1, "last_col": 19, "counts": counts, "threshold": 50}]

    cells, _ = server.normalize_cells(row, occupancy)

    assert cells[0][10:20] == ["20", "0만", "년", "전", "에", " ", "탄", "생", "하", "였"]

    already_grouped = [[" ", "호", "모", " ", "사", "피", "엔", "스", "는", " ", "20", "0만", "년", "전", "에", " ", "탄", "생", "하", "였"]]
    cells, _ = server.normalize_cells(already_grouped, occupancy)
    assert cells[0][10:20] == ["20", "0만", "년", "전", "에", " ", "탄", "생", "하", "였"]


def test_occupancy_alignment_splits_unjustified_hangul_pairs() -> None:
    row = [[" ", "현", "생", " ", "인", "류", "인", " ", "는", "호", " ", "모", "사", "피엔", "스", "는", " ", "10", "만", "년"]]
    counts = [0, 100, 100, 0, 100, 100, 100, 0, 100, 100, 0, 100, 100, 100, 100, 100, 0, 100, 100, 100]
    occupancy = [{"first_col": 1, "last_col": 19, "counts": counts, "threshold": 50}]

    cells, _ = server.normalize_cells(row, occupancy)

    assert "피엔" not in cells[0]
    assert cells[0][13:16] == ["피", "엔", "스"]
    assert "10" in cells[0]


def test_occupancy_alignment_splits_unjustified_digit_hangul_pair() -> None:
    row = [["인", "류", "의", " ", "끝", "2종", "으", "로"]]
    counts = [100, 100, 100, 0, 100, 100, 100, 100, 100] + [0] * 11
    occupancy = [{"first_col": 0, "last_col": 8, "counts": counts, "threshold": 50}]

    cells, _ = server.normalize_cells(row, occupancy)

    assert cells[0][:9] == ["인", "류", "의", " ", "끝", "2", "종", "으", "로"]


def test_occupancy_alignment_preserves_intentional_internal_spaces() -> None:
    row = [["발", "생", "한", " ", "새", "로", "운", " ", "종", "이", "라", "는", " ", "관", "점", "으", "로", " ", "아", "프"]]
    occupied = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]
    counts = [100 if index in occupied else 0 for index in range(20)]
    occupancy = [{"first_col": 0, "last_col": 19, "counts": counts, "threshold": 50}]

    cells, _ = server.normalize_cells(row, occupancy)

    assert cells[0] == row[0]


def test_period_uses_next_cell_unless_row_overflows() -> None:
    cells, _ = server.normalize_cells(["이다."])
    assert cells[0][:3] == ["이", "다", "."]

    occupancy = [
        {
            "first_col": 0,
            "last_col": 1,
            "counts": [100, 100] + [0] * 18,
            "threshold": 50,
        }
    ] + [{"first_col": None, "last_col": None} for _ in range(19)]
    aligned, _ = server.normalize_cells(["이다."], occupancy)
    assert aligned[0][:3] == ["이", "다", "."]

    overflow, _ = server.normalize_cells(["가" * 18 + "했다."])
    assert overflow[0][18:] == ["했", "다."]


def test_comma_uses_next_cell_unless_row_overflows() -> None:
    cells, validation = server.normalize_cells([["으", "며,", " ", "다"]])
    assert cells[0][:5] == ["으", "며", ",", " ", "다"]
    assert validation["cell_rules"]["row_final_punctuation_pair"] is True

    cells, _ = server.normalize_cells([["가"] * 10 + ["으", "며,", " ", "원", "주", " ", "집", "단", "과", "의"]])
    assert cells[0][10:20] == ["으", "며", ",", "원", "주", " ", "집", "단", "과", "의"]

    overflow, _ = server.normalize_cells([["가"] * 18 + ["며,"]])
    assert overflow[0][18:] == ["며", ","]

    overflow, _ = server.normalize_cells([["가"] * 19 + ["며,"]])
    assert overflow[0][19] == "며,"


def test_occupancy_alignment_splits_middle_commas_but_keeps_row_end_overflow() -> None:
    row = [["고,", "원", "류", " ", " ", "집", "단", "은", " ", "경", "쟁", "에", "서", " ", "전", "멸", "했", "습", "니", "다,"]]
    occupied = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
    counts = [100 if index in occupied else 0 for index in range(20)]
    occupancy = [{"first_col": 0, "last_col": 19, "counts": counts, "threshold": 50}]

    cells, _ = server.normalize_cells(row, occupancy)

    assert "고," not in cells[0]
    assert cells[0][:3] == ["고", ",", "원"]
    assert cells[0][19] == "다,"

    overflow_occupancy = [{"first_col": 0, "last_col": 19, "counts": [100] * 20, "threshold": 50}]
    overflow, _ = server.normalize_cells([["가"] * 19 + ["며,"]], overflow_occupancy)
    assert overflow[0][19] == "며,"


def test_occupancy_alignment_restores_internal_blank_cells() -> None:
    rows = ["가나다"] + [" " * 20 for _ in range(19)]
    occupancy = [
        {
            "first_col": 0,
            "last_col": 3,
            "counts": [100, 0, 120, 110] + [0] * 16,
            "threshold": 50,
        }
    ] + [{"first_col": None, "last_col": None} for _ in range(19)]
    cells, validation = server.normalize_cells(rows, occupancy)
    assert cells[0][:4] == ["가", " ", "나", "다"]
    assert validation["aligned_to_occupancy"] is True


def test_occupancy_alignment_preserves_model_supplied_spaces() -> None:
    row = ["발", "생", "한", " ", "새", "로", "운", " ", "종", "이", "라", "는", " ", "관", "점", "으", "로", " ", "아", "프"]
    occupancy = [
        {
            "first_col": 0,
            "last_col": 19,
            "counts": [100, 100, 100, 0, 100, 100, 100, 0, 100, 10, 100, 100, 0, 100, 100, 100, 100, 0, 100, 100],
            "threshold": 50,
        }
    ] + [{"first_col": None, "last_col": None} for _ in range(19)]
    cells, _ = server.normalize_cells([row], occupancy)
    assert cells[0] == row


def test_occupancy_alignment_shifts_model_spaces_to_detected_first_col() -> None:
    row = ["현", "생", " ", "인", "류"] + [" "] * 15
    occupancy = [
        {
            "first_col": 1,
            "last_col": 5,
            "counts": [0, 100, 100, 0, 100, 100] + [0] * 14,
            "threshold": 50,
        }
    ] + [{"first_col": None, "last_col": None} for _ in range(19)]
    cells, _ = server.normalize_cells([row], occupancy)
    assert cells[0][:6] == [" ", "현", "생", " ", "인", "류"]


def test_parse_nested_chandra_rows() -> None:
    raw = (
        '[{"rows": [{"text": "첫 번째 줄"}, {"text": "두 번째 줄"}]}, '
        '[{"text": "세 번째 줄"}]]'
    )
    parsed = server.parse_model_rows(raw)
    rows, validation = server.normalize_rows(parsed)
    assert rows[0].startswith("첫 번째 줄")
    assert rows[1].startswith("두 번째 줄")
    assert rows[2].startswith("세 번째 줄")
    assert validation["row_count"] == 3


def test_parse_malformed_chandra_text_fields() -> None:
    raw = (
        '[{"rows": [{"text": "첫 번째 줄"}, {"text": "두 번째 줄"}], '
        '[{"text": "세 번째 줄"}]]'
    )
    parsed = server.parse_model_rows(raw)
    assert parsed == ["첫 번째 줄", "두 번째 줄", "세 번째 줄"]


def test_parse_position_style_chandra_rows() -> None:
    raw = '{"rows":[{"x":0,"y":0,"text":"현 생"},{"x":2,"y":0,"text":"인 류"},{"x":0,"y":1,"text":"다 음"}]}'
    parsed = server.parse_model_rows(raw)
    assert parsed == ["현생인류", "다음"]


def test_parse_malformed_position_style_chandra_rows() -> None:
    raw = '[{"rows": [{"x": 0, "y": 0, "text": "현 생"}, {"x": 2, "y": 0, "text": "인 류"}], [{"x": 0, "y": 1, "text": "다 음"}]]'
    parsed = server.parse_model_rows(raw)
    assert parsed == ["현생인류", "다음"]


def test_parse_chandra_html_rows() -> None:
    raw = '<div data-bbox="1 2 3 4" data-label="Text"><p>현생 인류<br/>호모 사피엔스</p></div>'
    assert server.parse_model_rows(raw) == ["현생 인류", "호모 사피엔스"]


def test_bbox_only_chandra_output_is_rejected() -> None:
    raw = '[{"label": "Text", "bbox": "15 12 982 246"}]'
    with pytest.raises(ValueError, match="bounding boxes"):
        server.parse_model_rows(raw)


def test_crop_endpoint_creates_run_assets() -> None:
    image = server.list_images()[0]
    response = client.post("/api/crop", json={"image_id": image.id})
    assert response.status_code == 200
    data = response.json()
    run_dir = Path(server.RUNS_DIR) / data["run_id"]
    assert (run_dir / "crop.png").exists()
    assert (run_dir / "masked.png").exists()
    assert data["grid"]["rows"] == 20
    assert data["grid"]["columns"] == 20


def test_experiment_strategy_expansion_adds_vote_dependencies() -> None:
    assert server.expand_experiment_strategies(["vote_full_row_1_2"]) == [
        "full_grid",
        "row_1",
        "row_2",
        "vote_full_row_1_2",
    ]


def test_experiment_chunks_split_full_grid_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    exp_dir = run_dir / "experiments"
    run_dir.mkdir()
    exp_dir.mkdir()
    Image.new("RGB", (200, 400), "white").save(run_dir / "masked.png")

    row_chunks = server.build_experiment_chunks(run_dir, exp_dir, "row_1")
    two_row_chunks = server.build_experiment_chunks(run_dir, exp_dir, "row_2")

    assert len(row_chunks) == 20
    assert row_chunks[0]["row_start"] == 0
    assert row_chunks[-1]["row_start"] == 19
    assert all(Path(chunk["path"]).exists() for chunk in row_chunks)
    assert len(two_row_chunks) == 10
    assert two_row_chunks[0]["row_count"] == 2


def test_model_experiment_strategy_combines_row_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    exp_dir = run_dir / "experiments"
    run_dir.mkdir()
    exp_dir.mkdir()
    Image.new("RGB", (200, 400), "white").save(run_dir / "masked.png")
    calls: list[str] = []

    def fake_chandra(image_uri: str, prompt: str, settings: server.ModelSettings) -> str:
        calls.append(prompt)
        row_count = 2
        rows = [[f"{len(calls):02d}"] + [" "] * 19 for _ in range(row_count)]
        return '{"rows": ' + server.json.dumps(rows, ensure_ascii=False) + "}"

    monkeypatch.setattr(server, "call_chandra", fake_chandra)

    req = server.ExperimentRequest(run_id="abc123abc123", provider="chandra", strategies=["row_2"])
    result = server.run_model_experiment_strategy(req, run_dir, exp_dir, "row_2", [])

    assert len(calls) == 10
    assert result["validation"]["normalized_shape"] == [20, 20]
    assert result["cells"][0][0] == "01"
    assert result["cells"][18][0] == "10"
    assert result["trace"]["request_shape"]["calls"] == 10
    assert len(result["trace"]["input"]["chunks"]) == 10


def test_model_experiment_strategy_skips_blank_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    exp_dir = run_dir / "experiments"
    run_dir.mkdir()
    exp_dir.mkdir()
    Image.new("RGB", (200, 400), "white").save(run_dir / "masked.png")
    calls = 0

    def fake_chandra(image_uri: str, prompt: str, settings: server.ModelSettings) -> str:
        nonlocal calls
        calls += 1
        return '{"rows": [["가"], ["나"], ["다"], ["라"], ["마"]]}'

    monkeypatch.setattr(server, "call_chandra", fake_chandra)
    occupancy = [
        {"first_col": 0, "last_col": 4, "counts": [500, 400, 300, 200, 100] + [0] * 15, "threshold": 50}
        for _ in range(5)
    ] + [{"first_col": None, "last_col": None, "counts": [0] * 20, "threshold": 50} for _ in range(15)]

    req = server.ExperimentRequest(run_id="abc123abc123", provider="chandra", strategies=["row_5"])
    result = server.run_model_experiment_strategy(req, run_dir, exp_dir, "row_5", occupancy)
    raw_outputs = server.json.loads(result["raw_output"])

    assert calls == 1
    assert result["cells"][0][0] == "가"
    assert result["cells"][5] == [" "] * 20
    assert sum(1 for item in raw_outputs if item["validation"].get("skipped")) == 3


def test_chunk_has_handwriting_rejects_low_dark_noise() -> None:
    noisy_blank = [
        {"first_col": 1, "last_col": 19, "counts": [10] * 20, "threshold": 5}
        for _ in range(5)
    ]
    real_text = [
        {"first_col": 0, "last_col": 19, "counts": [120] * 20, "threshold": 50}
        for _ in range(5)
    ]

    assert server.chunk_has_handwriting(noisy_blank) is False
    assert server.chunk_has_handwriting(real_text) is True


def test_vote_experiment_cells_prefers_majority_then_full_grid() -> None:
    full = [["가"] + [" "] * 19] + [[" "] * 20 for _ in range(19)]
    row_1 = [["나"] + [" "] * 19] + [[" "] * 20 for _ in range(19)]
    row_2 = [["나"] + [" "] * 19] + [[" "] * 20 for _ in range(19)]
    cells, stats = server.vote_experiment_cells(
        {
            "full_grid": {"cells": full},
            "row_1": {"cells": row_1},
            "row_2": {"cells": row_2},
        }
    )

    assert cells[0][0] == "나"
    assert stats["majority_cells"] == 1
    assert stats["disagreement_cells"] == 1

    cells, stats = server.vote_experiment_cells(
        {
            "full_grid": {"cells": full},
            "row_1": {"cells": [["다"] + [" "] * 19] + [[" "] * 20 for _ in range(19)]},
        }
    )
    assert cells[0][0] == "가"
    assert stats["fallback_cells"] == 1


def test_run_payload_includes_experiment_results() -> None:
    image = sample_image()
    response = client.post("/api/crop", json={"image_id": image.id})
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    run_dir = Path(server.RUNS_DIR) / run_id
    exp_dir = run_dir / "experiments"
    exp_dir.mkdir()
    cells = [["가"] + [" "] * 19] + [[" "] * 20 for _ in range(19)]
    result = {
        "kind": "experiment",
        "provider": "chandra",
        "strategy": "full_grid",
        "trace": {"kind": "experiment", "strategy": "full_grid"},
        "raw_output": "raw",
        "rows": server.cells_to_text_rows(cells),
        "cells": cells,
        "validation": {"normalized_shape": [20, 20]},
    }
    server.write_json(exp_dir / "full_grid_chandra.json", result)
    server.write_json(
        exp_dir / "summary_chandra.json",
        {
            "run_id": run_id,
            "provider": "chandra",
            "strategies": ["full_grid"],
            "variants": [{"strategy": "full_grid"}],
        },
    )

    payload = client.get(f"/api/runs/{run_id}").json()

    assert payload["experiments"]["chandra"]["variants"][0]["strategy"] == "full_grid"
    assert payload["experiments"]["chandra"]["variants"][0]["cells"][0][0] == "가"


def test_audit_reports_api_lists_random_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    report_dir = work_dir / "random10_audit" / "20260707-135847"
    report_dir.mkdir(parents=True)
    image = sample_image()
    response = client.post("/api/crop", json={"image_id": image.id})
    run_id = response.json()["run_id"]
    server.write_json(
        report_dir / "report.json",
        {
            "seed": 1,
            "count": 1,
            "created_at": "2026-07-07 13:58:54",
            "summary": {"ok": 1, "needs_review": 0},
            "items": [{"index": 1, "image_name": image.name, "folder_name": image.folder_name, "run_id": run_id, "flags": ["ok"]}],
        },
    )
    (report_dir / "REPORT.md").write_text("# report", encoding="utf-8")
    (report_dir / "OCR_RESULTS.md").write_text("# ocr", encoding="utf-8")
    (report_dir / "contact_sheet.jpg").write_bytes(b"fake")
    monkeypatch.setattr(server, "WORK_DIR", work_dir)

    data = client.get("/api/audit-reports").json()

    assert data["reports"][0]["id"] == "20260707-135847"
    assert data["reports"][0]["items"][0]["run_id"] == run_id
    assert data["reports"][0]["ocr_results_url"].endswith("/OCR_RESULTS.md")
    assert data["reports"][0]["contact_sheet_url"].endswith("/contact_sheet.jpg")


def test_audit_report_file_is_served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    report_dir = work_dir / "random10_audit" / "20260707-135847"
    report_dir.mkdir(parents=True)
    (report_dir / "REPORT.md").write_text("# report", encoding="utf-8")
    monkeypatch.setattr(server, "WORK_DIR", work_dir)

    response = client.get("/reports/random10/20260707-135847/REPORT.md")

    assert response.status_code == 200
    assert response.text == "# report"


def test_runs_endpoint_lists_recent_runs() -> None:
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    if data["runs"]:
        assert "run_id" in data["runs"][0]
        assert "has_gpt" in data["runs"][0]


def test_gpt_request_forces_model_and_normalizes_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def json(self) -> dict:
            return self.payload

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
        calls.append(json)
        return FakeResponse({"id": "resp_test"})

    def fake_get(url: str, headers: dict, timeout: int) -> FakeResponse:
        return FakeResponse({"status": "completed", "output_text": "ok"})

    monkeypatch.setattr(server, "PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setattr(server.requests, "post", fake_post)
    monkeypatch.setattr(server.requests, "get", fake_get)

    settings = server.ModelSettings(model="gpt-4o", reasoning_effort="invalid")
    assert server.call_gpt_responses("https://example.com/crop.png", "prompt", settings) == "ok"
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["reasoning_effort"] == "low"


def test_repair_prompt_preserves_json_schema_braces() -> None:
    rows = '[["이","다","."]]'
    prompt = server.build_repair_prompt("gpt", rows)
    assert '{"cells"' in prompt
    assert rows in prompt
    assert "{rows}" not in prompt


def test_trace_exposes_input_prompt_and_request_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "PUBLIC_BASE_URL", "https://example.com")
    settings = server.ModelSettings(model="gpt-5.5", reasoning_effort="high")
    trace = server.build_ocr_trace("gpt", "abc123def456", "prompt text", settings)

    assert trace["input"]["image"]["public_url"] == "https://example.com/runs/abc123def456/masked.png"
    assert trace["prompt"] == "prompt text"
    assert trace["request_shape"]["body"]["model"] == "gpt-5.5"
    assert trace["request_shape"]["body"]["reasoning_effort"] == "high"
    assert "Authorization" not in str(trace)
    assert "DOGOK_PROXY_API_KEY" not in str(trace)


def test_repair_trace_includes_source_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "PUBLIC_BASE_URL", "https://example.com")
    rows = '[["으","며",","]]'
    settings = server.ModelSettings(model="gpt-5.5")
    prompt = server.build_repair_prompt("gpt", rows)

    trace = server.build_repair_trace("gpt", "gpt", "abc123def456", rows, prompt, settings)

    assert trace["kind"] == "repair"
    assert trace["input"]["source_provider"] == "gpt"
    assert trace["input"]["source_cells"] == [["으", "며", ","]]
    assert rows in trace["prompt"]
