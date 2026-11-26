"""CPU-only unit tests for shared LIBERO eval registry in project/briskvla/eval_registry.py.

Run with:
    uv run pytest project/briskvla/test_eval_registry.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import eval_registry
import pytest
import wandb_log
from common import SUITES


def test_merge_rows_deduplication_priority():
    readme_partial = {
        "model": "teacher",
        "hidden_map": "",
        "num_steps": 10,
        "episodes_per_task": 10,
        "n_action_steps": 10,
        "seed": 1000,
        "spatial": 79.0,
        "suites_completed": 1,
        "status": "partial",
        "source": "readme",
        "date": "2026-11-21T00:00:00Z",
    }
    readme_complete = {
        "model": "teacher",
        "hidden_map": "",
        "num_steps": 10,
        "episodes_per_task": 10,
        "n_action_steps": 10,
        "seed": 1000,
        "spatial": 79.0,
        "object": 91.0,
        "goal": 89.0,
        "long": 69.0,
        "avg": 82.0,
        "suites_completed": 4,
        "status": "complete",
        "source": "readme",
        "date": "2026-11-21T00:00:00Z",
    }
    wandb_complete = {
        **readme_complete,
        "evaluator": "djonnala",
        "wandb_run_id": "abc12345",
        "source": "wandb",
        "date": "2026-11-22T00:00:00Z",
    }
    wandb_newer = {
        **wandb_complete,
        "wandb_run_id": "def67890",
        "date": "2026-11-25T00:00:00Z",
    }

    merged = eval_registry.merge_rows([readme_partial, readme_complete], [wandb_complete, wandb_newer])
    assert len(merged) == 1
    assert merged[0]["status"] == "complete"
    assert merged[0]["source"] == "wandb"
    assert merged[0]["wandb_run_id"] == "def67890"
    assert merged[0]["wandb_url"] == (
        "https://wandb.ai/djonnala-carnegie-mellon-university/briskVLA/runs/def67890"
    )


def test_parse_readme_tables_extracts_canonical_models():
    readme = Path(__file__).resolve().parent / "README.md"
    rows = eval_registry.parse_readme_tables(readme)
    assert len(rows) == 26
    models = {r["model"] for r in rows}
    assert {
        "teacher",
        "student_8L_early",
        "student_8L_spread",
        "student_8L_late",
        "shallow_8L_early",
        "shallow_8L_spread",
        "shallow_8L_late",
        "student_8L_alternating_ffn_fused",
        "student_8L_alternating",
    }.issubset(models)

    # Spread appears in both same_depth and uniform hidden_map sections without collision.
    spread_k10 = [r for r in rows if r["model"] == "student_8L_spread" and r["num_steps"] == "10"]
    assert {r["hidden_map"] for r in spread_k10} == {"same_depth", "uniform"}


def test_record_local_eval_dir_and_check_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "teacher_K1"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "name": "teacher",
                "policy": "teacher",
                "num_steps": 1,
                "episodes_per_task": 10,
                "n_action_steps": 10,
                "seed": 1000,
                "gpu": "NVIDIA L4",
                "date": "2026-11-26T12:00:00Z",
            }
        )
    )
    (run_dir / wandb_log.ID_FILE).write_text("wb998877")
    suite_scores = {"libero_spatial": 80.0, "libero_object": 90.0, "libero_goal": 85.0, "libero_10": 65.0}
    for suite in SUITES:
        sdir = run_dir / suite
        sdir.mkdir()
        (sdir / "eval_info.json").write_text(json.dumps({"overall": {"pc_success": suite_scores[suite]}}))

    saved_store: list[dict[str, str]] = []
    monkeypatch.setattr(eval_registry, "load_registry", lambda repo_id: list(saved_store))

    def fake_save(rows, repo_id, *, commit_message="", private=False):
        saved_store.clear()
        saved_store.extend(eval_registry.merge_rows([], rows))
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(eval_registry, "save_registry", fake_save)

    row = eval_registry.record_local_eval_dir(run_dir, evaluator="djonnala")
    assert row["status"] == "complete"
    assert row["suites_completed"] == "4"
    assert row["avg"] == "80"
    assert row["wandb_url"] == "https://wandb.ai/djonnala-carnegie-mellon-university/briskVLA/runs/wb998877"

    hit = eval_registry.find_matching_eval(saved_store, name="teacher", steps=1, episodes=10)
    assert hit is not None
    assert hit["wandb_run_id"] == "wb998877"

    # Different episode count must not match ep=10.
    miss_ep = eval_registry.find_matching_eval(saved_store, name="teacher", steps=1, episodes=20)
    assert miss_ep is None
    miss_k = eval_registry.find_matching_eval(saved_store, name="teacher", steps=3, episodes=10)
    assert miss_k is None


def test_build_coverage_matrix_identifies_missing_jobs():
    rows = eval_registry.merge_rows(
        [],
        [
            {
                "model": "teacher",
                "num_steps": 10,
                "episodes_per_task": 10,
                "avg": 82.0,
                "suites_completed": 4,
                "status": "complete",
                "source": "readme",
            },
            {
                "model": "shallow_8L_alternating",
                "num_steps": 10,
                "episodes_per_task": 10,
                "avg": 63.75,
                "suites_completed": 4,
                "status": "complete",
                "wandb_run_id": "0f1aeccf",
                "source": "wandb",
            },
        ],
    )
    matrix, missing = eval_registry.build_coverage_matrix(
        rows,
        steps_list=[10, 3, 1],
        episodes=10,
        hf_models=["briskVLA/shallow_8L_alternating"],
    )
    by_model = {m["model"]: m for m in matrix}
    assert by_model["teacher"]["K=10"] == "82"
    assert by_model["teacher"]["K=3"] == "MISSING"
    assert by_model["shallow_8L_alternating"]["K=10"] == "63.75"
    assert "0f1aeccf" in by_model["shallow_8L_alternating"]["K=10_wandb"]
    assert ("teacher", "teacher", 3) in missing
    assert ("teacher", "teacher", 1) in missing
    assert ("shallow_8L_alternating", "briskVLA/shallow_8L_alternating", 3) in missing
    assert ("shallow_8L_alternating", "briskVLA/shallow_8L_alternating", 1) in missing
