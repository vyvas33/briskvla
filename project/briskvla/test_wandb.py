"""CPU-only unit tests for common.py naming/helpers and mandatory W&B logging in wandb_log.py.

Run with:
    uv run --with pytest python -m pytest project/briskvla/test_wandb.py -sv
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import common
import pytest
import wandb_log


class _FakeTable:
    def __init__(self, columns: list[str], data: list[list]) -> None:
        self.columns = columns
        self.data = data


class _FakeRun:
    def __init__(self, kwargs: dict) -> None:
        self.init_kwargs = kwargs
        self.defined_metrics: list[tuple[str, str | None]] = []
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.finished = False

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        self.defined_metrics.append((name, step_metric))

    def log(self, data: dict) -> None:
        self.logged.append(data)

    def finish(self) -> None:
        self.finished = True


class _FakeWandb(types.ModuleType):
    def __init__(self, logged_in: bool = True) -> None:
        super().__init__("wandb")
        self.logged_in = logged_in
        self.runs: list[_FakeRun] = []
        self.Table = _FakeTable

    def login(self, prompt: bool = False) -> bool:
        return self.logged_in

    def init(self, **kwargs) -> _FakeRun:
        run = _FakeRun(kwargs)
        self.runs.append(run)
        return run


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch, logged_in: bool = True) -> _FakeWandb:
    fake = _FakeWandb(logged_in=logged_in)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_resolve_layout_aliases_and_model_naming():
    assert common.canonical_layout_name("pairs") == "spread"
    assert common.canonical_layout_name("first8") == "early"
    assert common.canonical_layout_name("last8") == "late"
    assert common.canonical_layout_name("alt") == "alternating"
    assert common.canonical_layout_name("alternate") == "alternating"

    assert common.resolve_layout("spread") == common.resolve_layout("pairs") == [0, 1, 4, 5, 8, 9, 12, 13]
    assert common.resolve_layout("early") == common.resolve_layout("first8") == list(range(8))
    assert common.resolve_layout("late") == common.resolve_layout("last8") == list(range(8, 16))
    assert common.resolve_layout("alternating") == [0, 3, 4, 7, 8, 11, 12, 15]

    with pytest.raises(ValueError, match="Unknown layout"):
        common.resolve_layout("nonexistent")

    assert common.infer_model_role(1.0, 0.1, 1.0, steps=5000) == "student"
    assert common.infer_model_role(0.0, 0.0, 0.0, steps=5000) == "shallow"
    assert common.infer_model_role(1.0, 0.1, 1.0, steps=0) == "shallow"

    assert common.format_model_name("student", 8, "pairs") == "student_8L_spread"
    assert (
        common.format_model_name("student", 8, "alternating", "ffn_fused")
        == "student_8L_alternating_ffn_fused"
    )
    assert common.format_model_name("shallow", 8, "first8") == "shallow_8L_early"
    assert (
        common.format_model_name("shallow", 8, "alternating", "ffn_fused", "untrained")
        == "shallow_8L_alternating_ffn_fused_untrained"
    )
    with pytest.raises(ValueError, match="role must be"):
        common.format_model_name("teacher", 8, "spread")


def test_resolve_wandb_settings_defaults(monkeypatch: pytest.MonkeyPatch):
    for var in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_GROUP", "WANDB_TAGS", "WANDB_MODE"):
        monkeypatch.delenv(var, raising=False)
    p = argparse.ArgumentParser()
    wandb_log.add_wandb_args(p)
    ns = p.parse_args([])
    cfg = wandb_log.resolve_wandb_settings(ns)
    assert cfg == {
        "project": "briskVLA",
        "entity": "djonnala-carnegie-mellon-university",
        "group": None,
        "name": None,
        "tags": None,
        "mode": "online",
    }

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_GROUP", "ablation-1")
    monkeypatch.setenv("WANDB_TAGS", "libero, 8L")
    cfg_env = wandb_log.resolve_wandb_settings(ns)
    assert cfg_env["mode"] == "disabled"
    assert cfg_env["group"] == "ablation-1"
    assert cfg_env["tags"] == ["libero", "8L"]


def test_ensure_wandb_auth_blocks_unauthenticated_online_and_allows_offline_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _install_fake_wandb(monkeypatch, logged_in=False)

    with pytest.raises(SystemExit, match="W&B logging is required"):
        wandb_log.ensure_wandb_auth("online")

    # Offline and disabled modes do not require online credentials.
    wandb_log.ensure_wandb_auth("offline")
    wandb_log.ensure_wandb_auth("disabled")

    # Env API key satisfies online auth.
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    wandb_log.ensure_wandb_auth("online")


def test_distill_logger_logs_metrics_tracks_best_and_resumes_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = _install_fake_wandb(monkeypatch, logged_in=True)
    run_dir = tmp_path / "student_8L_spread"
    logger = wandb_log.WandbDistillLogger(
        run_dir,
        project="briskVLA",
        entity="djonnala-carnegie-mellon-university",
        group="k8-ablation",
        config={"args": {"w_velocity": 1.0, "w_hidden": 0.1, "w_output": 1.0, "steps": 100}},
        mode="online",
    )
    id_file = run_dir / wandb_log.ID_FILE
    assert id_file.is_file()
    first_id = id_file.read_text().strip()

    logger.log({"split": "val", "step": 0, "chunk_mse_teacher": 0.5})
    logger.log({"split": "train", "step": 50, "total": 0.25, "lr": 1e-4})
    logger.log({"split": "val", "step": 100, "chunk_mse_teacher": 0.12})
    logger.log({"split": "final", "step": 100, "best_chunk_mse_teacher": 0.12, "latency_ms_student": 48.5})
    logger.finish()

    run1 = fake.runs[0]
    assert run1.finished
    assert run1.init_kwargs["job_type"] == "distill"
    assert run1.init_kwargs["id"] == first_id
    assert "role:student" in run1.init_kwargs["tags"]
    assert ("step", None) in run1.defined_metrics
    assert ("*", "step") in run1.defined_metrics
    assert run1.summary["val/best_chunk_mse_teacher"] == 0.12
    assert run1.summary["val/best_step"] == 100
    assert run1.summary["train_device/latency_ms_student"] == 48.5
    assert run1.summary["final/best_chunk_mse_teacher"] == 0.12
    assert run1.logged[1] == {"step": 50, "train/total": 0.25, "train/lr": 1e-4}

    # Backfill CLI helper refuses duplicate upload without --force, and reuses ID with --force.
    (run_dir / "run_info.json").write_text(
        json.dumps({"role": "shallow", "val_episodes": [1, 2, 3], "args": {"w_gt": 1.0}})
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"split": "val", "step": 0, "chunk_mse_teacher": 0.4}) + "\n"
    )
    with pytest.raises(SystemExit, match="already has a wandb run"):
        wandb_log.upload_distill(run_dir, force=False)

    wandb_log.upload_distill(run_dir, force=True)
    run2 = fake.runs[1]
    assert run2.init_kwargs["id"] == first_id
    assert "val_episodes" not in run2.init_kwargs["config"]
    assert "role:shallow" in run2.init_kwargs["tags"]


def test_log_eval_run_and_distill_linkage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake = _install_fake_wandb(monkeypatch, logged_in=True)

    distill_dir = tmp_path / "distill" / "student_8L_spread"
    (distill_dir / "best").mkdir(parents=True)
    (distill_dir / wandb_log.ID_FILE).write_text("deadbeef")
    (distill_dir / "best" / "config.json").write_text(
        json.dumps({"expert_vlm_layers": [0, 1, 4, 5, 8, 9, 12, 13], "expert_intermediate_size": 2048})
    )

    eval_dir = tmp_path / "eval" / "student_8L_spread_K10"
    eval_dir.mkdir(parents=True)
    (eval_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "name": "student_8L_spread",
                "policy_dir": str(distill_dir / "best"),
                "num_steps": 10,
                "n_action_steps": 10,
                "gpu": "NVIDIA L4",
            }
        )
    )
    suite_scores = {
        "libero_spatial": 78.0,
        "libero_object": 82.0,
        "libero_goal": 80.0,
        "libero_10": 72.0,
    }
    for suite, pc in suite_scores.items():
        sdir = eval_dir / suite
        sdir.mkdir()
        (sdir / "eval_info.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "pc_success": pc,
                        "n_success": int(pc),
                        "n_episodes": 100,
                        "pc_success_ci95": [pc - 8.0, pc + 8.0],
                        "eval_s": 120.5,
                    },
                    "per_task": [
                        {
                            "task_id": 0,
                            "pc_success": pc,
                            "n_success": 8,
                            "n_episodes": 10,
                            "pc_success_ci95": [49.0, 94.0],
                        }
                    ],
                }
            )
        )

    assert wandb_log.discover_eval_dirs([tmp_path / "eval"]) == [eval_dir]
    assert wandb_log.log_eval_run(eval_dir, group="eval-k10") is True

    run = fake.runs[0]
    assert run.finished
    assert run.init_kwargs["job_type"] == "eval"
    assert run.init_kwargs["name"] == "student_8L_spread_K10"
    assert run.init_kwargs["config"]["distill_wandb_id"] == "deadbeef"
    assert run.init_kwargs["config"]["expert_intermediate_size"] == 2048
    assert "role:student" in run.init_kwargs["tags"]
    assert "K:10" in run.init_kwargs["tags"]
    assert run.summary["eval/avg_success"] == pytest.approx(78.0)
    assert run.summary["eval/suites_completed"] == 4
    assert run.summary["eval/total_success"] == 312
    assert run.summary["eval/total_episodes"] == 400
    assert run.summary["eval/libero_spatial/pc_success"] == 78.0
    table = run.logged[0]["eval/per_task"]
    assert isinstance(table, _FakeTable)
    assert len(table.data) == 4

    # Idempotent without --force, updates same run with --force.
    assert wandb_log.log_eval_run(eval_dir, force=False) is False
    assert wandb_log.log_eval_run(eval_dir, force=True) is True
    assert fake.runs[1].init_kwargs["id"] == run.init_kwargs["id"]


def test_log_latency_file_logs_per_k_summary_and_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake = _install_fake_wandb(monkeypatch, logged_in=True)
    lat_path = tmp_path / "rtx3080.json"
    lat_path.write_text(
        json.dumps(
            {
                "meta": {
                    "gpu_before": {"name": "NVIDIA GeForce RTX 3080 Laptop GPU"},
                    "cpu": "Intel Core i7",
                    "git_commit": "abc123",
                    "git_branch": "briskvla",
                },
                "models": {
                    "student_8L_alternating_ffn_fused": {
                        "path": "outputs/distill/student_8L_alternating_ffn_fused/best",
                        "expert_vlm_layers": [0, 3, 4, 7, 8, 11, 12, 15],
                        "num_expert_layers": 8,
                        "expert_intermediate_size": 4096,
                        "expert_params": 84512720,
                        "total_params": 436313056,
                        "flops": {"prefix": 725e9, "per_step": 9.33e9},
                        "fit": {"prefix_ms": 32.1, "per_step_ms": 4.71},
                        "steps": {
                            "1": {"wall_ms": 37.0, "wall_ms_p90": 38.0, "prefix_ms": 32.1, "denoise_ms": 4.9},
                            "10": {
                                "wall_ms": 79.3,
                                "wall_ms_p90": 80.5,
                                "prefix_ms": 30.7,
                                "denoise_ms": 48.5,
                                "ffn_ms": 5.7,
                                "non_ffn_ms": 42.7,
                                "samples": [{"wall_ms": 79.3}],
                            },
                        },
                    }
                },
                "ffn_width_sweep_us": {"2048": 42.5, "4096": 58.3},
            }
        )
    )

    assert wandb_log.log_latency_file(lat_path, group="rtx3080") == 1
    run = fake.runs[0]
    assert run.finished
    assert run.init_kwargs["job_type"] == "latency"
    assert run.init_kwargs["name"] == "student_8L_alternating_ffn_fused"
    assert run.init_kwargs["config"]["gpu"] == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert ("K", None) in run.defined_metrics
    assert ("latency/*", "K") in run.defined_metrics
    assert run.summary["latency/K10/wall_ms"] == 79.3
    assert run.summary["latency/K10/ffn_ms"] == 5.7
    assert run.summary["flops/per_step_gflops"] == pytest.approx(9.33)
    assert run.summary["latency/fit/per_step_ms"] == 4.71
    # Ensure per-sample raw list was not logged as a scalar
    assert all("latency/samples" not in entry for entry in run.logged)
    sweep_logs = [
        entry["latency/ffn_width_sweep"] for entry in run.logged if "latency/ffn_width_sweep" in entry
    ]
    assert len(sweep_logs) == 1 and sweep_logs[0].data == [[2048, 42.5], [4096, 58.3]]

    # Dedup via .wandb_ids.json
    assert wandb_log.log_latency_file(lat_path, force=False) == 0
    assert wandb_log.log_latency_file(lat_path, force=True) == 1
