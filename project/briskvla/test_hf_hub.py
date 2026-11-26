"""CPU-only unit tests for Hugging Face Hub helpers in project/briskvla/hf_hub.py.

Run with:
    uv run --with pytest python -m pytest project/briskvla/test_hf_hub.py -sv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import hf_hub
import pytest
import wandb_log


def test_parse_and_detect_hf_specs(tmp_path: Path):
    assert hf_hub.is_hf_repo_spec("briskvla/shallow_8L_alternating_ffn_fused") is True
    assert hf_hub.is_hf_repo_spec("briskvla/student_8L_spread@abc123") is True
    assert hf_hub.is_hf_repo_spec("/abs/path/model") is False
    assert hf_hub.is_hf_repo_spec("./rel/path") is False
    assert hf_hub.is_hf_repo_spec("teacher") is False

    # Existing local dir with a slash is NOT treated as an HF repo spec.
    local_dir = tmp_path / "owner" / "repo"
    local_dir.mkdir(parents=True)
    assert hf_hub.is_hf_repo_spec(local_dir) is False

    assert hf_hub.parse_hf_spec("briskvla/model_a") == ("briskvla/model_a", None)
    assert hf_hub.parse_hf_spec("briskvla/model_a@rev123") == ("briskvla/model_a", "rev123")
    with pytest.raises(ValueError, match="Invalid Hugging Face repo spec"):
        hf_hub.parse_hf_spec("not-a-repo")

    assert hf_hub.format_hf_repo_id("briskvla", "shallow_8L_alternating_ffn_fused") == (
        "briskvla/shallow_8L_alternating_ffn_fused"
    )
    assert hf_hub.format_hf_repo_id("devanshroyal-7", "shallow_8L_early", prefix="briskvla-") == (
        "devanshroyal-7/briskvla-shallow_8L_early"
    )
    assert hf_hub.format_hf_repo_id("ignored", "org/explicit") == "org/explicit"


def test_resolve_hf_settings_and_target_repo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(hf_hub.HF_NAMESPACE_ENV, raising=False)
    monkeypatch.delenv(hf_hub.HF_PREFIX_ENV, raising=False)
    p = argparse.ArgumentParser()
    hf_hub.add_hf_upload_args(p)

    ns_empty = p.parse_args([])
    cfg = hf_hub.resolve_hf_settings(ns_empty)
    assert (
        hf_hub.resolve_target_repo(
            "shallow_8L_alternating", repo=cfg["repo"], namespace=cfg["namespace"], prefix=cfg["prefix"]
        )
        == "briskVLA/shallow_8L_alternating"
    )

    monkeypatch.setenv(hf_hub.HF_NAMESPACE_ENV, "briskvla-team")
    monkeypatch.setenv(hf_hub.HF_PREFIX_ENV, "libero-")
    cfg_env = hf_hub.resolve_hf_settings(p.parse_args([]))
    assert (
        hf_hub.resolve_target_repo(
            "shallow_8L_alternating",
            repo=cfg_env["repo"],
            namespace=cfg_env["namespace"],
            prefix=cfg_env["prefix"],
        )
        == "briskvla-team/libero-shallow_8L_alternating"
    )

    cfg_disabled = hf_hub.resolve_hf_settings(p.parse_args(["--no_hf_push"]))
    assert (
        hf_hub.resolve_target_repo(
            "shallow_8L_alternating",
            repo=cfg_disabled["repo"],
            namespace=cfg_disabled["namespace"],
            prefix=cfg_disabled["prefix"],
        )
        is None
    )


def test_stage_upload_dir_preserves_provenance_and_wandb_linkage(tmp_path: Path):
    run_dir = tmp_path / "shallow_8L_alternating_ffn_fused"
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "model.safetensors").write_bytes(b"fake-weights")
    (best_dir / "config.json").write_text(
        json.dumps({"expert_vlm_layers": [0, 3, 4, 7, 8, 11, 12, 15], "expert_intermediate_size": 4096})
    )
    (best_dir / "policy_preprocessor.json").write_text("{}")
    (best_dir / "training_state.pt").write_bytes(b"should-not-upload")
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "role": "shallow",
                "layout_canonical": "alternating",
                "layers": [0, 3, 4, 7, 8, 11, 12, 15],
                "expert_intermediate_size": 4096,
                "expert_params_student": 84_512_720,
                "args": {"steps": 5000, "batch_size": 8, "w_velocity": 0, "w_hidden": 0, "w_output": 0, "w_gt": 1},
            }
        )
    )
    (run_dir / wandb_log.ID_FILE).write_text("75209064")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"split": "val", "step": 5000, "chunk_mse_teacher": 0.1593, "chunk_mse_gt": 0.2091}) + "\n"
    )

    stage = tmp_path / "stage"
    hf_hub.stage_upload_dir(run_dir, "briskvla/shallow_8L_alternating_ffn_fused", stage)

    assert (stage / "model.safetensors").is_file()
    assert (stage / "config.json").is_file()
    assert (stage / "policy_preprocessor.json").is_file()
    assert not (stage / "training_state.pt").exists()
    assert (stage / "run_info.json").is_file()
    assert (stage / wandb_log.ID_FILE).read_text() == "75209064"
    assert (stage / "metrics.jsonl").is_file()
    readme = (stage / "README.md").read_text()
    assert "briskvla/shallow_8L_alternating_ffn_fused" in readme
    assert "ffn-fusion" in readme
    assert "0.1593" in readme
    assert "75209064" in readme


def test_ensure_hf_write_auth_rejects_readonly_finegrained_and_accepts_write(monkeypatch: pytest.MonkeyPatch):
    class _ReadOnlyApi:
        def whoami(self):
            return {
                "name": "devanshroyal-7",
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "global": [],
                            "scoped": [{"entity": {"name": "devanshroyal-7"}, "permissions": ["repo.content.read"]}],
                        },
                    }
                },
            }

    monkeypatch.setattr(hf_hub, "HfApi", _ReadOnlyApi)
    with pytest.raises(SystemExit, match="no write permissions"):
        hf_hub.ensure_hf_write_auth("devanshroyal-7")

    class _WriteApi:
        def whoami(self):
            return {"name": "devanshroyal-7", "auth": {"accessToken": {"role": "write"}}}

    monkeypatch.setattr(hf_hub, "HfApi", _WriteApi)
    assert hf_hub.ensure_hf_write_auth("devanshroyal-7") == "devanshroyal-7"


def test_upload_model_and_resolve_policy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ckpt = tmp_path / "my_model"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"weights")
    (ckpt / "config.json").write_text(json.dumps({"expert_vlm_layers": [0, 3, 4, 7, 8, 11, 12, 15]}))

    calls: dict = {}

    class _FakeApi:
        def whoami(self):
            return {"name": "devanshroyal-7", "auth": {"accessToken": {"role": "write"}}}

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs
            staged_files = sorted(p.name for p in Path(kwargs["folder_path"]).iterdir())
            calls["staged_files"] = staged_files
            return SimpleNamespace(commit_url="https://huggingface.co/briskvla/my_model/commit/abc")

    monkeypatch.setattr(hf_hub, "HfApi", _FakeApi)
    url = hf_hub.upload_model(ckpt, "briskvla/my_model", private=True)
    assert url == "https://huggingface.co/briskvla/my_model/commit/abc"
    assert calls["create_repo"]["private"] is True
    assert calls["create_repo"]["repo_id"] == "briskvla/my_model"
    assert "README.md" in calls["staged_files"]
    assert "model.safetensors" in calls["staged_files"]

    # Local path resolves directly; HF spec delegates to snapshot_download.
    assert hf_hub.resolve_policy_path(ckpt) == ckpt
    monkeypatch.setattr(
        hf_hub,
        "snapshot_download",
        lambda repo_id, revision=None, local_dir=None: str(tmp_path / f"dl_{repo_id.replace('/', '_')}_{revision}"),
    )
    resolved = hf_hub.resolve_policy_path("briskvla/my_model@rev9")
    assert resolved.name == "dl_briskvla_my_model_rev9"
