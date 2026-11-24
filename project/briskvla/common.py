"""Shared constants, model naming helpers, and environment metadata for BriskVLA scripts."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

TEACHER = "lerobot/smolvla_libero"
# The revision evaluated on LIBERO so all runs share the exact same teacher.
TEACHER_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"

DATASET = "lerobot/libero"
# Same mapping as the teacher's train_config.json and policy_preprocessor.json.
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
CAMERAS = '{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_SHORT = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}

DEFAULT_N_ACTION_STEPS = 10
DEFAULT_SEED = 1000
DEFAULT_EMPTY_CAMERAS = 1

DEFAULT_WANDB_PROJECT = "briskVLA"
DEFAULT_WANDB_ENTITY = "djonnala-carnegie-mellon-university"

# Three primary placement-ablation layouts (kept at 3 so parametrized GPU tests stay fast).
LAYOUTS: dict[str, list[int]] = {
    "spread": [0, 1, 4, 5, 8, 9, 12, 13],
    "early": [0, 1, 2, 3, 4, 5, 6, 7],
    "late": [8, 9, 10, 11, 12, 13, 14, 15],
}
EXTRA_LAYOUTS: dict[str, list[int]] = {
    "alternating": [0, 3, 4, 7, 8, 11, 12, 15],
}
LEGACY_LAYOUT_ALIASES: dict[str, str] = {
    "pairs": "spread",
    "first8": "early",
    "last8": "late",
    "alt": "alternating",
    "alternate": "alternating",
}
ALL_LAYOUT_CHOICES = sorted({*LAYOUTS, *EXTRA_LAYOUTS, *LEGACY_LAYOUT_ALIASES})

MODEL_ROLES = ("student", "shallow")


def canonical_layout_name(name: str) -> str:
    """Resolve legacy layout aliases (`pairs`, `first8`, `last8`, `alt`) to canonical names."""
    canon = LEGACY_LAYOUT_ALIASES.get(name, name)
    if canon not in LAYOUTS and canon not in EXTRA_LAYOUTS:
        raise ValueError(f"Unknown layout {name!r}; expected one of {ALL_LAYOUT_CHOICES}")
    return canon


def resolve_layout(name: str) -> list[int]:
    """Return the VLM layer depths for a canonical layout name or legacy alias."""
    canon = canonical_layout_name(name)
    layers = LAYOUTS.get(canon) or EXTRA_LAYOUTS[canon]
    return list(layers)


def infer_model_role(w_velocity: float, w_hidden: float, w_output: float, steps: int = 1) -> str:
    """Return `student` when trained with any teacher distillation loss, else `shallow`."""
    if steps > 0 and (w_velocity > 0 or w_hidden > 0 or w_output > 0):
        return "student"
    return "shallow"


def format_model_name(
    role: str,
    num_layers: int,
    placement: str,
    modification: str | None = None,
    variant: str | None = None,
) -> str:
    """Build a self-explanatory BriskVLA model name: `<role>_<layers>L_<placement>[_<mod>][_<variant>]`."""
    if role not in MODEL_ROLES:
        raise ValueError(f"role must be one of {MODEL_ROLES}, got {role!r}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if placement in LAYOUTS or placement in EXTRA_LAYOUTS or placement in LEGACY_LAYOUT_ALIASES:
        placement = canonical_layout_name(placement)
    parts = [role, f"{num_layers}L", placement]
    if modification:
        parts.append(modification)
    if variant:
        parts.append(variant)
    return "_".join(parts)


def git_info(repo_dir: Path | None = None) -> dict[str, Any]:
    """Return the current git commit, branch name, and whether the working tree is dirty."""
    cwd = repo_dir or Path(__file__).resolve().parents[2]

    def run(args: list[str]) -> str | None:
        try:
            res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        except OSError:
            return None
        return res.stdout.strip() or None if res.returncode == 0 else None

    status = run(["status", "--porcelain"])
    return {
        "git_commit": run(["rev-parse", "HEAD"]),
        "git_branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": status is not None and bool(status),
    }


def gpu_state() -> dict[str, str] | None:
    """Query basic GPU metadata and clocks/temperature via nvidia-smi, or None if unavailable."""
    fields = "name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None
    return dict(zip(fields.split(","), (x.strip() for x in line.split(",")), strict=False))


def gpu_name() -> str:
    """Return the primary GPU name or `unknown`."""
    state = gpu_state()
    return state["name"] if state and "name" in state else "unknown"


def cpu_name() -> str:
    """Return the host CPU model name."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def read_policy_arch(policy_dir: str | Path) -> dict[str, Any]:
    """Extract expert depth and FFN width metadata from a local policy `config.json` if present."""
    cfg_path = Path(policy_dir) / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    layers = cfg.get("expert_vlm_layers")
    num_vlm = cfg.get("num_vlm_layers", 16)
    raw_num_expert = cfg.get("num_expert_layers")
    if isinstance(layers, list):
        num_expert = len(layers)
    elif isinstance(raw_num_expert, int) and raw_num_expert > 0:
        num_expert = raw_num_expert
    else:
        num_expert = num_vlm
    return {
        "expert_vlm_layers": layers,
        "num_expert_layers": num_expert,
        "expert_intermediate_size": cfg.get("expert_intermediate_size"),
        "num_vlm_layers": num_vlm,
    }
