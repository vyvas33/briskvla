"""Train or distill BriskVLA models in parallel on a Colab VM with mandatory W&B logging.

Run it in the kernel, not over ssh, so Colab sees the session as active:

    colab exec -s briskvla --timeout 0 -f project/briskvla/colab_distill.py

Each run is its own process and writes to Drive. Re-running after a disconnect skips runs that are
finished or running and resumes interrupted ones from their last checkpoint.

Runs come from named layouts (DISTILL_LAYOUTS) and custom students built on the VM with
build_student.py (DISTILL_STUDENTS, `name=build args` separated by `;`), for example:

    DISTILL_LAYOUTS=""
    DISTILL_STUDENTS="student_8L_alternating_ffn_fused=--group 2 --keep alternate --ffn fuse;student_8L_alternating=--group 2 --keep alternate --ffn drop"

Built untrained checkpoints are saved once under $DISTILL_OUT/students/<name> and trained with --student.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/briskvla")
REPO_URL = "https://github.com/vyvas33/briskvla.git"
BRANCH = os.environ.get("BRISKVLA_BRANCH", "briskvla")
DRIVE = Path(os.environ.get("BRISKVLA_DRIVE", "/content/drive/MyDrive/briskvla"))
OUT = Path(os.environ.get("DISTILL_OUT", str(DRIVE / "distill")))

LAYOUTS = [x.strip() for x in os.environ.get("DISTILL_LAYOUTS", "spread,early,late").split(",") if x.strip()]
STUDENTS = dict(
    entry.split("=", 1) for entry in os.environ.get("DISTILL_STUDENTS", "").split(";") if entry.strip()
)
RUNS = LAYOUTS + list(STUDENTS)
if not RUNS:
    raise SystemExit("Nothing to run: set DISTILL_LAYOUTS and/or DISTILL_STUDENTS.")

ARGS = os.environ.get("DISTILL_ARGS", "--steps 5000 --batch_size 8").split()
NUM_WORKERS = os.environ.get("DISTILL_WORKERS", "3")
SKIP_SETUP = os.environ.get("SKIP_SETUP", "0") == "1"
STATUS_EVERY = 300  # seconds

WANDB_PROJECT = os.environ.get("WANDB_PROJECT") or os.environ.get("DISTILL_WANDB_PROJECT") or "briskVLA"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or "djonnala-carnegie-mellon-university"
WANDB_GROUP = os.environ.get("WANDB_GROUP") or os.environ.get("DISTILL_WANDB_GROUP")
WANDB_TAGS = os.environ.get("WANDB_TAGS")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")

if WANDB_MODE == "online" and not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").is_file():
    raise SystemExit("WANDB_API_KEY must be set in the environment (or set WANDB_MODE=offline).")

if not Path("/content/drive/MyDrive").is_dir():
    raise SystemExit("Google Drive is not mounted. Run `colab drivemount -s <session>` first.")
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd: str, cwd: Path | None = None) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


if not SKIP_SETUP:
    if shutil.which("uv") is None:
        sh(f"{sys.executable} -m pip install -q uv")
    if REPO.exists():
        sh(
            f"git -C {REPO} fetch -q origin && git -C {REPO} checkout -q {BRANCH} && git -C {REPO} pull -q --ff-only"
        )
    else:
        sh(f"git clone -q -b {BRANCH} {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log --oneline -1")
    sh("uv sync --locked --inexact --python 3.12 --extra briskvla", cwd=REPO)

sh("uv run --no-sync python project/briskvla/wandb_log.py check-auth", cwd=REPO)

WANDB_ARGS = [
    "--wandb_project",
    WANDB_PROJECT,
    "--wandb_entity",
    WANDB_ENTITY,
    *(["--wandb_mode", WANDB_MODE] if WANDB_MODE in ("online", "offline") else []),
    *(["--wandb_group", WANDB_GROUP] if WANDB_GROUP else []),
    *(["--wandb_tags", WANDB_TAGS] if WANDB_TAGS else []),
]


def _parse_role_from_args(args: list[str]) -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--w_velocity", type=float, default=1.0)
    p.add_argument("--w_hidden", type=float, default=0.1)
    p.add_argument("--w_output", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=5000)
    known, _ = p.parse_known_args(args)
    return (
        "student"
        if known.steps > 0 and (known.w_velocity > 0 or known.w_hidden > 0 or known.w_output > 0)
        else "shallow"
    )


sys.path.insert(0, str(REPO / "project" / "briskvla"))


def run_folder_name(entry: str) -> str:
    if entry in STUDENTS:
        return entry
    # Named layouts default to intuitive `student_8L_<layout>` / `shallow_8L_<layout>` folders
    # unless an existing folder under the bare layout name already exists on Drive.
    if (OUT / entry).exists():
        return entry
    from common import canonical_layout_name, format_model_name, resolve_layout

    canon = canonical_layout_name(entry)
    role = _parse_role_from_args(ARGS)
    return format_model_name(role, len(resolve_layout(canon)), canon)


RUN_MAP = [(entry, run_folder_name(entry)) for entry in RUNS]


def is_running(out_name: str) -> bool:
    pattern = f"distill_student.py .* --out {OUT / out_name} "
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def records(out_name: str) -> list[dict]:
    path = OUT / out_name / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_finished(out_name: str) -> bool:
    return any(r["split"] == "final" for r in records(out_name))


def status(out_name: str) -> str:
    rs = records(out_name)
    if any(r["split"] == "final" for r in rs):
        return "finished"
    step = max((r["step"] for r in rs), default=0)
    val = [r for r in rs if r["split"] == "val"]
    best = min((r["chunk_mse_teacher"] for r in val), default=float("nan"))
    state = "running" if is_running(out_name) else "STOPPED"
    return f"{state}, step {step}, best chunk_mse_teacher {best:.4f}"


def build(name: str) -> Path:
    """Build an untrained model once under `$DISTILL_OUT/students/<name>`."""
    folder = OUT / "students" / name
    if not (folder / "model.safetensors").exists():
        cmd = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "project/briskvla/build_student.py",
            *STUDENTS[name].split(),
            "--out",
            str(folder),
        ]
        print("build:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    return folder


for entry, out_name in RUN_MAP:
    if is_finished(out_name):
        print(f"[skip] {out_name}: finished")
        continue
    if is_running(out_name):
        print(f"[skip] {out_name}: already running")
        continue
    which = ["--student", str(build(entry))] if entry in STUDENTS else ["--layout", entry]
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "project/briskvla/distill_student.py",
        *which,
        "--out",
        str(OUT / out_name),
        "--num_workers",
        NUM_WORKERS,
        *ARGS,
        *WANDB_ARGS,
    ]
    print("start:", " ".join(cmd), flush=True)
    with open(OUT / f"{out_name}.log", "a") as log:
        subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(20)

while any(is_running(out_name) for _, out_name in RUN_MAP):
    print(
        time.strftime("%H:%M"),
        " | ".join(f"{out_name}: {status(out_name)}" for _, out_name in RUN_MAP),
        flush=True,
    )
    time.sleep(STATUS_EVERY)

print(time.strftime("%H:%M"), "no run is running")
for _, out_name in RUN_MAP:
    print(f"{out_name}: {status(out_name)}")
    if not is_finished(out_name):
        print(f"  {out_name} did not finish, see {OUT / out_name}.log. Run this script again to resume it.")
