"""Run a queue of LIBERO success evals on a Colab VM with mandatory W&B logging.

Run it in the kernel, not over ssh, so Colab sees the session as active:

    colab exec -s briskvla --timeout 0 -f project/briskvla/colab_eval.py

Every job is one `run_eval.sh NAME POLICY` call with SUCCESS=1 LATENCY=0 (latency is measured on
the fixed RTX 3080). Jobs are split across EVAL_LANES detached lanes that survive disconnects.
Re-running skips running lanes and already-completed suites.

Required environment:
    EVAL_MODELS   `name=policy` entries separated by `;` (policy is "teacher" or a checkpoint dir),
                  for example:
                  EVAL_MODELS="teacher=teacher;student_8L_spread=/content/drive/MyDrive/briskvla/distill/student_8L_spread/best;shallow_8L_early=/content/drive/MyDrive/briskvla/distill/shallow_8L_early/best"
    WANDB_API_KEY required unless WANDB_MODE=offline (or ~/.netrc already has credentials)

Optional environment:
    BRISKVLA_BRANCH      git branch to sync (default: briskvla)
    BRISKVLA_DRIVE       Drive root folder (default: /content/drive/MyDrive/briskvla)
    EVAL_OUT             results go to $EVAL_OUT/eval (default: $BRISKVLA_DRIVE/outputs)
    EVAL_STEPS           denoising steps, each one a pass over all models (default: "10 3")
    EVAL_LANES           parallel lanes (default: 2)
    EVAL_SUITES          suites to evaluate (default: all four LIBERO suites)
    EVAL_EPISODES        episodes per task (default: 10)
    EVAL_N_ACTION_STEPS  actions before replanning (default: 10)
    EVAL_THREADS         OMP/MKL/OPENBLAS threads per process (default: 1)
    SKIP_SETUP           set to 1 to skip apt/git/uv setup
    WANDB_PROJECT        W&B project (default: briskVLA)
    WANDB_ENTITY         W&B entity (default: djonnala-carnegie-mellon-university)
    WANDB_GROUP          optional W&B group
"""

from __future__ import annotations

import glob
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
OUT = Path(os.environ.get("EVAL_OUT", str(DRIVE / "outputs")))

RAW_MODELS = os.environ.get("EVAL_MODELS", "")
MODELS = dict(entry.split("=", 1) for entry in RAW_MODELS.split(";") if entry.strip())
if not MODELS:
    raise SystemExit(
        "EVAL_MODELS must be set, e.g.:\n"
        '  EVAL_MODELS="teacher=teacher;'
        'student_8L_spread=/content/drive/MyDrive/briskvla/distill/student_8L_spread/best"'
    )

STEPS = os.environ.get("EVAL_STEPS", "10 3").split()
LANES = int(os.environ.get("EVAL_LANES", "2"))
SUITES = os.environ.get("EVAL_SUITES", "libero_spatial libero_object libero_goal libero_10")
SKIP_SETUP = os.environ.get("SKIP_SETUP", "0") == "1"
STATUS_EVERY = 300  # seconds

WANDB_MODE = os.environ.get("WANDB_MODE", "online")
if WANDB_MODE == "online" and not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").is_file():
    raise SystemExit("WANDB_API_KEY must be set in the environment (or set WANDB_MODE=offline).")

if not Path("/content/drive/MyDrive").is_dir():
    raise SystemExit("Google Drive is not mounted. Run `colab drivemount -s <session>` first.")

missing = [
    f"{name}={policy}"
    for name, policy in MODELS.items()
    if policy != "teacher" and not (Path(policy) / "model.safetensors").is_file()
]
if missing:
    raise SystemExit(f"Missing model.safetensors for: {missing}")

(OUT / "eval").mkdir(parents=True, exist_ok=True)


def sh(cmd: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=True)


if not SKIP_SETUP:
    sh("apt-get -qq install -y libegl1 libgl1 ffmpeg > /dev/null")
    # Colab ships NVIDIA's EGL library without a GLVND vendor JSON, so MUJOCO_GL=egl silently falls
    # back to llvmpipe CPU rendering (~3x slower). Register NVIDIA's ICD so MuJoCo renders on GPU.
    egl_vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    egl_vendor.parent.mkdir(parents=True, exist_ok=True)
    egl_vendor.write_text('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n')

    if shutil.which("uv") is None:
        sh(f"{sys.executable} -m pip install -q uv")

    if REPO.exists():
        sh(
            f"git -C {REPO} fetch -q origin && git -C {REPO} checkout -q {BRANCH} && git -C {REPO} pull -q --ff-only"
        )
    else:
        sh(f"git clone -q -b {BRANCH} {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log --oneline -1")
    sh("uv sync --locked --inexact --python 3.12 --extra briskvla --extra libero", cwd=REPO)

ENV = {
    **os.environ,
    "SUCCESS": "1",
    "LATENCY": "0",
    "SKIP_SETUP": "1",
    "SUITES": SUITES,
    "EPISODES": os.environ.get("EVAL_EPISODES", "10"),
    "N_ACTION_STEPS": os.environ.get("EVAL_N_ACTION_STEPS", "10"),
    "OUT": str(OUT),
    "CKPT": "/content/checkpoints",
    "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "briskVLA"),
    "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "djonnala-carnegie-mellon-university"),
    "WANDB_MODE": WANDB_MODE,
    # Prevent each env worker from spawning N_CPU BLAS threads and oversubscribing the VM.
    **{
        v: os.environ.get("EVAL_THREADS", "1")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    },
}

if not SKIP_SETUP:
    sh(
        "SUCCESS=0 LATENCY=0 bash project/briskvla/run_eval.sh setup teacher",
        cwd=REPO,
        env={**ENV, "SKIP_SETUP": "0"},
    )
sh("uv run --no-sync python project/briskvla/wandb_log.py check-auth", cwd=REPO, env=ENV)

jobs = [(k, name, policy) for k in STEPS for name, policy in MODELS.items()]
lanes = [jobs[i::LANES] for i in range(LANES)]


def lane_running(i: int) -> bool:
    return subprocess.run(["pgrep", "-f", f"BRISKVLA_EVAL_LANE={i} "], capture_output=True).returncode == 0


for i, lane in enumerate(lanes):
    if not lane:
        continue
    if lane_running(i):
        print(f"[skip] lane {i}: already running")
        continue
    calls = " ; ".join(
        f"STEPS={k} bash project/briskvla/run_eval.sh {name} {policy}" for k, name, policy in lane
    )
    log = OUT / f"eval_lane{i}.log"
    print(f"lane {i}: {[f'{name} K={k}' for k, name, _ in lane]}, log {log}", flush=True)
    # Trailing `true` prevents bash from exec-ing the last command and losing the pgrep marker.
    cmd = f"BRISKVLA_EVAL_LANE={i} ; {calls} ; true"
    with open(log, "a") as f:
        subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=REPO,
            env=ENV,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(30)


def status() -> str:
    done, active = [], []
    suite_list = SUITES.split()
    for k, name, _ in jobs:
        run_dir = OUT / "eval" / f"{name}_K{k}"
        finished = [s for s in suite_list if (run_dir / s / "eval_info.json").exists()]
        if len(finished) == len(suite_list):
            done.append(f"{name} K={k}")
        elif finished or (run_dir / "run_metadata.json").exists():
            current = next(s for s in suite_list if s not in finished)
            tasks = len(glob.glob(str(run_dir / current / "videos" / "*/")))
            active.append(
                f"{name} K={k} ({len(finished)}/{len(suite_list)} suites, {current} {tasks}/10 tasks)"
            )
    return f"done {len(done)}/{len(jobs)} | active: {', '.join(active) or '-'}"


while any(lane_running(i) for i in range(LANES)):
    print(time.strftime("%H:%M"), status(), flush=True)
    time.sleep(STATUS_EVERY)

print(time.strftime("%H:%M"), "no lane is running |", status(), flush=True)
subprocess.run(
    ["uv", "run", "--no-sync", "python", "project/briskvla/summarize_eval.py", "--root", str(OUT / "eval")],
    cwd=REPO,
)
