"""Distill all layouts in parallel on a Colab VM and keep the kernel busy until they finish.

Run it in the kernel, not over ssh, so Colab sees the session as active:

    colab exec -s briskvla --timeout 0 -f project/briskvla/colab_distill.py

Each run is its own process and writes to Drive. Running this again (after a disconnect, or on a new
VM after the old one was reclaimed) skips runs that are finished or still running and resumes the
others from their last checkpoint.

Runs are named layouts from distill_student.py (DISTILL_LAYOUTS) and students built on the VM with
build_student.py (DISTILL_STUDENTS, `name=build args` separated by `;`), for example

    DISTILL_LAYOUTS=pairs
    DISTILL_STUDENTS="k8_fuse=--group 2 --keep alternate --ffn fuse;k8_drop=--group 2 --keep alternate --ffn drop"

Built students are saved once under $DISTILL_OUT/students/<name> and distilled with --student.
"""

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/content/briskvla")
OUT = Path(os.environ.get("DISTILL_OUT", "/content/drive/MyDrive/briskvla/distill"))
LAYOUTS = [x for x in os.environ.get("DISTILL_LAYOUTS", "pairs,first8,last8").split(",") if x]
STUDENTS = dict(
    entry.split("=", 1) for entry in os.environ.get("DISTILL_STUDENTS", "").split(";") if entry.strip()
)
RUNS = LAYOUTS + list(STUDENTS)
# Every layout gets the same arguments, so the runs can be compared
ARGS = os.environ.get("DISTILL_ARGS", "--steps 5000 --batch_size 8").split()
NUM_WORKERS = os.environ.get("DISTILL_WORKERS", "3")  # per run; the runs share the VM's CPUs
STATUS_EVERY = 300  # seconds


def is_running(layout: str) -> bool:
    pattern = f"distill_student.py .* --out {OUT / layout} "
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def records(layout: str) -> list[dict]:
    path = OUT / layout / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_finished(layout: str) -> bool:
    return any(r["split"] == "final" for r in records(layout))


def status(layout: str) -> str:
    rs = records(layout)
    if any(r["split"] == "final" for r in rs):
        return "finished"
    step = max((r["step"] for r in rs), default=0)
    val = [r for r in rs if r["split"] == "val"]
    best = min((r["chunk_mse_teacher"] for r in val), default=float("nan"))
    state = "running" if is_running(layout) else "STOPPED"
    return f"{state}, step {step}, best chunk_mse_teacher {best:.4f}"


if not Path("/content/drive/MyDrive").is_dir():
    raise SystemExit("Google Drive is not mounted. Run `colab drivemount -s <session>` first.")
OUT.mkdir(parents=True, exist_ok=True)

def build(name: str) -> Path:
    """Build a student once. A folder with model.safetensors is complete (save_pretrained writes
    config.json first), so an interrupted build is redone."""
    folder = OUT / "students" / name
    if not (folder / "model.safetensors").exists():
        cmd = ["uv", "run", "--no-sync", "python", "project/briskvla/build_student.py",
               *STUDENTS[name].split(), "--out", str(folder)]  # fmt: skip
        print("build:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    return folder


for layout in RUNS:
    if is_finished(layout):
        print(f"[skip] {layout}: finished")
        continue
    if is_running(layout):
        print(f"[skip] {layout}: already running")
        continue
    which = ["--student", str(build(layout))] if layout in STUDENTS else ["--layout", layout]
    cmd = ["uv", "run", "--no-sync", "python", "project/briskvla/distill_student.py", *which,
           "--out", str(OUT / layout), "--num_workers", NUM_WORKERS, *ARGS]  # fmt: skip
    print("start:", " ".join(cmd))
    with open(OUT / f"{layout}.log", "a") as log:
        subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(20)  # stagger startup so the processes don't load the teacher at the same moment

# Wait here so the kernel stays busy. Stopping this loop does not stop the training processes.
while any(is_running(layout) for layout in RUNS):
    print(time.strftime("%H:%M"), " | ".join(f"{layout}: {status(layout)}" for layout in RUNS), flush=True)
    time.sleep(STATUS_EVERY)

print(time.strftime("%H:%M"), "no run is running")
for layout in RUNS:
    print(f"{layout}: {status(layout)}")
    if not is_finished(layout):
        print(f"  {layout} did not finish, see {OUT / layout}.log. Run this script again to resume it.")
