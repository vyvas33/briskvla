"""Distill all layouts in parallel on a Colab VM and keep the kernel busy until they finish.

Run it in the kernel, not over ssh, so Colab sees the session as active:

    colab exec -s briskvla --timeout 0 -f project/briskvla/colab_distill.py

Each layout runs as its own process and writes to Drive. Running this again (after a disconnect, or
on a new VM after the old one was reclaimed) skips layouts that are finished or still running and
resumes the others from their last checkpoint.
"""

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/content/briskvla")
OUT = Path(os.environ.get("DISTILL_OUT", "/content/drive/MyDrive/briskvla/distill"))
LAYOUTS = os.environ.get("DISTILL_LAYOUTS", "pairs,first8,last8").split(",")
# Every layout gets the same arguments, so the runs can be compared
ARGS = os.environ.get("DISTILL_ARGS", "--steps 5000 --batch_size 8").split()
NUM_WORKERS = os.environ.get("DISTILL_WORKERS", "3")  # per layout; 3 layouts share the VM's CPUs
STATUS_EVERY = 300  # seconds


def is_running(layout: str) -> bool:
    pattern = f"distill_student.py --layout {layout} --out {OUT / layout}"
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

for layout in LAYOUTS:
    if is_finished(layout):
        print(f"[skip] {layout}: finished")
        continue
    if is_running(layout):
        print(f"[skip] {layout}: already running")
        continue
    cmd = ["uv", "run", "--no-sync", "python", "project/briskvla/distill_student.py", "--layout", layout,
           "--out", str(OUT / layout), "--num_workers", NUM_WORKERS, *ARGS]  # fmt: skip
    print("start:", " ".join(cmd))
    with open(OUT / f"{layout}.log", "a") as log:
        subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(20)  # stagger startup so the processes don't load the teacher at the same moment

# Wait here so the kernel stays busy. Stopping this loop does not stop the training processes.
while any(is_running(layout) for layout in LAYOUTS):
    print(time.strftime("%H:%M"), " | ".join(f"{layout}: {status(layout)}" for layout in LAYOUTS), flush=True)
    time.sleep(STATUS_EVERY)

print(time.strftime("%H:%M"), "no layout is running")
for layout in LAYOUTS:
    print(f"{layout}: {status(layout)}")
    if not is_finished(layout):
        print(f"  {layout} did not finish, see {OUT / layout}.log. Run this script again to resume it.")
