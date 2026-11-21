"""Run a queue of LIBERO success evals on a Colab VM and keep the kernel busy until they finish.

    colab exec -s briskvla --timeout 0 -f project/briskvla/colab_eval.py

Sets up the repo on first use, then runs project/briskvla/run_eval.sh for every job in JOBS, WORKERS
at a time. Results go to Drive. Finished suites are skipped, so after a disconnect or on a new VM just
run the same command again.
"""

import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPO = Path("/content/briskvla")
REPO_URL = "https://github.com/vyvas33/briskvla.git"
BRANCH = os.environ.get("EVAL_BRANCH", "haozh/distill")
DRIVE = Path("/content/drive/MyDrive/briskvla")
OUT = Path(os.environ.get("EVAL_OUT", str(DRIVE / "outputs_colab")))
SUITES = os.environ.get("EVAL_SUITES", "libero_object libero_goal libero_10")
WORKERS = int(os.environ.get("EVAL_WORKERS", "2"))  # the simulation is CPU bound, so only a few at once

STUDENTS = {
    **{
        f"k8_{layout}_distilled": DRIVE / "distill" / layout / "best"
        for layout in ("pairs", "first8", "last8")
    },
    **{
        f"k8_{layout}_nokd": DRIVE / "distill_nokd" / layout / "best"
        for layout in ("pairs", "first8", "last8")
    },
}
# Most important first: all placements at K=10, then the two leading layouts at K=3 and K=1.
# The remaining last8 runs (no-KD K=10, both at K=3 and K=1) run on the RTX 3080 instead.
JOBS = [(name, 10) for name in ("k8_pairs_distilled", "k8_first8_distilled", "k8_last8_distilled",
                                 "k8_pairs_nokd", "k8_first8_nokd")]  # fmt: skip
JOBS += [(name, k) for k in (3, 1) for name in ("k8_pairs_distilled", "k8_first8_distilled",
                                                 "k8_pairs_nokd", "k8_first8_nokd")]  # fmt: skip


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup() -> None:
    sh("apt-get -qq install -y libegl1 libgl1 ffmpeg > /dev/null")
    if shutil.which("uv") is None:
        sh("pip install -q uv")
    if REPO.exists():
        sh(
            f"git -C {REPO} fetch -q origin && git -C {REPO} checkout -q {BRANCH} && git -C {REPO} pull -q --ff-only"
        )
    else:
        sh(f"git clone -q -b {BRANCH} {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log --oneline -1")
    sh(f"cd {REPO} && uv sync --locked --inexact --python 3.12 --extra smolvla --extra libero")
    # Writes ~/.libero/config.yaml; with SUCCESS=0 and LATENCY=0 the script stops right after setup
    student = STUDENTS["k8_pairs_distilled"]
    sh(f"cd {REPO} && SUCCESS=0 LATENCY=0 PYTHON=3.12 bash project/briskvla/run_eval.sh setup {student}")


def run(name: str, k: int) -> None:
    env = {**os.environ, "SUCCESS": "1", "LATENCY": "0", "SKIP_SETUP": "1", "STEPS": str(k), "SUITES": SUITES,
           "N_ACTION_STEPS": "10", "OUT": str(OUT), "CKPT": "/content/checkpoints"}  # fmt: skip
    proc = subprocess.Popen(["bash", "project/briskvla/run_eval.sh", name, str(STUDENTS[name])], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    for line in proc.stdout:
        if line.startswith(("===", "Success rate ", "[skip]", "[failed]", "No model")):
            print(time.strftime("%H:%M"), f"[{name} K={k}]", line.rstrip(), flush=True)
    proc.wait()


def worker(jobs: queue.Queue) -> None:
    while True:
        try:
            name, k = jobs.get_nowait()
        except queue.Empty:
            return
        run(name, k)


if not Path("/content/drive/MyDrive").is_dir():
    raise SystemExit("Google Drive is not mounted. Run `colab drivemount -s <session>` first.")
missing = [str(p) for p in STUDENTS.values() if not (p / "model.safetensors").exists()]
if missing:
    raise SystemExit(f"Missing checkpoints: {missing}")
OUT.mkdir(parents=True, exist_ok=True)
setup()

jobs: queue.Queue = queue.Queue()
for job in JOBS:
    jobs.put(job)
print(f"{len(JOBS)} jobs x [{SUITES}], {WORKERS} at a time, results in {OUT}", flush=True)
threads = [threading.Thread(target=worker, args=(jobs,)) for _ in range(WORKERS)]
for t in threads:
    t.start()
    time.sleep(60)  # stagger model loading
for t in threads:
    t.join()

print(time.strftime("%H:%M"), "ALL DONE", flush=True)
sh(f"cd {REPO} && uv run --no-sync python project/briskvla/summarize_eval.py --root {OUT}/eval")
