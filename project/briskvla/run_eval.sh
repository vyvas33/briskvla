#!/usr/bin/env bash
# Evaluate one model: inference latency, then LIBERO task success.
#
# Usage:
#   project/briskvla/run_eval.sh NAME POLICY
#
#   NAME    name for the output folders, e.g. teacher or k8_pairs_distilled
#   POLICY  "teacher" for lerobot/smolvla_libero at the evaluated revision, or a checkpoint folder
#
# Settings are environment variables. Defaults:
#   STEPS="10 3"          denoising steps to evaluate
#   SUITES="libero_spatial libero_object libero_goal libero_10"
#   EPISODES=10           episodes per task
#   BATCH=5 ASYNC=true SEED=1000
#   OUT=outputs           results go to $OUT/eval and $OUT/latency
#   CKPT=outputs/checkpoints
#   LATENCY=1             set to 0 to skip the latency measurement
#   SUCCESS=1             set to 0 to skip the LIBERO success evals
#   SKIP_SETUP=0          set to 1 to skip package sync and LIBERO setup
#   MUJOCO_GL=egl         use osmesa if egl does not work (e.g. under WSL)
#   PYTHON=               Python version for uv, e.g. 3.12 (empty keeps the current one)
#
# Example, quick check with one episode per task:
#   STEPS=10 SUITES=libero_spatial EPISODES=1 LATENCY=0 project/briskvla/run_eval.sh smoke teacher
#
# Finished suites are skipped, so the script can simply be rerun after a crash.
set -euo pipefail

NAME=${1:?usage: run_eval.sh NAME POLICY}
POLICY=${2:?usage: run_eval.sh NAME POLICY}

STEPS=${STEPS:-"10 3"}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
EPISODES=${EPISODES:-10}
BATCH=${BATCH:-5}
ASYNC=${ASYNC:-true}
SEED=${SEED:-1000}
OUT=${OUT:-outputs}
CKPT=${CKPT:-outputs/checkpoints}
LATENCY=${LATENCY:-1}
SUCCESS=${SUCCESS:-1}
SKIP_SETUP=${SKIP_SETUP:-0}
PYTHON=${PYTHON:-}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export MPLBACKEND=Agg

TEACHER=lerobot/smolvla_libero
TEACHER_REVISION=a24f1f380461b665158d52acd140152c79e8b36d
CAMERAS='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'

cd "$(dirname "$0")/../.."

if [[ "$SKIP_SETUP" != 1 ]]; then
  # --inexact keeps packages from other extras that are already installed
  uv sync --locked --inexact --extra smolvla --extra libero ${PYTHON:+--python "$PYTHON"}

  # hf-libero asks a question on first import if ~/.libero/config.yaml is missing, which would hang
  # the run. This writes it the same way as docker/Dockerfile.benchmark.libero. An existing config
  # is left alone.
  if [[ ! -f ~/.libero/config.yaml ]]; then
    echo "Setting up ~/.libero"
    uv run --no-sync python - <<'EOF'
import importlib.util
from pathlib import Path

from huggingface_hub import snapshot_download

home = Path.home() / ".libero"
home.mkdir(exist_ok=True)
snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset", local_dir=home / "assets")
libero_dir = Path(importlib.util.find_spec("libero").origin).parent / "libero"
(home / "config.yaml").write_text(
    f"assets: {home}/assets\n"
    f"bddl_files: {libero_dir}/bddl_files\n"
    f"datasets: {libero_dir}/../datasets\n"
    f"init_states: {libero_dir}/init_files\n"
)
EOF
  fi
fi

if [[ "$POLICY" == teacher ]]; then
  POLICY_DIR="$CKPT/smolvla_libero_${TEACHER_REVISION:0:8}"
  if [[ ! -f "$POLICY_DIR/model.safetensors" ]]; then
    uv run --no-sync hf download "$TEACHER" --revision "$TEACHER_REVISION" --local-dir "$POLICY_DIR"
  fi
else
  POLICY_DIR=$POLICY
fi
if [[ ! -f "$POLICY_DIR/model.safetensors" ]]; then
  echo "No model.safetensors in $POLICY_DIR" >&2
  exit 1
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
GPU=${GPU:-unknown}
GPU_TAG=$(echo "$GPU" | tr -c 'A-Za-z0-9\n' '_')
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)

echo "Model: $NAME ($POLICY_DIR)"
echo "GPU: $GPU"

# Latency first, while the GPU is idle and cool
LATENCY_FILE="$OUT/latency/${NAME}__${GPU_TAG}.json"
if [[ "$LATENCY" == 1 ]]; then
  if [[ -f "$LATENCY_FILE" ]]; then
    echo "[skip] latency already in $LATENCY_FILE"
  else
    uv run --no-sync python project/briskvla/latency.py --policy "$NAME=$POLICY_DIR" --out "$LATENCY_FILE"
  fi
fi

if [[ "$SUCCESS" != 1 ]]; then
  exit 0
fi

FAILED=()
for K in $STEPS; do
  RUN_DIR="$OUT/eval/${NAME}_K${K}"
  mkdir -p "$RUN_DIR"
  cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "name": "$NAME",
  "policy": "$POLICY",
  "policy_dir": "$POLICY_DIR",
  "teacher_revision": "$TEACHER_REVISION",
  "num_steps": $K,
  "suites": "$SUITES",
  "episodes_per_task": $EPISODES,
  "batch_size": $BATCH,
  "async_envs": $ASYNC,
  "seed": $SEED,
  "gpu": "$GPU",
  "git_commit": "$COMMIT",
  "mujoco_gl": "$MUJOCO_GL",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

  for SUITE in $SUITES; do
    SUITE_DIR="$RUN_DIR/$SUITE"
    if [[ -f "$SUITE_DIR/eval_info.json" ]]; then
      echo "[skip] $NAME K=$K $SUITE already done"
      continue
    fi
    # Remove what an interrupted run left behind
    rm -rf "$SUITE_DIR"
    echo "=== $NAME K=$K $SUITE ==="
    if ! uv run --no-sync lerobot-eval \
      --policy.path="$POLICY_DIR" \
      --policy.device=cuda \
      --policy.num_steps="$K" \
      --policy.empty_cameras=1 \
      --env.type=libero \
      --env.task="$SUITE" \
      --env.camera_name_mapping="$CAMERAS" \
      --eval.batch_size="$BATCH" \
      --eval.n_episodes="$EPISODES" \
      --eval.use_async_envs="$ASYNC" \
      --seed="$SEED" \
      --output_dir="$SUITE_DIR" 2>&1 | tee "$RUN_DIR/$SUITE.log"; then
      echo "[failed] $NAME K=$K $SUITE, log in $RUN_DIR/$SUITE.log"
      FAILED+=("K=$K $SUITE")
    fi
  done
done

uv run --no-sync python project/briskvla/summarize_eval.py --root "$OUT/eval"

if (( ${#FAILED[@]} )); then
  echo "Failed runs: ${FAILED[*]}" >&2
  exit 1
fi
