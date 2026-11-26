#!/usr/bin/env bash
# Evaluate one BriskVLA model: inference latency, then LIBERO task success, with mandatory W&B logging.
#
# Usage:
#   project/briskvla/run_eval.sh NAME POLICY
#
#   NAME    model run name, e.g. teacher, student_8L_spread, student_8L_alternating_ffn_fused, shallow_8L_early
#   POLICY  "teacher", a local checkpoint folder, or a Hugging Face repo id (owner/repo[@rev])
#
# Settings are environment variables. Defaults:
#   STEPS="10 3"          denoising steps to evaluate
#   SUITES="libero_spatial libero_object libero_goal libero_10"
#   EPISODES=10           episodes per task
#   BATCH=5 ASYNC=true SEED=1000
#   N_ACTION_STEPS=10     actions executed before replanning (set to "checkpoint" to use the raw 50)
#   OUT=outputs           results go to $OUT/eval and $OUT/latency
#   CKPT=outputs/checkpoints
#   LATENCY=1             set to 0 to skip the latency measurement
#   SUCCESS=0             set to 1 to run the LIBERO success evals (Colab does this)
#   SKIP_SETUP=0          set to 1 to skip package sync and LIBERO setup
#   MUJOCO_GL=egl         use osmesa if egl does not work (e.g. under WSL)
#   PYTHON=               Python version for uv, e.g. 3.12 (empty keeps the current one)
#   WANDB_PROJECT=briskVLA
#   WANDB_ENTITY=djonnala-carnegie-mellon-university
#   WANDB_GROUP=          optional W&B group
#   WANDB_MODE=online     set to "offline" on air-gapped machines (sync later with wandb sync)
#
# The defaults measure latency only, which is what the RTX 3080 runs:
#   project/briskvla/run_eval.sh teacher teacher
#
# Example, quick success check with one episode per task:
#   SUCCESS=1 LATENCY=0 STEPS=10 SUITES=libero_spatial EPISODES=1 project/briskvla/run_eval.sh smoke teacher
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
N_ACTION_STEPS=${N_ACTION_STEPS:-10}
OUT=${OUT:-outputs}
CKPT=${CKPT:-outputs/checkpoints}
LATENCY=${LATENCY:-1}
SUCCESS=${SUCCESS:-0}
SKIP_SETUP=${SKIP_SETUP:-0}
SKIP_EXISTING_REGISTRY=${SKIP_EXISTING_REGISTRY:-1}
PUSH_EVAL_REGISTRY=${PUSH_EVAL_REGISTRY:-1}
EVAL_REGISTRY_REPO=${EVAL_REGISTRY_REPO:-briskVLA/eval-results}
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
  uv sync --locked --inexact --extra briskvla --extra libero ${PYTHON:+--python "$PYTHON"}

  # Colab ships NVIDIA's EGL library without a GLVND vendor JSON, causing llvmpipe CPU rendering.
  if [[ -f /usr/lib64-nvidia/libEGL_nvidia.so.0 && ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    if [[ -w /usr/share/glvnd/egl_vendor.d ]]; then
      printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json || true
    fi
  fi

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

# Verify W&B authentication upfront unless this is a setup-only invocation (LATENCY=0 and SUCCESS=0).
if [[ "$LATENCY" == 1 || "$SUCCESS" == 1 ]]; then
  uv run --no-sync python project/briskvla/wandb_log.py check-auth
fi

if [[ "$POLICY" == teacher ]]; then
  POLICY_DIR="$CKPT/smolvla_libero_${TEACHER_REVISION:0:8}"
  if [[ ! -f "$POLICY_DIR/model.safetensors" ]]; then
    uv run --no-sync hf download "$TEACHER" --revision "$TEACHER_REVISION" --local-dir "$POLICY_DIR"
  fi
elif [[ -d "$POLICY" ]]; then
  POLICY_DIR=$POLICY
else
  # Support Hugging Face repo IDs (owner/repo or owner/repo@rev) transparently.
  POLICY_DIR=$(uv run --no-sync python project/briskvla/hf_hub.py resolve "$POLICY")
fi
if [[ ! -f "$POLICY_DIR/model.safetensors" ]]; then
  echo "No model.safetensors in $POLICY_DIR" >&2
  exit 1
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
GPU=${GPU:-unknown}
GPU_TAG=$(echo "$GPU" | tr -c 'A-Za-z0-9\n' '_')
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
DIRTY=$(test -n "$(git status --porcelain 2>/dev/null)" && echo true || echo false)

if [[ "$N_ACTION_STEPS" == checkpoint ]]; then
  N_ACTION_STEPS_JSON='"checkpoint"'
  N_ACTION_STEPS_FLAG=""
else
  N_ACTION_STEPS_JSON="$N_ACTION_STEPS"
  N_ACTION_STEPS_FLAG="--policy.n_action_steps=$N_ACTION_STEPS"
fi

echo "Model: $NAME ($POLICY_DIR)"
echo "GPU: $GPU"

# Latency first, while the GPU is idle and cool
LATENCY_FILE="$OUT/latency/${NAME}__${GPU_TAG}.json"
if [[ "$LATENCY" == 1 ]]; then
  if [[ -f "$LATENCY_FILE" ]]; then
    echo "[skip] latency already in $LATENCY_FILE"
    uv run --no-sync python project/briskvla/wandb_log.py latency "$LATENCY_FILE"
  else
    uv run --no-sync python project/briskvla/latency.py --policy "$NAME=$POLICY_DIR" --out "$LATENCY_FILE"
  fi
fi

if [[ "$SUCCESS" != 1 ]]; then
  exit 0
fi

FAILED=()
for K in $STEPS; do
  if [[ "$SKIP_EXISTING_REGISTRY" == 1 && "${WANDB_MODE:-online}" != "offline" && "$SUITES" == "libero_spatial libero_object libero_goal libero_10" && "$N_ACTION_STEPS" =~ ^[0-9]+$ ]]; then
    if uv run --no-sync python project/briskvla/eval_registry.py check \
      --name "$NAME" \
      --steps "$K" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --n_action_steps "$N_ACTION_STEPS" \
      --repo "$EVAL_REGISTRY_REPO"; then
      echo "[skip] $NAME K=$K already recorded in $EVAL_REGISTRY_REPO (set SKIP_EXISTING_REGISTRY=0 to force)"
      continue
    fi
  fi

  RUN_DIR="$OUT/eval/${NAME}_K${K}"
  mkdir -p "$RUN_DIR"
  cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "name": "$NAME",
  "policy": "$POLICY",
  "policy_dir": "$POLICY_DIR",
  "teacher_revision": "$TEACHER_REVISION",
  "num_steps": $K,
  "n_action_steps": $N_ACTION_STEPS_JSON,
  "suites": "$SUITES",
  "episodes_per_task": $EPISODES,
  "batch_size": $BATCH,
  "async_envs": $ASYNC,
  "seed": $SEED,
  "gpu": "$GPU",
  "git_commit": "$COMMIT",
  "git_branch": "$BRANCH",
  "git_dirty": $DIRTY,
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
    echo "=== $NAME K=$K $SUITE, started $(date +%H:%M) ==="
    # Output goes only to the log file. The progress bar redraws every step, which freezes Colab.
    if uv run --no-sync lerobot-eval \
      --policy.path="$POLICY_DIR" \
      --policy.device=cuda \
      --policy.num_steps="$K" \
      ${N_ACTION_STEPS_FLAG:+"$N_ACTION_STEPS_FLAG"} \
      --policy.empty_cameras=1 \
      --env.type=libero \
      --env.task="$SUITE" \
      --env.camera_name_mapping="$CAMERAS" \
      --eval.batch_size="$BATCH" \
      --eval.n_episodes="$EPISODES" \
      --eval.use_async_envs="$ASYNC" \
      --seed="$SEED" \
      --output_dir="$SUITE_DIR" > "$RUN_DIR/$SUITE.log" 2>&1; then
      echo "$(grep -ao 'Success rate[^)]*)' "$RUN_DIR/$SUITE.log" | tail -n 1), finished $(date +%H:%M)"
    else
      echo "[failed] $NAME K=$K $SUITE, log in $RUN_DIR/$SUITE.log"
      FAILED+=("K=$K $SUITE")
    fi
  done

  for f in "$RUN_DIR"/*/eval_info.json; do
    if [[ -f "$f" ]]; then
      uv run --no-sync python project/briskvla/wandb_log.py eval "$RUN_DIR" --force
      if [[ "$PUSH_EVAL_REGISTRY" == 1 && "${WANDB_MODE:-online}" != "offline" ]]; then
        uv run --no-sync python project/briskvla/eval_registry.py record "$RUN_DIR" --repo "$EVAL_REGISTRY_REPO" \
          || echo "[warn] failed to push eval result to $EVAL_REGISTRY_REPO"
      fi
      break
    fi
  done
done

uv run --no-sync python project/briskvla/summarize_eval.py --root "$OUT/eval"

if (( ${#FAILED[@]} )); then
  echo "Failed runs: ${FAILED[*]}" >&2
  exit 1
fi
