# BriskVLA Action-Expert Compression

This folder builds, trains, evaluates, and profiles reduced-depth SmolVLA action experts for BriskVLA. All models keep the teacher's VLM unchanged and replace the 16-layer action expert with fewer layers (and optionally fused parallel FFNs).

Teacher: `lerobot/smolvla_libero` at revision `a24f1f380461b665158d52acd140152c79e8b36d`, the revision used for the LIBERO teacher evaluation.

## Naming Convention

Every model and experiment name follows one self-explanatory format so an unfamiliar reader immediately knows whether teacher distillation was used, how many expert layers exist, where they sit, and what modification was applied:

`teacher` or `<role>_<layers>L_<placement>[_<modification>][_<variant>]`

- **`student`** = any reduced-depth model trained with **teacher distillation** (velocity, hidden-state, and/or output-chunk losses).
- **`shallow`** = any reduced-depth model trained **without distillation** (ground-truth flow matching only, or `untrained` copied/merged weights).
- **`<layers>L`** = expert layer count (`8L`, `4L`) — kept distinct from denoising steps (`K10`, `K3`) in eval folders (`<model>_K<steps>`).
- **`<placement>`** = canonical VLM depth layout (`spread`, `early`, `late`, `alternating`).
- **`<modification>`** = optional compression method (`ffn_fused`; plain layer dropping omits this).

| Canonical Name | Role | Expert Layers | Depth Layout | Notes |
|---|---|---|---|---|
| `teacher` | Teacher | `16L` (98.2M) | `0..15` | Pretrained SmolVLA baseline |
| `student_8L_spread` | Distilled student | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | Legacy layout alias: `pairs` |
| `student_8L_early` | Distilled student | `8L` (49.1M) | `0,1,2,3,4,5,6,7` | Legacy layout alias: `first8` |
| `student_8L_late` | Distilled student | `8L` (49.1M) | `8,9,10,11,12,13,14,15` | Legacy layout alias: `last8` |
| `student_8L_alternating` | Distilled student | `8L` (49.1M) | `0,3,4,7,8,11,12,15` | Matched drop control for FFN fusion (`alt`) |
| `student_8L_alternating_ffn_fused` | Distilled student | `8L` (84.5M) | `0,3,4,7,8,11,12,15` | Merged pairs with parallel 4096-wide FFNs |
| `student_8L_spread_shallow_pi` | Distilled student | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | Teacher velocity + hidden + GT (no output loss) |
| `shallow_8L_spread` / `early` / `late` | Shallow (no KD) | `8L` (49.1M) | same as above | Ground-truth flow matching only (`w_gt=1`) |
| `shallow_8L_spread_untrained` | Shallow (untrained) | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | Zero-shot copied weights before training |

## Files

| File | Purpose |
|---|---|
| `src/lerobot/policies/smolvla/configuration_smolvla.py` | Adds `expert_vlm_layers` and `expert_intermediate_size` config fields and validation |
| `src/lerobot/policies/smolvla/modeling_smolvla.py` | Passes `expert_vlm_layers` and `expert_intermediate_size` to the VLM + expert model |
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | Places expert layers at chosen depths, supports wider fused FFNs, records expert hidden states |
| `project/briskvla/common.py` | Shared constants, layout aliases, role/model naming helpers, git/GPU/arch metadata |
| `project/briskvla/hf_hub.py` | Hugging Face Hub upload/resolve/list helpers for team model sharing |
| `project/briskvla/wandb_log.py` | Mandatory W&B logging for distillation, LIBERO eval, and latency + backfill CLI |
| `project/briskvla/build_student.py` | Builds an untrained reduced-depth checkpoint: drops layers (`--layout`/`--layers`) or merges pairs with FFN fusion (`--group`) |
| `project/briskvla/distill_student.py` | Trains/distills a `student` (with KD) or `shallow` (GT-only) model on `lerobot/libero` |
| `project/briskvla/run_ablation.py` | Layer ablation: trains one model per layout, evaluates via `run_eval.sh`, writes summary |
| `project/briskvla/latency.py` | Inference latency per action chunk, split into prefix, denoising, and FFN breakdown |
| `project/briskvla/run_eval.sh` | Standardized latency and LIBERO success runner (`N_ACTION_STEPS=10`, W&B logging) |
| `project/briskvla/summarize_eval.py` | Success table for all evaluated models under `<root>` |
| `project/briskvla/colab_distill.py` | Runs parallel training/distillation jobs on Colab with auto-setup, Drive checkpoints, and W&B |
| `project/briskvla/colab_eval.py` | Runs parallel LIBERO success eval lanes on Colab with NVIDIA EGL GPU fix and W&B |
| `project/briskvla/eval_colab.ipynb` | Interactive Colab notebook wrapper around `run_eval.sh` (supersedes the old `vyvas/libero-eval` notebook) |
| `project/briskvla/plots.py` | Generates report PDF/PNG figures into `project/briskvla/figures/` (gitignored) |
| `project/briskvla/test_wandb.py` | Unit tests for `common.py` naming helpers and W&B logging (CPU-only) |
| `project/briskvla/test_hf_hub.py` | Unit tests for `hf_hub.py` repo parsing, staging, auth checks, and uploads (CPU-only) |
| `project/briskvla/test_student.py` | Tests for reduced-depth student checkpoint construction |
| `project/briskvla/test_distill.py` | Tests for distillation losses, shared KV cache, and checkpoint resume |
| `project/briskvla/test_fusion.py` | Tests for layer merging and exact FFN fusion |

## Setup

```bash
uv sync --locked --python 3.12 --extra briskvla                  # training / distillation / latency / plotting
uv sync --locked --python 3.12 --extra briskvla --extra libero   # + LIBERO simulator eval (Linux only)
uv run wandb login                                               # required before running experiments
```

Use Python 3.12 (with 3.14, `draccus` fails to parse the policy config). Install system `ffmpeg` (`sudo apt install ffmpeg`) so `torchcodec` loads without falling back to slower `pyav` decoding.

## Expert Layout

SmolVLA runs its VLM and action expert as two stacks of 16 layers. At depth `i` the forward pass runs VLM layer `i` and, if present, the expert layer paired with it. Even depths use joint attention and odd depths use cross-attention (decided by depth, not by the expert layer's own index).

`expert_vlm_layers` lists the depths that host an expert layer. Expert layer `j` sits at depth `expert_vlm_layers[j]` and reads that VLM layer's keys and values.

```
expert_vlm_layers = [0, 1, 4, 5, 8, 9, 12, 13]   # spread (legacy alias: pairs)

depth    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
VLM      L0  L1  L2  L3  L4  L5  L6  L7  L8  L9  L10 L11 L12 L13 L14 L15
expert   E0  E1  -   -   E2  E3  -   -   E4  E5  -   -   E6  E7  -   -
```

All VLM layers always run. At depths without an expert layer, the action tokens' hidden state passes through unchanged. `None` (the default) places an expert layer at every depth (the teacher).

### Why not `num_expert_layers`

Upstream SmolVLA's `num_expert_layers=k` is broken for `k=8`: it places every expert layer at an even depth while giving cross-attention projection shapes to odd expert indices, crashing on the first forward pass. `SmolVLMWithExpertModel` now raises an error if `num_expert_layers > 0` is set; use `expert_vlm_layers` instead.

## Building an Untrained Shallow Model & FFN Fusion

Build an untrained reduced-depth checkpoint by copying teacher layers (`--layout` or `--layers`):

```bash
uv run python project/briskvla/build_student.py \
    --layout spread --out outputs/students/shallow_8L_spread_untrained
```

Or merge every `g` consecutive expert layers into one layer (`--group g`):

- `--keep` picks which member's attention survives (`first`, `last`, or `alternate`: first in even groups, last in odd groups).
- `--ffn fuse` folds each member's post-attention RMSNorm gain into its SwiGLU `gate`/`up` weights and stacks all members' hidden units in parallel (width `2048 -> 4096` for pairs, exact in fp32 up to one bf16 rounding).
- `--ffn drop` keeps only the kept member's FFN (matched drop baseline with identical attention depths).
- `--source` merges an earlier checkpoint instead of the teacher (e.g. `16L -> 8L -> 4L`).

```bash
uv run python project/briskvla/build_student.py \
    --group 2 --keep alternate --ffn fuse \
    --out outputs/students/shallow_8L_alternating_ffn_fused_untrained
```

`latency.py --breakdown` on an RTX 5080 Laptop GPU (batch 1, `K=10`; report numbers come from the RTX 3080) showed FFNs are only ~10–12% of expert denoising time, so doubling FFN width (`2048 -> 4096`) keeps nearly all of the ~1.9× denoising speedup of halving depth (`48.5 ms` vs `47.8 ms` dropped, vs `91.1 ms` teacher) while retaining 86% of expert parameters (`84.5M` vs `49.1M`).

## Candidate Layouts

| Canonical Name (Aliases) | `--layers` | VLM layers read | VLM layers skipped by expert |
|---|---|---|---|
| `spread` (`pairs`) | `0,1,4,5,8,9,12,13` | spread over depth (4 joint, 4 cross) | `2, 3, 6, 7, 10, 11, 14, 15` |
| `early` (`first8`) | `0,1,2,3,4,5,6,7` | early half (4 joint, 4 cross) | `8–15` |
| `late` (`last8`) | `8,9,10,11,12,13,14,15` | deep half (4 joint, 4 cross) | `0–7` |
| `alternating` (`alt`, `alternate`) | `0,3,4,7,8,11,12,15` | first of even pairs, last of odd (4 joint, 4 cross) | `1, 2, 5, 6, 9, 10, 13, 14` |

## Training & Distillation

```bash
# Distilled student (teacher velocity + hidden + output losses):
uv run python project/briskvla/distill_student.py \
    --layout spread --out outputs/distill/student_8L_spread

# Non-distilled shallow baseline (ground-truth flow matching only):
uv run python project/briskvla/distill_student.py \
    --layout early --w_velocity 0 --w_hidden 0 --w_output 0 --w_gt 1 \
    --out outputs/distill/shallow_8L_early

# Distill a pre-built FFN-fused checkpoint:
uv run python project/briskvla/distill_student.py \
    --student outputs/students/shallow_8L_alternating_ffn_fused_untrained \
    --hidden_map uniform --out outputs/distill/student_8L_alternating_ffn_fused
```

### Losses

| Loss | Flag (default) | What it matches |
|---|---|---|
| Velocity distillation | `--w_velocity` (`1.0`) | Student velocity vs teacher velocity at the same `(x_t, t)` and prefix |
| Intermediate layer alignment | `--w_hidden` (`0.1`) | Layer-normalized expert hidden states (`same_depth` or `uniform` via `--hidden_map`) |
| Output-level distillation | `--w_output` (`1.0`) | Action chunk sampled by student (`--student_num_steps`, default 10) vs teacher |
| Ground-truth flow matching | `--w_gt` (`0.0`) | SmolVLA's original loss against dataset actions |

Teacher and student share the identical frozen VLM, so the prefix KV cache is computed once per batch (`encode_prefix`) and reused. Only expert layers, action projections, and time MLP receive gradients.

### Validation, Outputs, and Resuming

5% of episodes (`--val_fraction 0.05`, seed 1000) are held out. Every `--eval_freq` steps (and step 0), `chunk_mse_teacher`, `chunk_mse_gt`, `teacher_chunk_mse_gt`, and held-out losses are computed on `--val_batches` (25) fixed-noise batches:

- `best/`: lowest held-out `chunk_mse_teacher` checkpoint (saved atomically).
- `last/`: latest checkpoint plus `training_state.pt` (optimizer, scheduler, RNG states) for seamless resume after preemption.
- `metrics.jsonl` and `run_info.json`: full training trace, architecture, git commit/branch/dirty state, GPU, and role (`student` vs `shallow`).
- `wandb_id.txt`: persisted W&B run ID so resumed runs continue the exact same W&B run.

## Hugging Face Hub Model Sharing

All BriskVLA scripts support loading and auto-publishing checkpoints on Hugging Face Hub (`owner/repo` or `owner/repo@rev`), so teammates can share, evaluate, distill, or iteratively compress any model without Google Drive mounts:

1. **Authenticate once** with a **Write** token (`uv run hf auth login` or set `HF_TOKEN`), and set your team namespace (e.g. `export BRISKVLA_HF_NAMESPACE=<org-or-user>`).
2. **Auto-upload on training completion**:
   ```bash
   # Distill/train locally or on Colab SSD and push best/ + provenance (run_info.json, wandb_id.txt, metrics.jsonl) to HF Hub:
   BRISKVLA_HF_NAMESPACE=<org> uv run python project/briskvla/distill_student.py \
       --student outputs/students/shallow_8L_alternating_ffn_fused_untrained \
       --w_velocity 0 --w_hidden 0 --w_output 0 --w_gt 1 \
       --out outputs/distill/shallow_8L_alternating_ffn_fused
   ```
   Pass `--hf_repo <org>/<name>` to override the target repo, or `--no_hf_push` to skip uploading. On Colab, setting `BRISKVLA_HF_NAMESPACE` makes Google Drive optional (`colab_distill.py` trains on fast local SSD and pushes `best/` to HF Hub).
3. **Upload an existing checkpoint folder / list team models**:
   ```bash
   uv run python project/briskvla/hf_hub.py upload outputs/distill/shallow_8L_alternating_ffn_fused --namespace <org>
   uv run python project/briskvla/hf_hub.py list --namespace <org>
   ```
4. **Mix, match, evaluate, or compress directly from HF Hub**:
   ```bash
   SUCCESS=1 LATENCY=0 STEPS="10 1" project/briskvla/run_eval.sh shallow_8L_alternating_ffn_fused <org>/shallow_8L_alternating_ffn_fused
   uv run python project/briskvla/latency.py --policy shallow_8L_alternating_ffn_fused=<org>/shallow_8L_alternating_ffn_fused --out outputs/latency/rtx3080.json
   uv run python project/briskvla/build_student.py --source <org>/shallow_8L_alternating_ffn_fused --group 2 --keep alternate --ffn fuse --out outputs/students/shallow_4L_alternating_ffn_fused_untrained
   uv run python project/briskvla/distill_student.py --student <org>/shallow_8L_alternating_ffn_fused --out outputs/distill/student_8L_alternating_ffn_fused_kd
   ```

## Weights & Biases (Mandatory)

W&B logging is enabled by default for every run (`entity=djonnala-carnegie-mellon-university`, `project=briskVLA`).

1. **Authenticate once** (`uv run wandb login`) or set `WANDB_API_KEY`. On air-gapped machines, pass `--wandb_mode offline` (or `WANDB_MODE=offline`) and sync later with `wandb sync`. Online scripts fail immediately at startup if credentials are missing.
2. **Automatic live logging**:
   - `distill_student.py` (`job_type="distill"`): logs `train/*`, `val/*`, `val/best_chunk_mse_teacher`, `val/best_step`, and training-device latency under `train_device/latency_ms_*` (kept separate from RTX 3080 latency benchmarks).
   - `run_eval.sh` (`job_type="eval"`): logs per-suite `eval/<suite>/*`, `eval/avg_success`, and the `eval/per_task` table, linked automatically to the checkpoint's `distill_wandb_id` when available.
   - `latency.py` (`job_type="latency"`): logs one run per benchmarked model against `K`, flat `latency/K<k>/*` summaries, `flops/*_gflops`, linear-fit slopes, and `latency/ffn_width_sweep`.
3. **Backfill historical runs**:

```bash
uv run python project/briskvla/wandb_log.py distill --group <group> outputs/distill/...
uv run python project/briskvla/wandb_log.py eval --group <group> outputs/eval/
uv run python project/briskvla/wandb_log.py latency --group <group> outputs/latency/*.json
```

## Layer Ablation

`run_ablation.py` trains one model per layout with identical settings, evaluates via `run_eval.sh`, and writes `<out_root>/summary.{md,json}`:

```bash
uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/layers \
    --eval_suites libero_spatial --eval_teacher -- --steps 5000 --batch_size 8
```

## Tests

```bash
# Fast CPU unit tests for naming helpers & W&B logging:
uv run --with pytest python -m pytest project/briskvla/test_wandb.py -sv

# Model, FFN fusion, and distillation tests (downloads teacher on first use; no dataset needed):
WANDB_MODE=disabled uv run --with pytest python -m pytest \
    project/briskvla/test_student.py project/briskvla/test_fusion.py project/briskvla/test_distill.py -sv
```

## Evaluation, Colab, Latency, and Figures

Every model is evaluated with `run_eval.sh` (`N_ACTION_STEPS=10`, `BATCH=5`, `ASYNC=true`, `SEED=1000`, `EPISODES=10`, `STEPS="10 3"`):

```bash
# RTX 3080 latency only (default):
project/briskvla/run_eval.sh teacher teacher
project/briskvla/run_eval.sh student_8L_alternating_ffn_fused path/to/checkpoint

# LIBERO success evaluation:
SUCCESS=1 LATENCY=0 project/briskvla/run_eval.sh student_8L_spread path/to/checkpoint
```

### Colab runners

- **Parallel training (`colab_distill.py`)**:
  ```bash
  BRISKVLA_BRANCH=briskvla WANDB_GROUP=k8-ablation \
      colab exec -s briskvla --timeout 0 -f project/briskvla/colab_distill.py
  ```
- **Parallel LIBERO evaluation (`colab_eval.py`)**:
  ```bash
  EVAL_MODELS="teacher=teacher;student_8L_spread=/content/drive/MyDrive/briskvla/distill/student_8L_spread/best" \
      WANDB_GROUP=k8-libero colab exec -s briskvla --timeout 0 -f project/briskvla/colab_eval.py
  ```
- **Single-model interactive notebook**: open `project/briskvla/eval_colab.ipynb` (replaces the earlier `vyvas/libero-eval` notebook).

### Latency & Report Figures

```bash
uv run python project/briskvla/latency.py \
    --policy teacher=lerobot/smolvla_libero \
    --policy student_8L_spread=outputs/distill/student_8L_spread/best \
    --out outputs/latency/rtx3080.json

uv run python project/briskvla/plots.py   # writes PDF/PNG figures to project/briskvla/figures/ (gitignored)
```

## Results

Every trained model has 8 expert layers (`8L`), trained for 5000 steps at batch size 8, then evaluated with `run_eval.sh` (`seed=1000`, 10 episodes/task = 100 episodes/suite, `n_action_steps=10`). Values are LIBERO success rates (%).

| Variant | Role prefix | Loss weights (`velocity / hidden / output / gt`) |
|---|---|---|
| Distilled | `student` | `1 / 0.1 / 1 / 0` (defaults) |
| Shallow (no KD) | `shallow` | `0 / 0 / 0 / 1` (ground-truth flow matching only) |
| Shallow-π | `student` | `1 / 0.1 / 0 / 1` (teacher velocity & hidden + GT, no output loss) |

### Placement & Distillation vs Non-Distilled Shallow (`hidden_map=same_depth`)

| Model | K | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| Teacher (`16L`, 98.2M) | 10 | 79 | 91 | 89 | 69 | **82.0** |
| Student — 8L Early (distilled) | 10 | 77.3 | 86.3 | 83.0 | 66.3 | **78.3** |
| Student — 8L Spread (distilled) | 10 | 75.3 | 82.7 | 85.0 | 70.0 | **78.3** |
| Student — 8L Late (distilled) | 10 | 67 | 82 | 78 | 56 | 70.8 |
| Shallow — 8L Early (no KD) | 10 | 73.7 | 86.3 | 74.7 | 67.3 | 75.5 |
| Shallow — 8L Spread (no KD) | 10 | 63.0 | 80.3 | 79.3 | 64.0 | 71.7 |
| Shallow — 8L Late (no KD) | 10 | 63 | 66 | 75 | 48 | 63.0 |
| Teacher (`16L`, 98.2M) | 3 | 76 | 90 | 89 | 71 | **81.5** |
| Student — 8L Early (distilled) | 3 | 69.0 | 45.7 | 52.0 | 49.0 | 53.9 |
| Student — 8L Spread (distilled) | 3 | 53.7 | 63.7 | 69.3 | 49.0 | 58.9 |
| Student — 8L Late (distilled) | 3 | 53 | 66 | 69 | 45 | 58.3 |
| Shallow — 8L Early (no KD) | 3 | 81.0 | 85.3 | 75.3 | 73.0 | **78.7** |
| Shallow — 8L Spread (no KD) | 3 | 68.7 | 86.0 | 77.3 | 68.3 | 75.1 |
| Shallow — 8L Late (no KD) | 3 | 56 | 71 | 77 | 47 | 62.8 |
| Student — 8L Early (distilled) | 1 | 88 | 73 | 73 | 59 | 73.2 |
| Student — 8L Spread (distilled) | 1 | 70 | 74 | 78 | 57 | 69.8 |
| Student — 8L Late (distilled) | 1 | 59 | 74 | 69 | 46 | 62.0 |
| Shallow — 8L Early (no KD) | 1 | 78 | 81 | 81 | 66 | 76.5 |
| Shallow — 8L Spread (no KD) | 1 | 66 | 81 | 74 | 66 | 71.8 |
| Shallow — 8L Late (no KD) | 1 | 57 | 67 | 60 | 52 | 59.0 |

### FFN Fusion (`hidden_map=uniform`)

| Model | Expert params | K | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| Student — 8L Alternating + FFN Fusion | 84.5M | 10 | 76 | 83 | 85 | 61 | **76.3** |
| Student — 8L Alternating (matched drop) | 49.1M | 10 | 70 | 81 | 82 | 52 | 71.3 |
| Student — 8L Spread | 49.1M | 10 | 69 | 81 | 81 | 73 | 76.0 |
| Student — 8L Alternating + FFN Fusion | 84.5M | 3 | 62 | 73 | 71 | 44 | 62.5 |
| Student — 8L Alternating (matched drop) | 49.1M | 3 | 69 | 63 | 71 | 45 | 62.0 |
| Student — 8L Spread | 49.1M | 3 | 51 | 72 | 66 | 42 | 57.8 |

### Shallow-π (`hidden_map=same_depth`)

| Model | K | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| Student — 8L Early, Shallow-π | 10 | 67 | 79 | 74 | 69 | 72.2 |
| Student — 8L Spread, Shallow-π | 10 | 63 | 84 | 74 | 66 | 71.8 |
| Student — 8L Early, Shallow-π | 3 | 74 | 90 | 77 | 69 | **77.5** |
| Student — 8L Spread, Shallow-π | 3 | 72 | 82 | 74 | 62 | 72.5 |

### Shallow-π `libero_spatial` (incl. Late and `K=1`) & Untrained Baselines

| Model | K = 10 | K = 3 | K = 1 |
|---|---|---|---|
| `student_8L_spread_shallow_pi` | 63 | 72 | 64 |
| `student_8L_early_shallow_pi` | 67 | 74 | 79 |
| `student_8L_late_shallow_pi` | 55 | 56 | 60 |
| `shallow_8L_spread_untrained` / `early` / `late` | 3 / 0 / 0 | — | — |
