# BriskVLA

CMU 10-623 course project built on LeRobot + SmolVLA, reducing inference latency on LIBERO through shallower action experts, FFN fusion, and fewer denoising steps.
Original LeRobot README: [README_LEROBOT.md](README_LEROBOT.md)

## Setup (Linux + NVIDIA GPU; LIBERO doesn't install on macOS)

```bash
git clone https://github.com/vyvas33/briskvla.git
cd briskvla
git switch briskvla
uv sync --locked --python 3.12 --extra briskvla --extra libero   # omit --extra libero on macOS or distill-only VMs
uv run wandb login
uv run hf auth login
export MUJOCO_GL=egl   # headless servers
```

Use Python 3.12 (required for `draccus` config parsing) and install system `ffmpeg` (`sudo apt install ffmpeg`) for fast video decoding.

## Branch Workflow (Common Ground)

- `main`: untouched copy of `huggingface/lerobot`. Do not commit here.
- `briskvla`: **our single shared common ground**. All standardized code, W&B instrumentation, and verified methods live here.
- Personal experiments: always branch from `briskvla` (`git switch -c <name>/<topic> origin/briskvla`), then merge back via PR to `briskvla`.

## Team Standards

1. **Mandatory W&B logging**: Every training, distillation, LIBERO eval, and latency run logs automatically to Weights & Biases (`entity: djonnala-carnegie-mellon-university`, `project: briskVLA`). Runs fail fast if unauthenticated (set `WANDB_API_KEY` or run `uv run wandb login`; use `WANDB_MODE=offline` on air-gapped machines).
2. **Standard LIBERO eval settings** (enforced by [project/briskvla/run_eval.sh](project/briskvla/run_eval.sh)): `n_action_steps=10`, `seed=1000`, `episodes=10` per task (100 per suite), `batch_size=5`, `empty_cameras=1`, evaluated at `K=10` and `K=3` denoising steps.
3. **Comparable hardware**: Measure inference latency only on the fixed **RTX 3080 Laptop GPU**, and LIBERO task success on Google Colab (or equivalent CUDA GPU with fixed `run_eval.sh` settings).

## Model Naming

Every model and run name makes its training method, expert depth, layer placement, and compression immediately clear (`_K<steps>` is appended to eval folders for denoising steps):

`teacher` or `<role>_<layers>L_<placement>[_<modification>][_<variant>]`

| Component | Meaning | Values |
|---|---|---|
| `<role>` | Whether teacher distillation was used | `student` = distilled from teacher · `shallow` = **not distilled** (GT flow matching only, or untrained) |
| `<layers>L` | Number of action-expert transformer layers | e.g. `8L`, `4L` (teacher has `16L`; distinct from denoising steps `K10`, `K3`) |
| `<placement>` | Which of the 16 VLM depths host an expert layer | `spread` (`0,1,4,5,8,9,12,13`) · `early` (`0–7`) · `late` (`8–15`) · `alternating` (`0,3,4,7,8,11,12,15`) |
| `<modification>` | Optional structural change (omitted for plain drop) | `ffn_fused` (merged pair FFNs stacked in parallel) |
| `<variant>` | Optional training/eval qualifier | `untrained` · `shallow_pi` (teacher velocity/hidden + GT, no output loss) · `steps3` |

Legacy CLI layout aliases (`pairs`, `first8`, `last8`, `alt`, `alternate`) are still accepted by `--layout`. Every model, its architecture, and its results are listed in [Models & Results](#models--results).

## Quick Eval

```bash
# Latency only (RTX 3080 default):
project/briskvla/run_eval.sh teacher teacher

# Full LIBERO success eval (Colab):
SUCCESS=1 LATENCY=0 project/briskvla/run_eval.sh student_8L_spread path/to/checkpoint
```

## Models & Results

LIBERO success rate (%), 100 episodes per suite, seed 1000, `n_action_steps=10`. **Each result cell is `K=10 / K=3`** (number of denoising steps at eval). Each trained 8-layer (`8L`) model ran for 5000 steps at batch size 8. `K=1` results and per-variant details: [project/briskvla/README.md](project/briskvla/README.md#results).

| Model | Role | Depth / Params | VLM depths (`expert_vlm_layers`) | FFN width | Hidden map | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| `teacher` | Teacher | `16L` (98.2M) | `0..15` | 2048 | — | 79 / 76 | 91 / 90 | 89 / 89 | 69 / 71 | **82.0 / 81.5** |
| `student_8L_early` | Distilled | `8L` (49.1M) | `0,1,2,3,4,5,6,7` | 2048 | `same_depth` | 77.3 / 68 | 86.3 / 42 | 83.0 / 52 | 66.3 / 52 | **78.3** / 53.5 |
| `student_8L_spread` | Distilled | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | 2048 | `same_depth` | 75.3 / 55 | 82.7 / 64 | 85.0 / 71 | 70.0 / 49 | **78.3** / 59.8 |
| `student_8L_late` | Distilled | `8L` (49.1M) | `8,9,10,11,12,13,14,15` | 2048 | `same_depth` | 67 / 53 | 82 / 66 | 78 / 69 | 56 / 45 | 70.8 / 58.3 |
| `shallow_8L_early` | Shallow (GT-only) | `8L` (49.1M) | `0,1,2,3,4,5,6,7` | 2048 | — | 73.7 / 85 | 86.3 / 79 | 74.7 / 78 | 67.3 / 74 | 75.5 / **79.0** |
| `shallow_8L_spread` | Shallow (GT-only) | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | 2048 | — | 63.0 / 66 | 80.3 / 84 | 79.3 / 77 | 64.0 / 71 | 71.7 / 74.5 |
| `shallow_8L_late` | Shallow (GT-only) | `8L` (49.1M) | `8,9,10,11,12,13,14,15` | 2048 | — | 63 / 56 | 66 / 71 | 75 / 77 | 48 / 47 | 63.0 / 62.8 |
| `student_8L_early_shallow_pi` | Distilled (+GT, no output loss) | `8L` (49.1M) | `0,1,2,3,4,5,6,7` | 2048 | `same_depth` | 67 / 74 | — | — | — | — |
| `student_8L_spread_shallow_pi` | Distilled (+GT, no output loss) | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | 2048 | `same_depth` | 63 / 72 | — | — | — | — |
| `student_8L_late_shallow_pi` | Distilled (+GT, no output loss) | `8L` (49.1M) | `8,9,10,11,12,13,14,15` | 2048 | `same_depth` | 55 / 56 | — | — | — | — |
| `student_8L_alternating_ffn_fused` | Distilled + FFN Fusion | `8L` (84.5M) | `0,3,4,7,8,11,12,15` | 4096 | `uniform` | 76 / 62 | 83 / 73 | 85 / 71 | 61 / 44 | 76.3 / 62.5 |
| `student_8L_alternating` | Distilled (drop control) | `8L` (49.1M) | `0,3,4,7,8,11,12,15` | 2048 | `uniform` | 70 / 69 | 81 / 63 | 82 / 71 | 52 / 45 | 71.3 / 62.0 |
| `student_8L_spread` | Distilled | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | 2048 | `uniform` | 69 / 51 | 81 / 72 | 81 / 66 | 73 / 42 | 76.0 / 57.8 |
| `shallow_8L_alternating` | Shallow (GT-only drop control) | `8L` (49.1M) | `0,3,4,7,8,11,12,15` | 2048 | — | *not yet trained* | | | | |
| `shallow_8L_alternating_ffn_fused` | Shallow (GT-only) + FFN Fusion | `8L` (84.5M) | `0,3,4,7,8,11,12,15` | 4096 | — | *not yet trained* | | | | |
| `shallow_8L_<placement>[_ffn_fused]_untrained` | Untrained (zero-shot) | `8L` | any above | 2048 / 4096 | — | 0–3 / — | — | — | — | — |
| `student_4L_alternating_ffn_fused` / `shallow_4L_alternating_ffn_fused` | 4-layer iterative fusion (`16→8→4`) | `4L` | `0,7,8,15` | 8192 | — | *supported by `--source`, not yet trained* | | | | |

Shallow-π (`_shallow_pi`) models were evaluated on `libero_spatial` only.

The 95% confidence interval for a single 100-episode cell is roughly ±8–9 percentage points (about ±4–4.5 points for the 400-episode average).
