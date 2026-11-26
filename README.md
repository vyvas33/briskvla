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

Legacy CLI layout aliases (`pairs`, `first8`, `last8`, `alt`, `alternate`) are still accepted by `--layout`.

### Model Registry

| Canonical Name | Role | Depth / Params | VLM Depths (`expert_vlm_layers`) | FFN Width | Status |
|---|---|---|---|---|---|
| `teacher` | Teacher | `16L` (98.2M) | `0..15` | 2048 | Evaluated (`K=10, 3`) |
| `student_8L_spread` | Distilled | `8L` (49.1M) | `0,1,4,5,8,9,12,13` | 2048 | Evaluated (`K=10, 3, 1`) |
| `student_8L_early` | Distilled | `8L` (49.1M) | `0,1,2,3,4,5,6,7` | 2048 | Evaluated (`K=10, 3, 1`) |
| `student_8L_late` | Distilled | `8L` (49.1M) | `8,9,10,11,12,13,14,15` | 2048 | Evaluated (`K=10, 3, 1`) |
| `shallow_8L_spread` / `early` / `late` | Shallow (GT-only) | `8L` (49.1M) | same as above | 2048 | Evaluated (`K=10, 3, 1`) |
| `student_8L_spread` / `early` / `late` `_shallow_pi` | Distilled (+GT, no output) | `8L` (49.1M) | same as above | 2048 | Spatial only (`K=10, 3, 1`) |
| `student_8L_alternating` | Distilled (drop control) | `8L` (49.1M) | `0,3,4,7,8,11,12,15` | 2048 | Evaluated (`K=10, 3` only; `uniform`) |
| `student_8L_alternating_ffn_fused` | Distilled + FFN Fusion | `8L` (84.5M) | `0,3,4,7,8,11,12,15` | 4096 | Evaluated (`K=10, 3` only; `uniform`) |
| **`shallow_8L_alternating`** | Shallow (GT-only drop control) | `8L` (49.1M) | `0,3,4,7,8,11,12,15` | 2048 | **Not yet trained/evaluated** |
| **`shallow_8L_alternating_ffn_fused`** | Shallow (GT-only) + FFN Fusion | `8L` (84.5M) | `0,3,4,7,8,11,12,15` | 4096 | **Not yet trained/evaluated** |
| `shallow_8L_<placement>[_ffn_fused]_untrained` | Untrained | `8L` | any above | 2048 / 4096 | Zero-shot baselines |
| `student_4L_alternating_ffn_fused` / `shallow_4L_alternating_ffn_fused` | 4-layer iterative fusion (`16→8→4`) | `4L` | `0,7,8,15` | 8192 | Supported by `--source`, not yet trained |

## Quick Eval

```bash
# Latency only (RTX 3080 default):
project/briskvla/run_eval.sh teacher teacher

# Full LIBERO success eval (Colab):
SUCCESS=1 LATENCY=0 project/briskvla/run_eval.sh student_8L_spread path/to/checkpoint
```

## Results: 8-Layer Models on LIBERO

LIBERO success rate (%), 100 episodes per suite, seed 1000, `n_action_steps=10`. `K` is the number of denoising steps at eval. Each trained 8-layer (`8L`) model ran for 5000 steps at batch size 8. Details, `K=1`, Shallow-π, and untrained baselines: [project/briskvla/README.md](project/briskvla/README.md#results).

| Model | Hidden map | K | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| Teacher (`16L`, 98.2M expert) | — | 10 | 79 | 91 | 89 | 69 | **82.0** |
| Teacher (`16L`, 98.2M expert) | — | 3 | 76 | 90 | 89 | 71 | **81.5** |
| **Student — 8L Early** (distilled) | `same_depth` | 10 | 77.3 | 86.3 | 83.0 | 66.3 | **78.3** |
| **Student — 8L Spread** (distilled) | `same_depth` | 10 | 75.3 | 82.7 | 85.0 | 70.0 | **78.3** |
| **Student — 8L Late** (distilled) | `same_depth` | 10 | 67 | 82 | 78 | 56 | 70.8 |
| **Shallow — 8L Early** (no distillation) | — | 10 | 73.7 | 86.3 | 74.7 | 67.3 | 75.5 |
| **Shallow — 8L Spread** (no distillation) | — | 10 | 63.0 | 80.3 | 79.3 | 64.0 | 71.7 |
| **Shallow — 8L Late** (no distillation) | — | 10 | 63 | 66 | 75 | 48 | 63.0 |
| **Student — 8L Alternating + FFN Fusion** (84.5M) | `uniform` | 10 | 76 | 83 | 85 | 61 | **76.3** |
| **Student — 8L Alternating** (49.1M drop control) | `uniform` | 10 | 70 | 81 | 82 | 52 | 71.3 |
| **Student — 8L Spread** (49.1M) | `uniform` | 10 | 69 | 81 | 81 | 73 | 76.0 |
| **Shallow — 8L Early** (no distillation) | — | 3 | 85 | 79 | 78 | 74 | **79.0** |
| **Shallow — 8L Spread** (no distillation) | — | 3 | 66 | 84 | 77 | 71 | 74.5 |
| **Shallow — 8L Late** (no distillation) | — | 3 | 56 | 71 | 77 | 47 | 62.8 |
| **Student — 8L Early** (distilled) | `same_depth` | 3 | 68 | 42 | 52 | 52 | 53.5 |
| **Student — 8L Spread** (distilled) | `same_depth` | 3 | 55 | 64 | 71 | 49 | 59.8 |
| **Student — 8L Late** (distilled) | `same_depth` | 3 | 53 | 66 | 69 | 45 | 58.3 |
| **Student — 8L Alternating + FFN Fusion** (84.5M) | `uniform` | 3 | 62 | 73 | 71 | 44 | 62.5 |
| **Student — 8L Alternating** (49.1M) | `uniform` | 3 | 69 | 63 | 71 | 45 | 62.0 |
| **Student — 8L Spread** (49.1M) | `uniform` | 3 | 51 | 72 | 66 | 42 | 57.8 |

The 95% confidence interval for a single 100-episode cell is roughly ±8–9 percentage points (about ±4–4.5 points for the 400-episode average).
