# BriskVLA student

This folder builds the shallow-expert student used in BriskVLA. The student is SmolVLA with fewer
action expert layers. It keeps the teacher's VLM and a chosen subset of the teacher's expert layers,
and each kept layer starts as an exact copy of the teacher's layer.

Teacher: `lerobot/smolvla_libero` at revision `a24f1f380461b665158d52acd140152c79e8b36d`, the
revision used for the LIBERO teacher evaluation.

## Files

| File | Purpose |
|---|---|
| `src/lerobot/policies/smolvla/configuration_smolvla.py` | Adds the `expert_vlm_layers` config field and its validation |
| `src/lerobot/policies/smolvla/modeling_smolvla.py` | Passes `expert_vlm_layers` to the VLM + expert model |
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | Places expert layers at the chosen depths, can record expert hidden states |
| `project/briskvla/build_student.py` | Builds a student checkpoint from the teacher |
| `project/briskvla/distill_student.py` | Distills a student from the teacher on `lerobot/libero` |
| `project/briskvla/run_ablation.py` | Layer ablation: distills and evaluates one student per layout |
| `project/briskvla/test_student.py` | Tests for the student build |
| `project/briskvla/test_distill.py` | Tests for the distillation losses |
| `project/briskvla/latency.py` | Inference latency per action chunk, split into prefix and denoising |
| `project/briskvla/run_eval.sh` | Latency and LIBERO success for one model, with fixed settings |
| `project/briskvla/summarize_eval.py` | Success table for all evaluated models |
| `project/briskvla/eval_colab.ipynb` | Runs `run_eval.sh` on Colab |
| `project/briskvla/colab_distill.py` | Distills all layouts in parallel on a Colab VM, resuming interrupted ones (`colab exec -s <session> --timeout 0 -f project/briskvla/colab_distill.py`) |

## Setup

```bash
uv sync --locked --python 3.12 --extra smolvla --extra dataset   # distillation
uv sync --locked --python 3.12 --extra smolvla --extra dataset --extra libero   # + simulator eval
```

Use Python 3.12. With 3.14, draccus fails to parse the policy config. Without a system ffmpeg,
torchcodec cannot load and video decoding falls back to pyav, which works but is slower.
`sudo apt install ffmpeg` fixes that.

## Expert layout

SmolVLA runs its VLM and action expert as two stacks of 16 layers. At depth `i` the forward pass
runs VLM layer `i` and, if there is one, the expert layer paired with it. Even depths use joint
attention and odd depths use cross-attention. The type is decided by depth, not by the expert
layer's own index.

`expert_vlm_layers` lists the depths that have an expert layer. Expert layer `j` sits at depth
`expert_vlm_layers[j]` and reads that VLM layer's keys and values.

```
expert_vlm_layers = [0, 1, 4, 5, 8, 9, 12, 13]

depth    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
VLM      L0  L1  L2  L3  L4  L5  L6  L7  L8  L9  L10 L11 L12 L13 L14 L15
expert   E0  E1  -   -   E2  E3  -   -   E4  E5  -   -   E6  E7  -   -
```

All VLM layers always run. At a depth without an expert layer, the action tokens' hidden state is
passed to the next depth unchanged. `None` (the default) puts an expert layer at every depth, which
is the teacher.

The config rejects an empty list, a list that is not strictly increasing, indices outside
`[0, num_vlm_layers)`, and setting `expert_vlm_layers` together with `num_expert_layers`.

### Why not `num_expert_layers`

SmolVLA already has `num_expert_layers=k` for a shallower expert. We don't use it, and
`SmolVLMWithExpertModel` now raises an error if it is set. It only allows `k` that divides 16 and
places every expert layer at an even depth. It also gives cross-attention projections to odd expert
indices, which then run in joint-attention layers and fail with a shape error. With `k=8` this
fails on the first forward pass.

## Changes in `smolvlm_with_expert.py`

1. `__init__` stores the layout as `self.expert_vlm_layers`, defaulting to `range(num_vlm_layers)`.
2. The expert is built with `len(self.expert_vlm_layers)` layers.
3. The loop that resizes `k_proj` and `v_proj` for cross-attention picks the attention type from the
   depth of each expert layer. Cross-attention layers take the VLM key/value size (320) as input,
   joint-attention layers keep the expert width (720).
4. `get_model_layers` returns a 16-entry list of expert layers with `None` at depths without one.
5. In `forward`, a denoising pass (prefix cached, only action tokens passed in) skips depths with no
   expert layer. Without this the joint-attention code has nothing to concatenate and crashes. The
   teacher never hits this case.
6. `self.expert_hidden_states` (default `None`) can be set to a dict. `forward` then stores the
   expert hidden state after each expert layer under its depth. The distillation script uses this
   for the intermediate layer loss. With `None` nothing is recorded and behavior is unchanged.

## Building a student

```bash
uv run python project/briskvla/build_student.py --layers 0,1,4,5,8,9,12,13 --out outputs/students/k8_pairs
```

`--teacher` takes a Hub id or a local checkpoint folder, and `--revision` pins the Hub revision.
Both default to the teacher above. The teacher is downloaded to the Hugging Face cache on first use.

`build_student.py` does the following.

1. **Config.** `make_student_config` copies the teacher config and sets:
   - `expert_vlm_layers` to the given layers
   - `train_expert_only=True` and `freeze_vision_encoder=True`, so the VLM is frozen
   - `train_state_proj=False`, so the state token in the prefix stays the teacher's

   With these, the student's prefix (images, instruction, state) is identical to the teacher's and
   only the expert differs. `dataclasses.replace` is used so the config validation runs again.
2. **Skeleton.** A new `SmolVLAPolicy` is created from that config. It has the right number of
   expert layers with the right projection shapes. Its weights are overwritten in the next step.
3. **Weights.** `remap_state_dict` goes through the teacher's state dict. Expert layer weights at a
   kept depth are renamed from `lm_expert.layers.{depth}.` to `lm_expert.layers.{j}.`, where `j` is
   the depth's position in the list. Expert layers at other depths are dropped. Everything else
   (vision encoder, VLM, `state_proj`, action projections, time MLP, expert final norm) is copied
   unchanged. The result is loaded with `strict=True`, which fails if any weight is missing or extra.
   `load_state_dict` copies values, so the student shares no memory with the teacher.
4. **Save.** `save_pretrained` writes `config.json` and `model.safetensors`. The teacher's four
   `policy_*processor*` files are copied next to them. These hold the normalization stats and are
   the same for the student because the data is the same.

The output folder has the same layout as the teacher checkpoint, so `lerobot-eval` and
`lerobot-train --policy.path=...` load it without changes.

| | Teacher | k8 student |
|---|---|---|
| Expert parameters | 98.2M | 49.1M |
| `model.safetensors` | 907 MB | 807 MB |

The saved config records the device it was built on (`mps` on a Mac). On a machine without that
device the config falls back to an available one, and the eval command passes `--policy.device`.

## Candidate layouts

| Name | `--layers` | VLM layers read | VLM layers not read |
|---|---|---|---|
| pairs | `0,1,4,5,8,9,12,13` | spread over depth | 2, 3, 6, 7, 10, 11, 14, 15 |
| first 8 | `0,1,2,3,4,5,6,7` | early | 8–15 |
| last 8 | `8,9,10,11,12,13,14,15` | deep | 0–7 |

All three have four joint-attention and four cross-attention layers. VLM layers after the deepest
kept depth are still computed but nothing reads them. Stopping the VLM there would reduce prefix
latency, which matters most for "first 8". This is not implemented yet.

The plan is to distill each candidate for the same short budget, compare action-chunk error against
the teacher on held-out episodes, and train and evaluate only the best one on LIBERO.
`run_ablation.py` (see [Layer ablation](#layer-ablation)) runs this comparison. The names `pairs`,
`first8` and `last8` are defined in `LAYOUTS` in `distill_student.py`.

## Distillation

```bash
uv run python project/briskvla/distill_student.py --layout pairs --out outputs/distill/pairs
```

The student is given by one of `--layout pairs|first8|last8`, `--layers 0,2,4,...` (built from the
teacher in memory with `build_student.py`) or `--student <folder>` (a folder saved earlier). The
teacher is the same pinned checkpoint as above and stays frozen.

### Losses

Each loss has its own weight. A loss with weight 0 is not computed.

| Loss | Flag (default) | What it matches |
|---|---|---|
| Velocity distillation | `--w_velocity` (1.0) | Student velocity vs teacher velocity at the same `(x_t, t)` and prefix |
| Intermediate layer alignment | `--w_hidden` (0.1) | Student expert hidden states vs teacher expert hidden states |
| Output-level distillation | `--w_output` (1.0) | The action chunk the student samples vs the one the teacher samples from the same noise |
| Ground truth flow matching | `--w_gt` (0.0) | SmolVLA's original loss against the dataset actions |

- **Velocity.** `x_t` is built from the dataset action chunk with SmolVLA's own noise and time
  sampling, so the student is trained on the same `(x_t, t)` distribution as the teacher was.
- **Intermediate layers.** For each student expert layer, the hidden state of the 50 action tokens
  after that layer is compared with a teacher layer's. Both are layer-normalized first (no learned
  weights), so layers with larger activations don't dominate. `--hidden_map` picks the teacher layer:
  - `same_depth` (default): the teacher layer at the same depth, which is the layer the student
    layer was copied from.
  - `uniform`: student layer `j` is matched to teacher depth `(j+1)*16/k - 1` (1, 3, ..., 15 for
    k=8), so the last student layer is matched to the teacher's last layer.

  With `same_depth` and `first8`, the student's last layer (depth 7) is pulled towards the teacher's
  depth-7 state, while the output losses want it to produce what the teacher produces after depth
  15. The two losses then pull in different directions. Try `uniform` or a smaller `--w_hidden` there.
- **Output level.** The teacher integrates `num_steps` (10) Euler steps without gradients. The
  student integrates `--student_num_steps` steps (default 10) with gradients through every step.
  This is the most expensive loss. `--output_every N` computes it only every N steps.
  `--student_num_steps 3` trains a student that is meant to be run with fewer steps
  (evaluate it with `--policy.num_steps=3`).

All action losses use only the 7 real action dimensions, in normalized action space. The ground
truth loss also skips padded timesteps at the end of episodes.

### How a training step works

1. The batch goes through the teacher's preprocessor (rename cameras, tokenize the task, normalize
   state and action), the same as in `lerobot-train`.
2. The teacher's VLM encodes the prefix once without gradients and returns its KV cache. Teacher
   and student have the same frozen VLM, and the prefix never attends to the action tokens, so the
   student uses the same cache. The student also points at the teacher's VLM module, so the VLM is
   kept in GPU memory only once.
3. Teacher and student experts run `denoise_step` on that cache. The intermediate hidden states are
   collected through `expert_hidden_states` (change 6 above).
4. Only the student's expert, action projections and time MLP get gradients (`train_expert_only`,
   `train_state_proj=False` from `build_student.py`).

### Validation and outputs

A random 5% of episodes (`--val_fraction`, seeded) is held out. Every `--eval_freq` steps (and at
step 0, before training) the script computes these on `--val_batches` held-out batches with fixed
noise:

| Metric | Meaning |
|---|---|
| `chunk_mse_teacher` | Student vs teacher action chunk, both sampled with 10 steps from the same noise. Used to pick `best/` |
| `chunk_mse_gt` | Student chunk vs dataset actions. `teacher_chunk_mse_gt` is the teacher's value for reference |
| `velocity`, `hidden`, `gt_fm` | The training losses on held-out data |

Output folder:

| Path | Content |
|---|---|
| `best/`, `last/` | Student checkpoints in the teacher's layout. Pass to `lerobot-eval --policy.path=...`. `last/` also has `training_state.pt` for resuming |
| `metrics.jsonl` | One JSON line per train log, validation, and a `final` line with the median `sample_actions` latency (batch 1) of teacher and student |
| `run_info.json` | Arguments, layers, hidden layer pairs, held-out episode indices, parameter counts |

### Resuming

A run that stops early (for example a Colab machine that is reclaimed) continues where it stopped
when the same command is run again. At every validation (`--eval_freq`) `last/` gets a
`training_state.pt` with the step, the best `chunk_mse_teacher`, the optimizer, the LR schedule and
the random number generators. On start, if `--out` has one, the student's weights and this state are
loaded and training continues from that step, so at most `--eval_freq` steps are repeated.

- Checkpoints are written to `<name>.tmp/` and then moved into place, so stopping during a save
  leaves the previous complete checkpoint. `best/` is written before `last/`.
- The settings that affect training (steps, batch size, LR, loss weights, seed, ...) must be the
  same as in the first run, otherwise the script stops with the list of differences.
- The data order is not saved. A resumed run reshuffles with seed `--seed + step`, so the batches
  after a resume differ from an uninterrupted run, but the result is otherwise equivalent.
- `metrics.jsonl` lines after the resumed step are dropped and logged again.
- A fresh run refuses an `--out` that already has `best/` or `last/` without a training state.
  `--no_resume` makes the script fail instead of resuming.

### Defaults and memory

Optimizer defaults follow the teacher's `train_config.json` (AdamW, lr 1e-4, betas (0.9, 0.95),
warmup then cosine decay to 2.5e-6, gradient clipping at 10). `--steps` defaults to 5000 and
`--batch_size` to 8. Use `--grad_accum` for a larger effective batch. A smoke run with
`--batch_size 4` and all losses used 3.2 GB of GPU memory at about 0.8 s per step on an RTX 5070
Laptop GPU (8 GB). `--amp` turns on bf16 autocast, which uses less memory but is less exact.

`--empty_cameras` (default 1) must match the `lerobot-eval` command. The teacher evaluation passes
`--policy.empty_cameras=1`.

### Dataset

`lerobot/libero` (1693 episodes, 273k frames at 10 fps, about 1.9 GB) is downloaded on first use
to `~/.cache/huggingface/lerobot/lerobot/libero`. To download it before training:

```bash
uv run hf download lerobot/libero --repo-type dataset
```

`--max_episodes N` uses a random subset of N episodes and downloads only their files, which is
useful for a quick check:

```bash
uv run python project/briskvla/distill_student.py --layout pairs --out outputs/distill/debug \
    --max_episodes 20 --steps 20 --eval_freq 10 --val_batches 2 --batch_size 4
```

## Layer ablation

`run_ablation.py` distills one student per layout with the same arguments, optionally evaluates each
one in LIBERO, and writes a summary table. Arguments after `--` go to `distill_student.py`
unchanged, so every layout is trained with the same losses, steps and seed.

```bash
uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/layers \
    --eval_suites libero_spatial --eval_teacher -- --steps 5000 --batch_size 8
```

| Flag | Meaning |
|---|---|
| `--layouts` | Comma-separated layouts, default `pairs,first8,last8` |
| `--eval_suites` | LIBERO suites for `lerobot-eval` on each `best/` (empty = skip, needs the `libero` extra) |
| `--eval_episodes` | Episodes per task (default 10) |
| `--eval_num_steps` | Denoising steps at eval (default: the checkpoint's 10) |
| `--eval_n_action_steps` | Actions executed from each predicted chunk before replanning (default 10, see below) |
| `--eval_teacher` | Also evaluate the teacher on the same suites, as a reference row |
| `--summary_only` | Only rebuild the summary from finished runs |

The `lerobot-eval` command is the same as the teacher evaluation in the top-level README, plus
`--policy.n_action_steps=10`. The policy predicts a chunk of `chunk_size=50` actions per call, and
`n_action_steps` is how many of them are executed before the policy is queried again with a new
observation. The checkpoint default is 50 (the whole chunk, open loop). Replanning every 10 actions
comes much closer to the paper's success rates. It only affects evaluation: training always uses the
full 50-action chunk. Do not confuse it with `num_steps` (denoising steps per chunk) or with the
number of expert layers. Use the same value for the teacher and every student so the rows can be
compared. Results
go to `<out_root>/<layout>/eval/<suite>/`. Restarting the same command skips layouts that finished
distilling and suites that already have `eval_info.json`.

`<out_root>/summary.md` and `summary.json` have one row per layout: layers, expert parameters,
held-out `chunk_mse_teacher` before training and at the best step, `chunk_mse_gt`, the best
velocity and hidden losses, latency of student and teacher, and success rate per suite.

To compare loss terms instead of layouts, run the same command with different weights into
different `--out_root` folders, for example `-- --w_hidden 0` or `-- --w_output 0`.

## Tests

```bash
uv run --with pytest python -m pytest project/briskvla/test_student.py -sv
uv run --with pytest python -m pytest project/briskvla/test_distill.py -sv
```

pytest is not part of the default environment, hence `--with pytest`. The tests use the real
teacher with random inputs, so no dataset is needed. They take about a minute on a Mac CPU.

| Test | Checks |
|---|---|
| `test_full_student_matches_teacher` | A student with all 16 layers gives bit-identical actions to the teacher |
| `test_k8_layers_are_teacher_copies` | Student layer `j` has the same weights as the teacher layer at depth `layers[j]` |
| `test_k8_student_runs` | `sample_actions` returns shape (1, 50, 32) with finite values |
| `test_only_expert_side_trains` | Only expert layers and action projections have `requires_grad` |
| `test_save_and_reload` | The saved folder loads with `from_pretrained`, keeps `expert_vlm_layers`, has the processor files and gives the same actions |

`test_distill.py` (about 3 minutes on an RTX 5070 Laptop GPU):

| Test | Checks |
|---|---|
| `test_hidden_pairs` | `same_depth` and `uniform` give the expected student/teacher layer pairs |
| `test_full_student_has_zero_distill_loss` | With all 16 layers, the velocity, hidden and output losses are 0 (the shared KV cache path matches the teacher exactly) |
| `test_one_step_trains_only_the_student_expert` | For each layout, all losses are finite, every student expert layer gets a gradient, and the teacher gets no gradient and is unchanged after an optimizer step |

## Evaluation

Every model is evaluated with `run_eval.sh`, so all numbers use the same settings: batch size 5 with
parallel environments, seed 1000, 10 episodes per task, `empty_cameras=1`, and K = 10 and 3
denoising steps. It needs Linux and an NVIDIA GPU.

By default the script only measures latency, which is what the RTX 3080 runs:

```bash
project/briskvla/run_eval.sh teacher teacher                          # the teacher
project/briskvla/run_eval.sh k8_pairs_distilled path/to/checkpoint    # any other model
```

`SUCCESS=1` adds the LIBERO success evals, and `LATENCY=0` skips latency. The Colab notebook sets
both.

The script first syncs packages and writes `~/.libero/config.yaml` if it is missing. Both steps are
skipped when already done, or entirely with `SKIP_SETUP=1`. It then measures latency with
`latency.py` and, with `SUCCESS=1`, runs `lerobot-eval` on each suite. Results go to:

```
outputs/latency/<name>__<gpu>.json
outputs/eval/<name>_K<steps>/run_metadata.json
outputs/eval/<name>_K<steps>/<suite>/eval_info.json
outputs/eval/<name>_K<steps>/<suite>.log
```

Finished suites are skipped, so after a crash the same command continues where it stopped. Settings
are environment variables, listed at the top of the script. A quick check with one episode per task:

```bash
SUCCESS=1 LATENCY=0 STEPS=10 SUITES=libero_spatial EPISODES=1 project/briskvla/run_eval.sh smoke teacher
```

We measure latency only on the RTX 3080 laptop and task success only on Colab, so that each metric
comes from one machine type for every model. On Colab, open `eval_colab.ipynb`, set the model in the
Config cell and run all cells. It runs the success evals with latency off and saves results to Google
Drive. If rendering fails (for example under WSL), try `MUJOCO_GL=osmesa`.

`summarize_eval.py` prints the success table for every run under `outputs/eval` and saves it to
`summary.json`.

### Latency

`latency.py` times one `predict_action_chunk` call at batch size 1, with the same settings as the
eval. The prefix is everything before the first denoising step, timed with CUDA events around each
step. A line fit of total time over K = 1, 2, 3, 5, 10 is saved as a cross-check, along with exact
FLOPs for the prefix and for each step. Several models can be measured in one run. They are timed in
alternating rounds so that GPU temperature affects all of them equally:

```bash
uv run python project/briskvla/latency.py \
    --policy teacher=outputs/checkpoints/smolvla_libero_a24f1f38 \
    --policy k8_pairs=outputs/students/k8_pairs \
    --out outputs/latency/rtx3080.json
```
