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
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | Places expert layers at the chosen depths |
| `project/briskvla/build_student.py` | Builds a student checkpoint from the teacher |
| `project/briskvla/test_student.py` | Tests for the above |

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

## Tests

```bash
uv run --with pytest python -m pytest project/briskvla/test_student.py -sv
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
