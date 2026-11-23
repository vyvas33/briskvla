# BriskVLA

GenAI course project built on LeRobot + SmolVLA, evaluated on LIBERO.
Original LeRobot README: [README_LEROBOT.md](README_LEROBOT.md)

## Setup (Linux + NVIDIA GPU; LIBERO doesn't install on macOS)

```bash
git clone https://github.com/vyvas33/briskvla.git
cd briskvla
git switch briskvla
uv sync --locked --extra smolvla --extra libero
uv run hf auth login
export MUJOCO_GL=egl   # headless servers
```

## Branches

- `main`: untouched copy of huggingface/lerobot. Don't commit here.
- `briskvla`: our shared project branch.
- Personal work: `git switch -c <name>/<topic>` from `briskvla`, then open a PR with base `briskvla`.

## Quick eval

Pretrained checkpoint: `lerobot/smolvla_libero` (dataset `lerobot/libero`). Same command as the repo's CI:

```bash
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=1 --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --policy.empty_cameras=1 \
  --output_dir=outputs/eval/smolvla_libero
```

Full benchmark: `--env.task=libero_spatial,libero_object,libero_goal,libero_10` (see `docs/source/libero.mdx`).

## Results: 8-layer students

LIBERO success rate (%), 100 episodes per suite, seed 1000, `n_action_steps=10`. K is the number of
denoising steps at eval. Each student keeps 8 of the teacher's 16 expert layers and was trained for
5000 steps. Training, layouts and eval details: [project/briskvla/README.md](project/briskvla/README.md#results).

- **Distilled**: teacher velocity + hidden-state + output-chunk losses.
- **Shallow**: ground truth flow matching only, no teacher.
- Layouts: Spread = layers 0,1,4,5,8,9,12,13; Early = 0–7; Late = 8–15.

| Model | K | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| Spread, distilled | 10 | 78 | 79 | 79 | 75 | 77.8 |
| Early, distilled | 10 | 79 | 87 | 83 | 64 | 78.2 |
| Late, distilled | 10 | 67 | 82 | 78 | 56 | 70.8 |
| Spread, shallow | 10 | 62 | 79 | 78 | 69 | 72.0 |
| Early, shallow | 10 | 75 | 87 | 74 | 67 | 75.8 |
| Late, shallow | 10 | 63 | 66 | 75 | 48 | 63.0 |
| Spread, distilled | 3 | 55 | 64 | 71 | 49 | 59.8 |
| Early, distilled | 3 | 68 | 42 | 52 | 52 | 53.5 |
| Late, distilled | 3 | 53 | 66 | 69 | 45 | 58.3 |
| Spread, shallow | 3 | 66 | 84 | 77 | 71 | 74.5 |
| Early, shallow | 3 | 85 | 79 | 78 | 74 | 79.0 |
| Late, shallow | 3 | 56 | 71 | 77 | 47 | 62.8 |
| Spread, distilled | 1 | 70 | 74 | 78 | 57 | 69.8 |
| Early, distilled | 1 | 88 | 73 | 73 | 59 | 73.2 |
| Late, distilled | 1 | 59 | 74 | 69 | 46 | 62.0 |
| Spread, shallow | 1 | 66 | 81 | 74 | 66 | 71.8 |
| Early, shallow | 1 | 78 | 81 | 81 | 66 | 76.5 |
| Late, shallow | 1 | 57 | 67 | 60 | 52 | 59.0 |

The 95% interval of a single cell is about ±9 points. Shallow-π results (Spatial only) and the
untrained baselines are in the [project README](project/briskvla/README.md#results).
