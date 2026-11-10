# BriskVLA

GenAI course project built on LeRobot + SmolVLA, evaluated on LIBERO.
Original LeRobot README: [README_LEROBOT.md](README_LEROBOT.md)

## Setup (Linux + NVIDIA GPU; LIBERO doesn't install on macOS)

```bash
git clone https://github.com/vyvas33/lerobot.git
cd lerobot
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
