"""Distill a BriskVLA student from the SmolVLA teacher on lerobot/libero.

The student is built from the teacher with build_student.py (in memory, or loaded from a folder) and
trained with up to four losses, each with its own weight:

    velocity  MSE between student and teacher velocities at the same (x_t, t) and prefix
    hidden    MSE between student and teacher expert hidden states (intermediate layer alignment)
    output    MSE between the action chunks the student and teacher sample from the same noise
    gt        the original SmolVLA flow-matching loss against the dataset actions (off by default)

The VLM is frozen and identical in both models, so the prefix (images, instruction, state) is
encoded once per batch by the teacher and its KV cache is shared by the teacher and student experts.

Examples:
    uv run python project/briskvla/distill_student.py --layout pairs --out outputs/distill/pairs
    uv run python project/briskvla/distill_student.py --layers 0,2,4,6 --steps 200 --out outputs/distill/dbg
"""

import argparse
import json
import math
import random
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from build_student import (
    TEACHER,
    TEACHER_REVISION,
    build_student,
    count_params,
    download_teacher,
    save_student,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import make_pre_post_processors
from lerobot.policies.common.flow_matching import FlowConvention, euler_integrate, make_flow_matching_inputs
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# Expert layouts compared in the layer ablation. All have 4 joint- and 4 cross-attention layers.
LAYOUTS = {
    "pairs": [0, 1, 4, 5, 8, 9, 12, 13],
    "first8": [0, 1, 2, 3, 4, 5, 6, 7],
    "last8": [8, 9, 10, 11, 12, 13, 14, 15],
}

DATASET = "lerobot/libero"
# Same mapping as the teacher's train_config.json and policy_preprocessor.json
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------


def load_teacher(teacher_dir: Path, device: str, empty_cameras: int) -> SmolVLAPolicy:
    """Frozen teacher. `empty_cameras` must match what lerobot-eval is run with (1 in the README)."""
    config = PreTrainedConfig.from_pretrained(teacher_dir)
    config.device = device
    config.empty_cameras = empty_cameras
    config.validate_features()
    teacher = SmolVLAPolicy.from_pretrained(teacher_dir, config=config)
    teacher.eval().requires_grad_(False)
    return teacher


def load_student(teacher: SmolVLAPolicy, layers: list[int] | None, student_dir: Path | None) -> SmolVLAPolicy:
    """Build the student from the teacher, or load one saved earlier (e.g. by build_student.py)."""
    if student_dir is not None:
        config = PreTrainedConfig.from_pretrained(student_dir)
        config.device = teacher.config.device
        config.empty_cameras = teacher.config.empty_cameras
        config.validate_features()
        student = SmolVLAPolicy.from_pretrained(student_dir, config=config)
    else:
        assert layers is not None
        student = build_student(teacher, layers)
    # The VLM is frozen and has the same weights in both models. Point the student at the teacher's
    # copy so it is kept in GPU memory once.
    student.model.vlm_with_expert.vlm = teacher.model.vlm_with_expert.vlm
    student.to(teacher.config.device)
    return student


def trainable_parameters(policy: SmolVLAPolicy) -> list[torch.nn.Parameter]:
    return [p for p in policy.parameters() if p.requires_grad]


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------


def split_episodes(meta: LeRobotDatasetMetadata, val_fraction: float, max_episodes: int | None, seed: int):
    """Random episode-level train/val split. The val episodes are never trained on."""
    episodes = list(range(meta.total_episodes))
    random.Random(seed).shuffle(episodes)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    n_val = max(1, round(val_fraction * len(episodes)))
    return sorted(episodes[n_val:]), sorted(episodes[:n_val])


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        generator=torch.Generator().manual_seed(seed),
    )


def cycle(loader) -> Iterator[dict]:
    while True:
        yield from loader


def preprocess(batch: dict, preprocessor) -> dict:
    """Same steps as lerobot-train: uint8 images to [0, 1], rename keys, then the teacher's
    preprocessor (tokenize the task, move to device, normalize state and action)."""
    for key, value in batch.items():
        if key.startswith("observation.images") and torch.is_tensor(value) and value.dtype == torch.uint8:
            batch[key] = value.float() / 255.0
    return preprocessor(rename_batch_keys(batch, RENAME_MAP))


# ----------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------


def hidden_pairs(layers: list[int], mode: str, num_layers: int = 16) -> list[tuple[int, int]]:
    """(student depth, teacher depth) pairs for the intermediate layer loss.

    same_depth: student layer j is matched to the teacher layer it was copied from.
    uniform:    student layer j is matched to teacher depth (j+1)*16/k - 1, so the last student
                layer is matched to the last teacher layer (1, 3, 5, ..., 15 for k=8).
    """
    if mode == "same_depth":
        return [(d, d) for d in layers]
    if mode == "uniform":
        k = len(layers)
        return [(d, round((j + 1) * num_layers / k) - 1) for j, d in enumerate(layers)]
    raise ValueError(f"Unknown hidden_map {mode}")


@torch.no_grad()
def encode_prefix(model, images, img_masks, lang_tokens, lang_masks, state):
    """Run the frozen VLM over the prefix once and return its pad mask and KV cache."""
    embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embs, None],
        use_cache=True,
    )
    return pad_masks, past_key_values


def detach_cache(past_key_values) -> None:
    """denoise_step appends the suffix K/V and crops it off again. The cropped prefix is then a view
    of a tensor in the student's graph. Detach it so later steps don't keep that graph alive."""
    for layer in past_key_values.layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()


@contextmanager
def record_hidden(policy: SmolVLAPolicy, enabled: bool):
    """Collect {depth: expert hidden state} during the next forward passes."""
    model = policy.model.vlm_with_expert
    model.expert_hidden_states = {} if enabled else None
    try:
        yield model.expert_hidden_states
    finally:
        model.expert_hidden_states = None


def action_mse(pred: torch.Tensor, target: torch.Tensor, n_dims: int, is_pad: torch.Tensor | None = None):
    """MSE over the real action dimensions, ignoring padded timesteps when `is_pad` is given."""
    err = (pred[..., :n_dims].float() - target[..., :n_dims].float()).pow(2)
    if is_pad is None:
        return err.mean()
    mask = (~is_pad).unsqueeze(-1).float()
    return (err * mask).sum() / (mask.sum() * n_dims).clamp_min(1)


def hidden_mse(student_hidden: dict, teacher_hidden: dict, pairs: list[tuple[int, int]]) -> torch.Tensor:
    """MSE between layer-normalized hidden states, averaged over the matched layers.
    Layer norm (no weights) makes the loss independent of each layer's activation scale."""
    losses = []
    for s_depth, t_depth in pairs:
        hs, ht = student_hidden[s_depth].float(), teacher_hidden[t_depth].float()
        losses.append(F.mse_loss(F.layer_norm(hs, hs.shape[-1:]), F.layer_norm(ht, ht.shape[-1:])))
    return torch.stack(losses).mean()


def compute_losses(
    teacher: SmolVLAPolicy,
    student: SmolVLAPolicy,
    batch: dict,
    weights: dict[str, float],
    pairs: list[tuple[int, int]],
    student_num_steps: int,
    with_output: bool = True,
) -> dict[str, torch.Tensor]:
    """All distillation losses for one preprocessed batch. Losses with weight 0 are skipped."""
    T, S = teacher.model, student.model
    images, img_masks = teacher.prepare_images(batch)
    state = teacher.prepare_state(batch)
    actions = teacher.prepare_action(batch)
    lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    is_pad = batch.get("action_is_pad")
    n_dims = teacher.config.action_feature.shape[0]

    prefix_pad, kv = encode_prefix(T, images, img_masks, lang_tokens, lang_masks, state)

    def student_velocity(x_t, t):
        v = S.denoise_step(prefix_pad, kv, x_t, t)
        detach_cache(kv)
        return v

    def teacher_velocity(x_t, t):
        return T.denoise_step(prefix_pad, kv, x_t, t)

    losses: dict[str, torch.Tensor] = {}

    # Velocity, hidden and gt losses share one (x_t, t) built from the dataset actions
    if weights["velocity"] > 0 or weights["hidden"] > 0 or weights["gt"] > 0:
        noise = T.sample_noise(actions.shape, actions.device)
        t = T.sample_time(actions.shape[0], actions.device)
        x_t, u_t, _ = make_flow_matching_inputs(actions, noise, t, convention=FlowConvention.NOISE_AT_ONE)
        use_hidden = weights["hidden"] > 0
        with torch.no_grad(), record_hidden(teacher, use_hidden) as teacher_hidden:
            v_teacher = teacher_velocity(x_t, t)
        with record_hidden(student, use_hidden) as student_hidden:
            v_student = student_velocity(x_t, t)
        if weights["velocity"] > 0:
            losses["velocity"] = action_mse(v_student, v_teacher, n_dims)
        if use_hidden:
            losses["hidden"] = hidden_mse(student_hidden, teacher_hidden, pairs)
        if weights["gt"] > 0:
            losses["gt"] = action_mse(v_student, u_t, n_dims, is_pad)

    # Output level: both models integrate from the same noise; gradients flow through every
    # student denoising step
    if weights["output"] > 0 and with_output:
        noise = T.sample_noise(actions.shape, actions.device)
        with torch.no_grad():
            chunk_teacher = euler_integrate(teacher_velocity, noise, teacher.config.num_steps)
        chunk_student = euler_integrate(student_velocity, noise, student_num_steps)
        losses["output"] = action_mse(chunk_student, chunk_teacher, n_dims)

    losses["total"] = sum(weights[name] * value for name, value in losses.items())
    return losses


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------


@torch.no_grad()
def validate(teacher, student, val_loader, preprocessor, pairs, max_batches: int, seed: int) -> dict:
    """Held-out metrics with fixed noise, so numbers are comparable across steps and layouts.

    chunk_mse_teacher  student vs teacher action chunks, both sampled with num_steps from the same noise
    chunk_mse_gt       student chunk vs dataset actions; teacher_chunk_mse_gt is the teacher's value
    velocity / hidden / gt_fm  the training losses on held-out data
    """
    student.eval()
    T, S = teacher.model, student.model
    n_dims = teacher.config.action_feature.shape[0]
    sums: dict[str, float] = {}
    n = 0
    device = torch.device(teacher.config.device)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for i, raw in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = preprocess(raw, preprocessor)
            all_on = {"velocity": 1.0, "hidden": 1.0, "gt": 1.0, "output": 0.0}
            losses = compute_losses(teacher, student, batch, all_on, pairs, student.config.num_steps)

            images, img_masks = teacher.prepare_images(batch)
            state = teacher.prepare_state(batch)
            actions = teacher.prepare_action(batch)
            lang = (batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
            noise = T.sample_noise(actions.shape, actions.device)
            chunk_teacher = T.sample_actions(images, img_masks, *lang, state, noise=noise)
            chunk_student = S.sample_actions(images, img_masks, *lang, state, noise=noise)
            is_pad = batch.get("action_is_pad")
            metrics = {
                "velocity": losses["velocity"].item(),
                "hidden": losses["hidden"].item(),
                "gt_fm": losses["gt"].item(),
                "chunk_mse_teacher": action_mse(chunk_student, chunk_teacher, n_dims).item(),
                "chunk_mse_gt": action_mse(chunk_student, actions, n_dims, is_pad).item(),
                "teacher_chunk_mse_gt": action_mse(chunk_teacher, actions, n_dims, is_pad).item(),
            }
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
    student.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def measure_latency(policy: SmolVLAPolicy, batch: dict, iters: int = 20) -> float:
    """Median ms for one sample_actions call (prefix + all denoising steps) on a batch of 1."""
    one = {k: v[:1] if torch.is_tensor(v) else v for k, v in batch.items()}
    images, img_masks = policy.prepare_images(one)
    state = policy.prepare_state(one)
    args = (images, img_masks, one[OBS_LANGUAGE_TOKENS], one[OBS_LANGUAGE_ATTENTION_MASK], state)
    cuda = torch.device(policy.config.device).type == "cuda"
    times = []
    for i in range(iters + 3):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        policy.model.sample_actions(*args)
        if cuda:
            torch.cuda.synchronize()
        if i >= 3:  # warmup
            times.append((time.perf_counter() - start) * 1000)
    return sorted(times)[len(times) // 2]


# ----------------------------------------------------------------------------------------------
# Checkpoints and resuming
# ----------------------------------------------------------------------------------------------

TRAINING_STATE = "training_state.pt"
# Settings that must match for a run to be resumed. Everything else (logging, workers) may change.
RESUME_KEYS = (
    "steps", "batch_size", "grad_accum", "lr", "decay_lr", "warmup_steps", "weight_decay", "seed",
    "w_velocity", "w_hidden", "w_output", "w_gt", "hidden_map", "output_every", "student_num_steps",
    "val_fraction", "max_episodes", "amp", "empty_cameras",
)  # fmt: skip


def save_atomic(folder: Path, write) -> None:
    """Write a checkpoint folder so that an interruption at any point leaves a complete copy.

    `write(tmp)` fills `folder.tmp`. Then `folder` is moved to `folder.old`, `folder.tmp` to `folder`,
    and `folder.old` is deleted. If the machine dies between the two moves, `folder.old` is still
    complete and `find_resume_dir` falls back to it.
    """
    tmp, old = folder.with_name(folder.name + ".tmp"), folder.with_name(folder.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    write(tmp)
    shutil.rmtree(old, ignore_errors=True)
    if folder.exists():
        folder.rename(old)
    tmp.rename(folder)
    shutil.rmtree(old, ignore_errors=True)


def find_resume_dir(out: Path) -> Path | None:
    """The newest complete `last` checkpoint with a training state, or None for a fresh run."""
    for name in ("last", "last.old"):
        if (out / name / TRAINING_STATE).exists() and (out / name / "model.safetensors").exists():
            return out / name
    return None


def save_training_state(path: Path, step: int, best: float, optimizer, scheduler, args, layers) -> None:
    torch.save(
        {
            "step": step,
            "best": best,
            "layers": list(layers),
            "args": {k: getattr(args, k) for k in RESUME_KEYS},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_python": random.getstate(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_training_state(resume_dir: Path, student, optimizer, scheduler, args, layers) -> tuple[int, float]:
    """Restore the student's trainable weights, optimizer, LR schedule and RNG from `resume_dir`.
    Returns the step the checkpoint was saved at and the best validation chunk_mse_teacher so far."""
    from safetensors.torch import load_file

    state = torch.load(resume_dir / TRAINING_STATE, map_location="cpu", weights_only=False)
    saved = state["args"]
    mismatched = {k: (saved[k], getattr(args, k)) for k in RESUME_KEYS if saved.get(k) != getattr(args, k)}
    if list(state["layers"]) != list(layers):
        mismatched["layers"] = (state["layers"], list(layers))
    if mismatched:
        raise ValueError(
            f"Cannot resume from {resume_dir}: settings differ (saved, now): {mismatched}. "
            "Use the same arguments, or a new --out folder for a fresh run."
        )

    # The VLM is frozen and shared with the teacher, so only the student's own weights are loaded.
    weights = load_file(resume_dir / "model.safetensors", device=str(student.config.device))
    own = {k: v for k, v in weights.items() if ".vlm_with_expert.vlm." not in k}
    missing, unexpected = student.load_state_dict(own, strict=False)
    missing = [k for k in missing if ".vlm_with_expert.vlm." not in k]
    if missing or unexpected:
        raise ValueError(f"Checkpoint {resume_dir} does not match the student: {missing=} {unexpected=}")

    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng_python"])
    torch.set_rng_state(state["rng_torch"])
    if state["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["step"], state["best"]


def truncate_metrics(path: Path, step: int) -> None:
    """Drop log lines written after `step` (and the final line, which is written again at the end),
    so a resumed run does not log anything twice."""
    if not path.exists():
        return
    records = [(line, json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    kept = [line for line, r in records if r["step"] <= step and r["split"] != "final"]
    path.write_text("".join(line + "\n" for line in kept))


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def lr_lambda(step: int, peak_lr: float, decay_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to decay_lr, as in SmolVLA's training preset."""
    if step < warmup:
        return (step + 1) / (warmup + 1)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return (decay_lr + (peak_lr - decay_lr) * cosine) / peak_lr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--layout", choices=sorted(LAYOUTS), help="Named expert layout")
    which.add_argument("--layers", help="Comma-separated depths to keep an expert layer at")
    which.add_argument("--student", type=Path, help="Start from a student folder saved earlier")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--revision", default=TEACHER_REVISION)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--dataset_root", type=Path, default=None, help="Local dataset copy (default: HF cache)")
    p.add_argument("--val_fraction", type=float, default=0.05, help="Fraction of episodes held out")
    p.add_argument("--max_episodes", type=int, default=None, help="Use only this many episodes (debugging)")
    # Loss weights
    p.add_argument("--w_velocity", type=float, default=1.0)
    p.add_argument("--w_hidden", type=float, default=0.1)
    p.add_argument("--w_output", type=float, default=1.0)
    p.add_argument("--w_gt", type=float, default=0.0)
    p.add_argument("--hidden_map", choices=["same_depth", "uniform"], default="same_depth")
    p.add_argument("--output_every", type=int, default=1, help="Compute the output loss every N steps")
    p.add_argument("--student_num_steps", type=int, default=None, help="Student denoising steps in the "
                   "output loss (default: the teacher's num_steps)")
    # Optimization (defaults follow the teacher's train_config.json)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--decay_lr", type=float, default=2.5e-6)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=1e-10)
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA (lower memory, slightly less exact)")
    # Bookkeeping
    p.add_argument("--empty_cameras", type=int, default=1, help="Must match the lerobot-eval command")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=25)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--no_resume", action="store_true", help="Fail instead of resuming if --out has a checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    teacher_dir = download_teacher(args.teacher, args.revision)
    teacher = load_teacher(teacher_dir, args.device, args.empty_cameras)
    layers = LAYOUTS[args.layout] if args.layout else [int(x) for x in args.layers.split(",")] if args.layers else None
    student = load_student(teacher, layers, args.student)
    layers = student.model.vlm_with_expert.expert_vlm_layers
    student.train()
    pairs = hidden_pairs(layers, args.hidden_map, teacher.config.num_vlm_layers)
    weights = {"velocity": args.w_velocity, "hidden": args.w_hidden, "output": args.w_output, "gt": args.w_gt}
    student_num_steps = args.student_num_steps or teacher.config.num_steps

    preprocessor, _ = make_pre_post_processors(
        teacher.config,
        pretrained_path=teacher_dir,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    train_eps, val_eps = split_episodes(meta, args.val_fraction, args.max_episodes, args.seed)
    delta_timestamps = resolve_delta_timestamps(teacher.config, meta, RENAME_MAP)
    # Download the files of both splits first. A split built from a partial cache does not fetch the
    # files it is missing, it fails with "corresponds to no data".
    LeRobotDataset(args.dataset, root=args.dataset_root, episodes=None if args.max_episodes is None else train_eps + val_eps)
    train_ds = LeRobotDataset(args.dataset, root=args.dataset_root, episodes=train_eps, delta_timestamps=delta_timestamps)
    val_ds = LeRobotDataset(args.dataset, root=args.dataset_root, episodes=val_eps, delta_timestamps=delta_timestamps)
    # Fixed order (no shuffle) so validation always sees the same frames
    val_loader = make_loader(val_ds, args.batch_size, min(args.num_workers, 2), False, args.seed)

    params = trainable_parameters(student)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.lr, args.decay_lr, args.warmup_steps, args.steps)
    )

    # Resume from the last checkpoint in --out if there is one (e.g. after the machine was reclaimed)
    resume_dir = None if args.no_resume else find_resume_dir(args.out)
    start_step, best = 0, float("inf")
    if resume_dir is not None:
        start_step, best = load_training_state(resume_dir, student, optimizer, scheduler, args, layers)
        truncate_metrics(args.out / "metrics.jsonl", start_step)
        print(f"Resuming from {resume_dir} at step {start_step} (best chunk_mse_teacher {best:.5g})")
    elif any((args.out / name).exists() for name in ("best", "last", "last.old")):
        # A fresh run would leave the old best/ in place if validation never improves on step 0
        raise ValueError(f"{args.out} already has checkpoints. Remove them or use another --out folder.")
    elif (args.out / "metrics.jsonl").exists():
        # Lines from an earlier run without a checkpoint would be mixed with this run's
        (args.out / "metrics.jsonl").rename(args.out / "metrics.partial.jsonl")
    # The data order is not saved. Seeding with the start step gives a resumed run a new order
    # instead of replaying the batches from the start of the first epoch.
    train_iter = cycle(make_loader(train_ds, args.batch_size, args.num_workers, True, args.seed + start_step))

    run_info = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "layers": layers,
        "hidden_pairs": pairs,
        "teacher_dir": str(teacher_dir),
        "train_episodes": len(train_eps),
        "val_episodes": val_eps,
        "expert_params_teacher": count_params(teacher.model.vlm_with_expert.lm_expert),
        "expert_params_student": count_params(student.model.vlm_with_expert.lm_expert),
        "trainable_params": sum(p.numel() for p in params),
        "resumed_from_step": start_step,
    }
    (args.out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Layers {layers} | hidden pairs {pairs} | trainable {run_info['trainable_params'] / 1e6:.1f}M")
    print(f"Episodes: {len(train_eps)} train, {len(val_eps)} val | weights {weights}")

    metrics_file = (args.out / "metrics.jsonl").open("a")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    def run_validation(step: int) -> dict:
        metrics = validate(teacher, student, val_loader, preprocessor, pairs, args.val_batches, args.seed)
        log({"split": "val", "step": step, **metrics})
        return metrics

    def save_checkpoint(step: int, metrics: dict) -> None:
        nonlocal best
        # best/ first: if the run dies before last/ is written, it resumes from the previous last/
        # and repeats these steps, so best/ is never behind the best value stored in last/
        if metrics["chunk_mse_teacher"] < best:
            best = metrics["chunk_mse_teacher"]
            save_atomic(args.out / "best", lambda d: save_student(student, teacher_dir, d))
            print(f"New best chunk_mse_teacher={best:.5g}, saved to {args.out / 'best'}")

        def write_last(d: Path) -> None:
            save_student(student, teacher_dir, d)
            save_training_state(d / TRAINING_STATE, step, best, optimizer, scheduler, args, layers)

        save_atomic(args.out / "last", write_last)

    if start_step == 0:
        best = run_validation(0)["chunk_mse_teacher"]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.startswith("cuda"))
    running: dict[str, float] = {}
    start = time.perf_counter()

    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with_output = step % args.output_every == 0
        for _ in range(args.grad_accum):
            batch = preprocess(next(train_iter), preprocessor)
            with autocast:
                losses = compute_losses(teacher, student, batch, weights, pairs, student_num_steps, with_output)
            (losses["total"] / args.grad_accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_freq == 0:
            elapsed = time.perf_counter() - start
            log({
                "split": "train",
                "step": step,
                **{k: v / args.log_freq for k, v in running.items()},
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
                "s_per_step": elapsed / args.log_freq,
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            })
            running, start = {}, time.perf_counter()

        if step % args.eval_freq == 0 or step == args.steps:
            save_checkpoint(step, run_validation(step))
            start = time.perf_counter()

    if not (args.out / "best").exists():  # validation never improved on the copied weights
        save_atomic(args.out / "best", lambda d: save_student(student, teacher_dir, d))

    batch = preprocess(next(iter(val_loader)), preprocessor)
    student.eval()
    latency = {
        "latency_ms_teacher": measure_latency(teacher, batch),
        "latency_ms_student": measure_latency(student, batch),
    }
    log({"split": "final", "step": args.steps, "best_chunk_mse_teacher": best, **latency})
    metrics_file.close()


if __name__ == "__main__":
    main()
