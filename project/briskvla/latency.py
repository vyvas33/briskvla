"""Measure SmolVLA inference latency per action chunk, split into prefix and denoising.

One action chunk is one predict_action_chunk call at batch size 1. The prefix is everything before
the first denoising step (image preprocessing, vision encoder, VLM pass that fills the KV cache).
Denoising is the K expert passes. Settings match lerobot-eval: empty_cameras=1, TF32 on, no AMP.

Models are measured in alternating rounds so that GPU temperature drift affects all of them equally.
Any SmolVLA checkpoint works, including students from build_student.py.

--breakdown also times every expert FFN (CUDA events around each `mlp` call), so denoising time is
split into FFN and the rest (attention, norms, residuals, launch gaps). --compile runs the denoising
step through torch.compile, the baseline any structural speedup has to beat. --ffn_width_sweep times
one expert-sized SwiGLU alone at several hidden widths, which shows whether a wider (fused) FFN costs
more than a narrow one at batch size 1.

Example:
    uv run python project/briskvla/latency.py \
        --policy teacher=lerobot/smolvla_libero \
        --policy student_8L_spread=outputs/distill/student_8L_spread/best \
        --out outputs/latency/rtx3080.json
"""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from common import TEACHER, TEACHER_REVISION, cpu_name, git_info, gpu_state
from torch.utils.flop_counter import FlopCounterMode
from wandb_log import add_wandb_args, ensure_wandb_auth, log_latency_file, resolve_wandb_settings

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def load_policy(path: str, device: str) -> SmolVLAPolicy:
    revision = TEACHER_REVISION if path == TEACHER else None
    config = PreTrainedConfig.from_pretrained(path, revision=revision)
    config.device = device
    config.empty_cameras = 1
    policy = SmolVLAPolicy.from_pretrained(path, config=config, revision=revision)
    policy.eval()
    return policy


def make_batch(policy: SmolVLAPolicy, seed: int = 0) -> dict:
    """Random inputs in the format the preprocessor hands to the policy. Latency does not depend on
    the values, and the instruction is always padded to tokenizer_max_length."""
    cfg = policy.config
    g = torch.Generator().manual_seed(seed)
    batch = {}
    for key, feature in cfg.image_features.items():
        if "empty_camera" in key:
            continue
        batch[key] = torch.rand(1, *feature.shape, generator=g)
    batch[OBS_STATE] = torch.randn(1, *cfg.robot_state_feature.shape, generator=g)
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 1000, (1, cfg.tokenizer_max_length), generator=g)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool)
    return {k: v.to(cfg.device) for k, v in batch.items()}


class FFNTimer:
    """CUDA events around every expert FFN call. Adds two event records per layer and step."""

    def __init__(self, policy: SmolVLAPolicy):
        self.calls: list[tuple] = []
        self._open: dict[int, torch.cuda.Event] = {}
        for layer in policy.model.vlm_with_expert.lm_expert.layers:
            layer.mlp.register_forward_pre_hook(self._pre)
            layer.mlp.register_forward_hook(self._post)

    def _pre(self, module, args):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._open[id(module)] = event

    def _post(self, module, args, output):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.calls.append((self._open.pop(id(module)), event))

    def reset(self) -> None:
        self.calls = []

    def total_ms(self) -> float:
        return sum(start.elapsed_time(end) for start, end in self.calls)


class ChunkTimer:
    """Times predict_action_chunk and the denoising steps inside it.

    On CUDA the split uses CUDA events, which add no synchronization. On other devices each mark
    synchronizes, which slows things down a little, so treat those numbers as a code check only.
    """

    def __init__(self, policy: SmolVLAPolicy, breakdown: bool = False, compile: bool = False):
        self.policy = policy
        self.device = torch.device(policy.config.device)
        self.steps = []
        self.ffn = FFNTimer(policy) if breakdown else None
        denoise_step = policy.model.denoise_step
        if compile:
            denoise_step = torch.compile(denoise_step)

        def timed_denoise_step(*args, **kwargs):
            start = self._mark()
            out = denoise_step(*args, **kwargs)
            self.steps.append((start, self._mark()))
            return out

        # sample_actions looks up self.denoise_step at call time, so this replaces it
        policy.model.denoise_step = timed_denoise_step

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _mark(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        self._sync()
        return time.perf_counter()

    def _ms(self, start, end) -> float:
        if self.device.type == "cuda":
            return start.elapsed_time(end)
        return (end - start) * 1000

    def __call__(self, batch: dict) -> dict:
        self.policy.reset()
        self.steps = []
        if self.ffn is not None:
            self.ffn.reset()
        self._sync()
        wall_start = time.perf_counter()
        start = self._mark()
        self.policy.predict_action_chunk(batch)
        end = self._mark()
        self._sync()
        wall_ms = (time.perf_counter() - wall_start) * 1000

        first_step = self.steps[0][0]
        last_step = self.steps[-1][1]
        prefix_ms = self._ms(start, first_step)
        denoise_ms = self._ms(first_step, last_step)
        out = {
            "wall_ms": wall_ms,
            "prefix_ms": prefix_ms,
            "denoise_ms": denoise_ms,
            "other_ms": self._ms(start, end) - prefix_ms - denoise_ms,
        }
        if self.ffn is not None:
            # At inference the prefix pass has no action tokens, so the expert FFNs only run inside
            # denoising steps and every recorded call belongs to one
            out["ffn_ms"] = self.ffn.total_ms()
            out["non_ffn_ms"] = denoise_ms - out["ffn_ms"]
        return out


@torch.no_grad()
def ffn_width_sweep(policy: SmolVLAPolicy, widths: list[int], iters: int = 500) -> dict:
    """Time one SwiGLU with the expert's hidden size on one chunk of action tokens at several
    intermediate widths. Same dtype and TF32 settings as the policy."""
    expert = policy.model.vlm_with_expert
    hidden = expert.expert_hidden_size
    tokens = policy.config.chunk_size
    device = policy.config.device
    dtype = expert.lm_expert.layers[0].mlp.gate_proj.weight.dtype
    x = torch.randn(1, tokens, hidden, device=device, dtype=dtype)
    out = {}
    for width in widths:
        gate = torch.nn.Linear(hidden, width, bias=False, device=device, dtype=dtype)
        up = torch.nn.Linear(hidden, width, bias=False, device=device, dtype=dtype)
        down = torch.nn.Linear(width, hidden, bias=False, device=device, dtype=dtype)

        def ffn(x, gate=gate, up=up, down=down):
            return down(torch.nn.functional.silu(gate(x)) * up(x))

        for _ in range(50):
            ffn(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ffn(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        out[width] = statistics.median(times)
    return out


def count_flops(policy: SmolVLAPolicy, batch: dict) -> dict:
    """FLOPs per chunk. Running with 1 and 2 steps gives the prefix and per-step cost exactly."""
    totals = {}
    for steps in (1, 2):
        policy.config.num_steps = steps
        policy.reset()
        with FlopCounterMode(display=False) as counter:
            policy.predict_action_chunk(batch)
        totals[steps] = counter.get_total_flops()
    per_step = totals[2] - totals[1]
    return {"prefix": totals[1] - per_step, "per_step": per_step}


def summarize(samples: list[dict]) -> dict:
    wall = sorted(s["wall_ms"] for s in samples)
    out = {key: statistics.median(s[key] for s in samples) for key in samples[0]}
    out["wall_ms_p90"] = wall[int(0.9 * (len(wall) - 1))]
    out["expert_share"] = out["denoise_ms"] / (out["prefix_ms"] + out["denoise_ms"] + out["other_ms"])
    return out


def parse_policy_arg(arg: str) -> tuple[str, str]:
    if "=" in arg:
        name, path = arg.split("=", 1)
        return name, path
    return Path(arg).name, arg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="NAME=PATH, Hub id or local checkpoint folder. Repeat for several models.",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 5, 10], help="Denoising steps K")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed calls per model and K")
    parser.add_argument("--iters", type=int, default=100, help="Timed calls per model and K")
    parser.add_argument("--rounds", type=int, default=5, help="Alternate between models this many times")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--breakdown", action="store_true", help="Also time the expert FFNs (CUDA only)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the denoising step")
    parser.add_argument(
        "--ffn_width_sweep",
        type=int,
        nargs="*",
        default=None,
        help="Also time one expert SwiGLU at these intermediate widths (default 2048 3072 4096 8192)",
    )
    add_wandb_args(parser)
    args = parser.parse_args()
    if args.breakdown and args.compile:
        parser.error("--breakdown uses module hooks, which torch.compile traces through. Use one at a time.")
    if (args.breakdown or args.ffn_width_sweep is not None) and not args.device.startswith("cuda"):
        parser.error("--breakdown and --ffn_width_sweep need CUDA")
    wb = resolve_wandb_settings(args)
    ensure_wandb_auth(wb["mode"])

    # Same precision settings as lerobot-eval
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)

    models = {}
    for name, path in map(parse_policy_arg, args.policy):
        policy = load_policy(path, args.device)
        batch = make_batch(policy)
        timer = ChunkTimer(policy, breakdown=args.breakdown, compile=args.compile)
        models[name] = {"path": path, "policy": policy, "batch": batch, "timer": timer}

    results = {}
    for name, m in models.items():
        policy = m["policy"]
        expert = policy.model.vlm_with_expert
        results[name] = {
            "path": m["path"],
            "expert_vlm_layers": expert.expert_vlm_layers,
            "num_expert_layers": len(expert.expert_vlm_layers),
            "expert_intermediate_size": policy.config.expert_intermediate_size,
            "expert_params": sum(p.numel() for p in expert.lm_expert.parameters()),
            "total_params": sum(p.numel() for p in policy.parameters()),
            "flops": count_flops(policy, m["batch"]),
            "steps": {},
        }

    sweep = None
    if args.ffn_width_sweep is not None:
        widths = args.ffn_width_sweep or [2048, 3072, 4096, 8192]
        sweep = ffn_width_sweep(next(iter(models.values()))["policy"], widths)

    gpu_before = gpu_state()
    per_round = max(1, args.iters // args.rounds)
    for steps in args.steps:
        samples = {name: [] for name in models}
        for m in models.values():
            m["policy"].config.num_steps = steps
            for _ in range(args.warmup):
                m["timer"](m["batch"])
        for _ in range(args.rounds):
            for name, m in models.items():
                m["policy"].config.num_steps = steps
                samples[name].extend(m["timer"](m["batch"]) for _ in range(per_round))
        for name in models:
            results[name]["steps"][steps] = {**summarize(samples[name]), "samples": samples[name]}
    gpu_after = gpu_state()

    # Cross-check of the split: fit wall_ms = prefix + K * per_step over the measured K
    for r in results.values():
        ks = sorted(r["steps"])
        if len(ks) >= 2:
            slope, intercept = statistics.linear_regression(ks, [r["steps"][k]["wall_ms"] for k in ks])
            r["fit"] = {"prefix_ms": intercept, "per_step_ms": slope}

    meta = {
        "device": args.device,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        **git_info(),
        "warmup": args.warmup,
        "iters": per_round * args.rounds,
        "rounds": args.rounds,
        "batch_size": 1,
        "empty_cameras": 1,
        "tf32": True,
        "compile": args.compile,
        "breakdown": args.breakdown,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "models": results, "ffn_width_sweep_us": sweep}, indent=2))
    log_latency_file(args.out, **wb, force=True)

    gpu_name = (gpu_before or {}).get("name", args.device)
    print(f"\nGPU: {gpu_name}, CPU: {meta['cpu']}")
    print(f"\n{'model':<12}{'K':>4}{'total':>9}{'p90':>8}{'prefix':>9}{'denoise':>9}{'share':>7}")
    for name, r in results.items():
        for k, s in r["steps"].items():
            print(
                f"{name:<12}{k:>4}{s['wall_ms']:>9.1f}{s['wall_ms_p90']:>8.1f}"
                f"{s['prefix_ms']:>9.1f}{s['denoise_ms']:>9.1f}{s['expert_share']:>7.0%}"
            )
        if args.breakdown:
            for k, s in r["steps"].items():
                ffn = s["ffn_ms"] / k
                rest = s["non_ffn_ms"] / k
                print(
                    f"{name:<12}{k:>4}  per step: FFN {ffn:.2f} ms ({ffn / (ffn + rest):.0%}), rest {rest:.2f} ms"
                )
        f = r["flops"]
        fit = r.get("fit")
        fit_text = (
            f", fit: prefix {fit['prefix_ms']:.1f} ms + {fit['per_step_ms']:.1f} ms/step" if fit else ""
        )
        print(
            f"{name:<12}GFLOPs: prefix {f['prefix'] / 1e9:.1f}, per step {f['per_step'] / 1e9:.1f}{fit_text}"
        )
    if sweep:
        print("\nOne SwiGLU on one chunk (us):  " + "  ".join(f"{w}: {t:.0f}" for w, t in sweep.items()))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
