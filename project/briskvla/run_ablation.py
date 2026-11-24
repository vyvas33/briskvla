"""Layer ablation: train/distill one model per layout with identical settings, evaluate in LIBERO,
and write a summary table, with mandatory W&B logging.

Any argument this script does not know is passed to `distill_student.py` unchanged, so every
layout gets the same loss weights, steps, batch size and seed.

Examples:
    # Distill spread, early and late 8L students for 5000 steps, then evaluate on libero_spatial
    uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/layers \
        --eval_suites libero_spatial -- --steps 5000 --batch_size 8

    # Non-distilled shallow ablation (ground-truth flow matching only)
    uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/shallow \
        --eval_suites libero_spatial -- --w_velocity 0 --w_hidden 0 --w_output 0 --w_gt 1

    # Only rebuild the summary from finished runs
    uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/layers --summary_only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    ALL_LAYOUT_CHOICES,
    DEFAULT_N_ACTION_STEPS,
    DEFAULT_SEED,
    LEGACY_LAYOUT_ALIASES,
    canonical_layout_name,
    format_model_name,
    infer_model_role,
    resolve_layout,
)
from wandb_log import add_wandb_args, ensure_wandb_auth, resolve_wandb_settings

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_finished(run_dir: Path) -> bool:
    return (run_dir / "best").exists() and any(r["split"] == "final" for r in read_metrics(run_dir))


def infer_extra_role(extra: list[str]) -> str:
    """Infer whether `extra` arguments configure a distilled `student` or non-distilled `shallow`."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--w_velocity", type=float, default=1.0)
    p.add_argument("--w_hidden", type=float, default=0.1)
    p.add_argument("--w_output", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=5000)
    known, _ = p.parse_known_args(extra)
    return infer_model_role(known.w_velocity, known.w_hidden, known.w_output, known.steps)


def _wandb_cli_flags(wb: dict[str, Any], name: str) -> list[str]:
    flags = [
        "--wandb_project",
        wb["project"],
        "--wandb_name",
        name,
    ]
    if wb["mode"] in ("online", "offline"):
        flags.extend(["--wandb_mode", wb["mode"]])
    if wb["entity"]:
        flags.extend(["--wandb_entity", wb["entity"]])
    if wb["group"]:
        flags.extend(["--wandb_group", wb["group"]])
    if wb["tags"]:
        flags.extend(["--wandb_tags", ",".join(wb["tags"])])
    return flags


def _wandb_env(wb: dict[str, Any]) -> dict[str, str]:
    env = {
        **os.environ,
        "WANDB_PROJECT": wb["project"],
        "WANDB_MODE": wb["mode"],
    }
    if wb["entity"]:
        env["WANDB_ENTITY"] = wb["entity"]
    if wb["group"]:
        env["WANDB_GROUP"] = wb["group"]
    if wb["tags"]:
        env["WANDB_TAGS"] = ",".join(wb["tags"])
    return env


def distill(layout: str, model_name: str, run_dir: Path, extra: list[str], wb: dict[str, Any]) -> None:
    if is_finished(run_dir):
        print(f"[skip] {model_name}: already trained in {run_dir}")
        return
    # A partial run resumes from its last/ checkpoint. distill_student.py handles its metrics.jsonl.
    cmd = [
        sys.executable,
        str(HERE / "distill_student.py"),
        "--layout",
        layout,
        "--out",
        str(run_dir),
        *extra,
        *_wandb_cli_flags(wb, model_name),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate(
    name: str,
    policy_path: str,
    out_root: Path,
    suites: list[str],
    n_episodes: int,
    num_steps: int,
    n_action_steps: int,
    batch_size: int,
    async_envs: bool,
    seed: int,
    wb: dict[str, Any],
) -> None:
    """Delegate LIBERO evaluation to `run_eval.sh` so settings, metadata, and W&B logging match."""
    if not suites:
        return
    env = {
        **_wandb_env(wb),
        "SUCCESS": "1",
        "LATENCY": "0",
        "SKIP_SETUP": "1",
        "OUT": str(out_root),
        "STEPS": str(num_steps),
        "SUITES": " ".join(suites),
        "EPISODES": str(n_episodes),
        "N_ACTION_STEPS": str(n_action_steps),
        "BATCH": str(batch_size),
        "ASYNC": "true" if async_envs else "false",
        "SEED": str(seed),
    }
    cmd = ["bash", str(HERE / "run_eval.sh"), name, policy_path]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def success_rate(eval_dir: Path) -> float | None:
    info = eval_dir / "eval_info.json"
    return json.loads(info.read_text())["overall"]["pc_success"] if info.exists() else None


def _find_existing_run_dir(out_root: Path, model_name: str, layout: str) -> Path:
    candidates = [
        out_root / model_name,
        out_root / layout,
        *(out_root / legacy for legacy, canon in LEGACY_LAYOUT_ALIASES.items() if canon == layout),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return out_root / model_name


def _find_eval_rate(out_root: Path, run_dir: Path, model_name: str, suite: str, steps: int) -> float | None:
    for cand in (
        out_root / "eval" / f"{model_name}_K{steps}" / suite,
        run_dir / "eval" / suite,
    ):
        rate = success_rate(cand)
        if rate is not None:
            return rate
    return None


def summarize(
    out_root: Path,
    models: list[tuple[str, str]],
    suites: list[str],
    eval_steps: int,
) -> None:
    rows = []
    for layout, model_name in models:
        run_dir = _find_existing_run_dir(out_root, model_name, layout)
        records = read_metrics(run_dir)
        val = [r for r in records if r["split"] == "val"]
        final = next((r for r in records if r["split"] == "final"), {})
        info = (
            json.loads((run_dir / "run_info.json").read_text())
            if (run_dir / "run_info.json").exists()
            else {}
        )
        best = min(val, key=lambda r: r["chunk_mse_teacher"]) if val else {}
        row = {
            "model": model_name,
            "layout": layout,
            "layers": info.get("layers", resolve_layout(layout)),
            "expert_params_M": round(info.get("expert_params_student", 0) / 1e6, 1),
            "best_step": best.get("step"),
            "chunk_mse_teacher_init": val[0]["chunk_mse_teacher"] if val else None,
            "chunk_mse_teacher_best": best.get("chunk_mse_teacher"),
            "chunk_mse_gt_best": best.get("chunk_mse_gt"),
            "teacher_chunk_mse_gt": best.get("teacher_chunk_mse_gt"),
            "velocity_best": best.get("velocity"),
            "hidden_best": best.get("hidden"),
            "latency_ms_student": final.get("latency_ms_student"),
            "latency_ms_teacher": final.get("latency_ms_teacher"),
        }
        for suite in suites:
            row[f"success_{suite}"] = _find_eval_rate(out_root, run_dir, model_name, suite, eval_steps)
        rows.append(row)

    teacher_row = {
        f"success_{s}": _find_eval_rate(out_root, out_root / "teacher", "teacher", s, eval_steps)
        for s in suites
    }

    (out_root / "summary.json").write_text(json.dumps({"models": rows, "teacher": teacher_row}, indent=2))

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    columns = list(rows[0]) if rows else []
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    if any(v is not None for v in teacher_row.values()):
        lines.append(
            "| teacher | full | - | 98.2 | "
            + " | ".join(fmt(teacher_row.get(c)) if c.startswith("success_") else "-" for c in columns[4:])
            + " |"
        )
    table = "\n".join(lines)
    (out_root / "summary.md").write_text(table + "\n")
    print("\n" + table + f"\n\nWritten to {out_root / 'summary.md'} and summary.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument(
        "--layouts",
        default="spread,early,late",
        help=f"Comma-separated layouts, from {ALL_LAYOUT_CHOICES}",
    )
    p.add_argument(
        "--eval_suites",
        default="",
        help="LIBERO suites for run_eval.sh, e.g. libero_spatial (empty = skip simulator eval)",
    )
    p.add_argument("--eval_episodes", type=int, default=10, help="Episodes per task")
    p.add_argument("--eval_num_steps", type=int, default=10, help="Denoising steps at eval (default: 10)")
    p.add_argument(
        "--eval_n_action_steps",
        type=int,
        default=DEFAULT_N_ACTION_STEPS,
        help="Actions executed from each predicted 50-action chunk before replanning",
    )
    p.add_argument("--eval_batch_size", type=int, default=5)
    p.add_argument("--eval_async", choices=["true", "false"], default="true")
    p.add_argument("--eval_teacher", action="store_true", help="Also evaluate the teacher on the same suites")
    p.add_argument("--eval_seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--summary_only", action="store_true")
    add_wandb_args(p)
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    raw_layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    layouts = [canonical_layout_name(x) for x in raw_layouts]
    suites = [x.strip() for x in args.eval_suites.split(",") if x.strip()]
    role = infer_extra_role(extra)
    models = [(layout, format_model_name(role, len(resolve_layout(layout)), layout)) for layout in layouts]

    wb = resolve_wandb_settings(args)
    if not wb["group"]:
        wb["group"] = args.out_root.name

    args.out_root.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        ensure_wandb_auth(wb["mode"])
        for layout, model_name in models:
            distill(layout, model_name, args.out_root / model_name, extra, wb)
        if suites:
            for _, model_name in models:
                evaluate(
                    model_name,
                    str(args.out_root / model_name / "best"),
                    args.out_root,
                    suites,
                    args.eval_episodes,
                    args.eval_num_steps,
                    args.eval_n_action_steps,
                    args.eval_batch_size,
                    args.eval_async == "true",
                    args.eval_seed,
                    wb,
                )
            if args.eval_teacher:
                evaluate(
                    "teacher",
                    "teacher",
                    args.out_root,
                    suites,
                    args.eval_episodes,
                    args.eval_num_steps,
                    args.eval_n_action_steps,
                    args.eval_batch_size,
                    args.eval_async == "true",
                    args.eval_seed,
                    wb,
                )

    summarize(args.out_root, models, suites, args.eval_num_steps)


if __name__ == "__main__":
    main()
