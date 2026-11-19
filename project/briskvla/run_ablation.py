"""Layer ablation: distill one student per expert layout with the same settings, optionally run
lerobot-eval on each, and write a summary table.

Any argument this script does not know is passed to distill_student.py unchanged, so every layout
gets the same loss weights, steps, batch size and seed.

Examples:
    # Distill pairs, first8 and last8 for 5000 steps each, then evaluate on libero_spatial
    uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/layers \
        --eval_suites libero_spatial -- --steps 5000 --batch_size 8

    # Only rebuild the summary from finished runs
    uv run python project/briskvla/run_ablation.py --out_root outputs/ablation/layers --summary_only

A layout whose run folder already has best/ and a final metrics line is not trained again, and a
suite that already has eval_info.json is not evaluated again, so an interrupted ablation can be
restarted with the same command.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from distill_student import LAYOUTS

HERE = Path(__file__).resolve().parent
CAMERAS = '{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'


def read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_finished(run_dir: Path) -> bool:
    return (run_dir / "best").exists() and any(r["split"] == "final" for r in read_metrics(run_dir))


def distill(layout: str, run_dir: Path, extra: list[str]) -> None:
    if is_finished(run_dir):
        print(f"[skip] {layout}: already distilled in {run_dir}")
        return
    if (run_dir / "metrics.jsonl").exists():
        # A partial run would mix two runs' lines in metrics.jsonl
        (run_dir / "metrics.jsonl").rename(run_dir / "metrics.partial.jsonl")
    cmd = [sys.executable, str(HERE / "distill_student.py"), "--layout", layout, "--out", str(run_dir), *extra]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate(
    policy_path: str,
    out_dir: Path,
    suite: str,
    n_episodes: int,
    num_steps: int | None,
    n_action_steps: int,
    seed: int,
) -> None:
    """Same lerobot-eval command as the README's teacher evaluation, plus n_action_steps."""
    if (out_dir / "eval_info.json").exists():
        print(f"[skip] eval {suite}: results already in {out_dir}")
        return
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        "--policy.empty_cameras=1",
        f"--policy.n_action_steps={n_action_steps}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.camera_name_mapping={CAMERAS}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        "--eval.use_async_envs=false",
        f"--seed={seed}",
        f"--output_dir={out_dir}",
    ]
    if num_steps is not None:
        cmd.append(f"--policy.num_steps={num_steps}")
    print(" ".join(cmd), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def success_rate(eval_dir: Path) -> float | None:
    info = eval_dir / "eval_info.json"
    return json.loads(info.read_text())["overall"]["pc_success"] if info.exists() else None


def summarize(out_root: Path, layouts: list[str], suites: list[str]) -> None:
    rows = []
    for layout in layouts:
        run_dir = out_root / layout
        records = read_metrics(run_dir)
        val = [r for r in records if r["split"] == "val"]
        final = next((r for r in records if r["split"] == "final"), {})
        info = json.loads((run_dir / "run_info.json").read_text()) if (run_dir / "run_info.json").exists() else {}
        best = min(val, key=lambda r: r["chunk_mse_teacher"]) if val else {}
        row = {
            "layout": layout,
            "layers": info.get("layers", LAYOUTS[layout]),
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
            row[f"success_{suite}"] = success_rate(run_dir / "eval" / suite)
        rows.append(row)
    teacher_row = {f"success_{s}": success_rate(out_root / "teacher" / "eval" / s) for s in suites}

    (out_root / "summary.json").write_text(json.dumps({"layouts": rows, "teacher": teacher_row}, indent=2))

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
        lines.append("| teacher | all 16 | 98.2 | " + " | ".join(
            fmt(teacher_row.get(c)) if c.startswith("success_") else "-" for c in columns[3:]
        ) + " |")
    table = "\n".join(lines)
    (out_root / "summary.md").write_text(table + "\n")
    print("\n" + table + f"\n\nWritten to {out_root / 'summary.md'} and summary.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--layouts", default="pairs,first8,last8", help=f"Comma-separated, from {sorted(LAYOUTS)}")
    p.add_argument("--eval_suites", default="", help="LIBERO suites for lerobot-eval, e.g. libero_spatial "
                   "(empty = skip simulator eval; needs the libero extra)")
    p.add_argument("--eval_episodes", type=int, default=10, help="Episodes per task")
    p.add_argument("--eval_num_steps", type=int, default=None, help="Denoising steps at eval (default: config)")
    p.add_argument("--eval_n_action_steps", type=int, default=10, help="Actions executed from each "
                   "predicted 50-action chunk before replanning. 10 matches the paper better than 50")
    p.add_argument("--eval_teacher", action="store_true", help="Also evaluate the teacher on the same suites")
    p.add_argument("--eval_seed", type=int, default=1000)
    p.add_argument("--summary_only", action="store_true")
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    layouts = [x for x in args.layouts.split(",") if x]
    suites = [x for x in args.eval_suites.split(",") if x]
    unknown = set(layouts) - set(LAYOUTS)
    if unknown:
        p.error(f"Unknown layouts {sorted(unknown)}; choose from {sorted(LAYOUTS)}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for layout in layouts:
            distill(layout, args.out_root / layout, extra)
        for layout in layouts:
            for suite in suites:
                evaluate(str(args.out_root / layout / "best"), args.out_root / layout / "eval" / suite,
                         suite, args.eval_episodes, args.eval_num_steps, args.eval_n_action_steps, args.eval_seed)
        if args.eval_teacher:
            from build_student import download_teacher

            for suite in suites:
                evaluate(str(download_teacher()), args.out_root / "teacher" / "eval" / suite,
                         suite, args.eval_episodes, args.eval_num_steps, args.eval_n_action_steps, args.eval_seed)

    summarize(args.out_root, layouts, suites)


if __name__ == "__main__":
    main()
