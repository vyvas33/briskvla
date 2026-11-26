"""Mandatory Weights & Biases logging for BriskVLA distillation, LIBERO eval, and latency.

Every run logs to W&B by default (project `briskVLA`, entity `djonnala-carnegie-mellon-university`).
Log in once with `uv run wandb login` or set `WANDB_API_KEY` (use `WANDB_MODE=offline` to sync
later via `wandb sync`).

Live logging hooks into `distill_student.py`, `run_eval.sh`, and `latency.py`. Past runs can also be
backfilled from saved folders/JSONs:

    uv run python project/briskvla/wandb_log.py distill --group <group> outputs/distill/...
    uv run python project/briskvla/wandb_log.py eval --group <group> outputs/eval/
    uv run python project/briskvla/wandb_log.py latency --group <group> outputs/latency/*.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    SUITES,
    TEACHER,
    TEACHER_REVISION,
    cpu_name,
    git_info,
    gpu_name,
    infer_model_role,
    read_policy_arch,
)

ID_FILE = "wandb_id.txt"
LATENCY_IDS_SUFFIX = ".wandb_ids.json"
VALID_MODES = ("online", "offline", "disabled")


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Attach standard W&B CLI flags to an argparse parser."""
    parser.add_argument(
        "--wandb_project",
        default=None,
        help=f"W&B project (default: WANDB_PROJECT or {DEFAULT_WANDB_PROJECT})",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help=f"W&B entity/team (default: WANDB_ENTITY or {DEFAULT_WANDB_ENTITY})",
    )
    parser.add_argument(
        "--wandb_group",
        default=None,
        help="W&B group for organizing related runs (default: WANDB_GROUP)",
    )
    parser.add_argument("--wandb_name", default=None, help="Override W&B run name")
    parser.add_argument("--wandb_tags", default=None, help="Comma-separated W&B tags")
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline"],
        default=None,
        help="W&B mode (default: WANDB_MODE env var or online)",
    )


def resolve_wandb_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve W&B settings from CLI args, environment variables, and team defaults."""
    raw_tags = getattr(args, "wandb_tags", None) or os.environ.get("WANDB_TAGS")
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else None
    mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE") or "online"
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported WANDB_MODE {mode!r}; expected one of {VALID_MODES}")
    return {
        "project": getattr(args, "wandb_project", None)
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_WANDB_PROJECT,
        "entity": getattr(args, "wandb_entity", None)
        or os.environ.get("WANDB_ENTITY")
        or DEFAULT_WANDB_ENTITY,
        "group": getattr(args, "wandb_group", None) or os.environ.get("WANDB_GROUP") or None,
        "name": getattr(args, "wandb_name", None) or None,
        "tags": tags,
        "mode": mode,
    }


def ensure_wandb_auth(mode: str = "online") -> None:
    """Fail fast before starting a run if online W&B credentials are not configured."""
    if mode in ("offline", "disabled"):
        return
    if os.environ.get("WANDB_API_KEY"):
        return
    import wandb

    logged_in = False
    try:
        if hasattr(wandb, "login"):
            logged_in = bool(wandb.login(prompt=False))
        else:
            logged_in = bool(getattr(getattr(wandb, "api", None), "api_key", None))
    except Exception:
        logged_in = False
    if not logged_in:
        raise SystemExit(
            "W&B logging is required for all BriskVLA runs. "
            "Run `uv run wandb login` or set WANDB_API_KEY (or set WANDB_MODE=offline to sync later)."
        )


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _dedupe_tags(tags: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _enrich_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    for k, v in git_info().items():
        cfg.setdefault(k, v)
    cfg.setdefault("gpu", gpu_name())
    cfg.setdefault("cpu", cpu_name())
    cfg.setdefault("host", socket.gethostname())
    cfg.setdefault("teacher", TEACHER)
    cfg.setdefault("teacher_revision", TEACHER_REVISION)
    return cfg


def _infer_role_from_name_or_cfg(name: str, cfg: dict[str, Any]) -> str | None:
    if cfg.get("role") in ("student", "shallow", "teacher"):
        return cfg["role"]
    low = name.lower()
    if low == "teacher":
        return "teacher"
    if low.startswith("shallow_") or "nokd" in low or "untrained" in low:
        return "shallow"
    if low.startswith("student_") or "distilled" in low:
        return "student"
    args_cfg = cfg.get("args")
    if isinstance(args_cfg, dict) and any(k in args_cfg for k in ("w_velocity", "w_hidden", "w_output")):
        return infer_model_role(
            float(args_cfg.get("w_velocity", 0.0)),
            float(args_cfg.get("w_hidden", 0.0)),
            float(args_cfg.get("w_output", 0.0)),
            int(args_cfg.get("steps", 1)),
        )
    if cfg.get("distill_wandb_id"):
        return "student"
    return None


class WandbDistillLogger:
    """Logs `distill_student.py` metric records to one W&B run per output folder."""

    def __init__(
        self,
        out: Path,
        project: str = DEFAULT_WANDB_PROJECT,
        config: dict[str, Any] | None = None,
        entity: str | None = DEFAULT_WANDB_ENTITY,
        group: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
    ) -> None:
        """Start a distillation run or resume the run id recorded in `<out>/wandb_id.txt`."""
        ensure_wandb_auth(mode)
        import wandb

        out.mkdir(parents=True, exist_ok=True)
        id_file = out / ID_FILE
        run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)

        cfg = _enrich_config(config or {})
        run_name = name or out.name
        role = _infer_role_from_name_or_cfg(run_name, cfg)
        if role:
            cfg.setdefault("role", role)
        branch = cfg.get("git_branch")
        all_tags = _dedupe_tags(
            [
                "distill",
                f"role:{role}" if role else None,
                f"branch:{branch}" if branch else None,
                *(tags or []),
            ]
        )

        self.run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=run_name,
            job_type="distill",
            tags=all_tags,
            id=run_id,
            resume="allow",
            config=cfg,
            dir=str(out),
            mode=mode,
        )
        id_file.write_text(run_id)
        # Plot everything against the training step, not W&B's internal counter, so resumed runs
        # repeating steps since the last checkpoint do not trigger out-of-order warnings.
        self.run.define_metric("step")
        self.run.define_metric("*", step_metric="step")

    def log(self, record: dict[str, Any]) -> None:
        """Log one `metrics.jsonl` record. The `final` record goes to the run summary."""
        split = record.get("split", "train")
        if split == "final":
            values = {}
            for k, v in record.items():
                if k in ("split", "step") or not _is_number(v):
                    continue
                prefix = "train_device" if k.startswith("latency_ms_") else "final"
                values[f"{prefix}/{k}"] = v
            self.run.summary.update(values)
            return

        values = {
            f"{split}/{k}": v for k, v in record.items() if k not in ("split", "step") and _is_number(v)
        }
        self.run.log({"step": record["step"], **values})
        if split == "val" and "chunk_mse_teacher" in record:
            best = self.run.summary.get("val/best_chunk_mse_teacher")
            if best is None or record["chunk_mse_teacher"] < best:
                self.run.summary["val/best_chunk_mse_teacher"] = record["chunk_mse_teacher"]
                self.run.summary["val/best_step"] = record["step"]

    def finish(self) -> None:
        """Mark the W&B run as finished."""
        self.run.finish()


# Backwards-compatible alias
WandbLogger = WandbDistillLogger


def upload_distill(
    run_dir: Path,
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = DEFAULT_WANDB_ENTITY,
    group: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
    force: bool = False,
) -> None:
    """Backfill a distillation run folder from `metrics.jsonl` and `run_info.json`."""
    if (run_dir / ID_FILE).exists() and not force:
        raise SystemExit(
            f"{run_dir} already has a wandb run ({ID_FILE}); pass --force to log into that run again"
        )
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(f"Missing {metrics_path}")
    info_file = run_dir / "run_info.json"
    config = json.loads(info_file.read_text()) if info_file.exists() else {}
    config.pop("val_episodes", None)
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    logger = WandbDistillLogger(
        run_dir,
        project=project,
        config=config,
        entity=entity,
        group=group,
        name=name,
        tags=tags,
        mode=mode,
    )
    for record in records:
        logger.log(record)
    logger.finish()
    print(f"{run_dir}: uploaded {len(records)} distillation records")


def log_eval_run(
    run_dir: Path,
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = DEFAULT_WANDB_ENTITY,
    group: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
    force: bool = False,
) -> bool:
    """Log or update a LIBERO evaluation run folder (`<name>_K<steps>/`) in W&B."""
    ensure_wandb_auth(mode)
    id_file = run_dir / ID_FILE
    if id_file.exists() and not force:
        print(f"[skip] {run_dir}: already logged ({ID_FILE}; pass --force to update)")
        return False

    suite_files: list[tuple[str, Path]] = []
    seen_suites: set[str] = set()
    for suite in SUITES:
        p = run_dir / suite / "eval_info.json"
        if p.is_file():
            suite_files.append((suite, p))
            seen_suites.add(suite)
    for child in sorted(run_dir.iterdir()) if run_dir.is_dir() else []:
        if child.is_dir() and child.name not in seen_suites:
            p = child / "eval_info.json"
            if p.is_file():
                suite_files.append((child.name, p))
                seen_suites.add(child.name)

    if not suite_files:
        print(f"[skip] {run_dir}: no suite eval_info.json files found")
        return False

    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    cfg: dict[str, Any] = dict(meta)
    policy_dir = cfg.get("policy_dir") or cfg.get("policy")
    if policy_dir and Path(policy_dir).is_dir():
        for k, v in read_policy_arch(policy_dir).items():
            cfg.setdefault(k, v)
        for cand in (Path(policy_dir).parent / ID_FILE, Path(policy_dir) / ID_FILE):
            if cand.is_file():
                cfg.setdefault("distill_wandb_id", cand.read_text().strip())
                break
        for info_cand in (Path(policy_dir).parent / "run_info.json", Path(policy_dir) / "run_info.json"):
            if info_cand.is_file():
                try:
                    info = json.loads(info_cand.read_text())
                except (OSError, json.JSONDecodeError):
                    break
                if info.get("role") in ("student", "shallow", "teacher"):
                    cfg.setdefault("role", info["role"])
                break
    cfg = _enrich_config(cfg)

    run_name = name or run_dir.name
    role = _infer_role_from_name_or_cfg(str(cfg.get("name") or run_name), cfg)
    if role:
        cfg.setdefault("role", role)

    import wandb

    run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
    branch = cfg.get("git_branch")
    num_steps = cfg.get("num_steps")
    all_tags = _dedupe_tags(
        [
            "eval",
            "libero",
            f"role:{role}" if role else None,
            f"K:{num_steps}" if num_steps is not None else None,
            f"branch:{branch}" if branch else None,
            *(tags or []),
        ]
    )
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        job_type="eval",
        tags=all_tags,
        id=run_id,
        resume="allow",
        config=cfg,
        dir=str(run_dir),
        mode=mode,
    )
    id_file.write_text(run_id)

    scalars: dict[str, Any] = {}
    task_rows: list[list[Any]] = []
    suite_scores: list[float] = []
    total_success = 0
    total_episodes = 0

    for suite, info_path in suite_files:
        data = json.loads(info_path.read_text())
        overall = data.get("overall") or {}
        pc_success = overall.get("pc_success")
        n_success = overall.get("n_success", 0)
        n_episodes = overall.get("n_episodes", 0)
        ci95 = overall.get("pc_success_ci95")
        if pc_success is not None:
            scalars[f"eval/{suite}/pc_success"] = float(pc_success)
            suite_scores.append(float(pc_success))
        if isinstance(ci95, list) and len(ci95) == 2:
            scalars[f"eval/{suite}/ci95_lo"] = float(ci95[0])
            scalars[f"eval/{suite}/ci95_hi"] = float(ci95[1])
        scalars[f"eval/{suite}/n_success"] = int(n_success)
        scalars[f"eval/{suite}/n_episodes"] = int(n_episodes)
        if _is_number(overall.get("eval_s")):
            scalars[f"eval/{suite}/eval_s"] = float(overall["eval_s"])
        total_success += int(n_success)
        total_episodes += int(n_episodes)

        for task in data.get("per_task") or []:
            t_ci = task.get("pc_success_ci95") or [None, None]
            task_rows.append(
                [
                    suite,
                    task.get("task_id"),
                    task.get("pc_success"),
                    task.get("n_success"),
                    task.get("n_episodes"),
                    t_ci[0] if len(t_ci) == 2 else None,
                    t_ci[1] if len(t_ci) == 2 else None,
                ]
            )

    scalars["eval/suites_completed"] = len(suite_files)
    scalars["eval/total_success"] = total_success
    scalars["eval/total_episodes"] = total_episodes
    if suite_scores:
        scalars["eval/mean_completed_success"] = sum(suite_scores) / len(suite_scores)
    if all(s in seen_suites for s in SUITES):
        all_four = [scalars[f"eval/{s}/pc_success"] for s in SUITES]
        scalars["eval/avg_success"] = sum(all_four) / len(SUITES)

    payload: dict[str, Any] = dict(scalars)
    if task_rows:
        payload["eval/per_task"] = wandb.Table(
            columns=["suite", "task_id", "pc_success", "n_success", "n_episodes", "ci95_lo", "ci95_hi"],
            data=task_rows,
        )
    run.log(payload)
    run.summary.update(scalars)
    run.finish()
    print(f"{run_dir}: logged eval ({len(suite_files)} suites) to W&B run {run_id}")
    return True


def _is_eval_run_dir(path: Path) -> bool:
    if (path / "run_metadata.json").is_file():
        return True
    return any((child / "eval_info.json").is_file() for child in path.iterdir() if child.is_dir())


def discover_eval_dirs(paths: list[Path]) -> list[Path]:
    """Resolve direct eval run folders or parent `outputs/eval` roots into deduplicated run dirs."""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Path not found: {p}")
        candidates = (
            [p]
            if _is_eval_run_dir(p)
            else [c for c in sorted(p.iterdir()) if c.is_dir() and _is_eval_run_dir(c)]
        )
        for c in candidates:
            resolved = c.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(c)
    return discovered


def log_latency_file(
    json_path: Path,
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = DEFAULT_WANDB_ENTITY,
    group: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
    force: bool = False,
) -> int:
    """Log each model in a `latency.py` output JSON to W&B. Returns number of runs logged."""
    ensure_wandb_auth(mode)
    if not json_path.is_file():
        raise SystemExit(f"Latency JSON not found: {json_path}")
    data = json.loads(json_path.read_text())
    meta = data.get("meta") or {}
    models = data.get("models") or {}
    sweep = data.get("ffn_width_sweep_us")
    if not models:
        print(f"[skip] {json_path}: no models found")
        return 0

    ids_path = json_path.with_name(json_path.stem + LATENCY_IDS_SUFFIX)
    ids: dict[str, str] = json.loads(ids_path.read_text()) if ids_path.is_file() else {}

    import wandb

    logged = 0
    for model_name, mdata in models.items():
        if model_name in ids and not force:
            print(f"[skip] {json_path} ({model_name}): already logged ({ids_path.name}; --force to update)")
            continue
        run_id = ids.get(model_name) or secrets.token_hex(4)
        run_name = name if (name and len(models) == 1) else (f"{name}_{model_name}" if name else model_name)

        cfg: dict[str, Any] = {
            **meta,
            "model_name": model_name,
            "policy_path": mdata.get("path"),
            "expert_vlm_layers": mdata.get("expert_vlm_layers"),
            "num_expert_layers": mdata.get("num_expert_layers")
            or (len(mdata["expert_vlm_layers"]) if mdata.get("expert_vlm_layers") is not None else None),
            "expert_intermediate_size": mdata.get("expert_intermediate_size"),
            "expert_params": mdata.get("expert_params"),
            "total_params": mdata.get("total_params"),
            "flops": mdata.get("flops"),
            "latency_file": str(json_path),
        }
        if "gpu" not in cfg:
            gpu_info = meta.get("gpu_before") or meta.get("gpu_after")
            if isinstance(gpu_info, dict) and gpu_info.get("name"):
                cfg["gpu"] = gpu_info["name"]
        cfg = _enrich_config(cfg)
        role = _infer_role_from_name_or_cfg(model_name, cfg)
        if role:
            cfg.setdefault("role", role)

        branch = cfg.get("git_branch")
        gpu = str(cfg.get("gpu") or "")
        all_tags = _dedupe_tags(
            [
                "latency",
                f"role:{role}" if role else None,
                f"gpu:{_sanitize_tag(gpu)}" if gpu else None,
                f"branch:{branch}" if branch else None,
                *(tags or []),
            ]
        )

        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=run_name,
            job_type="latency",
            tags=all_tags,
            id=run_id,
            resume="allow",
            config=cfg,
            dir=str(json_path.parent),
            mode=mode,
        )
        ids[model_name] = run_id
        ids_path.write_text(json.dumps(ids, indent=2))

        run.define_metric("K")
        run.define_metric("latency/*", step_metric="K")

        summary_updates: dict[str, Any] = {}
        steps = mdata.get("steps") or {}
        for k_str in sorted(steps, key=lambda s: int(s)):
            k_int = int(k_str)
            s = steps[k_str]
            row = {"K": k_int}
            for metric_k, metric_v in s.items():
                if metric_k != "samples" and _is_number(metric_v):
                    row[f"latency/{metric_k}"] = metric_v
            run.log(row)
            for summary_key in ("wall_ms", "wall_ms_p90", "prefix_ms", "denoise_ms", "ffn_ms", "non_ffn_ms"):
                if summary_key in s and _is_number(s[summary_key]):
                    summary_updates[f"latency/K{k_int}/{summary_key}"] = s[summary_key]

        flops = mdata.get("flops") or {}
        if _is_number(flops.get("prefix")):
            summary_updates["flops/prefix_gflops"] = flops["prefix"] / 1e9
        if _is_number(flops.get("per_step")):
            summary_updates["flops/per_step_gflops"] = flops["per_step"] / 1e9
        fit = mdata.get("fit") or {}
        if _is_number(fit.get("prefix_ms")):
            summary_updates["latency/fit/prefix_ms"] = fit["prefix_ms"]
        if _is_number(fit.get("per_step_ms")):
            summary_updates["latency/fit/per_step_ms"] = fit["per_step_ms"]
        if _is_number(mdata.get("expert_params")):
            summary_updates["expert_params"] = mdata["expert_params"]
        if _is_number(mdata.get("total_params")):
            summary_updates["total_params"] = mdata["total_params"]

        if sweep and logged == 0:
            rows = [[int(w), float(us)] for w, us in sorted(sweep.items(), key=lambda kv: int(kv[0]))]
            run.log(
                {
                    "latency/ffn_width_sweep": wandb.Table(
                        columns=["intermediate_width", "median_us"], data=rows
                    )
                }
            )

        run.summary.update(summary_updates)
        run.finish()
        logged += 1
        print(f"{json_path} ({model_name}): logged latency to W&B run {run_id}")

    return logged


def main() -> None:
    """CLI entry point for checking W&B auth and uploading historical BriskVLA runs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    auth_p = sub.add_parser("check-auth", help="Verify W&B credentials before running experiments")
    auth_p.add_argument("--wandb_mode", choices=["online", "offline"], default=None)

    distill_p = sub.add_parser("distill", help="Upload finished distillation folders (metrics.jsonl)")
    distill_p.add_argument("run_dirs", nargs="+", type=Path, help="Distillation output folders")
    add_wandb_args(distill_p)
    distill_p.add_argument("--force", action="store_true", help="Re-upload into existing wandb_id.txt runs")

    eval_p = sub.add_parser("eval", help="Upload LIBERO eval folders (<name>_K<steps>/ or root)")
    eval_p.add_argument("paths", nargs="+", type=Path, help="Eval run folders or outputs/eval roots")
    add_wandb_args(eval_p)
    eval_p.add_argument(
        "--force", action="store_true", help="Re-upload/update runs with existing wandb_id.txt"
    )

    latency_p = sub.add_parser("latency", help="Upload latency.py JSON files")
    latency_p.add_argument("json_files", nargs="+", type=Path, help="Latency output JSON files")
    add_wandb_args(latency_p)
    latency_p.add_argument("--force", action="store_true", help="Re-upload models already in .wandb_ids.json")

    args = parser.parse_args()
    if args.command == "check-auth":
        mode = args.wandb_mode or os.environ.get("WANDB_MODE") or "online"
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported WANDB_MODE {mode!r}; expected one of {VALID_MODES}")
        ensure_wandb_auth(mode)
        return

    wb = resolve_wandb_settings(args)
    if args.command == "distill":
        for run_dir in args.run_dirs:
            upload_distill(run_dir, **wb, force=args.force)
    elif args.command == "eval":
        for run_dir in discover_eval_dirs(args.paths):
            log_eval_run(run_dir, **wb, force=args.force)
    elif args.command == "latency":
        for json_path in args.json_files:
            log_latency_file(json_path, **wb, force=args.force)


if __name__ == "__main__":
    main()
