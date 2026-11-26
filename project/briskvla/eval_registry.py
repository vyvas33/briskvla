"""Shared team registry of BriskVLA LIBERO evaluation results on Hugging Face Hub.

Maintains a single canonical ``results.csv`` (and leaderboard ``README.md``) in the HF dataset
``briskVLA/eval-results`` so all teammates can see what has already been evaluated, jump to the
corresponding W&B run, and skip duplicate evaluations in ``run_eval.sh``.

Examples:
    # Backfill 4-suite results documented in project/briskvla/README.md:
    uv run python project/briskvla/eval_registry.py backfill-readme

    # Sync finished W&B eval runs into briskVLA/eval-results:
    uv run python project/briskvla/eval_registry.py sync-wandb

    # Record a single local eval run folder (<name>_K<steps>/):
    uv run python project/briskvla/eval_registry.py record outputs/eval/teacher_K1

    # Check whether a specific (model, K, episodes, n_action_steps, seed) already exists:
    uv run python project/briskvla/eval_registry.py check --name teacher --steps 10

    # Print coverage matrix across all HF models + teacher and list missing evals:
    uv run python project/briskvla/eval_registry.py plan --steps "10 3 1" --episodes 10

    # Export results.csv and coverage matrix CSV for mirroring to Google Sheets:
    uv run python project/briskvla/eval_registry.py export-csv results.csv --matrix_csv matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_HF_EVAL_DATASET,
    DEFAULT_HF_NAMESPACE,
    DEFAULT_N_ACTION_STEPS,
    DEFAULT_SEED,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    SUITES,
    TEACHER,
    canonical_layout_name,
    layout_name_for_layers,
)
from hf_hub import ensure_hf_write_auth, list_models
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

COLUMNS = [
    "model",
    "policy",
    "role",
    "layers",
    "layout",
    "ffn_fused",
    "hidden_map",
    "num_steps",
    "episodes_per_task",
    "n_action_steps",
    "seed",
    "spatial",
    "object",
    "goal",
    "long",
    "avg",
    "suites_completed",
    "status",
    "evaluator",
    "gpu",
    "wandb_run_id",
    "wandb_url",
    "date",
    "source",
]

REGISTRY_FILENAME = "results.csv"

_SOURCE_PRIORITY = {"wandb": 3, "local": 2, "readme": 1}

_README_NAME_MAP: dict[str, str] = {
    "Teacher": "teacher",
    "Student — 8L Early": "student_8L_early",
    "Student — 8L Spread": "student_8L_spread",
    "Student — 8L Late": "student_8L_late",
    "Shallow — 8L Early": "shallow_8L_early",
    "Shallow — 8L Spread": "shallow_8L_spread",
    "Shallow — 8L Late": "shallow_8L_late",
    "Student — 8L Alternating + FFN Fusion": "student_8L_alternating_ffn_fused",
    "Student — 8L Alternating (matched drop)": "student_8L_alternating",
    "Student — 8L Alternating": "student_8L_alternating",
}


def wandb_url(
    run_id: str | None,
    *,
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
) -> str:
    """Format the canonical W&B run URL for ``run_id``, or ``""`` when absent."""
    clean = (run_id or "").strip()
    if not clean:
        return ""
    return f"https://wandb.ai/{entity}/{project}/runs/{clean}"


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes")


def _fmt_num(value: Any, digits: int = 2) -> str:
    val = _to_float(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}".rstrip("0").rstrip(".")


def _parse_model_meta(model: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer role, layers, layout, and ffn_fused from ``cfg`` or canonical model name."""
    cfg = cfg or {}
    low = model.lower()
    role = cfg.get("role")
    if role not in ("teacher", "student", "shallow"):
        if low == "teacher":
            role = "teacher"
        elif low.startswith("student_"):
            role = "student"
        elif low.startswith("shallow_"):
            role = "shallow"
        else:
            role = ""

    raw_layers = cfg.get("num_expert_layers")
    layers = _to_int(raw_layers)
    if layers is None:
        if role == "teacher":
            layers = 16
        else:
            m = re.search(r"_(\d+)l(?:_|$)", low)
            if m:
                layers = int(m.group(1))

    vlm_layers = cfg.get("expert_vlm_layers")
    layout = ""
    if isinstance(vlm_layers, list):
        layout = layout_name_for_layers(vlm_layers) or ""
    if not layout:
        for cand in ("spread", "early", "late", "alternating", "pairs", "first8", "last8", "alt", "alternate"):
            if re.search(rf"(?:^|_){cand}(?:_|$)", low):
                layout = canonical_layout_name(cand)
                break

    ffn_size = _to_int(cfg.get("expert_intermediate_size"))
    ffn_fused = (ffn_size is not None and ffn_size > 2048) or ("ffn_fused" in low) or ("fuse" in low)

    return {
        "role": role,
        "layers": "" if layers is None else str(layers),
        "layout": layout,
        "ffn_fused": "true" if ffn_fused else "false",
    }


def normalize_row(raw: dict[str, Any]) -> dict[str, str]:
    """Normalize any dict-like eval record into a canonical string-valued registry row."""
    row = dict.fromkeys(COLUMNS, "")
    for col in COLUMNS:
        if col in raw and raw[col] is not None:
            row[col] = str(raw[col]).strip()

    row["ffn_fused"] = "true" if _to_bool(row["ffn_fused"]) else "false"
    for num_col in ("num_steps", "episodes_per_task", "n_action_steps", "seed", "suites_completed"):
        val = _to_int(row[num_col])
        row[num_col] = "" if val is None else str(val)
    for score_col in ("spatial", "object", "goal", "long", "avg"):
        row[score_col] = _fmt_num(row[score_col], digits=2)

    if row["wandb_run_id"] and not row["wandb_url"]:
        row["wandb_url"] = wandb_url(row["wandb_run_id"])
    if not row["status"]:
        completed = _to_int(row["suites_completed"], 0) or 0
        row["status"] = "complete" if completed >= len(SUITES) else "partial"
    return row


def row_key(row: dict[str, Any]) -> tuple[str, str, int, int, int, int]:
    """Deduplication key: ``(model, hidden_map, num_steps, episodes_per_task, n_action_steps, seed)``."""
    return (
        str(row.get("model") or "").strip(),
        str(row.get("hidden_map") or "").strip(),
        _to_int(row.get("num_steps"), 0) or 0,
        _to_int(row.get("episodes_per_task"), 10) or 10,
        _to_int(row.get("n_action_steps"), DEFAULT_N_ACTION_STEPS) or DEFAULT_N_ACTION_STEPS,
        _to_int(row.get("seed"), DEFAULT_SEED) or DEFAULT_SEED,
    )


def _row_rank(row: dict[str, str]) -> tuple[int, int, int, str]:
    is_complete = 1 if row.get("status") == "complete" else 0
    has_wandb_url = 1 if row.get("wandb_url") else 0
    src_prio = _SOURCE_PRIORITY.get(row.get("source", ""), 0)
    date_str = row.get("date", "")
    return (is_complete, has_wandb_url, src_prio, date_str)


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Merge ``incoming`` eval rows into ``existing``, deduplicating by :func:`row_key`."""
    by_key: dict[tuple[str, str, int, int, int, int], dict[str, str]] = {}
    for raw in (*existing, *incoming):
        norm = normalize_row(raw)
        if not norm["model"] or not norm["num_steps"]:
            continue
        k = row_key(norm)
        prev = by_key.get(k)
        if prev is None or _row_rank(norm) >= _row_rank(prev):
            by_key[k] = norm

    return sorted(
        by_key.values(),
        key=lambda r: (
            r["model"],
            r["hidden_map"],
            _to_int(r["episodes_per_task"], 0) or 0,
            -(_to_int(r["num_steps"], 0) or 0),
            r["date"],
        ),
    )


def load_registry(repo_id: str = DEFAULT_HF_EVAL_DATASET) -> list[dict[str, str]]:
    """Download and parse ``results.csv`` from ``repo_id`` (returns ``[]`` if missing)."""
    try:
        local_csv = hf_hub_download(repo_id=repo_id, filename=REGISTRY_FILENAME, repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
        return []
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return []
        raise
    text = Path(local_csv).read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [normalize_row(row) for row in reader if row.get("model")]


def _render_csv(rows: list[dict[str, str]], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


def build_registry_readme(repo_id: str, rows: list[dict[str, str]]) -> str:
    """Generate a Hugging Face dataset card with a markdown leaderboard and W&B links."""
    lines = [
        "---",
        "pretty_name: BriskVLA LIBERO Evaluation Results",
        "tags:",
        "  - robotics",
        "  - libero",
        "  - smolvla",
        "  - briskvla",
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        f"        path: {REGISTRY_FILENAME}",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Shared team evaluation ledger for **BriskVLA** (`teacher`, `student_*`, `shallow_*`) on LIBERO.",
        "Updated automatically by `project/briskvla/run_eval.sh` and `project/briskvla/eval_registry.py` so teammates never repeat completed evaluations.",
        "",
        "## Recorded Evaluations",
        "",
        "| Model | Hidden map | K | Ep/task | Spatial | Object | Goal | Long | Avg | Status | Evaluator | W&B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        wb = f"[`{r['wandb_run_id']}`]({r['wandb_url']})" if r["wandb_url"] else "—"
        lines.append(
            f"| `{r['model']}` | `{r['hidden_map'] or '—'}` | {r['num_steps']} | {r['episodes_per_task']} | "
            f"{r['spatial'] or '—'} | {r['object'] or '—'} | {r['goal'] or '—'} | {r['long'] or '—'} | "
            f"**{r['avg'] or '—'}** | {r['status']} | {r['evaluator'] or r['source']} | {wb} |"
        )
    lines.extend(
        [
            "",
            "## Sync & Check Commands",
            "",
            "```bash",
            "# See missing K={10, 3, 1} evals across all briskVLA HF models:",
            "uv run python project/briskvla/eval_registry.py plan --steps \"10 3 1\" --episodes 10",
            "",
            "# Sync any finished W&B eval runs:",
            "uv run python project/briskvla/eval_registry.py sync-wandb",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def save_registry(
    rows: list[dict[str, Any]],
    repo_id: str = DEFAULT_HF_EVAL_DATASET,
    *,
    commit_message: str = "Update BriskVLA eval registry",
    private: bool = False,
) -> str:
    """Write sorted ``results.csv`` and ``README.md`` to ``repo_id`` on Hugging Face Hub."""
    namespace = repo_id.split("/", 1)[0]
    ensure_hf_write_auth(namespace)
    norm_rows = merge_rows([], rows)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="briskvla_eval_reg_") as tmp:
        stage = Path(tmp)
        (stage / REGISTRY_FILENAME).write_text(_render_csv(norm_rows, COLUMNS), encoding="utf-8")
        (stage / "README.md").write_text(build_registry_readme(repo_id, norm_rows), encoding="utf-8")
        info = api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    return str(getattr(info, "commit_url", None) or f"https://huggingface.co/datasets/{repo_id}")


def parse_wandb_eval_run(
    run: Any,
    *,
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
) -> dict[str, str] | None:
    """Convert one W&B eval run object into a normalized registry row, or None if not an eval."""
    if getattr(run, "job_type", None) != "eval" or getattr(run, "state", None) != "finished":
        return None
    cfg = dict(getattr(run, "config", {}) or {})
    summary = dict(getattr(run, "summary", {}) or {})

    raw_name = str(cfg.get("name") or getattr(run, "name", "") or "").strip()
    model = re.sub(r"_K\d+$", "", raw_name)
    num_steps = _to_int(cfg.get("num_steps"))
    if num_steps is None:
        m = re.search(r"_K(\d+)$", str(getattr(run, "name", "") or ""))
        if m:
            num_steps = int(m.group(1))
    if not model or num_steps is None:
        return None

    meta = _parse_model_meta(model, cfg)
    scores = {
        "spatial": summary.get("eval/libero_spatial/pc_success"),
        "object": summary.get("eval/libero_object/pc_success"),
        "goal": summary.get("eval/libero_goal/pc_success"),
        "long": summary.get("eval/libero_10/pc_success"),
    }
    completed = _to_int(summary.get("eval/suites_completed"))
    if completed is None:
        completed = sum(1 for v in scores.values() if v is not None)
    avg = summary.get("eval/avg_success")
    if avg is None and completed == len(SUITES):
        avg = sum(float(scores[k]) for k in ("spatial", "object", "goal", "long")) / len(SUITES)
    elif avg is None:
        avg = summary.get("eval/mean_completed_success")

    user_obj = getattr(run, "user", None)
    evaluator = str(getattr(user_obj, "username", None) or cfg.get("host") or "")
    run_id = str(getattr(run, "id", "") or "")
    date_val = str(cfg.get("date") or getattr(run, "created_at", "") or "")

    return normalize_row(
        {
            "model": model,
            "policy": cfg.get("policy") or "",
            "role": meta["role"],
            "layers": meta["layers"],
            "layout": meta["layout"],
            "ffn_fused": meta["ffn_fused"],
            "hidden_map": cfg.get("hidden_map") or "",
            "num_steps": num_steps,
            "episodes_per_task": cfg.get("episodes_per_task", 10),
            "n_action_steps": cfg.get("n_action_steps", DEFAULT_N_ACTION_STEPS),
            "seed": cfg.get("seed", DEFAULT_SEED),
            **scores,
            "avg": avg,
            "suites_completed": completed,
            "status": "complete" if completed >= len(SUITES) else "partial",
            "evaluator": evaluator,
            "gpu": cfg.get("gpu") or "",
            "wandb_run_id": run_id,
            "wandb_url": wandb_url(run_id, entity=entity, project=project),
            "date": date_val,
            "source": "wandb",
        }
    )


def sync_wandb_runs(
    *,
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
    repo_id: str = DEFAULT_HF_EVAL_DATASET,
) -> list[dict[str, str]]:
    """Fetch all finished W&B eval runs and merge them into ``repo_id``."""
    import wandb

    api = wandb.Api()
    incoming: list[dict[str, str]] = []
    for run in api.runs(f"{entity}/{project}"):
        row = parse_wandb_eval_run(run, entity=entity, project=project)
        if row is not None:
            incoming.append(row)

    existing = load_registry(repo_id)
    merged = merge_rows(existing, incoming)
    save_registry(
        merged,
        repo_id=repo_id,
        commit_message=f"Sync {len(incoming)} W&B eval runs from {entity}/{project}",
    )
    return merged


def _clean_md_cell(text: str) -> str:
    return re.sub(r"[`*]", "", text).strip()


def _map_readme_model_name(display_name: str) -> str | None:
    cleaned = _clean_md_cell(display_name)
    for prefix, canonical in _README_NAME_MAP.items():
        if cleaned.startswith(prefix):
            return canonical
    return None


def parse_readme_tables(readme_path: Path) -> list[dict[str, str]]:
    """Parse the 4-suite Placement and FFN Fusion LIBERO tables from ``project/briskvla/README.md``."""
    if not readme_path.is_file():
        raise FileNotFoundError(f"README not found: {readme_path}")

    section = ""
    rows: list[dict[str, str]] = []
    for raw_line in readme_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            if "Placement" in line:
                section = "placement"
            elif "FFN Fusion" in line:
                section = "ffn_fusion"
            else:
                section = ""
            continue
        if not section or not line.startswith("|"):
            continue
        cells = [_clean_md_cell(c) for c in line.strip("|").split("|")]
        if not cells or cells[0] in ("Model", "---") or cells[0].startswith("---"):
            continue

        if section == "placement" and len(cells) >= 7:
            model_raw, k_raw, sp, ob, go, lo, avg = cells[:7]
            model = _map_readme_model_name(model_raw)
            if not model:
                continue
            hidden_map = "same_depth" if model.startswith("student_") else ""
        elif section == "ffn_fusion" and len(cells) >= 8:
            model_raw, _params, k_raw, sp, ob, go, lo, avg = cells[:8]
            model = _map_readme_model_name(model_raw)
            if not model:
                continue
            hidden_map = "uniform"
        else:
            continue

        num_steps = _to_int(k_raw)
        if num_steps is None:
            continue
        meta = _parse_model_meta(model)
        policy = TEACHER if model == "teacher" else ""
        rows.append(
            normalize_row(
                {
                    "model": model,
                    "policy": policy,
                    "role": meta["role"],
                    "layers": meta["layers"],
                    "layout": meta["layout"],
                    "ffn_fused": meta["ffn_fused"],
                    "hidden_map": hidden_map,
                    "num_steps": num_steps,
                    "episodes_per_task": 10,
                    "n_action_steps": DEFAULT_N_ACTION_STEPS,
                    "seed": DEFAULT_SEED,
                    "spatial": sp,
                    "object": ob,
                    "goal": go,
                    "long": lo,
                    "avg": avg,
                    "suites_completed": len(SUITES),
                    "status": "complete",
                    "evaluator": "team_readme",
                    "gpu": "",
                    "wandb_run_id": "",
                    "wandb_url": "",
                    "date": "",
                    "source": "readme",
                }
            )
        )
    return rows


def record_local_eval_dir(
    run_dir: Path,
    *,
    repo_id: str = DEFAULT_HF_EVAL_DATASET,
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
    evaluator: str = "",
) -> dict[str, str]:
    """Read a local ``run_eval.sh`` run directory and upsert it into ``repo_id``."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Eval run directory not found: {run_dir}")

    meta_path = run_dir / "run_metadata.json"
    meta_cfg = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    raw_name = str(meta_cfg.get("name") or run_dir.name).strip()
    model = re.sub(r"_K\d+$", "", raw_name)
    num_steps = _to_int(meta_cfg.get("num_steps"))
    if num_steps is None:
        m = re.search(r"_K(\d+)$", run_dir.name)
        if m:
            num_steps = int(m.group(1))
    if not model or num_steps is None:
        raise ValueError(f"Could not determine model name or num_steps from {run_dir}")

    suite_key_map = {
        "libero_spatial": "spatial",
        "libero_object": "object",
        "libero_goal": "goal",
        "libero_10": "long",
    }
    scores: dict[str, float | None] = dict.fromkeys(suite_key_map.values())
    completed = 0
    for suite, col in suite_key_map.items():
        info_path = run_dir / suite / "eval_info.json"
        if info_path.is_file():
            overall = json.loads(info_path.read_text(encoding="utf-8")).get("overall") or {}
            pc = _to_float(overall.get("pc_success"))
            if pc is not None:
                scores[col] = pc
                completed += 1
    if completed == 0:
        raise ValueError(f"No suite eval_info.json files found in {run_dir}")

    done_vals = [v for v in scores.values() if v is not None]
    avg = sum(done_vals) / len(done_vals) if done_vals else None

    wb_file = run_dir / "wandb_id.txt"
    wb_id = wb_file.read_text(encoding="utf-8").strip() if wb_file.is_file() else ""
    arch_meta = _parse_model_meta(model, meta_cfg)

    row = normalize_row(
        {
            "model": model,
            "policy": meta_cfg.get("policy") or "",
            "role": arch_meta["role"],
            "layers": arch_meta["layers"],
            "layout": arch_meta["layout"],
            "ffn_fused": arch_meta["ffn_fused"],
            "hidden_map": meta_cfg.get("hidden_map") or "",
            "num_steps": num_steps,
            "episodes_per_task": meta_cfg.get("episodes_per_task", 10),
            "n_action_steps": meta_cfg.get("n_action_steps", DEFAULT_N_ACTION_STEPS),
            "seed": meta_cfg.get("seed", DEFAULT_SEED),
            **scores,
            "avg": avg,
            "suites_completed": completed,
            "status": "complete" if completed >= len(SUITES) else "partial",
            "evaluator": evaluator or str(meta_cfg.get("evaluator") or ""),
            "gpu": meta_cfg.get("gpu") or "",
            "wandb_run_id": wb_id,
            "wandb_url": wandb_url(wb_id, entity=entity, project=project),
            "date": str(meta_cfg.get("date") or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
            "source": "local",
        }
    )

    existing = load_registry(repo_id)
    merged = merge_rows(existing, [row])
    save_registry(merged, repo_id=repo_id, commit_message=f"Record eval {model} K={num_steps}")
    return row


def find_matching_eval(
    rows: list[dict[str, str]],
    *,
    name: str,
    steps: int,
    episodes: int = 10,
    seed: int = DEFAULT_SEED,
    n_action_steps: int = DEFAULT_N_ACTION_STEPS,
    hidden_map: str = "",
) -> dict[str, str] | None:
    """Return a matching ``status == 'complete'`` row if one exists, else None."""
    for raw in rows:
        r = normalize_row(raw)
        if r["status"] != "complete":
            continue
        if r["model"] != name:
            continue
        if _to_int(r["num_steps"]) != steps:
            continue
        if _to_int(r["episodes_per_task"], 10) != episodes:
            continue
        if _to_int(r["seed"], DEFAULT_SEED) != seed:
            continue
        if _to_int(r["n_action_steps"], DEFAULT_N_ACTION_STEPS) != n_action_steps:
            continue
        if hidden_map and r["hidden_map"] != hidden_map:
            continue
        return r
    return None


def build_coverage_matrix(
    rows: list[dict[str, str]],
    *,
    steps_list: list[int],
    episodes: int = 10,
    hf_models: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[str, str, int]]]:
    """Build coverage rows across ``teacher``, HF models, and registry models, plus missing jobs."""
    policy_by_model: dict[str, str] = {"teacher": "teacher"}
    for repo in hf_models or []:
        short = repo.split("/", 1)[-1]
        policy_by_model[short] = repo

    all_models = set(policy_by_model) | {r["model"] for r in rows if r.get("model")}
    ordered_models = sorted(all_models, key=lambda m: (0 if m == "teacher" else 1 if m in policy_by_model else 2, m))

    matrix: list[dict[str, str]] = []
    missing_jobs: list[tuple[str, str, int]] = []
    for model in ordered_models:
        policy_spec = policy_by_model.get(model, "")
        entry: dict[str, str] = {"model": model, "policy_hf": policy_spec or "not_on_hf"}
        missing_k: list[str] = []
        for k in steps_list:
            # Prefer a row with a W&B URL if multiple hidden_maps exist for the same K.
            matches = [
                r
                for r in rows
                if r["status"] == "complete"
                and r["model"] == model
                and _to_int(r["num_steps"]) == k
                and _to_int(r["episodes_per_task"], 10) == episodes
            ]
            if matches:
                best = max(matches, key=_row_rank)
                entry[f"K={k}"] = best["avg"]
                entry[f"K={k}_wandb"] = best["wandb_url"] or best["source"]
            else:
                entry[f"K={k}"] = "MISSING"
                entry[f"K={k}_wandb"] = ""
                missing_k.append(str(k))
                if policy_spec:
                    missing_jobs.append((model, policy_spec, k))
        entry["missing_K"] = ",".join(missing_k)
        matrix.append(entry)
    return matrix, missing_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sync_p = sub.add_parser("sync-wandb", help="Sync all finished W&B eval runs into HF Dataset registry")
    sync_p.add_argument("--repo", default=DEFAULT_HF_EVAL_DATASET)
    sync_p.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    sync_p.add_argument("--project", default=DEFAULT_WANDB_PROJECT)

    bf_p = sub.add_parser("backfill-readme", help="Backfill 4-suite results from project/briskvla/README.md")
    bf_p.add_argument("--readme", type=Path, default=Path(__file__).resolve().parent / "README.md")
    bf_p.add_argument("--repo", default=DEFAULT_HF_EVAL_DATASET)

    rec_p = sub.add_parser("record", help="Record a local eval run folder (<name>_K<steps>/)")
    rec_p.add_argument("run_dir", type=Path)
    rec_p.add_argument("--repo", default=DEFAULT_HF_EVAL_DATASET)
    rec_p.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    rec_p.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    rec_p.add_argument("--evaluator", default="")

    chk_p = sub.add_parser("check", help="Exit 0 if a complete eval exists in the registry, 1 if missing")
    chk_p.add_argument("--name", required=True)
    chk_p.add_argument("--steps", type=int, required=True)
    chk_p.add_argument("--episodes", type=int, default=10)
    chk_p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    chk_p.add_argument("--n_action_steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    chk_p.add_argument("--hidden_map", default="")
    chk_p.add_argument("--repo", default=DEFAULT_HF_EVAL_DATASET)

    plan_p = sub.add_parser("plan", help="Show coverage matrix and missing evals across HF models + teacher")
    plan_p.add_argument("--steps", default="10 3 1")
    plan_p.add_argument("--episodes", type=int, default=10)
    plan_p.add_argument("--namespace", default=DEFAULT_HF_NAMESPACE)
    plan_p.add_argument("--repo", default=DEFAULT_HF_EVAL_DATASET)

    exp_p = sub.add_parser("export-csv", help="Export results.csv and optional coverage matrix CSV locally")
    exp_p.add_argument("out_csv", type=Path)
    exp_p.add_argument("--matrix_csv", type=Path, default=None)
    exp_p.add_argument("--steps", default="10 3 1")
    exp_p.add_argument("--episodes", type=int, default=10)
    exp_p.add_argument("--namespace", default=DEFAULT_HF_NAMESPACE)
    exp_p.add_argument("--repo", default=DEFAULT_HF_EVAL_DATASET)

    args = parser.parse_args()
    if args.command == "sync-wandb":
        merged = sync_wandb_runs(entity=args.entity, project=args.project, repo_id=args.repo)
        print(f"Synced W&B eval runs -> {args.repo} ({len(merged)} total rows)")
    elif args.command == "backfill-readme":
        parsed = parse_readme_tables(args.readme)
        existing = load_registry(args.repo)
        merged = merge_rows(existing, parsed)
        url = save_registry(
            merged,
            repo_id=args.repo,
            commit_message=f"Backfill {len(parsed)} README eval rows",
        )
        print(f"Backfilled {len(parsed)} README rows -> {args.repo} ({len(merged)} total rows): {url}")
    elif args.command == "record":
        row = record_local_eval_dir(
            args.run_dir,
            repo_id=args.repo,
            entity=args.entity,
            project=args.project,
            evaluator=args.evaluator,
        )
        print(f"Recorded {row['model']} K={row['num_steps']} ({row['status']}, avg={row['avg']}) -> {args.repo}")
    elif args.command == "check":
        rows = load_registry(args.repo)
        hit = find_matching_eval(
            rows,
            name=args.name,
            steps=args.steps,
            episodes=args.episodes,
            seed=args.seed,
            n_action_steps=args.n_action_steps,
            hidden_map=args.hidden_map,
        )
        if hit is not None:
            ref = hit["wandb_url"] or hit["source"]
            who = hit["evaluator"] or "unknown"
            print(f"[exists] {args.name} K={args.steps} ep={args.episodes}: avg={hit['avg']}% by {who} ({ref})")
            raise SystemExit(0)
        print(f"[missing] {args.name} K={args.steps} ep={args.episodes}")
        raise SystemExit(1)
    elif args.command == "plan":
        rows = load_registry(args.repo)
        hf_models = list_models(args.namespace)
        steps_list = [int(s) for s in args.steps.split()]
        matrix, missing_jobs = build_coverage_matrix(
            rows, steps_list=steps_list, episodes=args.episodes, hf_models=hf_models
        )
        k_headers = "".join(f"{'K=' + str(k):>10}" for k in steps_list)
        print(f"\nCoverage at episodes={args.episodes} (registry: {args.repo}):\n")
        print(f"{'Model':<38} {'HF Policy':<44}{k_headers}  Missing")
        print("-" * (88 + 10 * len(steps_list)))
        for entry in matrix:
            vals = "".join(f"{entry[f'K={k}']:>10}" for k in steps_list)
            print(f"{entry['model']:<38} {entry['policy_hf']:<44}{vals}  {entry['missing_K'] or 'DONE'}")
        print(f"\nMissing HF-hosted / teacher jobs ({len(missing_jobs)}):")
        for model, policy_spec, k in missing_jobs:
            print(f"  - {model} (policy={policy_spec}) K={k}")
    elif args.command == "export-csv":
        rows = load_registry(args.repo)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        args.out_csv.write_text(_render_csv(rows, COLUMNS), encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {args.out_csv}")
        if args.matrix_csv is not None:
            hf_models = list_models(args.namespace)
            steps_list = [int(s) for s in args.steps.split()]
            matrix, _ = build_coverage_matrix(
                rows, steps_list=steps_list, episodes=args.episodes, hf_models=hf_models
            )
            m_cols = ["model", "policy_hf"]
            for k in steps_list:
                m_cols.extend([f"K={k}", f"K={k}_wandb"])
            m_cols.append("missing_K")
            args.matrix_csv.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_csv.write_text(_render_csv(matrix, m_cols), encoding="utf-8")
            print(f"Wrote {len(matrix)} matrix rows to {args.matrix_csv}")


if __name__ == "__main__":
    main()
