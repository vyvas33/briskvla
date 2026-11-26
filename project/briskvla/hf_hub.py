"""Hugging Face Hub helpers for sharing BriskVLA models across the team.

Supports uploading trained or untrained BriskVLA checkpoints (with `run_info.json`, `wandb_id.txt`,
`metrics.jsonl`, and an auto-generated model card) and resolving `owner/repo[@rev]` specs in
`distill_student.py`, `build_student.py`, `latency.py`, `run_eval.sh`, and `colab_eval.py`.

Examples:
    # Upload a finished distillation run (`best/` + provenance) to HF Hub:
    uv run python project/briskvla/hf_hub.py upload \
        outputs/distill/shallow_8L_alternating_ffn_fused \
        --namespace <org-or-user>

    # Resolve an HF repo id (or local path / "teacher") to a local snapshot folder:
    uv run python project/briskvla/hf_hub.py resolve <org-or-user>/shallow_8L_alternating_ffn_fused

    # List team models on HF Hub:
    uv run python project/briskvla/hf_hub.py list --namespace <org-or-user>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_HF_NAMESPACE,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    TEACHER,
    TEACHER_REVISION,
    layout_name_for_layers,
    read_policy_arch,
)
from huggingface_hub import HfApi, snapshot_download

# Optional default namespace/prefix via environment (e.g. BRISKVLA_HF_NAMESPACE=briskvla-cmu).
HF_NAMESPACE_ENV = "BRISKVLA_HF_NAMESPACE"
HF_PREFIX_ENV = "BRISKVLA_HF_PREFIX"

_REPO_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_HF_SPEC_RE = re.compile(rf"^(?P<repo>{_REPO_SEGMENT}/{_REPO_SEGMENT})(?:@(?P<rev>[^\s@]+))?$")

CHECKPOINT_REQUIRED = ("model.safetensors", "config.json")
PROVENANCE_FILES = ("run_info.json", "wandb_id.txt", "metrics.jsonl")


def is_hf_repo_spec(spec: str | Path) -> bool:
    """Return True if `spec` is an HF `owner/repo[@rev]` identifier (not a local filesystem path)."""
    text = str(spec).strip()
    if not text or text.startswith(("/", ".", "~")) or "\\" in text:
        return False
    if Path(text).exists():
        return False
    return _HF_SPEC_RE.match(text) is not None


def parse_hf_spec(spec: str) -> tuple[str, str | None]:
    """Split `owner/repo[@rev]` into `(repo_id, revision)`."""
    match = _HF_SPEC_RE.match(spec.strip())
    if match is None:
        raise ValueError(
            f"Invalid Hugging Face repo spec {spec!r}; expected 'owner/repo' or 'owner/repo@revision'."
        )
    return match.group("repo"), match.group("rev")


def format_hf_repo_id(namespace: str, model_name: str, prefix: str = "") -> str:
    """Build `<namespace>/<prefix><model_name>`, passing through explicit `owner/repo` values."""
    if "/" in model_name:
        repo_id, _ = parse_hf_spec(model_name)
        return repo_id
    clean_ns = namespace.strip().strip("/")
    clean_prefix = prefix.strip()
    clean_name = model_name.strip().strip("/")
    if not clean_ns or not clean_name:
        raise ValueError(f"Both namespace ({namespace!r}) and model_name ({model_name!r}) must be non-empty.")
    repo_id = f"{clean_ns}/{clean_prefix}{clean_name}"
    parse_hf_spec(repo_id)
    return repo_id


def resolve_hf_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve HF Hub upload settings from CLI args and environment variables."""
    repo = getattr(args, "hf_repo", None) or None
    env_ns = os.environ.get(HF_NAMESPACE_ENV)
    namespace = getattr(args, "hf_namespace", None)
    if namespace is None:
        namespace = DEFAULT_HF_NAMESPACE if env_ns is None else (env_ns.strip() or None)
    prefix = getattr(args, "hf_prefix", None)
    if prefix is None:
        prefix = os.environ.get(HF_PREFIX_ENV, "")
    private = bool(getattr(args, "hf_private", False))
    disabled = bool(getattr(args, "no_hf_push", False))
    return {
        "repo": None if disabled else repo,
        "namespace": None if disabled else namespace,
        "prefix": prefix,
        "private": private,
    }


def add_hf_upload_args(parser: argparse.ArgumentParser) -> None:
    """Attach standard Hugging Face Hub upload flags to an argparse parser."""
    parser.add_argument(
        "--hf_repo",
        default=None,
        help="Explicit HF model repo id to upload to (e.g. org/shallow_8L_alternating_ffn_fused)",
    )
    parser.add_argument(
        "--hf_namespace",
        default=None,
        help=f"HF user or org namespace to push `<ns>/<out.name>` (default: ${HF_NAMESPACE_ENV})",
    )
    parser.add_argument(
        "--hf_prefix",
        default=None,
        help=f"Optional prefix prepended to repo names under --hf_namespace (default: ${HF_PREFIX_ENV})",
    )
    parser.add_argument("--hf_private", action="store_true", help="Create the HF model repo as private")
    parser.add_argument("--no_hf_push", action="store_true", help="Disable HF upload even if env var is set")


def resolve_target_repo(
    model_name: str,
    *,
    repo: str | None = None,
    namespace: str | None = None,
    prefix: str = "",
) -> str | None:
    """Resolve the target `owner/repo` for upload, or None if HF upload is not configured."""
    if repo:
        repo_id, _ = parse_hf_spec(repo)
        return repo_id
    if namespace:
        return format_hf_repo_id(namespace, model_name, prefix=prefix)
    return None


def resolve_policy_path(
    policy: str | Path,
    *,
    teacher_revision: str = TEACHER_REVISION,
    local_dir: Path | None = None,
) -> Path:
    """Resolve `'teacher'`, a local checkpoint dir, or an HF `owner/repo[@rev]` to a local folder."""
    raw = str(policy).strip()
    if raw in ("teacher", TEACHER):
        return Path(snapshot_download(TEACHER, revision=teacher_revision, local_dir=local_dir))
    local = Path(raw)
    if local.is_dir():
        return local
    if is_hf_repo_spec(raw):
        repo_id, rev = parse_hf_spec(raw)
        return Path(snapshot_download(repo_id, revision=rev, local_dir=local_dir))
    raise FileNotFoundError(
        f"Policy {policy!r} is neither an existing local directory nor a valid HF repo id ('owner/repo[@rev]')."
    )


def ensure_hf_write_auth(namespace: str | None = None) -> str:
    """Verify HF credentials exist and have write permission before starting a long run."""
    api = HfApi()
    try:
        info = api.whoami()
    except Exception as exc:
        raise SystemExit(
            "Hugging Face authentication is required to upload models. "
            "Run `uv run hf auth login` with a Write token or set HF_TOKEN."
        ) from exc

    user = str(info.get("name") or "")
    token_info = info.get("auth", {}).get("accessToken", {})
    role = token_info.get("role")
    if role == "read":
        raise SystemExit(
            f"Logged in to Hugging Face as {user!r} with a read-only token. "
            "Log in with a Write token (`uv run hf auth login` or set HF_TOKEN) to upload models."
        )
    if role == "fineGrained":
        fg = token_info.get("fineGrained") or {}
        perms: set[str] = set(fg.get("global") or [])
        target_ns = namespace or user
        for entry in fg.get("scoped") or []:
            entity = entry.get("entity") or {}
            if entity.get("name") == target_ns or entity.get("name") == user:
                perms.update(entry.get("permissions") or [])
        if not any("write" in p for p in perms):
            raise SystemExit(
                f"Fine-grained HF token for {user!r} has no write permissions for {target_ns!r} "
                f"(found {sorted(perms)}). Create a Write token or grant repo.write and re-login."
            )
    return user


def _find_checkpoint_and_run_dir(path: Path) -> tuple[Path, Path | None]:
    """Locate the checkpoint folder (`model.safetensors`) and optional parent run dir."""
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint or run directory not found: {path}")
    if (path / "model.safetensors").is_file():
        parent = path.parent if (path.parent / "run_info.json").is_file() else None
        return path, parent
    best = path / "best"
    if (best / "model.safetensors").is_file():
        return best, path
    raise FileNotFoundError(
        f"No model.safetensors found in {path} or {best}. Pass a checkpoint folder or a training run dir."
    )


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _load_final_metrics(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return {}
    records: list[dict[str, Any]] = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {}
    final = next((r for r in reversed(records) if r.get("split") == "final"), None)
    vals = [r for r in records if r.get("split") == "val"]
    out: dict[str, Any] = {}
    if final:
        out["final"] = final
    if vals:
        out["last_val"] = vals[-1]
        best_val = min(vals, key=lambda r: r.get("chunk_mse_teacher", float("inf")))
        out["best_val"] = best_val
    return out


def build_model_card(
    repo_id: str,
    ckpt_dir: Path,
    *,
    run_info: dict[str, Any] | None = None,
    wandb_id: str | None = None,
    metrics_summary: dict[str, Any] | None = None,
) -> str:
    """Generate a self-contained Hugging Face README.md model card for a BriskVLA checkpoint."""
    arch = read_policy_arch(ckpt_dir)
    info = run_info or {}
    layers = arch.get("expert_vlm_layers") or info.get("layers")
    num_layers = arch.get("num_expert_layers") or (len(layers) if isinstance(layers, list) else None)
    ffn_width = arch.get("expert_intermediate_size") or info.get("expert_intermediate_size")
    layout = info.get("layout_canonical") or (
        layout_name_for_layers(layers) if isinstance(layers, list) else None
    )
    role = info.get("role") or ("student" if repo_id.split("/")[-1].startswith("student_") else "shallow")
    model_name = repo_id.split("/")[-1]
    expert_params = info.get("expert_params_student")
    args_cfg = info.get("args") or {}

    tags = ["lerobot", "smolvla", "briskvla", "libero", "robotics", f"role-{role}"]
    if num_layers is not None:
        tags.append(f"{num_layers}L")
    if layout:
        tags.append(f"layout-{layout}")
    if ffn_width and ffn_width > 2048:
        tags.append("ffn-fusion")

    lines = [
        "---",
        "library_name: lerobot",
        f"base_model: {TEACHER}",
        "tags:",
        *(f"  - {t}" for t in tags),
        "---",
        "",
        f"# {repo_id}",
        "",
        "BriskVLA reduced-depth SmolVLA action-expert checkpoint for LIBERO.",
        "",
        "## Architecture & Training",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Role | `{role}` |",
        f"| Teacher | `{TEACHER}` @ `{TEACHER_REVISION[:12]}` |",
        f"| Expert layers | `{num_layers}L` (`{layers}`) |",
        f"| Canonical layout | `{layout or 'custom'}` |",
        f"| Expert FFN width | `{ffn_width or 'default (2048)'}` |",
    ]
    if expert_params:
        lines.append(f"| Expert parameters | `{expert_params / 1e6:.2f}M` |")
    if args_cfg:
        loss_str = (
            f"velocity={args_cfg.get('w_velocity')}, hidden={args_cfg.get('w_hidden')}, "
            f"output={args_cfg.get('w_output')}, gt={args_cfg.get('w_gt')}"
        )
        lines.append(f"| Training steps / batch | `{args_cfg.get('steps')}` steps, batch `{args_cfg.get('batch_size')}` |")
        lines.append(f"| Loss weights | `{loss_str}` |")
        if args_cfg.get("hidden_map"):
            lines.append(f"| Hidden map | `{args_cfg.get('hidden_map')}` |")
    if wandb_id:
        wb_url = f"https://wandb.ai/{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}/runs/{wandb_id}"
        lines.append(f"| W&B training run | [`{wandb_id}`]({wb_url}) |")
    if metrics_summary:
        best_val = metrics_summary.get("best_val") or {}
        if "chunk_mse_teacher" in best_val:
            lines.append(
                f"| Best val `chunk_mse_teacher` | `{best_val['chunk_mse_teacher']:.4f}` (step {best_val.get('step')}) |"
            )
        if "chunk_mse_gt" in best_val:
            lines.append(f"| Val `chunk_mse_gt` at best | `{best_val['chunk_mse_gt']:.4f}` |")

    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```bash",
            "# Evaluate on LIBERO (K=10 and K=1):",
            f'SUCCESS=1 LATENCY=0 STEPS="10 1" project/briskvla/run_eval.sh {model_name} {repo_id}',
            "",
            "# Measure latency:",
            f"uv run python project/briskvla/latency.py --policy {model_name}={repo_id} --out outputs/latency/{model_name}.json",
            "",
            "# Continue training / distill / compress further (e.g. 8L -> 4L):",
            f"uv run python project/briskvla/distill_student.py --student {repo_id} --out outputs/distill/{model_name}_ft",
            f"uv run python project/briskvla/build_student.py --source {repo_id} --group 2 --keep alternate --ffn fuse --out outputs/students/{model_name}_4L",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def stage_upload_dir(
    run_or_ckpt_dir: Path,
    repo_id: str,
    stage_dir: Path,
    *,
    include_metrics: bool = True,
) -> Path:
    """Copy checkpoint weights, processors, provenance files, and model card into `stage_dir`."""
    ckpt_dir, run_dir = _find_checkpoint_and_run_dir(run_or_ckpt_dir)
    for required in CHECKPOINT_REQUIRED:
        if not (ckpt_dir / required).is_file():
            raise FileNotFoundError(f"Checkpoint {ckpt_dir} is missing required file {required}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    for item in ckpt_dir.iterdir():
        if item.is_file() and item.name != "training_state.pt":
            shutil.copy2(item, stage_dir / item.name)

    if run_dir is not None:
        for name in PROVENANCE_FILES:
            if name == "metrics.jsonl" and not include_metrics:
                continue
            src = run_dir / name
            if src.is_file() and not (stage_dir / name).exists():
                shutil.copy2(src, stage_dir / name)

    run_info = _load_json_if_exists(stage_dir / "run_info.json")
    wandb_id_path = stage_dir / "wandb_id.txt"
    wandb_id = wandb_id_path.read_text().strip() if wandb_id_path.is_file() else None
    metrics_summary = _load_final_metrics(run_dir)
    if not (stage_dir / "README.md").exists():
        (stage_dir / "README.md").write_text(
            build_model_card(
                repo_id,
                ckpt_dir,
                run_info=run_info,
                wandb_id=wandb_id,
                metrics_summary=metrics_summary,
            )
        )
    return stage_dir


def upload_model(
    run_or_ckpt_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str | None = None,
    include_metrics: bool = True,
) -> str:
    """Create (if needed) and upload a BriskVLA checkpoint + provenance to Hugging Face Hub."""
    repo_id, _ = parse_hf_spec(repo_id)
    namespace = repo_id.split("/", 1)[0]
    ensure_hf_write_auth(namespace)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="briskvla_hf_stage_") as tmp:
        stage_dir = stage_upload_dir(
            run_or_ckpt_dir,
            repo_id,
            Path(tmp),
            include_metrics=include_metrics,
        )
        info = api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload BriskVLA checkpoint {repo_id}",
        )
    commit_url = getattr(info, "commit_url", None) or f"https://huggingface.co/{repo_id}"
    return str(commit_url)


def list_models(namespace: str, prefix: str = "") -> list[str]:
    """List model repo IDs published under `namespace` (optionally filtered by name prefix)."""
    api = HfApi()
    repos = []
    for model in api.list_models(author=namespace):
        model_id = str(model.id)
        short = model_id.split("/", 1)[-1]
        if not prefix or short.startswith(prefix):
            repos.append(model_id)
    return sorted(repos)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    auth_p = sub.add_parser("check-auth", help="Verify HF write credentials before running experiments")
    auth_p.add_argument("--namespace", default=os.environ.get(HF_NAMESPACE_ENV))

    res_p = sub.add_parser("resolve", help="Resolve 'teacher', a local dir, or an HF repo id to a local path")
    res_p.add_argument("policy", help="'teacher', local checkpoint dir, or 'owner/repo[@rev]'")
    res_p.add_argument("--local_dir", type=Path, default=None, help="Optional destination dir for HF download")

    up_p = sub.add_parser("upload", help="Upload a trained run dir (with best/) or checkpoint dir to HF Hub")
    up_p.add_argument("path", type=Path, help="Training run folder or checkpoint folder")
    up_p.add_argument("--repo", default=None, help="Explicit target HF repo id ('owner/repo')")
    up_p.add_argument(
        "--namespace",
        default=os.environ.get(HF_NAMESPACE_ENV),
        help=f"Target HF user or org (default: ${HF_NAMESPACE_ENV})",
    )
    up_p.add_argument("--name", default=None, help="Model name under --namespace (default: folder name)")
    up_p.add_argument("--prefix", default=os.environ.get(HF_PREFIX_ENV, ""), help="Optional repo name prefix")
    up_p.add_argument("--private", action="store_true", help="Create the HF repo as private")
    up_p.add_argument("--message", default=None, help="Commit message for the HF upload")
    up_p.add_argument("--no_metrics", action="store_true", help="Omit metrics.jsonl from upload")

    ls_p = sub.add_parser("list", help="List BriskVLA models published under an HF user/org")
    ls_p.add_argument(
        "--namespace",
        default=os.environ.get(HF_NAMESPACE_ENV),
        required=not os.environ.get(HF_NAMESPACE_ENV),
        help=f"HF user or org to list (default: ${HF_NAMESPACE_ENV})",
    )
    ls_p.add_argument("--prefix", default=os.environ.get(HF_PREFIX_ENV, ""), help="Filter by repo name prefix")

    args = parser.parse_args()
    if args.command == "check-auth":
        user = ensure_hf_write_auth(args.namespace)
        print(f"HF write auth OK ({user})")
    elif args.command == "resolve":
        print(resolve_policy_path(args.policy, local_dir=args.local_dir))
    elif args.command == "upload":
        default_name = args.path.parent.name if args.path.name == "best" else args.path.name
        repo_id = resolve_target_repo(
            args.name or default_name,
            repo=args.repo,
            namespace=args.namespace,
            prefix=args.prefix,
        )
        if not repo_id:
            raise SystemExit(f"Provide --repo or --namespace (or set {HF_NAMESPACE_ENV}).")
        url = upload_model(
            args.path,
            repo_id,
            private=args.private,
            commit_message=args.message,
            include_metrics=not args.no_metrics,
        )
        print(f"Uploaded {args.path} -> {url}")
    elif args.command == "list":
        for repo_id in list_models(args.namespace, prefix=args.prefix):
            print(repo_id)


if __name__ == "__main__":
    main()
