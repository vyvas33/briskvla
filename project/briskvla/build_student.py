"""Build an untrained BriskVLA reduced-depth ("shallow") model from the SmolVLA teacher.

Two ways to make the expert shallower. Both keep the teacher's VLM unchanged.

--layout or --layers keeps a subset of the teacher's expert layers. Each kept layer stays at its
original depth and starts as an exact copy of the teacher's layer there.

--group merges every `group` consecutive expert layers of a source model (the teacher or an earlier
model) into one layer. The merged layer keeps the attention of one member (--keep). With --ffn fuse
it keeps the FFNs of all members side by side as one wider FFN, which reads the same input and adds
all their outputs to the residual stream. With --ffn drop it keeps only the kept member's FFN, which
is the layer-dropping baseline with the same attention depths.

Examples:
    uv run python project/briskvla/build_student.py \
        --layout spread --out outputs/students/shallow_8L_spread_untrained
    uv run python project/briskvla/build_student.py \
        --group 2 --keep alternate --ffn fuse \
        --out outputs/students/shallow_8L_alternating_ffn_fused_untrained
"""

import argparse
import copy
import dataclasses
import re
import shutil
from pathlib import Path

import torch
from common import ALL_LAYOUT_CHOICES, TEACHER, TEACHER_REVISION, resolve_layout
from hf_hub import (
    add_hf_upload_args,
    ensure_hf_write_auth,
    resolve_hf_settings,
    resolve_policy_path,
    resolve_target_repo,
    upload_model,
)
from huggingface_hub import snapshot_download

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

EXPERT_LAYER_KEY = re.compile(r"lm_expert\.layers\.(\d+)\.")


def download_teacher(teacher: str = TEACHER, revision: str = TEACHER_REVISION) -> Path:
    """Return a local folder with the teacher checkpoint, downloading it if needed."""
    if Path(teacher).is_dir():
        return Path(teacher)
    return Path(snapshot_download(teacher, revision=revision))


def make_student_config(
    teacher_config: SmolVLAConfig, layers: list[int], intermediate_size: int | None = None
) -> SmolVLAConfig:
    """Teacher config with a smaller expert. The VLM and state projection stay frozen so the
    student sees exactly the teacher's prefix. `intermediate_size` sets the expert FFN width
    (None keeps the source's)."""
    if intermediate_size is None:
        intermediate_size = teacher_config.expert_intermediate_size
    return dataclasses.replace(
        copy.deepcopy(teacher_config),
        expert_vlm_layers=list(layers),
        expert_intermediate_size=intermediate_size,
        train_expert_only=True,
        freeze_vision_encoder=True,
        train_state_proj=False,
    )


def remap_state_dict(teacher_state_dict: dict, layers: list[int]) -> dict:
    """Keep the teacher expert layers at the given depths and renumber them 0..k-1.
    Everything that is not an expert layer is copied unchanged."""
    new_index = {depth: j for j, depth in enumerate(layers)}
    student_state_dict = {}
    for key, tensor in teacher_state_dict.items():
        match = EXPERT_LAYER_KEY.search(key)
        if match is None:
            student_state_dict[key] = tensor
            continue
        depth = int(match.group(1))
        if depth not in new_index:
            continue
        new_key = key.replace(f"lm_expert.layers.{depth}.", f"lm_expert.layers.{new_index[depth]}.")
        student_state_dict[new_key] = tensor
    return student_state_dict


def build_student(teacher: SmolVLAPolicy, layers: list[int]) -> SmolVLAPolicy:
    student = SmolVLAPolicy(make_student_config(teacher.config, layers))
    # strict=True fails if the remap missed or added any weight
    student.load_state_dict(remap_state_dict(teacher.state_dict(), layers), strict=True)
    student.to(teacher.config.device)
    student.eval()
    return student


KEEP_CHOICES = ("first", "last", "alternate")
FFN_CHOICES = ("fuse", "drop")
EXPERT_PREFIX = "model.vlm_with_expert.lm_expert.layers"


def make_groups(num_layers: int, group: int, keep: str) -> list[tuple[list[int], int]]:
    """Split expert layer indices 0..num_layers-1 into consecutive groups of `group` and pick the
    member whose attention is kept. `alternate` keeps the first member in even groups and the last
    in odd groups, so with pairs of the teacher it keeps 4 joint- and 4 cross-attention layers and
    still reads the deepest VLM layer."""
    if group < 1 or num_layers % group != 0:
        raise ValueError(f"group={group} must divide the number of expert layers ({num_layers})")
    if keep not in KEEP_CHOICES:
        raise ValueError(f"keep must be one of {KEEP_CHOICES}, got {keep}")
    groups = []
    for g, start in enumerate(range(0, num_layers, group)):
        members = list(range(start, start + group))
        kept = members[0] if keep == "first" or (keep == "alternate" and g % 2 == 0) else members[-1]
        groups.append((members, kept))
    return groups


def fuse_ffns(layer_weights: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """One SwiGLU equal to the sum of several, each with its own pre-FFN RMSNorm gain.

    FFN_k(x) = down_k(silu(gate_k(g_k * n)) * up_k(g_k * n)), with n = rmsnorm(x) without a gain.
    Folding g_k into the input columns of gate_k and up_k and stacking the hidden units of all k
    gives one FFN whose output is the sum, with a norm gain of 1. The fold is done in float32 and
    rounded to the stored dtype once.
    """
    fused: dict[str, list[torch.Tensor]] = {"gate": [], "up": [], "down": []}
    for w in layer_weights:
        gain = w["post_attention_layernorm.weight"].float()
        fused["gate"].append(w["mlp.gate_proj.weight"].float() * gain[None, :])
        fused["up"].append(w["mlp.up_proj.weight"].float() * gain[None, :])
        fused["down"].append(w["mlp.down_proj.weight"].float())
    ref = layer_weights[0]
    return {
        "mlp.gate_proj.weight": torch.cat(fused["gate"], 0).to(ref["mlp.gate_proj.weight"].dtype),
        "mlp.up_proj.weight": torch.cat(fused["up"], 0).to(ref["mlp.up_proj.weight"].dtype),
        "mlp.down_proj.weight": torch.cat(fused["down"], 1).to(ref["mlp.down_proj.weight"].dtype),
        "post_attention_layernorm.weight": torch.ones_like(ref["post_attention_layernorm.weight"]),
    }


def is_ffn_key(name: str) -> bool:
    return name.startswith("mlp.") or name.startswith("post_attention_layernorm.")


def compress(source: SmolVLAPolicy, group: int, keep: str = "alternate", ffn: str = "fuse") -> SmolVLAPolicy:
    """Merge every `group` consecutive expert layers of `source` into one layer.

    `source` can be the teacher or any student built here, including an already merged one. The
    merged layer sits at the depth of the kept member, with that member's attention and input norm.
    `ffn="fuse"` gives it the FFNs of all members (see fuse_ffns), `ffn="drop"` only the kept one's.
    """
    if ffn not in FFN_CHOICES:
        raise ValueError(f"ffn must be one of {FFN_CHOICES}, got {ffn}")
    depths = source.model.vlm_with_expert.expert_vlm_layers
    groups = make_groups(len(depths), group, keep)

    # Split the state dict into per-layer weights (by source expert index) and everything else
    per_layer: dict[int, dict[str, torch.Tensor]] = {}
    other: dict[str, torch.Tensor] = {}
    for key, tensor in source.state_dict().items():
        match = EXPERT_LAYER_KEY.search(key)
        if match is None:
            other[key] = tensor
            continue
        per_layer.setdefault(int(match.group(1)), {})[key[match.end() :]] = tensor

    width = per_layer[0]["mlp.gate_proj.weight"].shape[0]
    new_width = width * group if ffn == "fuse" else width
    new_state = dict(other)
    for j, (members, kept) in enumerate(groups):
        weights = {k: v for k, v in per_layer[kept].items() if not is_ffn_key(k)}
        if ffn == "fuse":
            weights.update(fuse_ffns([per_layer[m] for m in members]))
        else:
            weights.update({k: v for k, v in per_layer[kept].items() if is_ffn_key(k)})
        new_state.update({f"{EXPERT_PREFIX}.{j}.{k}": v for k, v in weights.items()})

    new_depths = [depths[kept] for _, kept in groups]
    student = SmolVLAPolicy(make_student_config(source.config, new_depths, new_width))
    # strict=True fails if the merge missed or added any weight
    student.load_state_dict(new_state, strict=True)
    student.to(source.config.device)
    student.eval()
    return student


def save_student(student: SmolVLAPolicy, teacher_dir: Path, out_dir: Path) -> None:
    """Save the student in the same layout as the teacher so lerobot-eval and lerobot-train can load it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir)
    # Normalization stats and processor steps are the same as the teacher's.
    # copyfile skips the read-only permissions of the HF cache so the folder can be rebuilt.
    for path in teacher_dir.glob("policy_*processor*"):
        shutil.copyfile(path, out_dir / path.name)


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--layout", choices=ALL_LAYOUT_CHOICES, help="Named expert layer layout")
    which.add_argument("--layers", help="VLM depths to keep an expert layer at, e.g. 0,1,4,5")
    which.add_argument("--group", type=int, help="Merge this many consecutive expert layers into one")
    parser.add_argument(
        "--keep", choices=KEEP_CHOICES, default="alternate", help="With --group: kept attention"
    )
    parser.add_argument("--ffn", choices=FFN_CHOICES, default="fuse", help="With --group: fuse or drop FFNs")
    parser.add_argument("--out", required=True, type=Path, help="Folder to save the student in")
    parser.add_argument("--teacher", default=TEACHER, help="Hub repo id or local checkpoint folder")
    parser.add_argument("--revision", default=TEACHER_REVISION, help="Hub revision of the teacher")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="With --group: local checkpoint dir or HF repo id ('owner/repo[@rev]') to merge (default: teacher)",
    )
    add_hf_upload_args(parser)
    args = parser.parse_args()

    hf = resolve_hf_settings(args)
    hf_repo = resolve_target_repo(args.out.name, repo=hf["repo"], namespace=hf["namespace"], prefix=hf["prefix"])
    if hf_repo:
        ensure_hf_write_auth(hf_repo.split("/", 1)[0])

    teacher_dir = download_teacher(args.teacher, args.revision)
    teacher = SmolVLAPolicy.from_pretrained(teacher_dir)
    if args.layout or args.layers:
        layers = resolve_layout(args.layout) if args.layout else [int(x) for x in args.layers.split(",")]
        student = build_student(teacher, layers)
    else:
        source_path = resolve_policy_path(args.source) if args.source else None
        source = SmolVLAPolicy.from_pretrained(source_path) if source_path else teacher
        student = compress(source, args.group, args.keep, args.ffn)
    save_student(student, teacher_dir, args.out)

    expert = student.model.vlm_with_expert
    t = count_params(teacher.model.vlm_with_expert.lm_expert)
    s = count_params(expert.lm_expert)
    print(
        f"Expert layers at depths {expert.expert_vlm_layers}, FFN width {student.config.expert_intermediate_size}"
    )
    print(f"Expert params: teacher {t / 1e6:.1f}M, student {s / 1e6:.1f}M")
    print(f"Saved to {args.out}")
    if hf_repo:
        url = upload_model(args.out, hf_repo, private=hf["private"])
        print(f"Uploaded to Hugging Face Hub: {url}")


if __name__ == "__main__":
    main()
