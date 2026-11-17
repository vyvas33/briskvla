"""Build a BriskVLA student from the SmolVLA teacher.

The student keeps the teacher's VLM and a subset of its expert layers. Each kept layer stays at
its original depth and starts as an exact copy of the teacher's layer there.

Example:
    uv run python project/briskvla/build_student.py \
        --layers 0,1,4,5,8,9,12,13 --out outputs/students/k8_pairs
"""

import argparse
import copy
import dataclasses
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

TEACHER = "lerobot/smolvla_libero"
# The revision we evaluated on LIBERO
TEACHER_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"

EXPERT_LAYER_KEY = re.compile(r"lm_expert\.layers\.(\d+)\.")


def download_teacher(teacher: str = TEACHER, revision: str = TEACHER_REVISION) -> Path:
    """Return a local folder with the teacher checkpoint, downloading it if needed."""
    if Path(teacher).is_dir():
        return Path(teacher)
    return Path(snapshot_download(teacher, revision=revision))


def make_student_config(teacher_config: SmolVLAConfig, layers: list[int]) -> SmolVLAConfig:
    """Teacher config with a smaller expert. The VLM and state projection stay frozen so the
    student sees exactly the teacher's prefix."""
    return dataclasses.replace(
        copy.deepcopy(teacher_config),
        expert_vlm_layers=list(layers),
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", required=True, help="VLM depths to keep an expert layer at, e.g. 0,1,4,5")
    parser.add_argument("--out", required=True, type=Path, help="Folder to save the student in")
    parser.add_argument("--teacher", default=TEACHER, help="Hub repo id or local checkpoint folder")
    parser.add_argument("--revision", default=TEACHER_REVISION, help="Hub revision of the teacher")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    teacher_dir = download_teacher(args.teacher, args.revision)
    teacher = SmolVLAPolicy.from_pretrained(teacher_dir)
    student = build_student(teacher, layers)
    save_student(student, teacher_dir, args.out)

    t = count_params(teacher.model.vlm_with_expert.lm_expert)
    s = count_params(student.model.vlm_with_expert.lm_expert)
    print(f"Expert layers at depths {layers}")
    print(f"Expert params: teacher {t / 1e6:.1f}M, student {s / 1e6:.1f}M")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
