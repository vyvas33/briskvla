"""Checks that build_student.py builds the student correctly.

Run with:
    uv run --with pytest python -m pytest project/briskvla/test_student.py -sv

Downloads the teacher on first run. Uses random inputs, so no dataset is needed.
"""

import pytest
import torch
from build_student import build_student, download_teacher, save_student

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

K8_LAYERS = [0, 1, 4, 5, 8, 9, 12, 13]


@pytest.fixture(scope="module")
def teacher_dir():
    return download_teacher()


@pytest.fixture(scope="module")
def teacher(teacher_dir):
    return SmolVLAPolicy.from_pretrained(teacher_dir)


@pytest.fixture(scope="module")
def k8_student(teacher):
    return build_student(teacher, K8_LAYERS)


def random_inputs(policy: SmolVLAPolicy, batch_size: int = 1, seed: int = 0) -> dict:
    """Random tensors shaped like what sample_actions gets after preprocessing."""
    g = torch.Generator().manual_seed(seed)
    cfg = policy.config
    h, w = cfg.resize_imgs_with_padding
    inputs = {
        "images": [torch.rand(batch_size, 3, h, w, generator=g) * 2 - 1 for _ in range(2)],
        "img_masks": [torch.ones(batch_size, dtype=torch.bool) for _ in range(2)],
        "lang_tokens": torch.randint(0, 1000, (batch_size, 16), generator=g),
        "lang_masks": torch.ones(batch_size, 16, dtype=torch.bool),
        "state": torch.randn(batch_size, cfg.max_state_dim, generator=g),
        "noise": torch.randn(batch_size, cfg.chunk_size, cfg.max_action_dim, generator=g),
    }
    device = cfg.device
    return {k: [t.to(device) for t in v] if isinstance(v, list) else v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def test_full_student_matches_teacher(teacher):
    """Keeping all 16 layers must give back the teacher exactly."""
    student = build_student(teacher, list(range(16)))
    inputs = random_inputs(teacher)
    expected = teacher.model.sample_actions(**inputs)
    actual = student.model.sample_actions(**inputs)
    assert torch.equal(expected, actual), (expected - actual).abs().max()


def test_k8_layers_are_teacher_copies(teacher, k8_student):
    teacher_layers = teacher.model.vlm_with_expert.lm_expert.layers
    student_layers = k8_student.model.vlm_with_expert.lm_expert.layers
    assert len(student_layers) == len(K8_LAYERS)
    for j, depth in enumerate(K8_LAYERS):
        t_sd = teacher_layers[depth].state_dict()
        s_sd = student_layers[j].state_dict()
        assert t_sd.keys() == s_sd.keys()
        for name in t_sd:
            assert torch.equal(t_sd[name], s_sd[name]), f"student layer {j} vs teacher depth {depth}: {name}"


@torch.no_grad()
def test_k8_student_runs(teacher, k8_student):
    actions = k8_student.model.sample_actions(**random_inputs(teacher))
    assert actions.shape == (1, teacher.config.chunk_size, teacher.config.max_action_dim)
    assert torch.isfinite(actions).all()


def test_only_expert_side_trains(k8_student):
    trainable = [n for n, p in k8_student.named_parameters() if p.requires_grad]
    assert trainable
    frozen_by_mistake = [n for n in trainable if "lm_expert" not in n and ".action_" not in n]
    assert not frozen_by_mistake, frozen_by_mistake


@torch.no_grad()
def test_save_and_reload(teacher, teacher_dir, k8_student, tmp_path):
    """The saved folder loads like any checkpoint and gives the same actions."""
    save_student(k8_student, teacher_dir, tmp_path)
    reloaded = SmolVLAPolicy.from_pretrained(tmp_path)
    assert reloaded.config.expert_vlm_layers == K8_LAYERS
    assert sorted(p.name for p in tmp_path.glob("policy_*processor*")) == sorted(
        p.name for p in teacher_dir.glob("policy_*processor*")
    )
    inputs = random_inputs(teacher)
    assert torch.equal(k8_student.model.sample_actions(**inputs), reloaded.model.sample_actions(**inputs))
