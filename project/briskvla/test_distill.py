"""Checks the distillation losses in distill_student.py.

Run with:
    uv run --with pytest python -m pytest project/briskvla/test_distill.py -sv

Uses random inputs, so no dataset is needed.
"""

import pytest
import torch
from distill_student import LAYOUTS, compute_losses, hidden_pairs, load_student, load_teacher, trainable_parameters
from build_student import download_teacher

ALL_ON = {"velocity": 1.0, "hidden": 1.0, "output": 1.0, "gt": 1.0}


@pytest.fixture(scope="module")
def teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_teacher(download_teacher(), device, empty_cameras=1)


def random_batch(policy, batch_size: int = 2, seed: int = 0) -> dict:
    """Random tensors shaped like a preprocessed lerobot/libero batch."""
    g = torch.Generator().manual_seed(seed)
    cfg = policy.config
    batch = {
        "observation.images.camera1": torch.rand(batch_size, 3, 256, 256, generator=g),
        "observation.images.camera2": torch.rand(batch_size, 3, 256, 256, generator=g),
        "observation.state": torch.randn(batch_size, 8, generator=g),
        "action": torch.randn(batch_size, cfg.chunk_size, 7, generator=g),
        "action_is_pad": torch.zeros(batch_size, cfg.chunk_size, dtype=torch.bool),
        "observation.language.tokens": torch.randint(0, 1000, (batch_size, 16), generator=g),
        "observation.language.attention_mask": torch.ones(batch_size, 16, dtype=torch.bool),
    }
    return {k: v.to(cfg.device) for k, v in batch.items()}


def test_hidden_pairs():
    assert hidden_pairs(LAYOUTS["first8"], "same_depth") == [(d, d) for d in range(8)]
    assert hidden_pairs(LAYOUTS["first8"], "uniform") == list(zip(range(8), range(1, 16, 2), strict=True))
    assert hidden_pairs(LAYOUTS["last8"], "uniform")[-1] == (15, 15)


@torch.no_grad()
def test_full_student_has_zero_distill_loss(teacher):
    """A student with all 16 layers is the teacher, so every distillation loss must be 0."""
    student = load_student(teacher, list(range(16)), None).eval()
    torch.manual_seed(0)
    losses = compute_losses(teacher, student, random_batch(teacher), ALL_ON, hidden_pairs(list(range(16)), "same_depth"), 10)
    for name in ("velocity", "hidden", "output"):
        assert losses[name].item() < 1e-10, (name, losses[name].item())


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_one_step_trains_only_the_student_expert(teacher, layout):
    """All losses are finite, gradients reach only trainable student weights and the teacher is unchanged."""
    layers = LAYOUTS[layout]
    student = load_student(teacher, layers, None).train()
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items() if "lm_expert" in k}
    params = trainable_parameters(student)
    opt = torch.optim.AdamW(params, lr=1e-4)

    torch.manual_seed(0)
    losses = compute_losses(teacher, student, random_batch(teacher), ALL_ON, hidden_pairs(layers, "uniform"), 3)
    assert all(torch.isfinite(v) for v in losses.values()), losses
    assert losses["velocity"].item() > 0
    losses["total"].backward()

    assert all(p.grad is not None for p in student.model.vlm_with_expert.lm_expert.layers.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    opt.step()
    for k, v in teacher.state_dict().items():
        if "lm_expert" in k:
            assert torch.equal(v, teacher_before[k]), k
