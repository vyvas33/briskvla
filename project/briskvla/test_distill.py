"""Checks the distillation losses in distill_student.py.

Run with:
    uv run --with pytest python -m pytest project/briskvla/test_distill.py -sv

Uses random inputs, so no dataset is needed.
"""

import argparse
import json

import pytest
import torch
from build_student import download_teacher, save_student
from distill_student import (
    LAYOUTS,
    RESUME_KEYS,
    TRAINING_STATE,
    compute_losses,
    find_resume_dir,
    hidden_pairs,
    load_student,
    load_teacher,
    load_training_state,
    lr_lambda,
    save_atomic,
    save_training_state,
    trainable_parameters,
    truncate_metrics,
)

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
    assert hidden_pairs(LAYOUTS["early"], "same_depth") == [(d, d) for d in range(8)]
    assert hidden_pairs(LAYOUTS["early"], "uniform") == list(zip(range(8), range(1, 16, 2), strict=True))
    assert hidden_pairs(LAYOUTS["late"], "uniform")[-1] == (15, 15)


@torch.no_grad()
def test_full_student_has_zero_distill_loss(teacher):
    """A student with all 16 layers is the teacher, so every distillation loss must be 0."""
    student = load_student(teacher, list(range(16)), None).eval()
    torch.manual_seed(0)
    losses = compute_losses(
        teacher, student, random_batch(teacher), ALL_ON, hidden_pairs(list(range(16)), "same_depth"), 10
    )
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
    losses = compute_losses(
        teacher, student, random_batch(teacher), ALL_ON, hidden_pairs(layers, "uniform"), 3
    )
    assert all(torch.isfinite(v) for v in losses.values()), losses
    assert losses["velocity"].item() > 0
    losses["total"].backward()

    assert all(p.grad is not None for p in student.model.vlm_with_expert.lm_expert.layers.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    opt.step()
    for k, v in teacher.state_dict().items():
        if "lm_expert" in k:
            assert torch.equal(v, teacher_before[k]), k


def test_save_atomic_keeps_a_complete_copy(tmp_path):
    folder = tmp_path / "last"
    save_atomic(folder, lambda d: (d.mkdir(), (d / "a.txt").write_text("1")))
    save_atomic(folder, lambda d: (d.mkdir(), (d / "a.txt").write_text("2")))
    assert (folder / "a.txt").read_text() == "2"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last"]

    def dies_while_writing(d):
        d.mkdir()
        raise RuntimeError("machine reclaimed")

    with pytest.raises(RuntimeError):
        save_atomic(folder, dies_while_writing)
    assert (folder / "a.txt").read_text() == "2"


def test_find_resume_dir(tmp_path):
    assert find_resume_dir(tmp_path) is None
    old = tmp_path / "last.old"
    old.mkdir()
    (old / TRAINING_STATE).touch()
    (old / "model.safetensors").touch()
    # Died between moving last/ to last.old/ and last.tmp/ to last/
    assert find_resume_dir(tmp_path) == old
    last = tmp_path / "last"
    last.mkdir()
    (last / "model.safetensors").touch()
    assert find_resume_dir(tmp_path) == old  # last/ without a training state does not count
    (last / TRAINING_STATE).touch()
    assert find_resume_dir(tmp_path) == last


def test_truncate_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [{"split": "val", "step": 0}, {"split": "train", "step": 500}, {"split": "val", "step": 500},
               {"split": "train", "step": 550}, {"split": "final", "step": 500}]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    truncate_metrics(path, 500)
    assert [json.loads(line) for line in path.read_text().splitlines()] == records[:3]


def make_optimizer(student, total_steps: int = 100):
    opt = torch.optim.AdamW(trainable_parameters(student), lr=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, 1e-4, 2.5e-6, 10, total_steps))
    return opt, sched


def test_resume_restores_student_optimizer_and_schedule(teacher, tmp_path):
    """A checkpoint written mid-run restores the trained weights, AdamW moments, LR step and RNG."""
    layers = LAYOUTS["spread"]
    args = argparse.Namespace(**dict.fromkeys(RESUME_KEYS))
    teacher_dir = download_teacher()

    student = load_student(teacher, layers, None).train()
    opt, sched = make_optimizer(student)
    for step in range(3):
        torch.manual_seed(step)
        losses = compute_losses(
            teacher, student, random_batch(teacher), ALL_ON, hidden_pairs(layers, "same_depth"), 2
        )
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        sched.step()
    save_student(student, teacher_dir, tmp_path)
    save_training_state(tmp_path / TRAINING_STATE, 3, 0.25, opt, sched, args, layers)
    expected_rand = torch.rand(3)

    fresh = load_student(teacher, layers, None).train()
    fresh_opt, fresh_sched = make_optimizer(fresh)
    assert not torch.equal(fresh.model.action_out_proj.weight, student.model.action_out_proj.weight)
    step, best = load_training_state(tmp_path, fresh, fresh_opt, fresh_sched, args, layers)

    assert (step, best) == (3, 0.25)
    assert torch.equal(torch.rand(3), expected_rand)
    trained = {k: v for k, v in student.state_dict().items() if ".vlm_with_expert.vlm." not in k}
    for k, v in fresh.state_dict().items():
        if k in trained:
            assert torch.equal(v, trained[k]), k
    assert fresh_sched.last_epoch == sched.last_epoch and fresh_sched.get_last_lr() == sched.get_last_lr()
    for p_old, p_new in zip(trainable_parameters(student), trainable_parameters(fresh), strict=True):
        assert torch.equal(opt.state[p_old]["exp_avg_sq"], fresh_opt.state[p_new]["exp_avg_sq"])

    other = argparse.Namespace(**{**vars(args), "steps": 9999})
    with pytest.raises(ValueError, match="settings differ"):
        load_training_state(
            tmp_path, load_student(teacher, layers, None), *make_optimizer(fresh), other, layers
        )
