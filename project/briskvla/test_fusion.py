"""Checks that compress() in build_student.py merges expert layers correctly.

Run with:
    uv run --with pytest python -m pytest project/briskvla/test_fusion.py -sv

Downloads the teacher on first run. Uses random inputs, so no dataset is needed.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from build_student import build_student, compress, download_teacher, fuse_ffns, make_groups, save_student
from test_student import random_inputs

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ALTERNATE_PAIRS = [0, 3, 4, 7, 8, 11, 12, 15]


@pytest.fixture(scope="module")
def teacher_dir():
    return download_teacher()


@pytest.fixture(scope="module")
def teacher(teacher_dir):
    return SmolVLAPolicy.from_pretrained(teacher_dir)


@pytest.fixture(scope="module")
def shallow_ffn_fused(teacher):
    return compress(teacher, group=2, keep="alternate", ffn="fuse")


def expert_params(policy: SmolVLAPolicy) -> int:
    return sum(p.numel() for p in policy.model.vlm_with_expert.lm_expert.parameters())


def rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def swiglu(n: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(n, gate)) * F.linear(n, up), down)


def test_make_groups():
    depths = [kept for _, kept in make_groups(16, 2, "alternate")]
    assert depths == ALTERNATE_PAIRS
    assert [kept for _, kept in make_groups(16, 2, "first")] == list(range(0, 16, 2))
    assert [kept for _, kept in make_groups(16, 4, "last")] == [3, 7, 11, 15]
    with pytest.raises(ValueError):
        make_groups(16, 3, "first")


def test_fused_ffn_equals_parallel_ffns(teacher):
    """In float32 the fused FFN is exactly the sum of the member FFNs, each behind its own RMSNorm
    gain, all reading the same input."""
    layers = teacher.model.vlm_with_expert.lm_expert.layers
    members = [{k: v.float() for k, v in layers[i].state_dict().items()} for i in (2, 3)]
    eps = layers[2].post_attention_layernorm.variance_epsilon
    x = torch.randn(
        2, 50, members[0]["post_attention_layernorm.weight"].shape[0], device=teacher.config.device
    )

    n = rmsnorm(x, eps)
    expected = sum(
        swiglu(
            n * w["post_attention_layernorm.weight"],
            w["mlp.gate_proj.weight"],
            w["mlp.up_proj.weight"],
            w["mlp.down_proj.weight"],
        )
        for w in members
    )
    fused = fuse_ffns(members)
    assert torch.equal(
        fused["post_attention_layernorm.weight"], torch.ones_like(fused["post_attention_layernorm.weight"])
    )
    actual = swiglu(
        n, fused["mlp.gate_proj.weight"], fused["mlp.up_proj.weight"], fused["mlp.down_proj.weight"]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_compressed_layer_runs_the_fused_ffn(teacher, shallow_ffn_fused):
    """The built student's layer applies the fused FFN through its own modules (bf16 weights)."""
    t_layers = teacher.model.vlm_with_expert.lm_expert.layers
    s_layer = shallow_ffn_fused.model.vlm_with_expert.lm_expert.layers[1]  # pair (2, 3)
    dtype = s_layer.mlp.gate_proj.weight.dtype
    x = torch.randn(1, 50, s_layer.mlp.gate_proj.weight.shape[1], device=teacher.config.device, dtype=dtype)
    expected = sum(t_layers[i].mlp(t_layers[i].post_attention_layernorm(x)).float() for i in (2, 3))
    actual = s_layer.mlp(s_layer.post_attention_layernorm(x)).float()
    # One extra bf16 rounding from folding the norm gain into the weights
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_layout_and_size(teacher, shallow_ffn_fused):
    assert shallow_ffn_fused.config.expert_vlm_layers == ALTERNATE_PAIRS
    assert shallow_ffn_fused.config.expert_intermediate_size == 4096
    assert expert_params(shallow_ffn_fused) == pytest.approx(84.5e6, rel=0.01)
    shallow_dropped = compress(teacher, group=2, keep="alternate", ffn="drop")
    assert expert_params(shallow_dropped) == pytest.approx(49.1e6, rel=0.01)


@torch.no_grad()
def test_drop_equals_layer_selection(teacher):
    """ffn="drop" is the same model as keeping the kept members with build_student."""
    dropped = compress(teacher, group=2, keep="alternate", ffn="drop").state_dict()
    selected = build_student(teacher, ALTERNATE_PAIRS).state_dict()
    assert dropped.keys() == selected.keys()
    for key in dropped:
        assert torch.equal(dropped[key], selected[key]), key


@torch.no_grad()
def test_group_one_matches_teacher(teacher):
    inputs = random_inputs(teacher)
    expected = teacher.model.sample_actions(**inputs)
    assert torch.equal(compress(teacher, group=1, ffn="drop").model.sample_actions(**inputs), expected)
    # fuse with one member only folds the norm gain into the weights (one bf16 rounding)
    actual = compress(teacher, group=1, ffn="fuse").model.sample_actions(**inputs)
    assert (actual - expected).abs().max() < 0.1


@torch.no_grad()
def test_compress_twice(teacher, shallow_ffn_fused):
    """16 -> 8 -> 4: merging an already merged student."""
    k4 = compress(shallow_ffn_fused, group=2, keep="alternate", ffn="fuse")
    assert k4.config.expert_vlm_layers == [0, 7, 8, 15]
    assert k4.config.expert_intermediate_size == 8192
    actions = k4.model.sample_actions(**random_inputs(teacher))
    assert actions.shape == (1, teacher.config.chunk_size, teacher.config.max_action_dim)
    assert torch.isfinite(actions).all()


@torch.no_grad()
def test_save_and_reload(teacher, teacher_dir, shallow_ffn_fused, tmp_path):
    save_student(shallow_ffn_fused, teacher_dir, tmp_path)
    reloaded = SmolVLAPolicy.from_pretrained(tmp_path)
    assert reloaded.config.expert_vlm_layers == ALTERNATE_PAIRS
    assert reloaded.config.expert_intermediate_size == 4096
    inputs = random_inputs(teacher)
    assert torch.equal(
        shallow_ffn_fused.model.sample_actions(**inputs), reloaded.model.sample_actions(**inputs)
    )
