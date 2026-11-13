"""Qwen2-MoE, end to end."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, patched_checkpoint

from nnterp import StandardizedTransformer
from nnterp.families import qwen2_moe


class TestQwen2Moe(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
    FAMILY = qwen2_moe
    NATIVE = LLAMA_ROWS


def test_the_dense_block_is_an_mlp():
    """``mlp_only_layers=[0]`` (no released checkpoint): block 0 is a dense ``Qwen2MoeMLP``, the others mixtures."""
    model = StandardizedTransformer(patched_checkpoint(TestQwen2Moe.REPO, draw=True, mlp_only_layers=[0]), dispatch=True)
    dense, mixture = model.layers[0].mlp, model.layers[1].mlp
    assert type(dense) is qwen2_moe.Mlp and isinstance(mixture, qwen2_moe.Moe)
    assert isinstance(model.layers[1].mlp.shared_experts, qwen2_moe.Mlp)  # the shared expert is the same dense class
    assert all(reason is None for name, reason in model.support(layer=0).items() if name.startswith("mlp.") and name.removeprefix("mlp.") in qwen2_moe.Mlp.values())
    with model.trace(PROMPT):
        x = model.layers[0].layer_mid.save()
        down = dense.down_proj.output.save()
        out = dense.mlp_output.save()
        y = model.layers[0].layer_output.save()
    assert torch.equal(out, down) and torch.equal(x + out, y)
