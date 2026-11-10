"""Qwen3-MoE, end to end."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, patched_checkpoint

from nnterp import StandardizedTransformer
from nnterp.families import qwen3_moe


class TestQwen3Moe(FamilySuite):
    REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    FAMILY = qwen3_moe
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block is a mixture of experts; ``intermediate_size`` is the unused dense width


def test_the_dense_block_is_an_mlp():
    """``mlp_only_layers=[0]`` (no released checkpoint): block 0 is a dense ``Qwen3MoeMLP``, the others mixtures."""
    model = StandardizedTransformer(patched_checkpoint(TestQwen3Moe.REPO, draw=True, mlp_only_layers=[0]), dispatch=True)
    dense, mixture = model.layers[0].mlp, model.layers[1].mlp
    assert type(dense) is qwen3_moe.Mlp and isinstance(mixture, qwen3_moe.Moe)
    assert all(reason is None for name, reason in model.support(layer=0).items() if name.startswith("mlp.") and name.removeprefix("mlp.") in qwen3_moe.Mlp.values())
    with model.trace(PROMPT):
        x = model.layers[0].layer_mid.save()
        down = dense.down_proj.output.save()
        out = dense.mlp_output.save()
        y = model.layers[0].layer_output.save()
    assert torch.equal(out, down) and torch.equal(x + out, y)
