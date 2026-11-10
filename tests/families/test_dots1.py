"""dots.llm1, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, draw_routers, moe_host, patched_checkpoint

from nnterp.components import Moe
from nnterp.families import dots1

REPO = "hf-tiny-v2/tiny-random-Dots1ForCausalLM"


class TestDots1(FamilySuite):
    """The tiny checkpoint: every block a mixture that routes every token to all 8 experts, unnormalized and unscaled."""

    REPO = REPO
    FAMILY = dots1
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # the experts' width, where the first MLP block is a mixture


class TestDots1Released(TestDots1):
    """dots.llm1's routing (6 of 128 there): top 2 of 8, renormalized (``norm_topk_prob``), times
    ``routed_scaling_factor`` 2.5, a selection bias, after a dense first block (``first_k_dense_replace`` 1). The tiny
    config otherwise, weights drawn for it, the router and selection bias too."""

    REPO = built(
        patched_checkpoint, REPO, init=draw_routers, num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=2.5,
        first_k_dense_replace=1,
    )

    def test_the_branches_are_taken(self, model):
        assert type(model.layers[0].mlp) is dots1.Mlp and not isinstance(model.layers[0].mlp, Moe)
        host = moe_host(model)
        assert host.top_k == 2 < host.num_experts
