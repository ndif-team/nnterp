"""ERNIE 4.5 MoE, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, draw_routers, moe_host, patched_checkpoint

from nnterp.components import Moe
from nnterp.families import ernie4_5_moe

REPO = "hf-tiny-v2/tiny-random-Ernie4_5_MoeForCausalLM"


class TestErnie45Moe(FamilySuite):
    """The tiny checkpoint: every block a mixture, and a router of zeros (transformers' initialization, saved as is),
    so every token routes to the same two experts with weight 0.5."""

    REPO = REPO
    FAMILY = ernie4_5_moe
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # the experts' width, where the first MLP block is a mixture


class TestErnie45MoeReleased(TestErnie45Moe):
    """The released checkpoints' layout: a trained router and selection bias (drawn here), ERNIE-4.5-21B's dense
    first block (``moe_layer_start_index`` 1) and ERNIE-4.5-300B's mixture without shared experts
    (``moe_num_shared_experts`` 0). The tiny config otherwise, weights drawn for it."""

    REPO = built(patched_checkpoint, REPO, init=draw_routers, moe_layer_start_index=1, moe_num_shared_experts=0)

    def test_the_branches_are_taken(self, model):
        assert type(model.layers[0].mlp) is ernie4_5_moe.Mlp and not isinstance(model.layers[0].mlp, Moe)
        host = moe_host(model)
        assert host.support()["shared_expert_output"] == "this mixture has no shared expert"
        assert host.router._module.weight.abs().max() > 0
