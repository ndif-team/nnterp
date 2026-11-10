"""Solar Open, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, draw_biases, draw_routers, patched_checkpoint

from nnterp.families import solar_open


class TestSolarOpen(FamilySuite):
    REPO = "onnx-internal-testing/tiny-random-SolarOpenForCausalLM"
    FAMILY = solar_open
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block of the tiny checkpoint is a mixture of experts


def draw_routers_and_biases(model):
    draw_routers(model)
    draw_biases(model)


class TestSolarOpenWide(TestSolarOpen):
    """Solar-Open-100B's attention: heads (4 x 32 here, 64 x 128 there) twice the stream's width. Weights drawn for it,
    the router and its selection bias too."""

    REPO = built(patched_checkpoint, TestSolarOpen.REPO, init=draw_routers_and_biases, head_dim=32)
