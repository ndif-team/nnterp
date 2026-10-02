"""Solar Open, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import solar_open


class TestSolarOpen(FamilySuite):
    REPO = "onnx-internal-testing/tiny-random-SolarOpenForCausalLM"
    FAMILY = solar_open
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block of the tiny checkpoint is a mixture of experts
