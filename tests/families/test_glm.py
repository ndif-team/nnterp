"""GLM-4, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import glm


class TestGlm(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-GlmForCausalLM"
    FAMILY = glm
    NATIVE = LLAMA_ROWS
