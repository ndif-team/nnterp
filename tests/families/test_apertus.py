"""Apertus, end to end: Llama's block with its norms under their own names."""

from suite import FamilySuite, rows

from nnter.families import apertus


class TestApertus(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-ApertusForCausalLM"
    FAMILY = apertus
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1="attention_layernorm", ln2="feedforward_layernorm")
