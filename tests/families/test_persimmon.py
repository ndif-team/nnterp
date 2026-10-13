"""Persimmon, end to end: Llama's block with a fused qkv projection and a final norm of its own name."""

from suite import FamilySuite, rows

from nnterp.families import persimmon


class TestPersimmon(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-PersimmonForCausalLM"
    FAMILY = persimmon
    NATIVE = rows("model", "layers", "embed_tokens", "final_layernorm")
