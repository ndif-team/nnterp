"""Persimmon, end to end: Llama's block with a fused qkv projection and a final norm of its own name."""

import pytest
from suite import FamilySuite, rows

from nnterp import StandardizedTransformer
from nnterp.families import persimmon


class TestPersimmon(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-PersimmonForCausalLM"
    FAMILY = persimmon
    NATIVE = rows("model", "layers", "embed_tokens", "final_layernorm")

    @pytest.fixture(scope="class")
    def model(self):
        """With a pad token: the tokenizer has neither a pad nor an eos token, so a list of prompts cannot batch."""
        return StandardizedTransformer(self.REPO, dispatch=True, attn_implementation="eager", tokenizer_kwargs={"pad_token": "<unk>"})
