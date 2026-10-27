"""Gemma on vLLM, against the transformers engine: no ``lm_head`` module, embeddings scaled after the lookup."""

import torch
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import gemma


class TestVLLMGemma(VLLMFamilySuite):
    REPO = "unsloth/gemma-2b"
    FAMILY = gemma
    NATIVE = {name: path for name, path in LLAMA_ROWS.items() if name != "lm_head"}   # vLLM unembeds with embed_tokens
    MEMORY = 0.3

    def embedding_scale(self, model):
        return model.hidden_size ** 0.5   # vLLM scales after the module; transformers' module scales itself

    def test_first_block_input_is_the_scaled_embeddings(self, model, reference):
        with self.run(model, reference):
            embeddings = model.token_embeddings.cpu().save()
            stream = model.layers[0].layer_input.cpu().save()
        torch.testing.assert_close(embeddings * self.embedding_scale(model), stream)
