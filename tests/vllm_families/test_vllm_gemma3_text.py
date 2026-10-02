"""Gemma 3 (text) on vLLM, against the transformers engine: sandwich norms, embeddings scaled after the lookup."""

import torch
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import gemma3_text


class TestVLLMGemma3Text(VLLMFamilySuite):
    REPO = "unsloth/gemma-3-270m"
    FAMILY = gemma3_text
    NATIVE = LLAMA_ROWS

    def embedding_scale(self, model):
        return model.hidden_size ** 0.5   # vLLM scales after the module; transformers' module scales itself

    def test_first_block_input_is_the_scaled_embeddings(self, model, reference):
        with self.run(model, reference):
            embeddings = model.token_embeddings.cpu().save()
            stream = model.layers[0].layer_input.cpu().save()
        torch.testing.assert_close(embeddings * self.embedding_scale(model), stream)
