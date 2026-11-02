"""Granite on vLLM, against the transformers engine: contributions times residual_multiplier, embeddings and logits scaled."""

import torch
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import granite


class TestVLLMGranite(VLLMFamilySuite):
    REPO = "ibm-granite/granite-3.3-2b-instruct"
    FAMILY = granite
    NATIVE = LLAMA_ROWS
    MEMORY = 0.4

    def test_first_block_input_is_the_multiplied_embeddings(self, model, reference):
        with self.run(model, reference):
            embeddings = model.token_embeddings.cpu().save()
            stream = model.layers[0].layer_input.cpu().save()
        torch.testing.assert_close(embeddings * model.config.embedding_multiplier, stream)

    def test_contributions_are_the_scaled_module_outputs(self, model, reference):
        layer = model.layers[reference["middle"]]
        with self.run(model, reference):
            raw = layer.mlp.output.clone().cpu().save()
            served = layer.mlp.mlp_output.cpu().save()
        torch.testing.assert_close(served, raw.unsqueeze(0) * model.config.residual_multiplier)
