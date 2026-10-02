"""Gemma 2 on vLLM, against the transformers engine: sandwich norms, softcapped logits, no ``lm_head`` module."""

import nnsight
import torch
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import gemma2


class TestVLLMGemma2(VLLMFamilySuite):
    REPO = "google/gemma-2-2b-it"
    FAMILY = gemma2
    NATIVE = {name: path for name, path in LLAMA_ROWS.items() if name != "lm_head"}   # vLLM unembeds with embed_tokens
    MEMORY = 0.3

    def embedding_scale(self, model):
        return model.hidden_size ** 0.5   # vLLM scales after the module; transformers' module scales itself

    def test_contributions_are_the_post_norms(self, model, reference):
        layer = model.layers[0]
        with self.run(model, reference):
            attn = layer.self_attn.attention_output.cpu().save()
            post_attn = layer.post_attention_layernorm.output.clone().cpu().save()
            mlp = layer.mlp.mlp_output.cpu().save()
            post_ff = layer.post_feedforward_layernorm.output.clone().cpu().save()
        assert torch.equal(attn[0], post_attn) and torch.equal(mlp[0], post_ff)

    def test_first_block_input_is_the_scaled_embeddings(self, model, reference):
        with self.run(model, reference):
            embeddings = model.token_embeddings.cpu().save()
            stream = model.layers[0].layer_input.cpu().save()
        torch.testing.assert_close(embeddings * self.embedding_scale(model), stream)

    def test_lens_is_softcapped(self, model, reference):
        cap = model.config.final_logit_softcapping
        assert cap
        with self.run(model, reference):
            lens = nnsight.save(model.project_on_vocab(model.layers[-1].layer_output * 50).abs().max().item())
        assert lens <= cap
