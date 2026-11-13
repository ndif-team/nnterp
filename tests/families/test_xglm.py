"""XGLM, end to end: no MLP module, a scaled embedding, the interior on flattened ``bmm`` arithmetic."""

import math

import pytest
import torch
from suite import NO_ROTARY, FamilySuite, PROMPT, rows

from nnterp.components import OWN_ARITHMETIC
from nnterp.families import xglm


class TestXGLM(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-XGLMForCausalLM"
    FAMILY = xglm
    NATIVE = rows("model", "layers", "embed_tokens", "layer_norm", mlp=None, ln1="self_attn_layer_norm", ln2=None)
    MLP_NORM = "final_layer_norm"            # the block's own, native name
    EXPECTED_UNAVAILABLE = {**NO_ROTARY, "mlp_input": "no mlp module", "self_attn.interface": OWN_ARITHMETIC}

    def test_interior_per_invoke(self, model):
        """The queries, scores, pattern and head outputs are read where the batch leads; the keys and values are not (below)."""
        super().test_interior_per_invoke(model, values=("attention_queries", "attention_scores", "attention_probabilities", "attention_head_outputs"))

    @pytest.mark.parametrize("name", ["attention_keys", "attention_values"])
    @pytest.mark.xfail(strict=True, reason="the keys and values are served off the [batch * num_heads, ...] tensors after the cache update, which nnsight narrows to no invoke: each invoke reads the whole batch")
    def test_keys_and_values_per_invoke(self, model, name):
        super().test_interior_per_invoke(model, values=(name,))

    def test_no_mlp_module(self, model):
        """No block has an MLP module, so `support()` lists no ``mlp`` value at all."""
        assert not hasattr(model.layers[0], "mlp")
        assert not any(name.startswith("mlp.") for name in model.support())

    def test_contribution_identity(self, model):
        """No MLP module, but an MLP path on the block: ``fc2``'s output is what the block adds."""
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                parts[i] = (layer.input.save(), layer.self_attn.attention_output.save(), layer.fc2.output.save(), layer.layer_output.save())
        for i, (x, attn, mlp, out) in parts.items():
            eps = torch.finfo(out.dtype).eps
            torch.testing.assert_close(x.float() + attn.float() + mlp.float(), out.float(), rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")

    def test_token_embeddings_are_scaled(self, model):
        """``embed_tokens`` multiplies the lookup by ``sqrt(d_model)``; ``token_embeddings`` is that product."""
        assert model.config.scale_embedding
        with model.trace(PROMPT):
            ids = model.input_ids.save()
            embeddings = model.token_embeddings.save()
        lookup = model.embed_tokens._module.weight[ids]
        torch.testing.assert_close(embeddings, lookup * math.sqrt(model.config.d_model))

    def test_scores_are_scaled_queries_times_keys(self, model):
        """The queries arrive scaled, so the scores are ``q @ k^T`` where the mask lets them through."""
        with model.trace(PROMPT):
            q = model.layers[0].self_attn.attention_queries.save()
            k = model.layers[0].self_attn.attention_keys.save()
            scores = model.layers[0].self_attn.attention_scores.save()
        raw = q @ k.transpose(-1, -2)
        causal = torch.ones(raw.shape[-2:], dtype=torch.bool).tril()
        torch.testing.assert_close(scores[..., causal], raw[..., causal])

    def test_pattern_in_training_is_the_dropout_output(self, model):
        """Outside training the pattern is read at the softmax's output viewed batch first, which the dropout leaves as it
        is; in training it is the dropout's output, which differs from it."""
        attn = model.layers[0].self_attn
        model._module.train()
        try:
            torch.manual_seed(0)
            with model.trace(PROMPT):
                softmax = attn.source.attn_weights_reshaped_0.output.clone().save()
                pattern = attn.attention_probabilities.save()
                dropped = attn.source.nn_functional_dropout_0.output.save()
        finally:
            model._module.eval()
        torch.testing.assert_close(pattern, dropped.view(pattern.shape))
        assert not torch.equal(pattern, softmax)

    def test_interior_needs_no_eager_load(self):
        """XGLM has one attention implementation, so a default load serves the pattern."""
        from nnterp import StandardizedTransformer

        model = StandardizedTransformer(self.REPO, dispatch=True)
        assert model.support()["self_attn.attention_probabilities"] is None
        with model.trace(PROMPT):
            pattern = model.layers[0].self_attn.attention_probabilities.save()
        torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
