"""Falcon-H1, end to end: a Mamba-2 mixer and attention side by side in every block, µP multipliers around them."""

import torch
from ssd import StateSpaceChecks
from suite import LLAMA_ROWS, FamilySuite, PROMPT

from nnter.families import falcon_h1


class TestFalconH1(StateSpaceChecks, FamilySuite):
    REPO = "yujiepan/falcon-h1-tiny-random"   # the released multipliers; mamba_rms_norm on
    FAMILY = falcon_h1
    LOAD_KWARGS = {"dtype": torch.float32}    # in bf16 the small multipliers round some writes away
    NATIVE = {
        **LLAMA_ROWS,
        "norm": "model.final_layernorm",
        "layers.0.mlp": "model.layers.0.feed_forward",
        "layers.0.post_attention_layernorm": "model.layers.0.pre_ff_layernorm",
        "layers.0.linear_attn": "model.layers.0.mamba",
    }
    EXPECTED_UNAVAILABLE = {
        "linear_attn.state": "one tensor per call",
        "linear_attn.states": "chunk_per_token",
        "linear_attn.set_state_after": "one cumulative step",
    }

    def test_every_block_has_both_mixers(self, model):
        for layer in model.layers:
            assert layer.linear_attn is layer.mamba and layer.self_attn is not None and layer.mlp is layer.feed_forward

    def test_contributions_are_the_scaled_outputs(self, model):
        """What each mixer adds is its output times its multiplier; the embedding enters times its own."""
        config = model.config
        layer = model.layers[0]
        with model.trace(PROMPT):
            emb = model.token_embeddings.save()
            block_in = layer.input.save()
            mamba = layer.linear_attn.output.save()
            mamba_added = layer.linear_attn.attention_output.save()
            attn = layer.self_attn.output[0].save()
            attn_added = layer.self_attn.attention_output.save()
        assert config.ssm_out_multiplier != 1 and config.attention_out_multiplier != 1 and config.embedding_multiplier != 1
        assert config.lm_head_multiplier != 1  # project_on_vocab applies it; the suite's logit-lens tests check the equality
        torch.testing.assert_close(mamba_added, mamba * config.ssm_out_multiplier)
        torch.testing.assert_close(attn_added, attn * config.attention_out_multiplier)
        torch.testing.assert_close(block_in, emb * config.embedding_multiplier)

    def test_contribution_writes_reach_the_block(self, model):
        """Zeroing the Mamba-2 mixer's contribution in place: the block's output is then the input plus the other two."""
        layer = model.layers[0]
        with model.trace(PROMPT):
            x = layer.input.save()
            layer.linear_attn.attention_output[:] = 0          # in place: the tensor the block adds
            attn = layer.self_attn.attention_output.save()
            mlp = layer.mlp.mlp_output.save()
            out = layer.layer_output.save()
        eps = torch.finfo(out.dtype).eps
        torch.testing.assert_close(x.float() + attn.float() + mlp.float(), out.float(), rtol=8 * eps, atol=8 * eps)


class TestFalconH1GateInKernel(TestFalconH1):
    """Multipliers of one, and ``mamba_rms_norm`` off: a decode step passes the gate into the update kernel."""

    REPO = "hf-tiny-v2/tiny-random-FalconH1ForCausalLM"
    LOAD_KWARGS = {}

    def test_contributions_are_the_scaled_outputs(self, model):
        assert model.config.ssm_out_multiplier == 1 and not model.config.mamba_rms_norm
