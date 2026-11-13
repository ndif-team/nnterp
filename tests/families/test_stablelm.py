"""StableLM-2, end to end: a parallel block, or a sequential one."""

import torch
from suite import LLAMA_ROWS, PROMPT, FamilySuite, built, draw_biases, patched_checkpoint, rows

from nnterp.families import stablelm

REPO = "stabilityai/tiny-random-stablelm-2"


class TestStableLm(FamilySuite):
    """The tiny checkpoint: StableLM-2-12B's layout, a parallel block (``use_parallel_residual``) with q/k norms."""

    REPO = REPO
    FAMILY = stablelm
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel
    EXPECTED_UNAVAILABLE = {"layer_mid": "a parallel block"}   # use_parallel_residual on the tiny checkpoint

    def test_block_is_parallel(self, model):
        assert model.config.use_parallel_residual


class TestStableLmSequential(FamilySuite):
    """stablelm-2-1_6b's, stablelm-3b-4e1t's and their chat models' layout: a sequential block
    (``use_parallel_residual`` false) with ``post_attention_layernorm``, no q/k norms, biased q/k/v projections. The
    tiny config with those set, weights drawn for it, the biases too."""

    REPO = built(patched_checkpoint, REPO, init=draw_biases, use_parallel_residual=False, qk_layernorm=False, use_qkv_bias=True)
    FAMILY = stablelm
    NATIVE = LLAMA_ROWS

    def test_block_is_sequential(self, model):
        config = model.config
        assert not config.use_parallel_residual and not config.qk_layernorm and config.use_qkv_bias
        assert model.layers[0].self_attn._module.q_proj.bias.abs().max() > 0
        with model.trace(PROMPT):
            mid = model.layers[0].layer_mid.save()
            normed = model.layers[0].post_attention_layernorm.output.save()
            entering = model.layers[0].mlp.input.save()
        assert torch.equal(normed, entering) and mid.shape == normed.shape
