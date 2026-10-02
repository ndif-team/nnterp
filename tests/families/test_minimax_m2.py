"""MiniMax-M2, end to end: Llama's block, whole-projection q/k norms, a mixture of experts in every block."""

import glob
import os
import shutil
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import minimax_m2


class TestMiniMaxM2(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-MiniMaxM2ForCausalLM"
    FAMILY = minimax_m2
    NATIVE = LLAMA_ROWS

    def test_qk_norms_span_the_whole_projection(self, model):
        attn = model.layers[0].self_attn._module
        assert attn.q_norm.weight.shape[0] == model.num_heads * model.head_dim
        assert attn.k_norm.weight.shape[0] == model.num_kv_heads * model.head_dim


def _grouped_checkpoint(repo="hf-tiny-v2/tiny-random-MiniMaxM2ForCausalLM"):
    """A randomly initialised checkpoint in the released models' shape, with the tiny checkpoint's tokenizer.

    The tiny checkpoint has as many kv heads as heads, ``head_dim == hidden_size //
    num_heads`` and a full rotary; MiniMax-M2.5 has 8 kv heads for 48, ``head_dim``
    128 on a 3072 residual and a half-width rotary (``rotary_dim`` 64). This copy
    has 2 kv heads for 4, ``head_dim`` 16 on a 32 residual and
    ``partial_rotary_factor`` 0.5.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="minimax-m2-grouped-")
    config = AutoConfig.from_pretrained(snapshot)
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 16
    config.rope_parameters = {**config.rope_parameters, "partial_rotary_factor": 0.5}
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(patched)
    for name in os.listdir(snapshot):
        if name.startswith("tokenizer") or name.endswith(".jinja"):
            shutil.copy(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    return patched


class TestMiniMaxM2Grouped(FamilySuite):
    """The released models' attention shape: grouped kv heads, a ``head_dim`` of its own, a partial rotary."""

    REPO = _grouped_checkpoint()
    FAMILY = minimax_m2
    NATIVE = LLAMA_ROWS

    def test_grouped_shapes(self, model):
        assert (model.num_heads, model.num_kv_heads, model.head_dim) == (4, 2, 16)
        assert model.num_heads * model.head_dim != model.hidden_size
        assert model._module.model.rotary_emb.inv_freq.shape[0] == 16 // 4  # the rotary spans half of each head
        with model.trace(PROMPT):
            keys = model.layers[0].self_attn.attention_keys.save()
            queries = model.layers[1].self_attn.attention_queries.save()
        assert keys.shape[1] == 2 and keys.shape[-1] == 16
        assert queries.shape[1] == 4 and queries.shape[-1] == 16
