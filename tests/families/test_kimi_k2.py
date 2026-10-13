"""Kimi K2, end to end: DeepSeek-V3's family under Moonshot's ``kimi_k2`` model type."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp import StandardizedTransformer
from nnterp.components import Moe
from nnterp.families import deepseek_v3, kimi_k2


def _checkpoint(repo="hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration"):
    """The tiny Kimi-K2.5 checkpoint with the released configs' text model type and routing.

    The released K2.5, K2.6 and K2.7-Code configs nest a ``text_config`` whose
    ``model_type`` is ``kimi_k2``; transformers builds it as a
    ``DeepseekV3Config`` and keeps that type, so the family is looked up under
    ``kimi_k2``. The tiny checkpoint says ``deepseek_v3`` there, groups its
    experts in two and sends every token to all eight; the rewrite sets
    ``kimi_k2``, one expert group as K2's router has, and two experts per
    token, so the routing is sparse as on K2 (8 of 384). Only ``config.json`` is rewritten; the weights and the tokenizer
    are symlinked. Written once per snapshot.
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-kimi-k2-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "config.json")):
        return patched
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        if name != "config.json" and not os.path.exists(os.path.join(patched, name)):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["text_config"].update(model_type="kimi_k2", n_group=1, num_experts_per_tok=2)
    partial = os.path.join(patched, f"config.json.{os.getpid()}")
    json.dump(config, open(partial, "w"))
    os.replace(partial, os.path.join(patched, "config.json"))
    return patched


class TestKimiK2(FamilySuite):
    REPO = _checkpoint()
    FAMILY = kimi_k2
    NATIVE = {key: f"model.language_model.{native.removeprefix('model.')}" if native.startswith("model.") else native for key, native in LLAMA_ROWS.items()}
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_the_text_model_type_is_kimi_k2(self, model):
        assert model.config.model_type == "kimi_k25" and model.config.text_config.model_type == "kimi_k2"
        assert type(model.config.text_config).__name__ == "DeepseekV3Config"

    def test_the_family_is_deepseek_v3s(self, model):
        assert kimi_k2.ENVOYS is deepseek_v3.ENVOYS and kimi_k2.RENAME is deepseek_v3.RENAME
        assert type(model.layers[0].mlp) is deepseek_v3.Mlp and type(model.layers[1].mlp) is deepseek_v3.Moe
        assert isinstance(model.layers[1].mlp, Moe)

    def test_latent_attention_widths(self, model):
        text = model.config.text_config
        assert model.qk_head_dim == text.qk_nope_head_dim + text.qk_rope_head_dim
        assert model.head_dim == text.v_head_dim

    def test_a_text_only_checkpoint_loads_from_its_module(self, model):
        """The text-only K2 checkpoints carry ``kimi_k2`` at the top, which ``AutoConfig`` does not map: built with
        transformers' own ``DeepseekV3ForCausalLM`` and handed over as a module, they resolve here too."""
        from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

        text = model.config.text_config.to_dict()
        config = DeepseekV3Config(**{k: v for k, v in text.items() if k != "model_type"})
        config.model_type = "kimi_k2"
        torch.manual_seed(0)
        k2 = StandardizedTransformer(DeepseekV3ForCausalLM(config).eval(), tokenizer=model.tokenizer, attn_implementation="eager")
        assert k2.family is kimi_k2 and type(k2.layers[1].mlp) is deepseek_v3.Moe
        parts = []   # bound outside: a name bound in the block does not survive it
        with k2.trace(PROMPT):
            for layer in k2.layers:   # in forward order
                parts.append((layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save()))
            logits = k2.logits.save()
        for x, attn, mlp, out in parts:
            torch.testing.assert_close(x + attn + mlp, out)
        assert logits.shape[-1] == config.vocab_size
