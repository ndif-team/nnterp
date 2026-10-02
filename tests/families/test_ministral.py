"""Ministral, end to end: Llama's block with sliding-window layers."""

import glob
import os
import shutil
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS

from nnter.families import ministral


def _with_a_tokenizer(repo="hf-tiny-v2/tiny-random-MinistralForCausalLM", tokenizer="hf-internal-testing/tiny-random-MistralForCausalLM"):
    """The tiny checkpoint's config, randomly initialised, with the tiny Mistral checkpoint's tokenizer.

    Every tiny ``ministral`` checkpoint on the Hub ships only ``tekken.json``, which
    loads through ``mistral_common``; this copy takes Mistral's ``tokenizer.json``
    instead and widens the vocabulary to it (99 to 32000).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    def snapshot(name):
        return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/*"))[0]

    source, vocab = snapshot(repo), snapshot(tokenizer)
    patched = tempfile.mkdtemp(prefix="ministral-")
    config = AutoConfig.from_pretrained(source)
    config.vocab_size = AutoConfig.from_pretrained(vocab).vocab_size
    config.bos_token_id, config.eos_token_id = 1, 2
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(patched)
    for name in os.listdir(vocab):
        if name.startswith("tokenizer") or name == "special_tokens_map.json":
            shutil.copy(os.path.realpath(os.path.join(vocab, name)), os.path.join(patched, name))
    return patched


class TestMinistral(FamilySuite):
    REPO = _with_a_tokenizer()
    FAMILY = ministral
    NATIVE = LLAMA_ROWS

    def test_every_block_has_a_sliding_window(self, model):
        assert model.config.layer_types == ["sliding_attention"] * model.num_layers
        assert all(layer.self_attn._module.sliding_window == model.config.sliding_window for layer in model.layers)
