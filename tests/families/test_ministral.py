"""Ministral, end to end: Llama's block with sliding-window layers."""

import os
import shutil
import tempfile

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, cached_snapshot, built

from nnterp.families import ministral


def _with_a_tokenizer(repo="hf-tiny-v2/tiny-random-MinistralForCausalLM", tokenizer="hf-internal-testing/tiny-random-MistralForCausalLM"):
    """The tiny checkpoint's config, randomly initialised, with the tiny Mistral checkpoint's tokenizer.

    Every tiny ``ministral`` checkpoint on the Hub ships only ``tekken.json``, which
    loads through ``mistral_common``; this copy takes Mistral's ``tokenizer.json``
    instead and widens the vocabulary to it (99 to 32000). Its blocks are
    Ministral-8B's mix, a full-attention block and a sliding one (every block
    slides on the tiny config), with a window of 3 tokens, shorter than the
    suite's prompt, so the sliding block's mask bites.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    source, vocab = cached_snapshot(repo), cached_snapshot(tokenizer)
    patched = tempfile.mkdtemp(prefix="ministral-")
    config = AutoConfig.from_pretrained(source)
    config.vocab_size = AutoConfig.from_pretrained(vocab).vocab_size
    config.bos_token_id, config.eos_token_id = 1, 2
    config.layer_types, config.sliding_window = ["full_attention", "sliding_attention"], 3
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(patched)
    for name in os.listdir(vocab):
        if name.startswith("tokenizer") or name == "special_tokens_map.json":
            shutil.copy(os.path.realpath(os.path.join(vocab, name)), os.path.join(patched, name))
    return patched


class TestMinistral(FamilySuite):
    REPO = built(_with_a_tokenizer)
    FAMILY = ministral
    NATIVE = LLAMA_ROWS

    def test_each_block_slides_as_layer_types_says(self, model):
        """A ``sliding_attention`` block has the config's window, a ``full_attention`` block none."""
        kinds = model.config.layer_types
        assert len(kinds) == model.num_layers and set(kinds) <= {"full_attention", "sliding_attention"}
        windows = [layer.self_attn._module.sliding_window for layer in model.layers]
        assert windows == [model.config.sliding_window if kind == "sliding_attention" else None for kind in kinds]

    def test_the_window_masks_the_pattern_on_sliding_blocks_only(self, model):
        """Where a window is shorter than the prompt, a sliding block's pattern is zero past it and a full block's is not."""
        n = len(model.tokenizer(PROMPT).input_ids)
        window = model.config.sliding_window
        if window is None or window >= n or "full_attention" not in model.config.layer_types:
            pytest.skip("no window shorter than the prompt beside a full block on this checkpoint")
        far = torch.ones(n, n, dtype=torch.bool).tril(-window)  # keys at least `window` positions back
        for layer, kind in zip(model.layers, model.config.layer_types):
            with model.trace(PROMPT):
                probs = layer.self_attn.attention_probabilities.save()
            assert (probs[..., far] == 0).all() == (kind == "sliding_attention"), (layer.path, kind)
