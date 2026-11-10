"""Qwen2-MoE, end to end."""

import glob
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import qwen2_moe


def _random_checkpoint(repo, **changes):
    """The tiny checkpoint's config with ``changes``, randomly initialised and saved (seeded, so every load is one model), with its tokenizer."""
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="qwen2_moe-")
    config = AutoConfig.from_pretrained(snapshot)
    for name, value in changes.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(patched)
    for name in os.listdir(snapshot):
        target = os.path.join(patched, name)
        if (name.startswith(("tokenizer", "special_tokens")) or name in ("vocab.json", "merges.txt") or name.endswith(".jinja")) and not os.path.exists(target):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


class TestQwen2Moe(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
    FAMILY = qwen2_moe
    NATIVE = LLAMA_ROWS


class TestQwen2MoeMlpOnly(TestQwen2Moe):
    """``mlp_only_layers=[0]``: block 0 is a dense ``Qwen2MoeMLP`` (no released checkpoint), the others mixtures."""

    REPO = _random_checkpoint(TestQwen2Moe.REPO, mlp_only_layers=[0])

    def test_the_dense_block_is_an_mlp(self, model):
        dense, mixture = model.layers[0].mlp, model.layers[1].mlp
        assert type(dense) is qwen2_moe.Mlp and isinstance(mixture, qwen2_moe.Moe)
        assert isinstance(model.layers[1].mlp.shared_experts, qwen2_moe.Mlp)  # the shared expert is the same dense class
        assert all(reason is None for name, reason in model.support(layer=0).items() if name.startswith("mlp.") and name.removeprefix("mlp.") in qwen2_moe.Mlp.values())
        with model.trace(PROMPT):
            x = model.layers[0].layer_mid.save()
            down = dense.down_proj.output.save()
            out = dense.mlp_output.save()
            y = model.layers[0].layer_output.save()
        assert torch.equal(out, down) and torch.equal(x + out, y)
