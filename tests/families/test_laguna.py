"""Laguna, end to end: gated attention, a dense block then a mixture, and per-block head counts."""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp import Unavailable
from nnterp.families import laguna

REPO = "hf-tiny-v2/tiny-random-LagunaForCausalLM"
#: The released checkpoints' shape: fewer heads on the full-attention block than on the sliding one.
HEADS = [2, 4]


def _per_layer_heads_checkpoint(repo=REPO):
    """The tiny checkpoint with ``num_attention_heads_per_layer`` rewritten to `HEADS`.

    The tiny checkpoint has two heads on every block. A second block with four
    changes the shapes of its ``q_proj``, ``g_proj`` and ``o_proj``, so the copy
    is written once per snapshot: the tiny weights where the shapes agree, a
    seeded initialisation where they do not; the tokenizer is symlinked.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-laguna-heads-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    config = AutoConfig.from_pretrained(snapshot)
    config.num_attention_heads_per_layer = HEADS
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    tiny = load_file(os.path.join(snapshot, "model.safetensors"))
    state = model.state_dict()
    model.load_state_dict({name: tiny[name] if name in tiny and tiny[name].shape == tensor.shape else tensor for name, tensor in state.items()})
    model.save_pretrained(patched)
    for name in os.listdir(snapshot):
        if name.startswith("tokenizer") or name == "chat_template.jinja":
            target = os.path.join(patched, name)
            if not os.path.exists(target):
                os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


class TestLaguna(FamilySuite):
    REPO = REPO
    FAMILY = laguna
    NATIVE = LLAMA_ROWS

    def test_dense_then_mixture(self, model):
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        expected = {"dense": "LagunaMLP", "sparse": "LagunaSparseMoeBlock"}
        assert kinds == [expected[kind] for kind in model.config.mlp_layer_types]

    def test_shared_expert_has_no_contribution(self, model):
        block = next(layer for layer in model.layers if type(layer.mlp._module).__name__ == "LagunaSparseMoeBlock")
        shared = block.mlp.shared_experts
        assert "shared expert" in shared.support()["mlp_output"]
        with pytest.raises(Unavailable, match="shared expert"):
            with model.trace(PROMPT):
                shared.mlp_output.save()

    def test_head_outputs_are_before_the_gate(self, model):
        attn = model.layers[0].self_attn
        with model.trace(PROMPT):
            heads = attn.attention_head_outputs.save()
            projected = attn.o_proj.input.save()
        assert not torch.allclose(heads.flatten(2), projected)


class TestLagunaPerLayerHeads(FamilySuite):
    """Blocks with different head counts, as the released checkpoints have."""

    REPO = _per_layer_heads_checkpoint()
    FAMILY = laguna
    NATIVE = LLAMA_ROWS

    def test_each_block_reports_its_own_heads(self, model):
        assert [layer.self_attn.num_heads for layer in model.layers] == HEADS
        assert model.num_heads == model.config.num_attention_heads == HEADS[0]
        assert all(layer.self_attn.num_kv_heads == model.num_kv_heads for layer in model.layers)

    def test_each_blocks_interior_has_its_heads(self, model):
        for layer, heads in zip(model.layers, HEADS):
            with model.trace(PROMPT):
                probs = layer.self_attn.attention_probabilities.save()
            with model.trace(PROMPT):
                queries = layer.self_attn.attention_queries.save()
            with model.trace(PROMPT):
                outputs = layer.self_attn.attention_head_outputs.save()
            assert probs.shape[1] == queries.shape[1] == outputs.shape[2] == heads

    def test_pattern_across_layers_and_traces(self, model):
        """The suite's check with each block's own head count: the two blocks' patterns differ in shape here."""
        first, last = model.layers[0], model.layers[-1]
        with model.trace(PROMPT):
            a = first.self_attn.attention_probabilities.save()
            b = last.self_attn.attention_probabilities.save()
        with model.trace(PROMPT):
            again = first.self_attn.attention_probabilities.save()
        assert (a.shape[1], b.shape[1]) == (first.self_attn.num_heads, last.self_attn.num_heads) == tuple(HEADS)
        assert a.shape[2:] == b.shape[2:] and torch.equal(a, again)

    def test_residual_identity_on_every_block(self, model):
        read = []  # bound outside: a name bound inside the block does not survive it
        with model.trace(PROMPT):
            for layer in model.layers:
                read.append((layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save()))
        for x, attn, mlp, out in read:
            torch.testing.assert_close(x + attn + mlp, out)
