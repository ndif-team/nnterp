"""Kimi-Linear, end to end: a Kimi Delta Attention hybrid with latent attention and a mixture of experts."""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import FamilySuite, LINEAR, LLAMA_ROWS, PROMPT, VALUES, listed

from nnterp import StandardizedTransformer, Unavailable, route_delta_rule
from nnterp.components import Moe

from nnterp.families import kimi_linear

LINEAR_BLOCKS = (0, 1, 2)       # ``config.layer_types``: three KDA blocks, then two latent-attention blocks
ATTENTION_BLOCKS = (3, 4)


def _checkpoint(repo="yujiepan/kimi-linear-tiny-random"):
    """The tiny checkpoint with its last block typed.

    Its ``linear_attn_config`` lists blocks 1-3 as KDA and block 4 as full
    attention (1-indexed) on a 5-block model, and the weights hold a latent
    attention in block 5, so native transformers, which types each block from
    the two lists, refuses the config. The released Kimi-Linear-48B ends on a
    full-attention block the same way and lists it; the rewrite adds block 5 to
    ``full_attn_layers``. Only ``config.json`` is rewritten; the weights (in the
    released checkpoints' own layout, which transformers converts at load) and
    the tokenizer are symlinked. Written once per snapshot.
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-kimi-linear-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "config.json")):
        return patched
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        if name != "config.json" and not os.path.exists(os.path.join(patched, name)):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["linear_attn_config"]["full_attn_layers"] = [4, 5]
    partial = os.path.join(patched, f"config.json.{os.getpid()}")
    json.dump(config, open(partial, "w"))
    os.replace(partial, os.path.join(patched, "config.json"))
    return patched


class TestKimiLinear(FamilySuite):
    REPO = _checkpoint()
    FAMILY = kimi_linear
    # The KDA mixer's native name is ``self_attn``, which the family clears on its blocks, so ``linear_attn``
    # has no native path to compare with; `test_the_kda_mixer_is_native_self_attn` checks the module instead.
    NATIVE = {**LLAMA_ROWS, "layers.0.self_attn": None, "layers.3.self_attn": "model.layers.3.self_attn"}
    NATIVE = {k: v for k, v in NATIVE.items() if v is not None}
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",    # missing on the attention blocks, and needing route_kernels on the others
        "linear_attn.states": "",
        **{f"mlp.{name}": "no " for name in ("router_logits", "expert_weights", "expert_indices", "expert_outputs", "routed_output", "shared_expert_output")},  # block 0's dense MLP
    }

    def test_every_layer_is_renamed(self, model):
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i in LINEAR_BLOCKS)
            assert (layer.self_attn is not None) == (i in ATTENTION_BLOCKS)
            assert hasattr(layer, "mlp") and hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm")

    def test_layer_types_match_the_tree(self, model):
        kinds = ["linear_attention" if hasattr(layer, "linear_attn") else "full_attention" for layer in model.layers]
        assert kinds == list(model.config.layer_types)

    def test_the_kda_mixer_is_native_self_attn(self, model):
        """transformers names both mixers ``self_attn``; the KDA one is ``linear_attn`` and ``self_attn`` names nothing on its block."""
        for i in LINEAR_BLOCKS:
            layer = model.layers[i]
            assert type(layer._module.self_attn).__name__ == "KimiLinearDeltaAttention"
            assert layer.self_attn is None and layer.linear_attn._module is layer._module.self_attn
        for i in ATTENTION_BLOCKS:
            assert type(model.layers[i].self_attn._module).__name__ == "KimiLinearAttention"

    def test_support_is_per_block_on_a_hybrid(self, model):
        support = model.support()
        assert set(support["self_attn.attention_probabilities"]) == set(LINEAR_BLOCKS)
        assert set(support["linear_attn.state_output"]) == set(ATTENTION_BLOCKS)
        assert support["layer_output"] is None and support["mlp.mlp_output"] is None

    def test_a_dense_block_then_mixtures(self, model):
        assert type(model.layers[0].mlp) is kimi_linear.Mlp and not isinstance(model.layers[0].mlp, Moe)
        assert all(type(model.layers[i].mlp) is kimi_linear.Moe for i in range(1, len(model.layers)))

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim

    def test_linear_values_shapes(self, model):
        mix = model.layers[0].linear_attn
        m = mix._module
        got = {}
        for name in LINEAR:
            if name in ("state", "states"):
                continue  # need route_delta_rule(family, 'recurrent'); their own tests below
            with model.trace(PROMPT):
                value = getattr(mix, name)
                got[name] = value.save() if value is not None else None
        batch, seq = got["attention_queries"].shape[:2]
        assert got["attention_queries"].shape == got["attention_keys"].shape == (batch, seq, m.num_heads, m.head_dim)
        assert got["attention_values"].shape == (batch, seq, m.num_heads, m.head_dim)
        assert got["decays"].shape == (batch, seq, m.num_heads, m.head_dim)   # one decay per key channel
        assert got["betas"].shape == (batch, seq, m.num_heads)
        assert (got["decays"] <= 0).all() and got["decays"].dtype == torch.float32
        assert (got["betas"] > 0).all() and (got["betas"] < 1).all()
        assert got["state_input"] is None  # a fresh prompt starts from nothing
        assert got["attention_head_outputs"].shape == (batch, seq, m.num_heads, m.head_dim)
        assert got["state_output"].shape == (batch, m.num_heads, m.head_dim, m.head_dim)
        assert got["attention_output"].shape == (batch, seq, model.hidden_size)

    def test_decays_are_the_forget_gates_output(self, model):
        mix = model.layers[0].linear_attn
        with model.trace(PROMPT):
            gate = mix.forget_gate.output.save()
        with model.trace(PROMPT):
            decays = mix.decays.save()
        assert torch.equal(decays, gate)

    def test_linear_writes_are_causal(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        for name in ("attention_queries", "attention_values", "decays", "betas", "attention_head_outputs"):
            with model.trace(PROMPT):
                mix = model.layers[0].linear_attn
                setattr(mix, name, getattr(mix, name) * 0)
                edited = model.logits.save()
            assert not torch.equal(clean, edited), name
        with model.trace(PROMPT):
            model.layers[0].linear_attn.attention_head_outputs[:, -1] = 0
            inplace = model.logits.save()
        assert not torch.equal(clean, inplace)

    def test_one_decay_channel_moves_the_state(self, model):
        """The decay is per channel: zeroing one key channel's log decay changes that row of the state."""
        mix = model.layers[0].linear_attn
        with model.trace(PROMPT):
            clean = mix.state_output.save()
        with model.trace(PROMPT):
            decays = mix.decays
            decays[..., 0] = -1e4   # forget everything in key channel 0 at every token
            forgot = mix.state_output.save()
        assert not torch.equal(clean[:, :, 0], forgot[:, :, 0])

    def test_state_output_is_the_state_the_last_token_leaves(self, model):
        with model.trace(PROMPT):
            full = model.layers[0].linear_attn.state_output.save()
        ids = model.tokenizer(PROMPT, return_tensors="pt").input_ids
        with model.trace(ids[:, :-1]):
            prefix = model.layers[0].linear_attn.state_output.save()
        assert full.shape == prefix.shape and not torch.equal(full, prefix)

    def test_linear_values_listed_in_the_repr(self, model):
        text = repr(model.layers[0].linear_attn)
        for name in LINEAR:
            assert listed(name, text), name
        assert "(decays) -> ChannelGates [batch seq heads key_dim]" in text

    def test_values_follow_the_step_under_generate(self, model):
        """A prompt runs the chunked kernel and each decode step the recurrent one; the state hands off from step to step."""
        mix = model.layers[0].linear_attn
        with model.trace(PROMPT):
            traced = mix.state_output.save()
        outs, ins, seqs = [], [], []
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                entering = mix.state_input
                ins.append(entering.save() if entering is not None else None)
                seqs.append(mix.attention_queries.shape[1])
                outs.append(mix.state_output.save())
        assert len(outs) == 3 and seqs == [len(model.tokenizer(PROMPT).input_ids), 1, 1]
        torch.testing.assert_close(outs[0], traced)
        assert ins[0] is None and all(torch.equal(ins[k], outs[k - 1]) for k in range(1, 3))
        assert all(not torch.equal(outs[k], outs[k - 1]) for k in range(1, 3))

    def test_per_token_state_needs_the_recurrent_kernel(self, model):
        reason = model.support()["linear_attn.states"]
        assert all("route_kernels(model.family, 'torch')" in reason[i] for i in LINEAR_BLOCKS)
        assert all("no linear_attn module" in reason[i] for i in ATTENTION_BLOCKS)
        with pytest.raises(Unavailable, match="route_kernels"):
            model.layers[0].linear_attn.state_after(0)

    def test_per_token_state_is_the_kda_recurrence(self):
        """Routed through the token-by-token kernel, `states` is KDA's update, recomputed here from the read values:
        decay each key channel of the state, then the delta rule on the L2-normed key, scaled by beta."""
        route_delta_rule(self.FAMILY, "recurrent")
        recurrent = StandardizedTransformer(self.REPO, dispatch=True, attn_implementation="eager")
        try:
            mix = recurrent.layers[0].linear_attn
            assert recurrent.support()["linear_attn.states"] == {i: "no linear_attn module on this block" for i in ATTENTION_BLOCKS}
            got = {}
            for name in ("attention_keys", "attention_values", "decays", "betas"):
                with recurrent.trace(PROMPT):
                    got[name] = getattr(mix, name).save()
            with recurrent.trace(PROMPT):
                states = mix.states.save()
                final = mix.state_output.save()
            k, v, g, beta = (got[name].float() for name in ("attention_keys", "attention_values", "decays", "betas"))
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            state = torch.zeros_like(states[:, 0])
            for t in range(k.shape[1]):
                state = state * g[:, t, :, :, None].exp()
                delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
                state = state + k[:, t, :, :, None] * delta[:, :, None, :]
                torch.testing.assert_close(states[:, t], state, rtol=1e-4, atol=1e-5)
            assert torch.equal(states[:, -1], final)
        finally:
            route_delta_rule(self.FAMILY, "chunked")   # restore the family's default kernel

    def test_per_token_state_with_the_recurrent_kernel(self):
        route_delta_rule(self.FAMILY, "recurrent")
        recurrent = StandardizedTransformer(self.REPO, dispatch=True, attn_implementation="eager")
        try:
            mix = recurrent.layers[0].linear_attn
            with recurrent.trace(PROMPT):
                states = mix.states.save()
                clean = recurrent.logits.save()
            n = len(recurrent.tokenizer(PROMPT).input_ids)
            assert states.shape[:2] == (1, n)
            assert all(not torch.equal(states[:, t], states[:, t - 1]) for t in range(1, n))
            walked = []   # bound outside: a name bound in the block does not survive it
            with recurrent.trace(PROMPT) as tracer:
                for t in tracer.iter[:n]:
                    walked.append(mix.state.save())
            assert all(torch.equal(w, states[:, t]) for t, w in enumerate(walked))
            with recurrent.trace(PROMPT):
                before = mix.state_after(0).save()
                mix.set_state_after(1, torch.zeros_like(before))
                after = mix.state_after(2).save()
                written = recurrent.logits.save()
            assert torch.equal(before, states[:, 0])
            assert not torch.equal(after, states[:, 2]) and not torch.equal(written, clean)
        finally:
            route_delta_rule(self.FAMILY, "chunked")
