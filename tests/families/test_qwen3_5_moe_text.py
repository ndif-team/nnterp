"""Qwen3.5-MoE text, end to end: a gated DeltaNet hybrid with a mixture-of-experts MLP."""

import pytest
import torch
from suite import FamilySuite, LINEAR, LLAMA_ROWS, PROMPT, VALUES, listed

from nnter import StandardizedTransformer, Unavailable, route_delta_rule

from nnter.families import qwen3_5_moe_text

LINEAR_BLOCKS = (0, 1, 2)   # ``config.layer_types``: three DeltaNet blocks, then one attention block
ATTENTION_BLOCK = 3


class TestQwen3_5Moe(FamilySuite):
    REPO = "yujiepan/qwen3.5-moe-tiny-random"
    FAMILY = qwen3_5_moe_text
    NATIVE = {**LLAMA_ROWS, "layers.0.self_attn": None, "layers.0.linear_attn": "model.layers.0.linear_attn", "layers.3.self_attn": "model.layers.3.self_attn"}
    NATIVE = {k: v for k, v in NATIVE.items() if v is not None}
    QUERY_GATED = True  # q_proj yields the query and its gate side by side
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",    # missing on the attention block, and needing route_kernels on the others
        "linear_attn.states": "",
    }

    def test_every_layer_is_renamed(self, model):
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i in LINEAR_BLOCKS)
            assert (getattr(layer, "self_attn", None) is not None) == (i == ATTENTION_BLOCK)
            assert hasattr(layer, "mlp") and hasattr(layer, "input_layernorm")

    def test_layer_types_match_the_tree(self, model):
        kinds = ["linear_attention" if hasattr(layer, "linear_attn") else "full_attention" for layer in model.layers]
        assert kinds == list(model.config.layer_types)

    def test_support_is_per_block_on_a_hybrid(self, model):
        support = model.support()
        assert set(support["self_attn.attention_probabilities"]) == set(LINEAR_BLOCKS)
        assert set(support["linear_attn.state_output"]) == {ATTENTION_BLOCK}
        assert support["layer_output"] is None and support["mlp.mlp_output"] is None

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
        assert got["attention_queries"].shape == got["attention_keys"].shape == (batch, seq, m.num_v_heads, m.head_k_dim)
        assert got["attention_values"].shape == (batch, seq, m.num_v_heads, m.head_v_dim)
        assert got["decays"].shape == got["betas"].shape == (batch, seq, m.num_v_heads)
        assert (got["decays"] <= 0).all() and got["decays"].dtype == torch.float32
        assert (got["betas"] > 0).all() and (got["betas"] < 1).all()
        assert got["state_input"] is None  # a fresh prompt starts from nothing
        assert got["attention_head_outputs"].shape == (batch, seq, m.num_v_heads, m.head_v_dim)
        assert got["state_output"].shape == (batch, m.num_v_heads, m.head_k_dim, m.head_v_dim)
        assert got["attention_output"].shape == (batch, seq, model.hidden_size)

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

    def test_state_output_is_the_state_the_last_token_leaves(self, model):
        """Running the prompt minus its last token and then that token as a decode step must land on the same state."""
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

    def test_values_follow_the_step_under_generate(self, model):
        """A prompt runs the chunked kernel and each decode step the recurrent one; the values
        follow the forward's own branch, and the state hands off from step to step."""
        mix = model.layers[0].linear_attn
        with model.trace(PROMPT):
            traced = mix.state_output.save()
        outs, ins, seqs = [], [], []   # made outside the block: names bound inside do not survive it
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                entering = mix.state_input
                ins.append(entering.save() if entering is not None else None)
                seqs.append(mix.attention_queries.shape[1])
                outs.append(mix.state_output.save())
        assert len(outs) == 3 and seqs == [len(model.tokenizer(PROMPT).input_ids), 1, 1]  # the prompt, then one forward per new token
        assert torch.equal(outs[0], traced)
        assert ins[0] is None and all(torch.equal(ins[k], outs[k - 1]) for k in range(1, 3))
        assert all(not torch.equal(outs[k], outs[k - 1]) for k in range(1, 3))

    def test_per_token_state_needs_the_recurrent_kernel(self, model):
        reason = model.support()["linear_attn.states"]
        assert all("route_kernels(model.family, 'torch')" in reason[i] for i in LINEAR_BLOCKS)
        assert "no linear_attn module" in reason[ATTENTION_BLOCK]
        with pytest.raises(Unavailable, match="route_kernels"):
            model.layers[0].linear_attn.state_after(0)

    def test_per_token_state_with_the_recurrent_kernel(self):
        """Like eager attention: routed through the slower kernel, the state after every token is a value."""
        route_delta_rule(self.FAMILY, "recurrent")
        recurrent = StandardizedTransformer(self.REPO, dispatch=True, attn_implementation="eager")
        try:
            mix = recurrent.layers[0].linear_attn
            assert recurrent.support()["linear_attn.states"] == {ATTENTION_BLOCK: "no linear_attn module on this block"}
            with recurrent.trace(PROMPT):
                states = mix.states.save()
                final = mix.state_output.save()
                clean = recurrent.logits.save()
            n = len(recurrent.tokenizer(PROMPT).input_ids)
            assert states.shape[:2] == (1, n) and states.shape[2:] == final.shape[1:]
            assert torch.equal(states[:, -1], final)
            assert all(not torch.equal(states[:, t], states[:, t - 1]) for t in range(1, n))
            with recurrent.trace(PROMPT):
                one = mix.state_after(1).save()
            assert torch.equal(one, states[:, 1])
            with recurrent.trace(PROMPT):
                before = mix.state_after(0).save()          # reads follow the forward: earlier positions first
                mix.set_state_after(1, torch.zeros_like(one))
                after = mix.state_after(2).save()           # then the positions the write flows into
                final_written = mix.state_output.save()
                written = recurrent.logits.save()
            assert torch.equal(before, states[:, 0])                            # before the write: untouched
            assert not torch.equal(after, states[:, 2])                         # the tokens after it continue from the write
            assert not torch.equal(final_written, final) and not torch.equal(written, clean)
            with pytest.raises(AttributeError, match="read-only"):
                mix.states = states * 0
        finally:
            route_delta_rule(self.FAMILY, "chunked")   # restore the family's default kernel

    def test_state_iterates_with_the_users_own_iter(self):
        """`state` is a per-token location: the user's `tracer.iter` walks it, and an assignment there writes."""
        route_delta_rule(self.FAMILY, "recurrent")
        recurrent = StandardizedTransformer(self.REPO, dispatch=True, attn_implementation="eager")
        try:
            mix = recurrent.layers[0].linear_attn
            n = len(recurrent.tokenizer(PROMPT).input_ids)
            with recurrent.trace(PROMPT):
                stacked = mix.states.save()
            walked = []
            with recurrent.trace(PROMPT) as tracer:
                for t in tracer.iter[:n]:
                    walked.append(mix.state.save())
            assert len(walked) == n and all(torch.equal(w, stacked[:, t]) for t, w in enumerate(walked))
            with recurrent.trace(PROMPT) as tracer:
                first = mix.state.save()                       # outside any iter: after token 0
            assert torch.equal(first, stacked[:, 0])
            with recurrent.trace(PROMPT) as tracer:
                for t in tracer.iter[1]:
                    mix.state = torch.zeros_like(first)         # a write at token 1 ...
                for t in tracer.iter[2]:
                    after = mix.state.save()                    # ... the next token continues from
            assert not torch.equal(after, stacked[:, 2])
            with recurrent.trace(PROMPT) as tracer:
                for t in tracer.iter[2]:
                    late_first = mix.state.save()              # a first read pinned past token 0 still resolves the call
            assert torch.equal(late_first, stacked[:, 2])
        finally:
            route_delta_rule(self.FAMILY, "chunked")

    def test_per_token_state_within_a_generate(self):
        """Under generate, the prompt's tokens are an inner loop on step 0 and each later step is one token."""
        route_delta_rule(self.FAMILY, "recurrent")
        recurrent = StandardizedTransformer(self.REPO, dispatch=True, attn_implementation="eager")
        try:
            mix = recurrent.layers[0].linear_attn
            n = len(recurrent.tokenizer(PROMPT).input_ids)
            with recurrent.trace(PROMPT):
                prompt_states = mix.states.save()
            per_token, finals, stacks = [], [], []
            with recurrent.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
                for step in tracer.iter[:]:
                    if step == 0:
                        for t in tracer.iter[:n]:              # the prompt's tokens, nested
                            per_token.append(mix.state.save())
                    else:
                        stacks.append(mix.states.save())       # a decode step's one-token call
                        per_token.append(mix.state_output.save())   # one token per decode step
                    finals.append(mix.state_output.save())      # after the token loop too: the branch was decided
            assert len(per_token) == n + 2                      # 3 forwards: n prompt tokens, then 2 more tokens
            assert all(torch.equal(per_token[t], prompt_states[:, t]) for t in range(n))
            assert torch.equal(finals[0], per_token[n - 1])
            assert all(s.shape[1] == 1 for s in stacks) and all(torch.equal(stacks[k][:, 0], per_token[n + k]) for k in (0, 1))
            assert all(not torch.equal(per_token[k], per_token[k - 1]) for k in range(1, n + 2))
        finally:
            route_delta_rule(self.FAMILY, "chunked")
