"""The checks every Mamba-1 (selective-scan) family passes on top of `FamilySuite`.

A family's test class subclasses both, `SelectiveScanSuite` first, and names
a block whose mixer is a `SelectiveScan` (`SCAN_BLOCK`) and a real checkpoint
to meta-build (`REAL`). Every test runs on transformers' pure-torch kernels:
the class routes the family before its first trace and restores it after.
"""

import pytest
import torch
from suite import LINEAR, PROMPT, listed

from nnter import StandardizedTransformer, route_kernels


def scan(mix, values, keys, queries, betas, decays):
    """The Mamba-1 recurrence over the read values: the states and ``y`` before the gate."""
    h = values.new_zeros(values.shape[0], values.shape[2], keys.shape[-1])
    states, ys = [], []
    for t in range(values.shape[1]):
        h = decays[:, t].exp() * h + (betas[:, t] * values[:, t])[..., None] * keys[:, t, 0][:, None]
        states.append(h)
        ys.append((h * queries[:, t, 0][:, None]).sum(-1) + mix._module.D * values[:, t])
    return torch.stack(states, 1), torch.stack(ys, 1)


class SelectiveScanSuite:
    """Mix in before `FamilySuite`: the selective-scan values, the per-token state, the decode step."""

    #: A block whose mixer is a `SelectiveScan`.
    SCAN_BLOCK = 0
    #: A real checkpoint whose config is meta-built with this family: repo, blocks, hidden size, mixer channels.
    REAL: tuple

    def scan_mixer(self, model):
        return model.layers[self.SCAN_BLOCK].linear_attn

    def scan_blocks(self, model):
        return [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]

    @pytest.fixture(scope="class", autouse=True)
    def torch_kernels(self):
        """``mamba_ssm`` dispatches to CUDA kernels with no source; every test here runs the pure-torch ones."""
        route_kernels(self.FAMILY, "torch")
        yield
        route_kernels(self.FAMILY, "default")

    def test_support_before_routing(self, model):
        """Unrouted, the per-token state and the values inside the kernels say how to reach them."""
        route_kernels(self.FAMILY, "default")
        try:
            support = model.support()
        finally:
            route_kernels(self.FAMILY, "torch")
        blocks = self.scan_blocks(model)
        for name in ("states", "state_output", "attention_head_outputs"):
            assert all("route_kernels(model.family, 'torch')" in support[f"linear_attn.{name}"][i] for i in blocks), name
        assert not set(blocks) & set(model.support()["linear_attn.states"] or {})

    def test_linear_values_shapes(self, model):
        mix = self.scan_mixer(model)
        m = mix._module
        got = {}
        for name in LINEAR:
            if name == "state":
                continue  # a per-token location; its own tests below
            with model.trace(PROMPT):
                value = getattr(mix, name)
                got[name] = value.save() if value is not None else None
        batch, seq = got["attention_queries"].shape[:2]
        channels, state_dim = m.intermediate_size, m.ssm_state_size
        assert got["attention_queries"].shape == got["attention_keys"].shape == (batch, seq, 1, state_dim)
        assert got["attention_values"].shape == got["betas"].shape == got["attention_head_outputs"].shape == (batch, seq, channels)
        assert got["decays"].shape == (batch, seq, channels, state_dim) and (got["decays"] <= 0).all()
        assert (got["betas"] > 0).all()
        assert got["state_input"] is None  # a prompt's scan starts from zeros
        assert got["state_output"].shape == (batch, channels, state_dim)
        assert got["states"].shape == (batch, seq, channels, state_dim)
        assert got["attention_output"].shape == (batch, seq, model.hidden_size)

    def test_values_are_the_recurrence(self, model):
        """Running ``h = exp(dt*A) h + dt B x``, ``y = C.h + D x`` over the read values lands on the scan's own states and read."""
        mix = self.scan_mixer(model)
        got = {}
        for name in ("attention_values", "attention_keys", "attention_queries", "betas", "decays", "states", "attention_head_outputs"):
            with model.trace(PROMPT):
                got[name] = getattr(mix, name).save()
        floats = {name: value.float() for name, value in got.items()}
        states, ys = scan(mix, floats["attention_values"], floats["attention_keys"], floats["attention_queries"], floats["betas"], floats["decays"])
        # The scan reads the state in the model's dtype: exact in float32, to bfloat16's precision otherwise.
        tol = {} if got["attention_values"].dtype == torch.float32 else {"atol": 1e-2, "rtol": 1e-2}
        torch.testing.assert_close(states, floats["states"], **tol)
        torch.testing.assert_close(ys, floats["attention_head_outputs"], **tol)

    def test_linear_writes_are_causal(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        for name in ("attention_queries", "attention_keys", "attention_values", "attention_head_outputs"):
            with model.trace(PROMPT):
                mix = self.scan_mixer(model)
                setattr(mix, name, getattr(mix, name) * 0)
                edited = model.logits.save()
            assert not torch.equal(clean, edited), name
        for name in ("attention_values", "attention_head_outputs"):
            with model.trace(PROMPT):
                getattr(self.scan_mixer(model), name)[:, -1] = 0   # a view: in place reaches the kernel
                inplace = model.logits.save()
            assert not torch.equal(clean[:, -1], inplace[:, -1]) and torch.equal(clean[:, :-1], inplace[:, :-1]), name

    def test_zeroed_head_outputs_leave_only_the_projection(self, model):
        """With ``y`` zeroed the gate multiplies nothing: the mixer adds ``out_proj``'s bias alone, the same at every position."""
        with model.trace(PROMPT):
            mix = self.scan_mixer(model)
            mix.attention_head_outputs = mix.attention_head_outputs * 0
            out = mix.attention_output.save()
        torch.testing.assert_close(out[:, 0], out[:, -1])

    def test_derived_values_are_read_only(self, model):
        for name in ("betas", "decays", "states"):
            with pytest.raises(AttributeError, match="read-only"):
                with model.trace(PROMPT):
                    setattr(self.scan_mixer(model), name, None)

    def test_state_output_is_the_state_the_last_token_leaves(self, model):
        with model.trace(PROMPT):
            full = self.scan_mixer(model).state_output.save()
        ids = model.tokenizer(PROMPT, return_tensors="pt").input_ids
        with model.trace(ids[:, :-1]):
            prefix = self.scan_mixer(model).state_output.save()
        assert full.shape == prefix.shape and not torch.equal(full, prefix)

    def test_linear_values_listed_in_the_repr(self, model):
        text = repr(self.scan_mixer(model))
        for name in LINEAR:
            assert listed(name, text), name

    def test_values_follow_the_step_under_generate(self, model):
        """A prompt runs the scan and each decode step the single-step update; the values follow the
        forward's own branch, and the state hands off from step to step."""
        mix = self.scan_mixer(model)
        with model.trace(PROMPT):
            traced = mix.state_output.save()
        outs, ins, seqs = [], [], []   # made outside the block: names bound inside do not survive it
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                entering = mix.state_input
                ins.append(entering.save() if entering is not None else None)
                seqs.append(mix.attention_queries.shape[1])
                outs.append(mix.state_output.save())
        n = len(model.tokenizer(PROMPT).input_ids)
        assert len(outs) == 3 and seqs == [n, 1, 1]  # the prompt, then one forward per new token
        assert torch.equal(outs[0], traced)
        assert ins[0] is None and all(torch.equal(ins[k], outs[k - 1]) for k in range(1, 3))
        assert all(not torch.equal(outs[k], outs[k - 1]) for k in range(1, 3))

    def test_read_order_differs_between_the_kernels(self, model):
        """``y`` comes before the returned state in the scan, after the updated state in the decode step."""
        mix = self.scan_mixer(model)
        heads, outs = [], []
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                if step == 0:
                    heads.append(mix.attention_head_outputs.save())
                    outs.append(mix.state_output.save())
                else:
                    outs.append(mix.state_output.save())
                    heads.append(mix.attention_head_outputs.save())
        n = len(model.tokenizer(PROMPT).input_ids)
        assert [h.shape[1] for h in heads] == [n, 1, 1] and len(outs) == 3

    def test_state_input_write_on_a_decode_step(self, model):
        """Assigning ``state_input`` on a decode step replaces what the step starts from; on a prompt there is none."""
        mix = self.scan_mixer(model)
        clean, written = [], []
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                clean.append(mix.state_output.save())
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                if step == 1:
                    mix.state_input = torch.zeros_like(clean[0])
                written.append(mix.state_output.save())
        assert torch.equal(written[0], clean[0]) and not torch.equal(written[1], clean[1])
        with pytest.raises(ValueError, match="starts from zeros"):
            with model.trace(PROMPT):
                mix.state_input = torch.zeros_like(clean[0])

    def test_per_token_state(self, model):
        """The pure-torch scan is the token loop: the state after every token is a value, read and written."""
        mix = self.scan_mixer(model)
        with model.trace(PROMPT):
            states = mix.states.save()
            final = mix.state_output.save()
            clean = model.logits.save()
        n = len(model.tokenizer(PROMPT).input_ids)
        assert states.shape[:2] == (1, n) and states.shape[2:] == final.shape[1:]
        assert torch.equal(states[:, -1], final)
        assert all(not torch.equal(states[:, t], states[:, t - 1]) for t in range(1, n))
        with model.trace(PROMPT):
            one = mix.state_after(1).save()
        assert torch.equal(one, states[:, 1])
        with model.trace(PROMPT):
            before = mix.state_after(0).save()          # reads follow the forward: earlier positions first
            mix.set_state_after(1, torch.zeros_like(one))
            after = mix.state_after(2).save()           # then the positions the write flows into
            final_written = mix.state_output.save()
            written = model.logits.save()
        assert torch.equal(before, states[:, 0])
        assert not torch.equal(after, states[:, 2])
        assert not torch.equal(final_written, final) and not torch.equal(written[:, 1:], clean[:, 1:])
        assert torch.equal(written[:, 0], clean[:, 0])  # the write lands after token 0's output

    def test_state_iterates_with_the_users_own_iter(self, model):
        mix = self.scan_mixer(model)
        n = len(model.tokenizer(PROMPT).input_ids)
        with model.trace(PROMPT):
            stacked = mix.states.save()
        walked = []
        with model.trace(PROMPT) as tracer:
            for t in tracer.iter[:n]:
                walked.append(mix.state.save())
        assert len(walked) == n and all(torch.equal(w, stacked[:, t]) for t, w in enumerate(walked))
        with model.trace(PROMPT) as tracer:
            first = mix.state.save()                       # outside any iter: after token 0
        assert torch.equal(first, stacked[:, 0])
        with model.trace(PROMPT) as tracer:
            for t in tracer.iter[1]:
                mix.state = torch.zeros_like(first)         # a write at token 1 ...
            for t in tracer.iter[2]:
                after = mix.state.save()                    # ... the next token continues from
        assert not torch.equal(after, stacked[:, 2])
        with model.trace(PROMPT) as tracer:
            for t in tracer.iter[2]:
                late_first = mix.state.save()              # a first read pinned past token 0 still resolves the call
        assert torch.equal(late_first, stacked[:, 2])

    def test_per_token_state_within_a_generate(self, model):
        """Under generate, the prompt's tokens are an inner loop on step 0 and each later step is one token."""
        mix = self.scan_mixer(model)
        n = len(model.tokenizer(PROMPT).input_ids)
        with model.trace(PROMPT):
            prompt_states = mix.states.save()
        per_token, finals, stacks, entering = [], [], [], []
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                if step == 0:
                    for t in tracer.iter[:n]:              # the prompt's tokens, nested
                        per_token.append(mix.state.save())
                else:
                    entering.append(mix.state_input.save())   # a read before the per-token one, in the same step
                    stacks.append(mix.states.save())       # a decode step's one-token call
                    per_token.append(mix.state_output.save())
                finals.append(mix.state_output.save())      # after the token loop too: the branch was decided
        assert len(per_token) == n + 2
        assert all(torch.equal(per_token[t], prompt_states[:, t]) for t in range(n))
        assert torch.equal(finals[0], per_token[n - 1])
        assert all(s.shape[1] == 1 for s in stacks) and all(torch.equal(stacks[k][:, 0], per_token[n + k]) for k in (0, 1))
        assert all(torch.equal(entering[k], per_token[n - 1 + k]) for k in (0, 1))
        assert all(not torch.equal(per_token[k], per_token[k - 1]) for k in range(1, n + 2))

    def test_real_checkpoint_config_resolves(self):
        """The real checkpoint builds on the meta device with this family (its config only): the tree is the tiny one's."""
        repo, blocks, hidden, channels = self.REAL
        real = StandardizedTransformer(repo)
        assert real.family is self.FAMILY and real.num_layers == blocks and real.hidden_size == hidden
        mix = real.layers[self.SCAN_BLOCK].linear_attn
        assert type(mix) is self.FAMILY.SelectiveScan and mix._module.intermediate_size == channels

