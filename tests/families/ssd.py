"""Checks every family with a Mamba-2 (SSD) mixer passes, written once: mix into a `FamilySuite` subclass.

The mixer is `nnterp.StateSpace` at ``layers[i].linear_attn``. Its values are
read inside transformers' pure-torch scan kernels, so the class routes the
family's kernels to them for its whole run (``mamba_ssm`` is installed in the
test environment, and its kernels have no Python source) and restores the
default after. A test that sets `chunk_per_token` restores the chunk size
before it returns: the model is shared by the class.
"""

import pytest
import torch
from suite import LINEAR, PROMPT, listed

from nnterp import Unavailable, chunk_per_token, route_kernels
from nnterp.components.state_space import NO_STATE_OCCURRENCES, NO_STATE_WRITES


class StateSpaceChecks:
    """Set `SSD_BLOCK` to a block holding a Mamba-2 mixer."""

    #: A block whose ``linear_attn`` is the SSD mixer.
    SSD_BLOCK = 0

    @pytest.fixture(scope="class", autouse=True)
    def torch_kernels(self, request):
        """Transformers' pure-torch scan kernels for the class, before the model is loaded or traced."""
        route_kernels(request.cls.FAMILY, "torch")
        yield
        route_kernels(request.cls.FAMILY, "default")

    def ssd(self, model):
        return model.layers[self.SSD_BLOCK].linear_attn

    def expected_values(self, model):
        """The suite's list, plus ``set_state_after``: listed on a state-space mixer because it is unavailable there."""
        return super().expected_values(model) | {"linear_attn.set_state_after"}

    @staticmethod
    def _recurrence(prev, B, x, betas, decays, repeat):
        """One token of SSD, key side first: ``exp(decay) * h + beta * (B outer x)``, per head."""
        B = B.float().repeat_interleave(repeat, dim=1)                    # [batch, heads, state_dim]
        write = betas.float()[..., None, None] * B[..., :, None] * x.float()[..., None, :]
        return decays.exp()[..., None, None] * prev.float() + write

    def _read_all(self, model):
        mix = self.ssd(model)
        got = {}
        for name in LINEAR:
            if name in ("state", "states"):
                continue  # not materialized by the SSD kernels; see the availability test
            with model.trace(PROMPT):  # one trace each: reads within a trace follow forward order
                value = getattr(mix, name)
                got[name] = value.save() if value is not None else None
        return got

    def test_ssd_values_shapes(self, model):
        mix = self.ssd(model)
        m = mix._module
        got = self._read_all(model)
        batch, seq = 1, len(model.tokenizer(PROMPT).input_ids)
        assert got["attention_queries"].shape == got["attention_keys"].shape == (batch, seq, m.n_groups, m.ssm_state_size)
        assert got["attention_values"].shape == got["attention_head_outputs"].shape == (batch, seq, m.num_heads, m.head_dim)
        assert got["betas"].shape == got["decays"].shape == (batch, seq, m.num_heads)
        assert got["decays"].dtype == torch.float32 and (got["decays"] <= 0).all() and (got["betas"] > 0).all()
        torch.testing.assert_close(got["decays"], -torch.exp(m.A_log.float()) * got["betas"].float())
        assert got["state_input"] is None  # a fresh prompt starts from nothing
        assert got["state_output"].shape == (batch, m.num_heads, m.ssm_state_size, m.head_dim)  # key side first
        assert got["attention_output"].shape == (batch, seq, model.hidden_size)

    def test_ssd_writes_are_causal(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        for name in ("attention_queries", "attention_keys", "attention_values", "betas", "decays", "attention_head_outputs"):
            with model.trace(PROMPT):
                mix = self.ssd(model)
                setattr(mix, name, getattr(mix, name) * 0)
                edited = model.logits.save()
            assert not torch.equal(clean, edited), name
        with model.trace(PROMPT):
            self.ssd(model).attention_head_outputs[:, -1] = 0
            inplace = model.logits.save()
        assert not torch.equal(clean, inplace)

    def test_state_hands_off_under_generate(self, model):
        """A prompt runs the chunk scan and each decode step the token update; the state one step leaves is
        the state the next starts from, and each step's update is SSD's recurrence over the step's values."""
        mix = self.ssd(model)
        m = mix._module
        with model.trace(PROMPT):
            traced = mix.state_output.save()
        ins, outs, seqs, parts = [], [], [], []
        with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
            for step in tracer.iter[:]:
                entering = mix.state_input
                ins.append(entering.save() if entering is not None else None)
                seqs.append(mix.attention_queries.shape[1])
                parts.append((mix.attention_keys.save(), mix.attention_values.save(), mix.betas.save(), mix.decays.save()))
                outs.append(mix.state_output.save())
        assert seqs == [len(model.tokenizer(PROMPT).input_ids), 1, 1]
        assert torch.equal(outs[0], traced)
        assert ins[0] is None and all(torch.equal(ins[k], outs[k - 1]) for k in (1, 2))
        repeat = m.num_heads // m.n_groups
        for k in (1, 2):
            B, x, dt, log_decay = parts[k]
            B = B[:, 0].float().repeat_interleave(repeat, dim=1)             # [batch, heads, state_dim]
            write = dt[:, 0].float()[..., None, None] * B[..., :, None] * x[:, 0].float()[..., None, :]
            expected = log_decay[:, 0].exp()[..., None, None] * ins[k].float() + write
            torch.testing.assert_close(outs[k].float(), expected, rtol=2e-2, atol=2e-3)

    def test_mixer_input_before_the_kernel_values(self, model):
        """The kernel choice is read inside the forward, so the mixer's own input can be read first, on a prompt and a decode step."""
        mix = self.ssd(model)
        seqs = []
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[:2]:
                x = mix.input
                seqs.append((x.shape[1], mix.attention_values.shape[1]))
        assert seqs == [(len(model.tokenizer(PROMPT).input_ids),) * 2, (1, 1)]

    def test_written_state_input_moves_the_step(self, model):
        mix = self.ssd(model)
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[1]:
                clean = mix.attention_head_outputs.save()
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[1]:
                mix.state_input = torch.zeros_like(mix.state_input)
                edited = mix.attention_head_outputs.save()
        assert clean.shape[1] == 1 and not torch.equal(clean, edited)

    # -- the state after every token: chunk_per_token ------------------------------------

    def test_states_per_token(self, model):
        """With a chunk size of 1 the chunk scan's boundaries are the tokens: ``states`` is SSD's recurrence, token by token."""
        mix = self.ssd(model)
        m = mix._module
        n = len(model.tokenizer(PROMPT).input_ids)
        t = n // 2
        with model.trace(PROMPT):
            default = model.logits.save()
        chunk_per_token(model)
        try:
            with model.trace(PROMPT):
                B, x = mix.attention_keys.save(), mix.attention_values.save()
                betas, decays = mix.betas.save(), mix.decays.save()
                states = mix.states.save()
                final = mix.state_output.save()
                logits = model.logits.save()
            with model.trace(PROMPT):
                after = mix.state_after(t).save()
        finally:
            chunk_per_token(model, False)
        assert states.shape == (1, n, m.num_heads, m.ssm_state_size, m.head_dim)
        torch.testing.assert_close(states[:, -1], final)
        torch.testing.assert_close(logits, default)
        torch.testing.assert_close(after, states[:, t])
        prev = torch.zeros_like(states[:, 0])                                  # a fresh prompt starts from nothing
        for t in range(n):
            expected = self._recurrence(prev, B[:, t], x[:, t], betas[:, t], decays[:, t], m.num_heads // m.n_groups)
            torch.testing.assert_close(states[:, t].float(), expected, rtol=1e-4, atol=1e-5, msg=f"token {t}")
            prev = states[:, t]

    def test_states_on_a_decode_step(self, model):
        """A decode step's ``states`` is its one token's state, the update's, from the step's ``state_input``."""
        mix = self.ssd(model)
        m = mix._module
        chunk_per_token(model)
        try:
            with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
                for step in tracer.iter[1]:
                    entering = mix.state_input.save()
                    B, x = mix.attention_keys.save(), mix.attention_values.save()
                    betas, decays = mix.betas.save(), mix.decays.save()
                    states = mix.states.save()
                    final = mix.state_output.save()
        finally:
            chunk_per_token(model, False)
        assert states.shape == (1, 1, m.num_heads, m.ssm_state_size, m.head_dim)
        assert torch.equal(states[:, 0], final)
        expected = self._recurrence(entering, B[:, 0], x[:, 0], betas[:, 0], decays[:, 0], m.num_heads // m.n_groups)
        torch.testing.assert_close(states[:, 0].float(), expected, rtol=2e-2, atol=2e-3)

    def test_states_need_chunk_per_token(self, model):
        mix = self.ssd(model)
        reason = model.support(layer=self.SSD_BLOCK)["linear_attn.states"]
        assert f"chunk_size={mix._module.chunk_size}" in reason and "chunk_per_token" in reason
        with pytest.raises(Unavailable, match="chunk_per_token"):
            mix.state_after(0)
        chunk_per_token(model)
        try:
            assert model.support(layer=self.SSD_BLOCK)["linear_attn.states"] is None
        finally:
            chunk_per_token(model, False)

    def test_chunk_per_token_restores_the_config(self, model):
        config = model.config
        built = config.mamba_chunk_size if hasattr(config, "mamba_chunk_size") else config.chunk_size
        mixers = [layer.linear_attn._module for layer in model.layers if getattr(layer, "linear_attn", None) is not None]
        chunk_per_token(model)
        assert all(m.chunk_size == 1 for m in mixers)
        chunk_per_token(model, False)
        assert all(m.chunk_size == built for m in mixers)

    def test_no_per_token_occurrence_and_no_per_token_write(self, model):
        """``state`` and ``set_state_after`` stay unavailable, with or without ``chunk_per_token``, and say why."""
        mix = self.ssd(model)
        chunk_per_token(model)
        try:
            support = model.support(layer=self.SSD_BLOCK)
            assert support["linear_attn.state"] == NO_STATE_OCCURRENCES
            assert support["linear_attn.set_state_after"] == NO_STATE_WRITES
            with pytest.raises(Unavailable, match="one cumulative step"):
                mix.set_state_after(0, torch.zeros(1))
            with pytest.raises(Unavailable, match="one tensor per call"):
                mix.state
        finally:
            chunk_per_token(model, False)

    # -- betas and decays are the kernel's dt ------------------------------------------------

    def test_written_betas(self, model):
        """A zero write strength on a decode step stops the write and the decay: the state passes through unchanged.
        On a prompt a written value reads back, and ``decays`` follows it."""
        mix = self.ssd(model)
        m = mix._module
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[1]:
                entering = mix.state_input.save()
                mix.betas = torch.zeros_like(mix.betas)
                back = mix.betas.save()
                leaving = mix.state_output.save()
        assert (back == 0).all()
        torch.testing.assert_close(leaving.float(), entering.float())
        with model.trace(PROMPT):
            betas = mix.betas
            written = (betas * 2).save()
            mix.betas = written
            back = mix.betas.save()
            decays = mix.decays.save()
        torch.testing.assert_close(back, written)
        torch.testing.assert_close(decays, -torch.exp(m.A_log.float()) * back.float())

    def test_written_decays(self, model):
        """A written log decay on a decode step reads back and runs: the state is SSD's recurrence with it (and the
        write strength it implies, ``decays / A``)."""
        mix = self.ssd(model)
        m = mix._module
        with model.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
            for step in tracer.iter[1]:
                entering = mix.state_input.save()
                B, x = mix.attention_keys.save(), mix.attention_values.save()
                written = (mix.decays * 0.5).save()
                mix.decays = written
                back = mix.decays.save()
                betas = mix.betas.save()
                leaving = mix.state_output.save()
        # float32 decays through the kernel's dt, which is bf16 on a bf16 checkpoint: exact to that precision
        torch.testing.assert_close(back, written, rtol=2e-2, atol=1e-5)
        torch.testing.assert_close(betas.float() * -torch.exp(m.A_log.float()), written, rtol=2e-2, atol=1e-5)
        expected = self._recurrence(entering, B[:, 0], x[:, 0], betas[:, 0], written[:, 0], m.num_heads // m.n_groups)
        torch.testing.assert_close(leaving.float(), expected, rtol=2e-2, atol=2e-3)

    # -- one argument read per call ------------------------------------------------------------

    def test_values_in_forward_order_in_one_trace(self, model):
        """Several values of one call in one trace, whatever each needs of the call's arguments."""
        mix = self.ssd(model)
        with model.trace(PROMPT):
            final = mix.state_output.save()
            y = mix.attention_head_outputs.save()
        with model.trace(PROMPT):
            values = mix.attention_values.save()
            first = mix.state_output.save()
        with model.trace(PROMPT):
            queries = mix.attention_queries.save()
            betas = mix.betas.save()
            heads = mix.attention_head_outputs.save()
            second = mix.state_output.save()
        assert torch.equal(first, final) and torch.equal(second, final) and torch.equal(heads, y)
        assert values.dim() == queries.dim() == 4 and betas.dim() == 3
        chunk_per_token(model)
        try:
            with model.trace(PROMPT):
                values = mix.attention_values.save()
                states = mix.states.save()
                third = mix.state_output.save()
        finally:
            chunk_per_token(model, False)
        torch.testing.assert_close(third, final)
        torch.testing.assert_close(states[:, -1], third)

    def test_optimized_kernels_are_reported(self, model):
        """With ``mamba_ssm``'s kernels bound, every kernel value says so and names the way out."""
        pytest.importorskip("mamba_ssm")
        route_kernels(self.FAMILY, "default")
        try:
            reason = model.support(layer=self.SSD_BLOCK)["linear_attn.attention_queries"]
            assert "mamba_ssm" in reason and "route_kernels" in reason
            assert model.support(layer=self.SSD_BLOCK)["linear_attn.attention_output"] is None
        finally:
            route_kernels(self.FAMILY, "torch")

    def test_ssd_values_listed_in_the_repr(self, model):
        text = repr(self.ssd(model))
        for name in LINEAR:
            assert listed(name, text), name
