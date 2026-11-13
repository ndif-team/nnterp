"""CodeGen, end to end: GPT-J's tuple parallel block, arithmetic in ``_attn`` with the mask added before the scaling."""

import math

import pytest
import torch
from suite import FamilySuite, rows, PROMPT, built, patched_checkpoint, random_weights

from nnterp.families import codegen


class TestCodeGen(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-CodeGenForCausalLM"
    FAMILY = codegen
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel: ln_1 feeds both sublayers
    EXPECTED_UNAVAILABLE = {"layer_mid": "a parallel block"}

    def test_an_in_place_edit_before_the_rotary(self, model):
        """The head split reshapes a ``torch.split`` output: with one head per ``mp_num`` group (the tiny checkpoint's 4
        heads) the result is a view torch refuses to edit in place; with more (every released CodeGen: 16 to 24
        heads) it is a copy, and the edit lands."""
        attn = model.layers[0].self_attn
        if model.num_heads // 4 == 1:
            with pytest.raises(RuntimeError, match="view"):
                with model.trace(PROMPT):
                    attn.attention_queries_prerope[:, 0, -1] += 1.0
            return
        with model.trace(PROMPT):
            clean = attn.attention_queries.save()
        with model.trace(PROMPT):
            attn.attention_queries_prerope[:, 0, -1] += 1.0
            moved = attn.attention_queries.save()
        assert not torch.equal(moved[:, 0, -1], clean[:, 0, -1]) and torch.equal(moved[:, 1:], clean[:, 1:])

    def test_block_returns_a_tuple(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].output.save()
        assert isinstance(raw, tuple)

    def test_scores_are_scaled_after_the_mask(self, model):
        """The scores are ``(q @ k^T + mask) / sqrt(head_dim)`` where the mask lets them through."""
        with model.trace(PROMPT):
            q = model.layers[0].self_attn.attention_queries.save()
            k = model.layers[0].self_attn.attention_keys.save()
            scores = model.layers[0].self_attn.attention_scores.save()
        raw = q.float() @ k.float().transpose(-1, -2) / math.sqrt(model.head_dim)
        causal = torch.ones(raw.shape[-2:], dtype=torch.bool).tril()
        torch.testing.assert_close(scores[..., causal], raw[..., causal])

    def test_interior_needs_no_eager_load(self):
        """CodeGen has one attention implementation, so a default load serves the pattern."""
        from nnterp import StandardizedTransformer

        model = StandardizedTransformer(self.REPO, dispatch=True)
        assert model.support()["self_attn.attention_probabilities"] is None
        with model.trace(PROMPT):
            pattern = model.layers[0].self_attn.attention_probabilities.save()
        torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))


def test_value_weight_with_two_heads_per_part():
    """The tiny checkpoint has one head in each of the four ``mp_num`` parts, where a part and a head within it cannot be
    told apart; two per part can, so random weights."""
    model = random_weights(TestCodeGen.REPO, n_head=8, rotary_dim=4)
    assert model.num_heads // 4 == 2
    FamilySuite().test_value_weight_reproduces_the_values(model)


class TestCodeGenHeads(TestCodeGen):
    """Every released CodeGen's head layout: several heads per ``mp_num`` group (16 heads of 4 here; 16 of 64 on
    codegen-350M), where the head split copies. The tiny config with 16 heads, weights drawn for it."""

    REPO = built(patched_checkpoint, TestCodeGen.REPO, draw=True, n_head=16, n_embd=64)
