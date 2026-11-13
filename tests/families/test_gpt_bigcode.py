"""GPT-BigCode, end to end: GPT-2's names with multi-query attention."""

from suite import NO_ROTARY, FamilySuite, PROMPT, random_weights, rows

from nnterp.families import gpt_bigcode


class TestGPTBigCode(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-GPTBigCodeForCausalLM"
    FAMILY = gpt_bigcode
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v are split views of one c_attn tensor
    EXPECTED_UNAVAILABLE = NO_ROTARY

    def test_one_kv_head_under_multi_query(self, model):
        assert model.config.multi_query and model.num_kv_heads == 1
        with model.trace(PROMPT):
            keys = model.layers[0].self_attn.attention_keys.save()
        assert keys.shape[1] == 1

    def test_intermediate_size_is_n_inner(self, model):
        assert model.intermediate_size == (model.config.n_inner or 4 * model.hidden_size)


def test_value_weight_without_multi_query():
    """``c_attn`` without multi-query holds each head's query, key and value; no tiny checkpoint has it, so random weights."""
    model = random_weights(TestGPTBigCode.REPO, multi_query=False)
    assert model.num_kv_heads == model.num_heads
    FamilySuite().test_value_weight_reproduces_the_values(model)
