"""GPT-BigCode, end to end: GPT-2's names with multi-query attention."""

from suite import FamilySuite, PROMPT, rows

from nnter.families import gpt_bigcode


class TestGPTBigCode(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-GPTBigCodeForCausalLM"
    FAMILY = gpt_bigcode
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v are split views of one c_attn tensor

    def test_one_kv_head_under_multi_query(self, model):
        assert model.config.multi_query and model.num_kv_heads == 1
        with model.trace(PROMPT):
            keys = model.layers[0].self_attn.attention_keys.save()
        assert keys.shape[1] == 1

    def test_intermediate_size_is_n_inner(self, model):
        assert model.intermediate_size == (model.config.n_inner or 4 * model.hidden_size)
