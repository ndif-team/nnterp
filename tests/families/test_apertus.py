"""Apertus, end to end: Llama's block with its norms under their own names."""

from suite import FamilySuite, rows, built, patched_checkpoint

from nnterp.families import apertus


class TestApertus(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-ApertusForCausalLM"
    FAMILY = apertus
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1="attention_layernorm", ln2="feedforward_layernorm")


def draw_xielu(model):
    """The xIELU activation's learned ``alpha_p`` / ``alpha_n`` drawn per block: an initialization makes them all alike."""
    for layer in model.model.layers:
        layer.mlp.act_fn.alpha_p.normal_(0.5, 0.3)
        layer.mlp.act_fn.alpha_n.normal_(0.5, 0.3)


class TestApertusReleased(TestApertus):
    """Every released Apertus's layout: the xIELU activation, grouped-query attention (4 heads, 2 key/value heads here;
    32 and 8 on Apertus-8B). The tiny config otherwise, weights drawn for it, the activation's parameters too."""

    REPO = built(patched_checkpoint, TestApertus.REPO, init=draw_xielu, hidden_act="xielu", num_attention_heads=4, num_key_value_heads=2)
