"""GPT-OSS (``GptOssForCausalLM``).

Llama's tree with a mixture-of-experts MLP that returns ``(hidden_states,
router_scores)``, sliding-window layers, and an **attention sink**: each head
carries a learned logit that joins the softmax as one extra key column and is
dropped afterwards. So the pattern's rows sum to *less* than one, and the
softmax's input is one key wider than the pattern and shifted by its row max;
the standard ``attention_scores`` is therefore read one step earlier, at the
masked scores just before the sink joins them (a binding that exists only
when an attention mask is passed, as it is on every prompt).
"""

from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssDecoderLayer, GptOssMLP

from ..components import Attention, EProperty, INTERFACE, Layer, Moe, Pattern, interface_reason

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

class Layer(Layer):
    """GPT-OSS's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GPT-OSS's attention: the shared interface, with a sink column in the softmax.

    The pattern (the dropout output) is already without the sink column, so
    the base holds there; the scores are the masked scores bound just before
    the sink column is concatenated.
    """

    #: The pattern's rows sum to less than one: the sink takes the rest.
    SINK = True

    @EProperty(f"source.{INTERFACE}.source.attn_weights_1.output", description=Attention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value


class Mlp(Moe):
    """GPT-OSS's mixture of experts returns ``(hidden_states, router_scores)``; the base takes the first."""

    SCORING = "topk_softmax"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GptOssDecoderLayer: Layer, GptOssAttention: Attention, GptOssMLP: Mlp}
