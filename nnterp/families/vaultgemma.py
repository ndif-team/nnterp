"""VaultGemma (``VaultGemmaForCausalLM``).

Gemma-2's tree without the post-norms: a pre-norm block,
``x + attn(input_layernorm(x))`` and then ``+ mlp(pre_feedforward_layernorm(x))``,
so the contributions are the modules' outputs and the base holds.
``pre_feedforward_layernorm`` is the pre-MLP norm, aliased
``post_attention_layernorm`` (the Llama meaning; VaultGemma has no norm after the
attention). The embedding module scales its own output by ``sqrt(hidden_size)``,
so ``token_embeddings`` is ``layers[0].input``. The attention runs the shared
eager forward with Gemma-2's score softcapping inside it (``attention_scores`` are
the capped scores entering the softmax); ``final_logit_softcapping`` applies to
the logits after ``lm_head``, which the root's ``project_on_vocab`` reads.
"""

from transformers.models.vaultgemma.modeling_vaultgemma import VaultGemmaAttention, VaultGemmaDecoderLayer, VaultGemmaMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "pre_feedforward_layernorm": "post_attention_layernorm",
}


class Layer(Layer):
    """VaultGemma's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """VaultGemma's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """VaultGemma's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {VaultGemmaDecoderLayer: Layer, VaultGemmaAttention: Attention, VaultGemmaMLP: Mlp}
