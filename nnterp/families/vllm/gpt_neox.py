"""GPT-NeoX (Pythia) on vLLM, through its transformers backend (``GPTNeoXForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`).
The tree is transformers' under ``model`` rather than ``gpt_neox`` (the backend
builds the bare ``GPTNeoXModel``): ``model.{embed_in, layers[i].{input_layernorm,
attention, post_attention_layernorm, mlp}, final_layer_norm}`` and ``lm_head``.
The block adds both contributions itself, in parallel or in sequence as the
config's ``use_parallel_residual`` says, and returns the stream.
"""

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXMLP

from ...components import PARALLEL, Residual
from ...components.vllm import Flat
from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_in": "embed_tokens",
    "model.layers": "layers",
    "model.final_layer_norm": "norm",
    "attention": "self_attn",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "act": "act_fn",
    "dense_4h_to_h": "down_proj",
}


class Layer(Layer):
    """GPT-NeoX's block: called with the stream and returning it."""

    # With ``use_parallel_residual`` (Pythia's default) the block is parallel: no stream between the sublayers,
    # and the MLP reads its own norm of the block's input.

    @Flat(
        "post_attention_layernorm.input",
        batch=True,
        unavailable=lambda block: PARALLEL if block.root.config.use_parallel_residual else None,
        description="The residual stream after the attention add, [1, tokens, hidden_size]",
        in_place=True,
    )
    def layer_mid(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTNeoXLayer: Layer, GPTNeoXAttention: Attention, GPTNeoXMLP: Mlp}
