"""GPT-NeoX (Pythia) on vLLM (``vllm.model_executor.models.gpt_neox``).

Its tree is transformers' (``gpt_neox.{embed_in, layers[i].{input_layernorm,
attention, post_attention_layernorm, mlp}, final_layer_norm}`` plus
``embed_out``). The block is called with the positions and the residual
stream and returns the stream, adding both contributions itself, in parallel
or in sequence as the config's ``use_parallel_residual`` says.
"""

from vllm.model_executor.models.gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXMLP

from ...components import PARALLEL, Residual
from ...components.vllm import Attention, Flat, Layer, Mlp

RENAME = {
    "gpt_neox.embed_in": "embed_tokens",
    "gpt_neox.layers": "layers",
    "gpt_neox.final_layer_norm": "norm",
    "embed_out": "lm_head",
    "attention": "self_attn",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "act": "act_fn",
    "dense_4h_to_h": "down_proj",
}


class Layer(Layer):
    """The block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1

    # With ``use_parallel_residual`` (Pythia's default) the block is parallel: no stream between the sublayers,
    # and the MLP reads its own norm of the block's input.

    @Flat(
        "post_attention_layernorm.input",
        unavailable=lambda block: PARALLEL if block.root.config.use_parallel_residual else None,
        description="The residual stream after the attention add, [1, tokens, hidden_size]",
    )
    def layer_mid(self, value) -> Residual:
        return value


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTNeoXLayer: Layer, GPTNeoXAttention: Attention, GPTNeoXMLP: Mlp}
