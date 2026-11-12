"""Doge: Llama's tree with a gated residual stream and a dynamic mask on the attention scores."""

MODEL_TYPE = "doge"
TITLE = "Doge"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Doge-260M is the largest dense checkpoint whose weights load in transformers.
REFERENCE = "SmallDoge/Doge-260M"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-DogeForCausalLM"
#: Dense checkpoints first, then the cross-domain mixtures (is_moe), then the ones whose weights do not load.
CHECKPOINTS = [
    "SmallDoge/Doge-20M", "SmallDoge/Doge-40M", "SmallDoge/Doge-260M",
    "SmallDoge/Doge-40M-MoE", "SmallDoge/Doge-180M-MoE",
    "SmallDoge/Doge-20M-Instruct", "SmallDoge/Doge-60M", "SmallDoge/Doge-60M-Instruct",
    "SmallDoge/Doge-160M", "SmallDoge/Doge-160M-Instruct", "SmallDoge/Doge-320M", "SmallDoge/Doge-320M-Instruct",
    "SmallDoge/Doge-20M-MoE", "SmallDoge/Doge-120M-MoE",
]

#: Checkpoints whose safetensors (read from their headers on 2026-10-07) do not fit transformers'
#: DogeForCausalLM: they hold the layout of the repositories' own remote code. Their configs build a model, but
#: each block's ``A`` and ``dt_proj`` are twice ``num_key_value_heads`` wide where transformers builds
#: ``num_key_value_heads``, and the attention has no ``q_norm`` or ``k_norm``, so ``from_pretrained`` raises on the
#: size mismatch. The page lists them greyed out with the reason.
GREYED = {
    checkpoint: "transformers does not load its weights: A and dt_proj are twice num_key_value_heads wide, "
                "and the attention has no q_norm or k_norm"
    for checkpoint in (
        "SmallDoge/Doge-20M-Instruct", "SmallDoge/Doge-60M", "SmallDoge/Doge-60M-Instruct",
        "SmallDoge/Doge-160M", "SmallDoge/Doge-160M-Instruct", "SmallDoge/Doge-320M", "SmallDoge/Doge-320M-Instruct",
        "SmallDoge/Doge-20M-MoE", "SmallDoge/Doge-120M-MoE",
    )
}

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 21}
QUIRKS = ["scaled-residual-adds", "qk-norm", "dynamic-mask"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads over {num_kv_heads} kv, dynamic mask",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
        },
    ],
    "identity": "post_attention_residual * (input_residual * layers[i].input + self_attn.attention_output) "
                "+ mlp.mlp_output == layer_output",
    "identity_note": "input_residual and post_attention_residual are the block's own [hidden_size] parameters, on "
                     "layers[i]._module.",
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The block scales the stream, not what it adds

```
h   = input_residual * x + self_attn(input_layernorm(x))
out = post_attention_residual * h + mlp(post_attention_layernorm(h))
```

The gates are per-channel parameters of the block, below one almost everywhere on Doge-260M, so a
term added at block `i` reaches the logits multiplied, channel by channel, by every later gate. A
direct logit attribution has to include them. Zeroing `attention_output` does not leave the block's
input in place, since the stream still goes on as `input_residual * x`. On Doge-260M the last block's
`post_attention_residual` nearly zeroes the stream's one massive channel, so a steer along that
channel is erased before the final norm.

## A write to `attention_values` does not move the dynamic mask

The bias each key adds to the scores is computed from the values before the attention call, so a
write to `attention_values` changes what the pattern mixes and leaves the bias as it was. To move a
key's bias with its value, write `self_attn.v_proj.output`.
"""
