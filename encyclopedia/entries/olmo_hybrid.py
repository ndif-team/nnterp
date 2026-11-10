"""OLMo-Hybrid: AI2's gated DeltaNet hybrid (OlmoHybridForCausalLM), pre-norm DeltaNet blocks and OLMo-3's post-norm attention blocks."""

MODEL_TYPE = "olmo_hybrid"
TITLE = "OLMo-Hybrid"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/Olmo-Hybrid-7B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-OlmoHybridForCausalLM"
CHECKPOINTS = [
    "allenai/Olmo-Hybrid-7B",
    "allenai/Olmo-Hybrid-Instruct-SFT-7B", "allenai/Olmo-Hybrid-Instruct-DPO-7B",
    "allenai/Olmo-Hybrid-Think-SFT-7B",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 110}
QUIRKS = [
    "hybrid", "post-norms", "qk-norm",
    # A config with no rope_theta builds no rotary embedding (OlmoHybridModel.__init__); Think-SFT's null
    # rope_parameters is filled with a theta of 10000, and so is the pinned tiny checkpoint's.
    {"slug": "nope-blocks", "when": lambda config: not (config.rope_parameters or {}).get("rope_theta")},
]

#: Every checkpoint is 7B, so no real weights were run: shapes, identities and read places are from the
#: pinned tiny checkpoint with the released config's rope_theta null and linear_allow_neg_eigval written in
#: (tests/families/test_olmo_hybrid.py's copy), routed, float32, CPU; sizes from the checkpoints' configs.

#: The two block classes put the same native norm in two roles and the one MLP in two places: on a DeltaNet
#: block post_attention_layernorm is the MLP's pre-norm, on an attention block the attention's post-norm,
#: and the MLP is pre-normed on one and post-normed on the other. Each sublayer's `block` key names the
#: native block class it is drawn on, so `mlp` is listed once per class with its own norms.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "block": "OlmoHybridLinearAttentionDecoderLayer",
            "kind": "mixer",
            "label": "Linear attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "decays", "betas",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "gated DeltaNet, {linear_num_value_heads} heads",
        },
        {
            "host": "self_attn",
            "block": "OlmoHybridAttentionDecoderLayer",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "post_norm_note": "Follows the attention on this block; on a DeltaNet block the same name is the MLP's pre-norm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "block": "OlmoHybridLinearAttentionDecoderLayer",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm on this block; on an attention block the same name follows the attention.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
        },
        {
            "host": "mlp",
            "block": "OlmoHybridAttentionDecoderLayer",
            "kind": "mlp",
            "label": "MLP",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## On attention blocks the contributions are the post-norms' outputs

```
# a DeltaNet block: pre-norm
h   = x + linear_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))

# an attention block: post-norm, no input_layernorm
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

On a DeltaNet block `attention_output` and `mlp_output` are the modules' outputs. On an attention
block they are the outputs of the norms after the sublayers, so scaling `self_attn.output` or
`mlp.output` there does nothing to the stream; scale, steer and decompose through `attention_output`
and `mlp_output`.

## `q_norm` and `k_norm` span every head

On the attention blocks each takes one RMS over the whole projection before the heads are split, so an
edit to one head's slice of `q_proj.output` reaches the other heads. Edit one head at
`attention_queries` or `attention_keys`, which no norm follows.

## Think-SFT's attention is rotary

The released configs set `rope_theta` to null, and the attention blocks then apply no rotary embedding.
Olmo-Hybrid-Think-SFT-7B sets `rope_parameters` itself to null, which transformers fills with a theta
of 10000, so its attention blocks do rotate their queries and keys.
"""
