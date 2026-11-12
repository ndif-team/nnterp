"""ZAYA1: residual scaling modules after every sublayer, convolutional queries and keys, and a one-expert mixture with a skip class."""

MODEL_TYPE = "zaya"
TITLE = "ZAYA1"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Zyphra/ZAYA1-8B"
#: The tiny checkpoint the test suite builds the page from (the suite runs a copy with qk_norm.temp set to one).
PINNED = "hf-tiny-v2/tiny-random-ZayaForCausalLM"
#: The -base, -reasoning-base and -legacy checkpoints keep the original layout's config (`rope_scaling: false`),
#: which transformers does not parse; the page lists them greyed out with that reason.
CHECKPOINTS = [
    "Zyphra/ZAYA1-8B", "Zyphra/ZAYA1-8B-FP8-Experts", "Zyphra/ZAYA1-8B-MXFP4-Experts", "Zyphra/ZAYA1-74B-preview",
    "Zyphra/ZAYA1-base", "Zyphra/ZAYA1-reasoning-base", "Zyphra/ZAYA1-8B-legacy", "Zyphra/ZAYA1-74B-preview-legacy",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 354}
QUIRKS = [
    "scaled-residual-adds", "embedding-multiplier", "partial-rotary", "conv-qk",
    "sliding-window", "mixture-of-experts", "unnormalized-routing",
]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "merge": "post_attention_residual_scale",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads over {num_kv_heads} kv, conv q/k",
            "variants": {
                "hybrid": "full causal, conv q/k",
                "hybrid_sliding": "sliding window, conv q/k",
            },
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, + skip",
            "pre_norm_note": "The mixture's input norm, over the stream after the attention's merge.",
        },
    ],
    "identity": "h = (layers[i].input + a.residual_bias) * a.residual_scale + self_attn.attention_output; "
                "(h + m.residual_bias) * m.residual_scale + mlp.mlp_output == layer_output",
    "identity_note": "a and m are layers[i].post_attention_residual_scale._module and post_mlp_residual_scale._module, "
                     "the block's two merges.",
}

STRIP = {
    "embed": "layers[0].input is (token_embeddings + input_hidden_states_bias) * input_hidden_states_scale, per channel.",
}

NOTES = """
## ZAYA1-8B's tokenizer appends `<|im_end|>` to every prompt

On ZAYA1-8B a plain prompt's last position is `<|im_end|>`, so the logits there are not the prediction
after your text. Tokenize with `add_special_tokens=False` and prepend `<bos>` (id 2) yourself, or read
position -2. ZAYA1-74B-preview's tokenizer appends nothing.

## The contributions are the merges' scaled terms

```
merge(o, r) = (o + hidden_states_bias) * hidden_states_scale + (r + residual_bias) * residual_scale
h   = post_attention_residual_scale(self_attn(input_layernorm(x)), x)
out = post_mlp_residual_scale(mlp(post_attention_layernorm(h)), h)
```

`attention_output` and `mlp_output` are each merge's first term, and the merge also shifts and
rescales the stream, so zeroing `attention_output` leaves `(x + residual_bias) * residual_scale`, not
`x`, and zeroing the chosen expert's weight leaves `hidden_states_bias * hidden_states_scale`. A write
to a contribution is divided back by `hidden_states_scale` over the whole tensor, which in bfloat16
rounds the positions it does not touch; load in float32 for small edits.

## An early term reaches the logits shrunk by every later merge

Each merge multiplies the stream by its `residual_scale`, channel by channel, so a contribution reaches
the last block's stream multiplied by the product of every later `residual_scale`. On ZAYA1-8B that
product is far below one on every channel, and a direct logit attribution has to carry it.

## The router carries a state from block to block

Each block's router adds the previous block's router state to its input, and the block returns that
state beside the stream, so `layer_output` is not the whole state between blocks. A patch of
`layer_output` alone leaves this run's router state for the later routers; for a full interchange
between two prompts of one length, patch `layers[i].output`, both elements.

## The second key/value head holds the previous token's values

On ZAYA1-8B and 74B-preview, with two key/value heads, `attention_values` head 0 is `v_proj_current`
of each token and head 1 is `v_proj_delayed` of the token before. The query heads that read key/value
head 1, the second half, move the previous token's value, so a weight on key `j` in their pattern
carries token `j - 1`'s.
"""
