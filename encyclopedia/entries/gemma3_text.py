"""Gemma 3, text: Gemma 2's sandwich block with per-head query and key norms, and no softcaps."""

MODEL_TYPE = "gemma3_text"
TITLE = "Gemma 3"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-3-1b-pt"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
CHECKPOINTS = [
    "google/gemma-3-270m", "google/gemma-3-270m-it",
    "google/gemma-3-1b-pt", "google/gemma-3-1b-it",
    # Vision-language wrappers around a Gemma 3 text model: model_type gemma3, a key of WRAPPERS.
    "google/gemma-3-4b-pt", "google/gemma-3-4b-it",
    "google/gemma-3-12b-pt", "google/gemma-3-12b-it",
    "google/gemma-3-27b-pt", "google/gemma-3-27b-it",
]

#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "gemma3": {
        "title": "Gemma 3",
        "pinned": "yujiepan/gemma-3-tiny-random",
        "projector": "multi_modal_projector: a 4 × 4 average pool over vision.tower_output's 64 × 64 patches, "
                     "an RMSNorm (mm_soft_emb_norm) and a matrix product (mm_input_projection_weight)",
        "projector_input": "`vision.tower_output`",
        "quirks": ["pooled-projector"],
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.

## An image's tokens attend to each other in both directions

With the processor's `token_type_ids`, the tokens of one image attend to each other, later ones
included, on sliding and full blocks alike. An image token still attends causally to earlier images
and text, and text stays causal.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 159}
QUIRKS = ["sandwich-norms", "qk-norm", "sliding-window", "scaled-embeddings", "gain-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads, head_dim {head_dim}; "
                      "q_norm and k_norm per head before the rotary; scores scaled by {query_pre_attn_scalar}^-1/2",
            "variants": {
                "sliding_attention": "{sliding_window}-token window, local RoPE",
                "full_attention": "full causal, global RoPE",
            },
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_feedforward_layernorm",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_activation}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by √hidden_size inside the embedding module, so token_embeddings equals layers[0].input on text.",
}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. RMSNorm is scale-invariant, so halving `mlp.output` does almost nothing to the stream.
Scale, steer and decompose through `attention_output` and `mlp_output`.

## Gemma Scope 2's `attn_out` SAEs read `o_proj.input`

That is the heads' outputs concatenated, `num_heads × head_dim` wide, not `attention_output`. The
`mlp_out` SAEs read `mlp_output`.
"""
