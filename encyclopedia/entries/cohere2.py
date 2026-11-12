"""Cohere 2: Command R7B and Command A (Cohere2ForCausalLM), and the Aya Vision and Command A Vision wrappers."""

MODEL_TYPE = "cohere2"
TITLE = "Command R7B / Command A"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Every CohereLabs repository is gated; this is an ungated copy of c4ai-command-r7b-12-2024 (its card names that base).
REFERENCE = "mlx-community/c4ai-command-r7b-12-2024-bf16"
#: The Hub org the index files the family under: the copy is mlx-community's; the model is CohereLabs'.
ORG = "CohereLabs"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Cohere2ForCausalLM"
CHECKPOINTS = [
    "CohereLabs/c4ai-command-r7b-12-2024",
    "CohereLabs/c4ai-command-r7b-arabic-02-2025",
    "CohereLabs/c4ai-command-a-03-2025",
    # Ungated copies of Command R7B and Command A.
    "mlx-community/c4ai-command-r7b-12-2024-bf16",
    "unsloth/c4ai-command-a-03-2025",
    # Vision-language wrappers around a Cohere 2 text model, each a key of WRAPPERS by its config's model_type.
    "CohereLabs/aya-vision-8b",
    "CohereLabs/command-a-vision-07-2025",
    "unsloth/aya-vision-8b",  # an ungated copy of aya-vision-8b
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "aya_vision": {
        "title": "Aya Vision",
        "pinned": "hf-tiny-v2/tiny-random-AyaVisionForConditionalGeneration",
        "projector": "multi_modal_projector: a 2 × 2 pixel shuffle of the last block's patches, a LayerNorm, "
                     "then linear_1, a SwiGLU and linear_2",
        "projector_input": "the last block's layer_output, before vision.norm",
        "quirks": ["pooled-projector", "tiled-images"],
        "notes": """
## The projector reads the last block, before `vision.norm`

A write to `vision.tower_output` does not reach the text model; write `vision.layers[-1].layer_output`.
""",
    },
    "cohere2_vision": {
        "title": "Command A Vision",
        "pinned": "hf-tiny-v2/tiny-random-Cohere2VisionForConditionalGeneration",
        "projector": "multi_modal_projector: a 2 × 2 pixel shuffle of vision.tower_output, then linear_1, a SwiGLU "
                     "and linear_2, with no norm",
        "projector_input": "`vision.tower_output`",
        "quirks": ["pooled-projector", "tiled-images"],
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.
""",
    },
}

#: Set by hues.py (lineage: Cohere).
PALETTE = {"hue": 177}
QUIRKS = [
    "parallel-blocks", "layernorm", "sliding-window", "nope-blocks", "interleaved-rotary",
    # Cohere2ForCausalLM multiplies lm_head's output by logit_scale; the Aya Vision and Command A Vision wrappers return it
    # unscaled. A causal-LM checkpoint's config names its class in `architectures`; a wrapper's text config names none.
    {"slug": "scaled-logits", "when": lambda config: "Cohere2ForCausalLM" in (config.architectures or [])},
]

#: What the visualization draws: one input_layernorm whose output both sublayers read (drawn once per branch,
#: one node), no post-norms; the two contributions join the stream in one add.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
            "variants": {
                "sliding_attention": "{sliding_window}-token window, rotary",
                "full_attention": "full causal, no rotary",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "logits": "logits = lm_head.output × logit_scale on Command R7B and Command A, and project_on_vocab multiplies "
              "too; the vision wrappers apply no scale.",
}

NOTES = """
## The MLP does not read the attention

```
h   = input_layernorm(x)
out = x + self_attn(h) + mlp(h)
```

Both sublayers read the one norm's output, so ablating or patching a block's attention leaves that
block's MLP unchanged. The first MLP to read block `i`'s attention is block `i + 1`'s.

## The logits are the head's output times `logit_scale`

On Command R7B and Command A `logit_scale` is 0.25. `logits` is the scaled tensor and
`project_on_vocab` multiplies too, so a lens at the last block equals `logits`. A softmax or a
direct logit attribution taken on `lm_head.output` is off by that factor. The Aya Vision and Command
A Vision wrappers apply no scale.
"""
