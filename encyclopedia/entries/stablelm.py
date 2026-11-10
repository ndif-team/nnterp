"""StableLM: Stability AI's StableLM-3B-4E1T and StableLM 2, which load as StableLmForCausalLM."""

MODEL_TYPE = "stablelm"
TITLE = "StableLM 2 / StableLM-3B"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "stabilityai/stablelm-2-1_6b"
#: The tiny checkpoint the test suite builds the page from (StableLM-2-12B's shape: parallel, qk_layernorm).
PINNED = "stabilityai/tiny-random-stablelm-2"
CHECKPOINTS = [
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "stabilityai/stablelm-2-12b",
    "stabilityai/stablelm-3b-4e1t",
    "stabilityai/stablelm-zephyr-3b",
]

#: Set by hues.py (lineage: GPT-NeoX).
PALETTE = {"hue": 69}
QUIRKS = [
    "layernorm", "partial-rotary",
    {"slug": "qkv-bias", "when": lambda config: config.use_qkv_bias},  # StableLM 2 1.6B and its zephyr
    {"slug": "parallel-blocks", "when": lambda config: config.use_parallel_residual},  # StableLM-2-12B
    {"slug": "qk-norm", "when": lambda config: config.qk_layernorm},  # StableLM-2-12B
]

ATTENTION_INTERIOR = [
    "attention_queries", "attention_keys", "attention_values",
    "attention_scores", "attention_probabilities", "attention_head_outputs",
]
MLP_DETAIL = "{hidden_size} → {intermediate_size} → {hidden_size}, gated {hidden_act}"

#: StableLM-2-1.6B and StableLM-3B (use_parallel_residual false): input_layernorm before the attention,
#: post_attention_layernorm before the MLP, each sublayer adding to the stream in turn.
SEQUENTIAL = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": ATTENTION_INTERIOR,
            "detail": "{num_heads} heads, {num_kv_heads} kv; rotary on ¼",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": MLP_DETAIL,
        },
    ],
}

#: StableLM-2-12B and the pinned checkpoint (use_parallel_residual true): one input_layernorm whose output both
#: sublayers read (drawn once per branch, one node), per-head q/k LayerNorms inside the attention, one add.
PARALLEL_NORM_NOTE = "Both sublayers read this one LayerNorm's output, the same tensor."
PARALLEL = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": PARALLEL_NORM_NOTE,
            "contribution": "attention_output",
            "interior": ATTENTION_INTERIOR,
            "detail": "{num_heads} heads, {num_kv_heads} kv; q/k norms",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": PARALLEL_NORM_NOTE,
            "contribution": "mlp_output",
            "detail": MLP_DETAIL,
        },
    ],
}

#: What the visualization draws, by the checkpoint's config: parallel where use_parallel_residual is set
#: (StableLM-2-12B, the pinned checkpoint), sequential otherwise (the configs that leave it out take false).
BLOCK = [
    (lambda config: getattr(config, "use_parallel_residual", False), PARALLEL),
    (lambda config: True, SEQUENTIAL),
]

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## On 12B the MLP does not read the attention

```
n   = input_layernorm(x)            # stablelm-2-12b (use_parallel_residual)
out = x + self_attn(n) + mlp(n)
```

On `stablelm-2-12b` both sublayers read the same normed input, so ablating or patching a block's
attention leaves that block's MLP unchanged, and the block has no `layer_mid`. The 1.6B and 3B
blocks are sequential, so check `model.config.use_parallel_residual` before reusing a recipe across
sizes.
"""
