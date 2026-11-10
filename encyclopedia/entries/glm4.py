"""GLM-4-0414 and GLM-Z1: ``Glm4ForCausalLM``, GLM's block with a norm after each sublayer."""

MODEL_TYPE = "glm4"
TITLE = "GLM-4-0414 / GLM-Z1"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-4-9B-0414"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Glm4ForCausalLM"
CHECKPOINTS = [
    "zai-org/GLM-4-9B-0414", "zai-org/GLM-Z1-9B-0414",
    "zai-org/GLM-4-32B-Base-0414", "zai-org/GLM-4-32B-0414",
    "zai-org/GLM-Z1-32B-0414", "zai-org/GLM-Z1-Rumination-32B-0414",
]

#: Set by hues.py (lineage: GLM).
PALETTE = {"hue": 119}
QUIRKS = [
    "sandwich-norms",
    {"slug": "qkv-bias", "when": lambda config: config.attention_bias},  # the 9B models, not the 32B ones
    "partial-rotary",
    "interleaved-rotary",
]

#: Every claim in the notes was checked on the pinned tiny checkpoint (shapes, identities, the rotary
#: recomputed exactly) or read off the listed checkpoints' configs and tokenizers; no real weights were run.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_self_attn_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, {num_kv_heads} kv, × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "post_norm": "post_mlp_layernorm",
            "contribution": "mlp_output",
            "detail": "fused gate_up_proj: {hidden_size} → 2 × {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "The MLP's input norm, as on Llama; on Gemma 2 the same name follows the attention.",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_self_attn_layernorm(self_attn(input_layernorm(x)))
out = h + post_mlp_layernorm(mlp(post_attention_layernorm(h)))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. RMSNorm is scale-invariant, so halving `mlp.output` does almost nothing to the stream.
Scale, steer and decompose through `attention_output` and `mlp_output`.

## The tokenizer prepends nothing

A raw prompt starts at position 0 with its first token, without the `[gMASK]<sop>` the chat template
writes, so the sink that trained models form at position 0 falls on that token. On GLM-Z1-9B-0414
the generation prompt ends in `<think>`, so the next token starts a reasoning trace, not the answer.
"""
