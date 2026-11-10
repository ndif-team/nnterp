"""Qwen3.5 and Qwen3.6, dense: the text model of the multimodal checkpoints, a gated DeltaNet hybrid."""

MODEL_TYPE = "qwen3_5_text"
TITLE = "Qwen3.5 / Qwen3.6"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3.5-0.8B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen3.5-tiny-random"
CHECKPOINTS = [
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.6-27B",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
WRAPPERS = {
    "qwen3_5": {
        "title": "Qwen3.5",
        "pinned": "yujiepan/qwen3.5-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm on each patch, then an MLP (linear_fc1, GELU, "
                     "linear_fc2) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
        "notes": """
## The projector reads the vision encoder's last block

`model.projector` is the encoder's own `merger`, and with no DeepStack the image reaches the text
model only through `vision.image_features`.
""",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 310}
QUIRKS = ["hybrid", "gated-query", "qk-norm", "partial-rotary", "gain-norm", "multimodal-rotary"]

#: Real-value numbers below come from runs on Qwen3.5-0.8B (float32) and Qwen3.5-9B (bfloat16), with
#: transformers' pure-torch DeltaNet kernels (flash-linear-attention and causal-conv1d not installed).

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``);
#: every block has the dense MLP.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Linear attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "decays", "betas",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "gated DeltaNet, {linear_num_value_heads} heads × {linear_value_head_dim}",
        },
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
            "detail": "{num_heads} heads, {num_kv_heads} kv, output-gated",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The attention's output gate is mostly closed

```
q, gate = split per head(q_proj(x))
heads = attention(rope(q_norm(q)), rope(k_norm(k_proj(x))), v_proj(x))   # attention_head_outputs
out   = o_proj(heads * sigmoid(gate))           # o_proj's input is attention_premix, after the gate
```

On 0.8B and 9B the gate's sigmoid is small for most entries, so `attention_head_outputs` reaches
the stream scaled far down, head by head. Attribute and patch heads at `attention_premix`,
`o_proj`'s input, which is after the gate.

## The DeltaNet's queries and keys are copied per value head on 4B and larger

On 4B, 9B and 27B the 16 key heads' queries and keys are repeated up to the value heads before the
kernel, so `attention_queries` and `attention_keys` are `num_v_heads` wide, each key head copied into
consecutive heads. An edit to one copy reaches only its own value head.

## Prepending `<|endoftext|>` changes the predictions

The tokenizer adds no beginning-of-sequence token, and position 0 draws much of the attention without
growing into an outlier in the stream. Prepending `<|endoftext|>` as a sink changes what the model
predicts, so leave the prompt as it is.

## Whether the template opens a reasoning trace depends on the size

With `add_generation_prompt=True` the 0.8B and 2B templates close an empty thinking block unless
`enable_thinking=True` is passed, and the 4B, 9B, 27B and Qwen3.6-27B templates end at `<think>\\n`
unless `enable_thinking=False` is passed. After `<think>\\n` the next token starts a reasoning trace,
so pass the flag either way before taking a next-token metric.
"""
