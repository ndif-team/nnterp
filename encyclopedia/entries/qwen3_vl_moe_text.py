"""Qwen3-VL-MoE: the text model of the Qwen3-VL mixture-of-experts checkpoints, with DeepStack after its first blocks."""

MODEL_TYPE = "qwen3_vl_moe_text"
TITLE = "Qwen3-VL-MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-VL-30B-A3B-Instruct"
#: The tiny checkpoint the test suite builds the page from: the wrapper itself, as every checkpoint is.
PINNED = "yujiepan/qwen3-vl-moe-tiny-random"
#: Every Qwen3-VL-MoE checkpoint is the vision-language wrapper (model_type qwen3_vl_moe); there is no text-only release.
CHECKPOINTS = [
    "Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen/Qwen3-VL-235B-A22B-Thinking",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
#: Shapes and identities were checked on the pinned tiny wrapper (and on a copy with five text blocks, for the blocks
#: past the taps); the numbers are from docs/usage/vision.md, on the dense Qwen3-VL-4B-Instruct.
WRAPPERS = {
    "qwen3_vl_moe": {
        "title": "Qwen3-VL-MoE",
        "pinned": "yujiepan/qwen3-vl-moe-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm on each patch, then an MLP (linear_fc1, GELU, "
                     "linear_fc2) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
        "quirks": ["deepstack"],
        "notes": """
## The DeepStack mergers read three earlier vision blocks

Beside `model.projector`, which reads the last block, each `vision.deepstack_merger_list[k]` reads a
block that `vision_config.deepstack_visual_indexes` names, and its image tokens join the text stream
as `layers[k].deepstack_output`.
""",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 304}
QUIRKS = ["qk-norm", "multimodal-rotary", "mixture-of-experts"]

#: What the visualization draws. The DeepStack add after blocks 0-2 is the text model's, outside the block: `between`
#: draws it after `layer_output` on those blocks. Every block of every checkpoint, and of the pinned tiny, is a mixture.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv, M-RoPE",
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
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
    "between": [
        {
            "value": "deepstack_output",
            "mask": "vision.image_token_mask",
            "blocks": [0, 1, 2],
            "label": "DeepStack",
            "note": "Its rows are `vision.deepstack_merger_list[k].output`; a text prompt has none.",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since the DeepStack add is drawn between the blocks.
STRIP: dict[str, str] = {}

NOTES = """
## The image enters the stream again after blocks 0 to 2

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
x'  = out + deepstack_output[k] at the image tokens    # after blocks 0, 1 and 2, outside the block
```

On an image prompt `layer_output` of blocks 0 to 2 is not what the next block reads, so patching block
`k`'s `layer_output` into block `k + 1`'s input drops the DeepStack add. Zeroing `vision.image_features` alone leaves the image in the stream;
zero `layers[k].deepstack_output` on blocks 0 to 2 as well.

## The Thinking template ends where a reasoning trace starts

With `add_generation_prompt=True` the Thinking templates add `<think>\\n` after
`<|im_start|>assistant\\n`, so a next-token metric there measures the start of a reasoning trace. The
Instruct templates end at `<|im_start|>assistant\\n`.
"""
