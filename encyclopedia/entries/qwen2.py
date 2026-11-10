"""Qwen2 and Qwen2.5, and every other line that loads as ``Qwen2ForCausalLM``."""

MODEL_TYPE = "qwen2"
TITLE = "Qwen2 / Qwen2.5"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen2.5-0.5B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen2-tiny-random"
CHECKPOINTS = [
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-7B",
    "Qwen/Qwen2-0.5B", "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B", "Qwen/Qwen2-72B",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # Vision-language wrappers around a Qwen2 text model, each a key of WRAPPERS by its config's model_type.
    "llava-hf/llava-interleave-qwen-0.5b-hf",  # model_type llava
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf",
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "llava": {
        "title": "llava-interleave",
        # The family suite's TestLlavaInterleaveVision runs on this real checkpoint (where CUDA is); the meta build is cheap.
        "pinned": "llava-hf/llava-interleave-qwen-0.5b-hf",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2) over the last block's patches, before post_layernorm",
        "projector_input": "the last block's `layer_output`, before `vision.norm`",
        "notes": """
## The projector reads the last block, before the final norm

A write to `vision.tower_output` does not reach the text model; write to `vision.layers[-1].layer_output`.
""",
    },
    "llava_onevision": {
        "title": "LLaVA-OneVision",
        "pinned": "hf-tiny-v2/tiny-random-LlavaOnevisionForConditionalGeneration",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2) over each crop's last-block patches; the wrapper unpads its output and adds newline tokens",
        "projector_input": "each crop's last-block `layer_output`, before `vision.norm`",
        "quirks": ["tiled-images", "unpadded-features"],
        "notes": """
## The projector reads each crop's last block, before the final norm

A write to `vision.tower_output` does not reach the text model; write to `vision.layers[-1].layer_output`.
""",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 282}
QUIRKS = ["qkv-bias"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k, v biased",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no biases",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The query bias outweighs the input

`q_proj`, `k_proj` and `v_proj` add a bias before the rotary, and on 0.5B the query bias is larger
than the input's part of the query on every block. A query-key analysis from the weights alone misses
most of each query, and zeroing the attention's input leaves it writing `o_proj` of the value bias at
every position. To remove the attention, ablate `attention_output`.

## The prompt's first token becomes the attention sink

The tokenizer prepends nothing, so through most blocks the prompt's first token draws much of the
attention, carries a norm far above the other positions' and holds nothing of the word. Leave position 0
out of means, probes and steering-vector norms, or prepend `<|endoftext|>`, which takes the sink and
leaves the first word its own position.
"""
