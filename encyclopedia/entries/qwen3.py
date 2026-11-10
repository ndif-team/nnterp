"""Qwen3: the dense checkpoints, and every other line that loads as ``Qwen3ForCausalLM``."""

MODEL_TYPE = "qwen3"
TITLE = "Qwen3"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-0.6B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Qwen3ForCausalLM"
CHECKPOINTS = [
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-SafeRL",
    "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B", "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3Guard-Gen-0.6B", "Qwen/Qwen3Guard-Gen-4B", "Qwen/Qwen3Guard-Gen-8B",
    "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 295}
QUIRKS = ["qk-norm"]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k RMS-normed",
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

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes most of the
attention, its stream has a norm many times the other positions' and a probe or lens there reads the
sink, not the word. Leave position 0 out of activation statistics and steering-vector norms, or prepend
`<|endoftext|>`, which takes the sink and leaves the first word its own position.

## The chat template ends where a reasoning trace starts

On the 0.6B to 32B checkpoints, Base included, `add_generation_prompt=True` ends the prompt at
`<|im_start|>assistant\\n` and the next token is `<think>`, so a next-token metric there measures
the start of a reasoning trace. Pass `enable_thinking=False`, which appends an empty thinking block,
to put the answer next. Qwen3-4B-Thinking-2507's template ignores the flag and always opens the block.

## Scaling a head's slice of `q_proj.output` does nothing

`q_norm` and `k_norm` are RMSNorms over each head's dimensions, applied after the projections, so
they undo a scale on a head's slice of `q_proj.output` or `k_proj.output`. To sharpen or flatten a
head's pattern, edit `attention_queries` or `attention_keys`, which are read after the norms.
"""
