"""GPT-Neo: EleutherAI's GPT-Neo and the TinyStories models that load as GPTNeoForCausalLM."""

MODEL_TYPE = "gpt_neo"
TITLE = "GPT-Neo / TinyStories"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "EleutherAI/gpt-neo-125m"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
CHECKPOINTS = [
    "EleutherAI/gpt-neo-125m", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
    "roneneldan/TinyStories-1M", "roneneldan/TinyStories-3M", "roneneldan/TinyStories-8M",
    "roneneldan/TinyStories-28M", "roneneldan/TinyStories-33M",
    "roneneldan/TinyStories-1Layer-21M", "roneneldan/TinyStories-2Layers-33M",
    "roneneldan/TinyStories-Instruct-33M",
]

#: Set by hues.py (lineage: GPT-J).
PALETTE = {"hue": 50}
QUIRKS = ["sliding-window", "position-embeddings", "layernorm"]

#: What the visualization draws: GPT-2's sequential block, ln_1 before the attention, ln_2 before the MLP.
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
            "detail": "{num_heads} heads × {head_dim}, unscaled",
            "variants": {
                "global": "{num_heads} heads × {head_dim}, unscaled",
                "local": "{num_heads} heads × {head_dim}, window {window_size}",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The learned position embedding transformer.wpe is added after it, so layers[0].input is "
             "token_embeddings + transformer.wpe.output.",
}

NOTES = """
## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes most of the global
blocks' attention and its stream has a norm many times the other positions'. Leave position 0 out of
activation statistics and steering-vector norms.
"""
