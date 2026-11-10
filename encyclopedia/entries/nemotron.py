"""Nemotron: the Nemotron-4 block as Minitron and Nemotron-Mini ship it (NemotronForCausalLM)."""

MODEL_TYPE = "nemotron"
TITLE = "Nemotron-4 / Minitron"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "nvidia/Nemotron-Mini-4B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-NemotronForCausalLM"
CHECKPOINTS = [
    "nvidia/Minitron-4B-Base", "nvidia/Nemotron-Mini-4B-Instruct",
    "nvidia/Minitron-8B-Base",
    "nvidia/Nemotron-4-Mini-Hindi-4B-Base", "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct",
]

#: Set by hues.py (lineage: Nemotron).
PALETTE = {"hue": 223}
QUIRKS = ["layernorm", "gain-norm", "partial-rotary", "squared-relu"]

#: What the visualization draws: Llama's sequential block, a LayerNorm before each sublayer
#: and none after; the MLP is two projections around a squared ReLU.
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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads; "
                      "rotary on {partial_rotary_factor} of each {head_dim}-wide head",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## Position 0 and one or two other tokens are attention sinks

The tokenizer adds no beginning-of-sequence token, so the prompt's first token, and on most prompts one
or two later tokens, take most of the attention and have norms many times the other positions'. Leave
them out of activation statistics and steering-vector norms.
"""
