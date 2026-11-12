"""XGLM: Meta's multilingual XGLM-564M to XGLM-7.5B, OPT's block with a scaled embedding and sinusoidal positions."""

MODEL_TYPE = "xglm"
TITLE = "XGLM"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "facebook/xglm-564M"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-XGLMForCausalLM"
CHECKPOINTS = [
    "facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B",
]

#: Set by hues.py (lineage: OPT).
PALETTE = {"hue": 81}
QUIRKS = ["no-mlp", "scaled-embeddings", "position-embeddings", "layernorm", "qkv-bias"]

#: What the visualization draws: the attention and its pre-norm, then the MLP path, which has no module:
#: final_layer_norm, fc1, activation_fn and fc2 sit on the block, so the sublayer is hosted on the native fc2.
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
            "detail": "{num_heads} heads × {head_dim}, own bmm",
        },
        {
            "host": "fc2",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "final_layer_norm",
            "reads": "fc1",
            "pre_norm_note": "The MLP's input norm; the model's final norm is model.norm (native layer_norm).",
            "contribution": "fc2.output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
            "host_note": "No mlp module: the MLP path is fc1, activation_fn and fc2 on the block.",
        },
    ],
    "identity": "layers[i].input + self_attn.attention_output + fc2.output == layer_output",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by sqrt(d_model); the sinusoidal model.embed_positions is added after it, so layers[0].input "
             "is token_embeddings + embed_positions.output.",
}

NOTES = """
## The MLP's contribution is `layers[i].fc2.output`

There is no `mlp` module, so `mlp_output` and the other `mlp` values do not exist. The MLP path adds
`layers[i].fc2.output` to the stream, and its neurons are `layers[i].activation_fn.output`.

## An edit to the keys or values reaches every invoke

`attention_keys` and `attention_values` are read after the cache update, over the whole batch, so
under several invokes each invoke reads every invoke's keys and values and an edit in one reaches
them all. Edit keys and values in a trace with one invoke.
"""
