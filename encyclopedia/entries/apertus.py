"""Apertus: Llama's block under other norm names, per-head query/key norms and a gateless MLP with a learned xIELU."""

MODEL_TYPE = "apertus"
TITLE = "Apertus"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "swiss-ai/Apertus-8B-2509"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-ApertusForCausalLM"
CHECKPOINTS = [
    "swiss-ai/Apertus-8B-2509", "swiss-ai/Apertus-8B-Instruct-2509",
    "swiss-ai/Apertus-70B-2509", "swiss-ai/Apertus-70B-Instruct-2509",
    "swiss-ai/Apertus-v1.1-0.5B", "swiss-ai/Apertus-v1.1-0.5B-Instruct",
    "swiss-ai/Apertus-v1.1-1.5B", "swiss-ai/Apertus-v1.1-1.5B-Instruct",
    "swiss-ai/Apertus-v1.1-4B", "swiss-ai/Apertus-v1.1-4B-Instruct",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 238}
QUIRKS = ["qk-norm", "learned-activation"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## A negative pre-activation still writes

The MLP is `down_proj(xielu(up_proj(z)))`, with no gate. On the negative side xIELU dips below zero
and then grows again, like `(alpha_n − beta)·|x|` as `x` falls, so a large negative pre-activation
writes its `down_proj` column as a large positive one does, and the activations are not sparse.
Read `mlp.act_fn.output`, not the sign of `mlp.act_fn.input`, to tell whether a neuron fires.
"""
