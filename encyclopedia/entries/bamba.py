"""Bamba: IBM's Mamba-2 hybrid (BambaForCausalLM), Llama's pre-norm block with a Mamba-2 mixer on most blocks."""

MODEL_TYPE = "bamba"
TITLE = "Bamba"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-ai-platform/Bamba-9B-v2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-BambaForCausalLM"
#: Bamba-9B on the Hub redirects to Bamba-9B-v1. The -fp8 copies are left out: their compressed-tensors
#: config does not build without that package.
CHECKPOINTS = [
    "ibm-ai-platform/Bamba-9B-v2", "ibm-ai-platform/Bamba-9B-v1",
    "ibm-ai-platform/Bamba-9B-1.8T", "ibm-ai-platform/Bamba-9B-2T",
]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 198}
QUIRKS = ["hybrid", "mamba2", "partial-rotary"]

#: Every checkpoint is 9B, so no real weights were run: the shapes and identities below are from the pinned
#: tiny checkpoint (routed, float32, CPU), the sizes from the checkpoints' configs.

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``);
#: every block has the same MLP.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD scan, a state per head",
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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, half rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, native name pre_ff_layernorm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## A block holds one mixer, and three blocks attend

```
h   = x + mixer(input_layernorm(x))      # linear_attn (Mamba-2) or self_attn
out = h + mlp(post_attention_layernorm(h))
```

Each block has `linear_attn` or `self_attn`, never both, as `config.layer_types` lists. Three of the
32 blocks attend, so attention patterns and head edits exist on those blocks alone. Choose blocks
from `layer_types` outside the trace.
"""
