"""Jamba: AI21's hybrid (JambaForCausalLM), Mamba-1 blocks and attention blocks, a mixture of experts on every other block."""

MODEL_TYPE = "jamba"
TITLE = "Jamba / Jamba 1.5–1.7 / Jamba2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ai21labs/Jamba-v0.1"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-JambaForCausalLM"
#: Jamba 1.5, 1.6 and 1.7 are gated on the Hub: the build lists them greyed unless the token has access.
CHECKPOINTS = [
    "ai21labs/Jamba-v0.1",
    "ai21labs/AI21-Jamba-Mini-1.5", "ai21labs/AI21-Jamba-Large-1.5",
    "ai21labs/AI21-Jamba-Mini-1.6", "ai21labs/AI21-Jamba-Large-1.6",
    "ai21labs/AI21-Jamba-Mini-1.7", "ai21labs/AI21-Jamba-Large-1.7",
    "ai21labs/AI21-Jamba-Reasoning-3B",
    "ai21labs/AI21-Jamba2-3B", "ai21labs/AI21-Jamba2-Mini",
]

#: Set by hues.py (lineage: Mamba).
PALETTE = {"hue": 8}
QUIRKS = ["hybrid", "mamba1", "nope-blocks", "mixture-of-experts", "unnormalized-routing"]

#: No real weights were run: shapes and identities are from the pinned tiny checkpoint (routed, float32,
#: CPU), sizes and periods from the checkpoints' configs, routing from transformers' JambaSparseMoeBlock.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "state_input", "attention_values", "betas", "decays", "attention_keys", "attention_queries",
                "states", "attention_head_outputs", "state_output",
            ],
            "detail": "selective scan, normed dt, B, C",
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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, no rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm, the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm, the MLP's input norm.",
            "contribution": "mlp_output",
            "interior": ["router_logits", "expert_weights", "expert_indices", "expert_outputs", "routed_output"],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Each block has one mixer, Mamba or attention

```
h   = x + mixer(input_layernorm(x))            # linear_attn (Mamba) or self_attn
out = h + mlp(post_attention_layernorm(h))     # a SwiGLU MLP or a mixture
```

`config.layer_types` lists each block's mixer. A Mamba block has no `self_attn` values, so pick
blocks from that list outside the trace. On Jamba-v0.1 and Jamba2-Mini the mixture sits on the odd
blocks and the attention on four even ones, so an attention block always has the dense MLP. The 3B
checkpoints have no mixture.
"""
