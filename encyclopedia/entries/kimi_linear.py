"""Kimi-Linear: a hybrid whose blocks come in three shapes (a KDA mixer or latent attention, a dense MLP or a mixture)."""

MODEL_TYPE = "kimi_linear"
TITLE = "Kimi-Linear"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
#: The tiny checkpoint the test suite builds the page from (the suite types its last block in a copy of the config).
PINNED = "yujiepan/kimi-linear-tiny-random"
CHECKPOINTS = ["moonshotai/Kimi-Linear-48B-A3B-Base", "moonshotai/Kimi-Linear-48B-A3B-Instruct"]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 340}
QUIRKS = ["hybrid", "latent-attention", "nope-blocks", "mixture-of-experts", "dense-first-blocks"]

#: Every number on this page comes from the reference's config or a run on the pinned tiny checkpoint;
#: none has been checked on real weights, which are not cached (48B parameters).


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device. The reference's tokenizer loads through its remote
    code and is not cached; the page reads no tokens, so the Kimi K2.5 test checkpoint's stands in."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    tokenizer = AutoTokenizer.from_pretrained("hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration")
    return StandardizedTransformer(checkpoint, tokenizer=tokenizer, **kwargs)


#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``)
#: and the MLP its ``mlp`` is (the dense one on block 0, the mixture after).
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
            "detail": "KDA, {linear_num_heads} heads × {linear_head_dim}",
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
            "detail": "latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}, no rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Each block has KDA or latent attention

```
h   = x + mixer(input_layernorm(x))          # linear_attn (KDA) or self_attn (latent attention)
out = h + mlp(post_attention_layernorm(h))   # a dense MLP on block 0, a mixture after
```

transformers names both mixers `self_attn`. nnterp names the KDA mixer `linear_attn` and sets
`self_attn` to `None` on its blocks, so pick blocks from `config.layer_types` outside the trace.

## The expert weights sum to `routed_scaling_factor`

The router picks experts by their sigmoid plus a selection bias, so the top `router_logits` are not
always the chosen experts. `expert_weights` are the chosen sigmoids without the bias, renormalized
and multiplied by `routed_scaling_factor` (2.446), so they sum to it at every token. Every token also
runs through a shared expert, and `mlp_output` is `routed_output + shared_expert_output`.
"""
