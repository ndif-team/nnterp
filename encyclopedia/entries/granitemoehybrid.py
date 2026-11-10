"""GraniteMoE-Hybrid: Granite 4.0, Mamba-2 and attention blocks in Granite's scaled block, with a shared MLP and routed experts."""

MODEL_TYPE = "granitemoehybrid"
TITLE = "Granite 4.0"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-4.0-h-tiny-base"
#: The tiny checkpoint the test suite builds the page from (the suite routes it to the pure-torch kernels).
PINNED = "hf-tiny-v2/tiny-random-GraniteMoeHybridForCausalLM"
CHECKPOINTS = [
    "ibm-granite/granite-4.0-h-350m-base", "ibm-granite/granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-1b-base", "ibm-granite/granite-4.0-h-1b",
    "ibm-granite/granite-4.0-h-micro-base", "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-h-tiny-base", "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-small-base", "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-tiny-base-preview", "ibm-granite/granite-4.0-tiny-preview",
    "ibm-research/granite-4.0-h-3b-ar",
    # attention on every block: their blocks hold no linear_attn, and the page draws the attention block alone
    "ibm-granite/granite-4.0-350m-base", "ibm-granite/granite-4.0-350m",
    "ibm-granite/granite-4.0-1b-base", "ibm-granite/granite-4.0-1b",
    "ibm-granite/granite-4.0-micro-base", "ibm-granite/granite-4.0-micro",
]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 201}
QUIRKS = [
    "scaled-residual-adds",
    # the H models; the non-H 350M, 1B and Micro are attention with rotary on every block
    {"slug": "hybrid", "when": lambda config: "linear_attention" in config.layer_types},
    {"slug": "mamba2", "when": lambda config: "linear_attention" in config.layer_types},
    {"slug": "nope-blocks", "when": lambda config: config.position_embedding_type != "rope"},
    {"slug": "mixture-of-experts", "when": lambda config: config.num_local_experts > 0},  # H-Tiny, H-Small, the Tiny previews
    "embedding-multiplier", "scaled-logits",
]

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
            "detail": "SSD scan, gated RMSNorm after",
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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
        },
        {
            # The dense layout (H-350M, H-1B, H-Micro and the non-H models): the shared MLP is the whole feed-forward.
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the mixer's add.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
        },
        {
            # The mixture layout (H-Tiny, H-Small, the previews): mlp is the shared MLP and hosts the block's mixture.
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The input norm of the mixture and the shared MLP, named for its place after the mixer's add.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, + shared",
        },
    ],
    "identity_note": "The terms are the mixer's output and the feed-forward's sum, each times residual_multiplier, "
                     "so the plain sum is exact.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The model multiplies it by embedding_multiplier before block 0, so layers[0].input is "
             "token_embeddings · embedding_multiplier.",
    "logits": "logits = lm_head.output / logits_scaling, and project_on_vocab divides too.",
}

NOTES = """
## The contributions are scaled, and `mlp` is the shared MLP

```
h   = x + mixer(input_layernorm(x)) * residual_multiplier       # linear_attn or self_attn
n   = post_attention_layernorm(h)
out = h + (block_sparse_moe(n) + mlp(n)) * residual_multiplier
```

`attention_output`, on either mixer, and `mlp_output` are the tensors the block adds, outputs times
`residual_multiplier`. `mlp` is the shared MLP every block has, so `mlp.output` is its output
alone; on the mixture checkpoints (H-Tiny, H-Small, the Tiny previews) `mlp_output` also holds the
routed experts' term, `layers[i].block_sparse_moe.output`, and a write to it replaces both. In
bfloat16 a write to either contribution moves the other positions by rounding; load in float32 for
small edits.

## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes most of the
attention and its stream has a norm many times the other positions'. Leave position 0 out of activation
statistics and steering-vector norms.

## The logits are divided by `logits_scaling`, and the embeddings multiplied

`model.logits` is `lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a softmax
over `lm_head.output` is sharper than the model's own distribution. `layers[0].input` is
`token_embeddings` times `embedding_multiplier`, and embeddings passed as `inputs_embeds` are
multiplied too, so to set what block 0 reads, edit `layers[0].input`.
"""
