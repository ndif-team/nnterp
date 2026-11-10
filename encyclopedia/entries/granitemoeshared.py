"""GraniteMoE-Shared: GraniteMoE's scaled block with a dense shared expert beside the mixture."""

MODEL_TYPE = "granitemoeshared"
TITLE = "GraniteMoE-Shared"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-research/moe-7b-1b-active-shared-experts"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GraniteMoeSharedForCausalLM"
CHECKPOINTS = ["ibm-research/moe-7b-1b-active-shared-experts"]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 217}
QUIRKS = ["scaled-residual-adds", "mixture-of-experts", "embedding-multiplier", "scaled-logits"]

#: No public checkpoint of this family is small enough to run here (the one release is 7B), so the notes' shapes,
#: identities and snippets come from the pinned tiny checkpoint and a copy of it with Granite's multipliers away
#: from 1.0, and their numbers from the reference's config.
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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The input norm of the mixture and the shared expert, named for its place after the "
                             "attention's add.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, + shared",
        },
    ],
    "identity_note": "The terms are the attention's output and the two experts' sum, each times residual_multiplier, "
                     "so the plain sum is exact.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The model multiplies it by embedding_multiplier before block 0, so layers[0].input is "
             "token_embeddings · embedding_multiplier.",
    "logits": "logits = lm_head.output / logits_scaling, and project_on_vocab divides too.",
}

NOTES = """
## The contributions are scaled, and `mlp_output` holds both experts

```
h   = x + self_attn(input_layernorm(x)) * residual_multiplier
n   = post_attention_layernorm(h)
out = h + (mlp(n) + shared_mlp(n)) * residual_multiplier
```

`attention_output` is the attention's output times `residual_multiplier`, and `mlp_output` the
mixture's plus the shared expert's, times it. `expert_outputs`, `routed_output` and
`shared_expert_output` are read before the multiplier, and the shared expert runs on every token
with no weight; to ablate it alone, edit `layers[i].shared_mlp.output`. In bfloat16 a write to either
contribution moves the other positions by rounding; load in float32 for small edits.

## The logits are divided by `logits_scaling`, and the embeddings multiplied

`model.logits` is `lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a softmax
over `lm_head.output` is sharper than the model's own distribution. `layers[0].input` is
`token_embeddings` times `embedding_multiplier`, and embeddings passed as `inputs_embeds` are
multiplied too, so to set what block 0 reads, edit `layers[0].input`.
"""
