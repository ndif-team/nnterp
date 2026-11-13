"""GraniteMoE SWA: Granite Swash's attention in GraniteMoE's scaled block, with a shared expert beside the mixture."""

MODEL_TYPE = "granitemoe_swa"
TITLE = "Granite Swash MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-swash-3b-a600m"
#: The tiny checkpoint the test suite builds the page from (it has no shared expert: shared_intermediate_size 0).
PINNED = "hf-tiny-v2/tiny-random-GraniteMoeSWAForCausalLM"
CHECKPOINTS = ["ibm-granite/granite-swash-3b-a600m"]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 214}
QUIRKS = ["scaled-residual-adds", "sliding-window", "sink-scaled-heads", "mixture-of-experts", "embedding-multiplier", "scaled-logits"]

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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}, sink",
            "variants": {
                "sliding_attention": "window of {sliding_window} tokens, sink",
                "full_attention": "full causal attention, sink",
            },
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
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
    "identity": "layers[i].input + self_attn.attention_output + mlp.mlp_output + shared_mlp.output * residual_multiplier "
                "== layer_output",
    "identity_note": "The shared expert is in neither attention_output nor mlp_output, so its scaled output is the "
                     "third term.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The model multiplies it by embedding_multiplier before block 0, so layers[0].input is "
             "token_embeddings · embedding_multiplier.",
    "logits": "logits = lm_head.output / logits_scaling, and project_on_vocab divides too.",
}

NOTES = """
## `mlp_output` leaves out the shared expert

```
h   = x + self_attn(input_layernorm(x)) * residual_multiplier
n   = post_attention_layernorm(h)
out = h + (mlp(n) + shared_mlp(n)) * residual_multiplier
```

`attention_output` and `mlp_output` are the modules' outputs times `residual_multiplier`, and
`expert_outputs` and `routed_output` are read inside the mixture, before the multiplier. The shared expert's
term, `layers[i].shared_mlp.output` times `residual_multiplier`, is about as large as `mlp_output`
and in no standard value (`shared_expert_output` is unavailable), so decompose, attribute or ablate
the feed-forward with both. In bfloat16 a write to either contribution moves the other positions by
rounding; load in float32 for small edits.

## The pattern overstates what a head writes

After the softmax, each head's output is multiplied by the share a learned sink leaves to the real
keys, `sigmoid(logsumexp(attention_scores) - sink)`, and that share is small on most blocks. Weight
`attention_probabilities` by it to read how much a head moves from each token, and edit
`attention_head_outputs`, which is read after the scale, to set what a head writes. An edit to
`attention_probabilities` leaves the share as it was.

## The logits are divided by `logits_scaling`, and the embeddings multiplied

`model.logits` is `lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a softmax
over `lm_head.output` is sharper than the model's own distribution. `layers[0].input` is
`token_embeddings` times `embedding_multiplier`, and embeddings passed as `inputs_embeds` are
multiplied too, so to set what block 0 reads, edit `layers[0].input`.
"""
