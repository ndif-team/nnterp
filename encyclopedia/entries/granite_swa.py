"""Granite SWA: Granite's scaled block with sliding-window attention and a per-head sink that scales the head outputs."""

MODEL_TYPE = "granite_swa"
TITLE = "Granite Swash"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-swash-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GraniteSWAForCausalLM"
CHECKPOINTS = ["ibm-granite/granite-swash-2b"]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 208}
QUIRKS = ["scaled-residual-adds", "sliding-window", "sink-scaled-heads", "embedding-multiplier", "scaled-logits"]

#: Numbers in the notes come from granite-swash-2b's weights in bfloat16 on an RTX A6000; the snippets also ran
#: on the pinned tiny checkpoint, and on a copy of it with Granite's multipliers away from 1.0.
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
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
    "identity_note": "The terms are the modules' outputs times residual_multiplier, so the plain sum is exact.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The model multiplies it by embedding_multiplier before block 0, so layers[0].input is "
             "token_embeddings · embedding_multiplier.",
    "logits": "logits = lm_head.output / logits_scaling, and project_on_vocab divides too.",
}

NOTES = """
## The contributions are scaled copies of the modules' outputs

```
x   = embed_tokens(ids) * embedding_multiplier
h   = x + self_attn(input_layernorm(x)) * residual_multiplier
out = h + mlp(post_attention_layernorm(h)) * residual_multiplier
```

`attention_output` and `mlp_output` are the modules' outputs times `residual_multiplier`, the tensors
the block adds; decompose, attribute and steer through them.

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
