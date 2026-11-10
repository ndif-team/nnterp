"""Granite: Llama's block with four config scalars around it; the first of the Granite lineage."""

MODEL_TYPE = "granite"
TITLE = "Granite"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-3.0-2b-base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GraniteForCausalLM"
CHECKPOINTS = [
    "ibm-granite/granite-3.0-2b-base", "ibm-granite/granite-3.0-2b-instruct",
    "ibm-granite/granite-3.0-8b-base", "ibm-granite/granite-3.0-8b-instruct",
    "ibm-granite/granite-3.1-2b-base", "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-3.1-8b-base", "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-3.2-2b-instruct", "ibm-granite/granite-3.2-8b-instruct",
    "ibm-granite/granite-3.3-2b-base", "ibm-granite/granite-3.3-2b-instruct",
    "ibm-granite/granite-3.3-8b-base", "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-3b-base", "ibm-granite/granite-4.1-3b",
    "ibm-granite/granite-4.1-8b-base", "ibm-granite/granite-4.1-8b",
    "ibm-granite/granite-4.1-30b-base", "ibm-granite/granite-4.1-30b",
    "ibm-granite/granite-4.2-3b", "ibm-granite/granite-4.2-8b", "ibm-granite/granite-4.2-30b",
]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 205}
QUIRKS = ["scaled-residual-adds", "embedding-multiplier", "scaled-logits"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` is formatted with the sizes and the config keys.
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
the block adds; decompose, attribute and steer through them. In bfloat16 a write to either
contribution moves the other positions by rounding; load in float32 for small edits. On Granite 4.2 the embedding, residual and logit
multipliers are 1.

## The logits are divided by `logits_scaling`, and the embeddings multiplied

`model.logits` is `lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a softmax
over `lm_head.output` is sharper than the model's own distribution. `layers[0].input` is
`token_embeddings` times `embedding_multiplier`, and embeddings passed as `inputs_embeds` are
multiplied too, so to set what block 0 reads, edit `layers[0].input`.

## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes most of the
attention and its stream has a norm many times the other positions'. Leave position 0 out of activation
statistics and steering-vector norms.
"""
