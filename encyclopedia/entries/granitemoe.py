"""GraniteMoE: Granite's four multipliers around a block whose MLP is a mixture of experts."""

MODEL_TYPE = "granitemoe"
TITLE = "GraniteMoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-3.0-1b-a400m-base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GraniteMoeForCausalLM"
CHECKPOINTS = [
    "ibm-granite/granite-3.0-1b-a400m-base", "ibm-granite/granite-3.0-1b-a400m-instruct",
    "ibm-granite/granite-3.0-3b-a800m-base", "ibm-granite/granite-3.0-3b-a800m-instruct",
    "ibm-granite/granite-3.1-1b-a400m-base", "ibm-granite/granite-3.1-1b-a400m-instruct",
    "ibm-granite/granite-3.1-3b-a800m-base", "ibm-granite/granite-3.1-3b-a800m-instruct",
    "ibm-granite/granite-guardian-3.2-3b-a800m", "ibm-research/PowerMoE-3b",
]

#: Checkpoints that ship their tokenizer as `vocab.json` and `merges.txt` only, with no `tokenizer.json`.
GPT2_FILES_ONLY = {"ibm-granite/granite-3.0-3b-a800m-instruct"}


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device. ``ibm-granite/granite-3.0-3b-a800m-instruct`` has no ``tokenizer.json``,
    and transformers' ``AutoTokenizer`` maps ``granitemoe`` to the generic ``TokenizersBackend``, which cannot
    build from ``vocab.json`` and ``merges.txt``; the GPT-2 tokenizer its ``tokenizer_config.json`` names
    can, so it is loaded by type and handed over."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    if checkpoint in GPT2_FILES_ONLY:
        kwargs["tokenizer"] = AutoTokenizer.from_pretrained(checkpoint, tokenizer_type="gpt2")
    return StandardizedTransformer(checkpoint, **kwargs)


#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 211}
QUIRKS = ["scaled-residual-adds", "mixture-of-experts", "embedding-multiplier", "scaled-logits"]

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
            "pre_norm_note": "The mixture's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
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
## The contributions are scaled, and the expert outputs are not

```
x   = embed_tokens(ids) * embedding_multiplier
h   = x + self_attn(input_layernorm(x)) * residual_multiplier
out = h + mlp(post_attention_layernorm(h)) * residual_multiplier
```

`attention_output` and `mlp_output` are the modules' outputs times `residual_multiplier`, the tensors
the block adds. `expert_outputs` and `routed_output` are read inside the mixture, before the
multiplier, so one expert's term in the stream is its `expert_outputs` slot times
`residual_multiplier`. In bfloat16 a write to either contribution moves the other positions by
rounding; load in float32 for small edits.

## The first token routes mostly to one expert

The tokenizer adds no beginning-of-sequence token, and the prompt's first token routes mostly to one
expert whose output is far larger than the stream at the other positions. Leave position 0 out of
expert-usage counts and ablation sweeps.

## The logits are divided by `logits_scaling`, and the embeddings multiplied

`model.logits` is `lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a softmax
over `lm_head.output` is sharper than the model's own distribution. `layers[0].input` is
`token_embeddings` times `embedding_multiplier`, and embeddings passed as `inputs_embeds` are
multiplied too, so to set what block 0 reads, edit `layers[0].input`.
"""
