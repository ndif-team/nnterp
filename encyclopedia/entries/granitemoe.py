"""GraniteMoE: Granite's four multipliers around a block whose MLP is a mixture of experts."""

MODEL_TYPE = "granitemoe"
TITLE = "GraniteMoE"
SUBTITLE = (
    "Granite's block with a mixture of experts for its MLP: the embeddings are multiplied before block 0, "
    "each sublayer's output is multiplied by residual_multiplier before it is added, and the logits are "
    "divided by logits_scaling, so mlp_output is the mixture's output times 0.22."
)

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

#: Numbers in the notes come from the reference's weights (float32 on an RTX A6000) unless they name
#: granite-3.1-3b-a800m-base; the snippets ran on both and on the pinned tiny checkpoint (block 1 for 5 and 20).
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
`residual_multiplier`.

## The first token's expert writes a vector of norm in the thousands

The tokenizer adds no beginning-of-sequence token, and the prompt's first token, whatever it is, is
routed mostly to one expert that writes a vector hundreds of times the norm of the stream elsewhere
(block 5's expert 4 on granite-3.0-1b-a400m, block 3's expert 11 on granite-3.1-3b-a800m). Leave
position 0 out of expert-usage counts and ablation sweeps. On granite-3.0-1b-a400m, ablating that
expert everywhere lowers a factual answer's log-probability by 9.4 nats by removing the first token's
vector; with position 0 left out, no expert moves it by more than 1.5.

## The logits are divided by `logits_scaling`, and the embeddings multiplied

`model.logits` is `lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a softmax
over `lm_head.output` runs at a sixth of the model's temperature. `layers[0].input` is
`token_embeddings` times `embedding_multiplier`, and embeddings passed as `inputs_embeds` are
multiplied too, so to set what block 0 reads, edit `layers[0].input`.

## In bfloat16 an edit at one position moves the others

A write to `mlp_output` or `attention_output` reaches the model divided by `residual_multiplier` over
the whole tensor, which rounds every position in bfloat16. An edit that changes the token's routing in
a later mixture also changes how the other tokens' expert outputs round, which can move them more
than the edit moves the edited token. Load in float32 for small edits.
"""
