"""GPT-NeoX: Pythia, GPT-NeoX-20B, and the other checkpoints that load as GPTNeoXForCausalLM."""

MODEL_TYPE = "gpt_neox"
TITLE = "GPT-NeoX"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "EleutherAI/pythia-70m-deduped"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
CHECKPOINTS = [
    "EleutherAI/pythia-14m-deduped", "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m-deduped", "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b-deduped", "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b-deduped", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b-deduped", "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b-deduped", "EleutherAI/pythia-12b",
    "EleutherAI/gpt-neox-20b",
]

#: Set by hues.py (lineage: GPT-NeoX).
PALETTE = {"hue": 63}
QUIRKS = ["parallel-blocks", "partial-rotary", "layernorm"]

#: What the visualization draws: Pythia's parallel block, two pre-norms of the block input,
#: no post-norms; the two contributions join the stream in one add.
BLOCK = {
    "topology": "parallel",
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
            "detail": "{num_heads} heads × {head_dim}, fused qkv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Despite its name it norms the block's input, beside input_layernorm, with its own weight "
                             "and bias.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The MLP does not read the attention

```
out = x + self_attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))
```

Every Pythia checkpoint and GPT-NeoX-20B run this parallel block, where `post_attention_layernorm`
norms the block's input despite its name. Ablating or patching a block's attention leaves that
block's MLP unchanged, and the first MLP to read block `i`'s attention is block `i + 1`'s.

## `query_key_value` is laid out per head

The fused projection's output holds each head's query, key and value side by side, head after
head, so its first `hidden_size` columns are not the queries. View it as
`[batch, seq, num_heads, 3 * head_dim]` before splitting, or read `attention_queries`,
`attention_keys` and `attention_values`.

## The final LayerNorm's bias is a fixed logit vector

A logit lens adds `lm_head.weight @ norm.bias` at every block whatever the input. On
`pythia-70m-deduped` that vector is nearly as large as the whole readout at an early block, with
punctuation and function words on top.

## A tuned lens for block `N` reads `layers[N].input`

The `tuned-lens` lenses for Pythia and GPT-NeoX-20B are indexed by transformers' `hidden_states`, so
the lens for block `N` reads `model.layers[N].input`, the stream before that block.
"""
