"""GPT-2: LayerNorm, learned absolute positions, a fused query/key/value projection."""

MODEL_TYPE = "gpt2"
TITLE = "GPT-2"
SUBTITLE = (
    "Llama's sequential block with LayerNorm in place of RMSNorm, a learned position embedding added "
    "to the token embeddings before block 0, and queries, keys and values cut from one fused c_attn projection."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "openai-community/gpt2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-gpt2"
CHECKPOINTS = [
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "distilbert/distilgpt2",
    "stanford-crfm/alias-gpt2-small-x21", "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81", "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-crfm/arwen-gpt2-medium-x21", "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/celebrimbor-gpt2-medium-x81", "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
]

#: Set by hues.py (lineage: GPT-2).
PALETTE = {"hue": 35}
QUIRKS = ["position-embeddings", "layernorm"]

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
            "detail": "{num_heads} heads × {head_dim}, fused c_attn",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function} (c_fc, c_proj)",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The learned position embedding transformer.wpe is added after it, so layers[0].input is "
             "token_embeddings + transformer.wpe.output.",
}

NOTES = """
## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes most of the
attention and its stream has a norm tens of times the other positions'. Leave position 0 out of
activation statistics and steering-vector norms, or prepend `<|endoftext|>`, as TransformerLens does
by default.

## `gpt2-small-res-jb` reads the centered stream

TransformerLens' default `HookedTransformer.from_pretrained` centers every weight that writes to the
stream, so its `hook_resid_pre`, and the `gpt2-small-res-jb` SAEs trained on it, see
`layers[i].input` minus its mean over the hidden axis. Subtract that mean before encoding. Fed the raw stream, block 8's
SAE leaves 250% of the variance unexplained, against 15% on the centered one.

## The Stanford CRFM checkpoints need a different load for the attention interior

They set `reorder_and_upcast_attn`, so the queries, the pattern and the other interior values are
unavailable. Load them in float32 with `reorder_and_upcast_attn=False` and
`attn_implementation="eager"`: that load computes the flagged path's logits to rounding, and a float16
load does not.
"""
