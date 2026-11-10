"""VaultGemma: Gemma 2's tree with the pre-norms only, so the contributions are the modules' outputs."""

MODEL_TYPE = "vaultgemma"
TITLE = "VaultGemma"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/vaultgemma-1b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-VaultGemmaForCausalLM"
CHECKPOINTS = ["google/vaultgemma-1b"]

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 168}
QUIRKS = [
    "scaled-embeddings", "gain-norm",
    # every block is full_attention on google/vaultgemma-1b; the pinned tiny checkpoint alternates
    {"slug": "sliding-window", "when": lambda config: "sliding_attention" in (config.layer_types or [])},
    # null on google/vaultgemma-1b; set (30) on the pinned tiny checkpoint
    {"slug": "softcapped-logits", "when": lambda config: config.final_logit_softcapping is not None},
]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads × {head_dim}",
            "variants": {
                "full_attention": "full causal attention",
                "sliding_attention": "sliding window of {sliding_window} tokens",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_feedforward_layernorm",
            "pre_norm_note": "The MLP's input norm, also bound as post_attention_layernorm; no norm follows the attention.",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_activation}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by √hidden_size inside the embedding module, so token_embeddings equals layers[0].input.",
}

NOTES = """
## `model.tokenizer` turns any text into two tokens

`google/vaultgemma-1b` ships `tokenizer.model` and no `tokenizer.json`, and the tokenizer
transformers builds from it has five tokens, so every string encodes to `<bos> <unk>` without a
warning. Tokenize with sentencepiece on the repository's `tokenizer.model`, put `<bos>` (id 2) first,
and trace the ids.
"""
