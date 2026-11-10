"""GPT-BigCode: BigCode's StarCoder and SantaCoder, and the fine-tunes that load as GPTBigCodeForCausalLM."""

MODEL_TYPE = "gpt_bigcode"
TITLE = "StarCoder / SantaCoder"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "bigcode/gpt_bigcode-santacoder"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GPTBigCodeForCausalLM"
CHECKPOINTS = [
    "bigcode/gpt_bigcode-santacoder", "bigcode/tiny_starcoder_py",
    "bigcode/starcoderbase-1b", "bigcode/starcoderbase-3b", "bigcode/starcoderbase-7b",
    "bigcode/starcoderbase", "bigcode/starcoder", "bigcode/starcoderplus",
    "bigcode/octocoder", "HuggingFaceH4/starchat-alpha", "WizardLMTeam/WizardCoder-15B-V1.0",
]

#: Set by hues.py (lineage: GPT-2).
PALETTE = {"hue": 38}
QUIRKS = ["position-embeddings", "layernorm"]

#: What the visualization draws: GPT-2's sequential block, ln_1 before the attention, ln_2 before the MLP.
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
            "detail": "{num_heads} q heads, {num_kv_heads} kv × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
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

The tokenizer adds no beginning-of-sequence token, so in the later blocks the prompt's first token
takes most of the attention. Leave position 0 out of activation statistics and steering-vector norms.
"""
