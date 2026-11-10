"""Youtu-LLM: DeepSeek-V2's latent attention on Llama's dense block, with tied embeddings."""

MODEL_TYPE = "youtu"
TITLE = "Youtu-LLM"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tencent/Youtu-LLM-2B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-YoutuForCausalLM"
#: Every public id whose config is youtu (Youtu-LLM-2B-GGUF has no config.json).
CHECKPOINTS = ["tencent/Youtu-LLM-2B-Base", "tencent/Youtu-LLM-2B"]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 334}
QUIRKS = ["latent-attention", "interleaved-rotary"]

#: Shapes, identities and the rotary order were run on the pinned tiny checkpoint (queries and keys 64 wide,
#: values 32). Sizes, the scale, tied weights and the tokenizer are the 2B checkpoints', read off their
#: configs, meta builds and tokenizers; no real weights were run.

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
            "detail": "latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The rotary key is one vector copied into every head

The last `qk_rope_head_dim` dimensions of every head's `attention_keys` are the same rotary key,
computed once per token and copied into each head before nnterp reads it. An edit to one head's copy
reaches that head only.
"""
