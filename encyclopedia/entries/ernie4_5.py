"""ERNIE 4.5 dense: Llama's block and Llama's names, with rotary turning adjacent pairs of dimensions."""

MODEL_TYPE = "ernie4_5"
TITLE = "ERNIE 4.5"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "baidu/ERNIE-4.5-0.3B-Base-PT"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Ernie4_5ForCausalLM"
CHECKPOINTS = ["baidu/ERNIE-4.5-0.3B-Base-PT", "baidu/ERNIE-4.5-0.3B-PT"]

#: Set by hues.py (lineage: ERNIE).
PALETTE = {"hue": 186}
QUIRKS = ["interleaved-rotary"]

#: Shapes, identities and the rotary layout ran on the pinned tiny checkpoint and on ERNIE-4.5-0.3B-Base-PT in
#: float32 on a GPU; the norm gains are 0.3B-Base's. Every snippet ran on both.

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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
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

NOTES = ""
