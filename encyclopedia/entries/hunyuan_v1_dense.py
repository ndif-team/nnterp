"""Hunyuan dense V1: Llama's block whose attention RMS-norms each query and key head after the rotary embedding."""

MODEL_TYPE = "hunyuan_v1_dense"
TITLE = "Hunyuan"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tencent/Hunyuan-0.5B-Pretrain"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HunYuanDenseV1ForCausalLM"
CHECKPOINTS = [
    "tencent/Hunyuan-0.5B-Pretrain", "tencent/Hunyuan-0.5B-Instruct",
    "tencent/Hunyuan-1.8B-Pretrain", "tencent/Hunyuan-1.8B-Instruct",
    "tencent/Hunyuan-4B-Pretrain", "tencent/Hunyuan-4B-Instruct",
    "tencent/Hunyuan-7B-Pretrain", "tencent/Hunyuan-7B-Instruct", "tencent/Hunyuan-7B-Instruct-0124",
    "tencent/Hunyuan-MT-7B", "tencent/Hunyuan-MT-Chimera-7B",
]

#: Set by hues.py (lineage: HunYuan).
PALETTE = {"hue": 26}
QUIRKS = ["qk-norm"]

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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, q/k normed",
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
}

STRIP: dict[str, str] = {}

NOTES = """
## Queries and keys are normed after the rotary

`query_layernorm` and `key_layernorm` norm each head's rotated query and key, and `attention_queries`
and `attention_keys` are read after them. The norm undoes a rescaling of a head's slice of
`q_proj.output` or `k_proj.output`, so scale or steer a head's query or key at `attention_queries` or
`attention_keys`.

## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token to a plain prompt, so its first token takes most of
the attention and its stream has a norm many times the other positions'; a chat template puts its
beginning-of-sequence token there instead. Leave position 0 out of activation statistics and
steering-vector norms.
"""
