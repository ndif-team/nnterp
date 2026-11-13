"""Hunyuan MoE V1 (Hunyuan-A13B): Hunyuan's attention and a mixture of 64 experts plus a shared expert on every block."""

MODEL_TYPE = "hunyuan_v1_moe"
TITLE = "Hunyuan-A13B"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tencent/Hunyuan-A13B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HunYuanMoEV1ForCausalLM"
#: Hunyuan-A13B-Pretrain's config says ``model_type`` ``hunyuan`` and needs its own remote code: listed, greyed.
CHECKPOINTS = ["tencent/Hunyuan-A13B-Instruct", "tencent/Hunyuan-A13B-Pretrain"]

#: Set by hues.py (lineage: HunYuan).
PALETTE = {"hue": 29}
QUIRKS = ["qk-norm", "mixture-of-experts"]

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
            "kind": "moe",
            "part_order": ["shared", "router", "experts"],
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The mixture's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, 1 shared",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = """
## `mlp_output` includes the shared expert

Every block's mixture runs one shared expert on every token and adds the routed experts' sum to it,
so `mlp_output` is `shared_expert_output + routed_output`. Zeroing `shared_expert_output` leaves the
routed term in place.

## Queries and keys are normed after the rotary

`query_layernorm` and `key_layernorm` norm each head's rotated query and key, and `attention_queries`
and `attention_keys` are read after them. The norm undoes a rescaling of a head's slice of
`q_proj.output` or `k_proj.output`, so scale or steer a head's query or key at `attention_queries` or
`attention_keys`.
"""
