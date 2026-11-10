"""DeepSeek-V2: multi-head latent attention, a dense first block and a softmax-routed mixture with shared experts after."""

MODEL_TYPE = "deepseek_v2"
TITLE = "DeepSeek-V2 / DeepSeek-Coder-V2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V2-Lite"
#: The tiny checkpoint the test suite builds the page from (a mixture on both of its blocks).
PINNED = "hf-tiny-v2/tiny-random-DeepseekV2ForCausalLM"
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V2-Lite", "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V2", "deepseek-ai/DeepSeek-V2-Chat", "deepseek-ai/DeepSeek-V2-Chat-0628",
    "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/DeepSeek-V2.5-1210",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Base", "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct-0724",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 322}
QUIRKS = ["latent-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on block 0, a mixture after).
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
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Every head shares the rotary part of the keys

Keys are expanded per head from one latent per token, but their last `qk_rope_head_dim` dimensions
come from one projection shared by every head: `attention_keys[:, h, :, -qk_rope_head_dim:]` is the
same tensor for every `h`. An edit to that part acts as an edit to the shared key only when it is
made on every head. `config.head_dim` is the rotary part's width, which no served value has.

## The score scale carries YaRN's factor

Every checkpoint extends its context with YaRN, and the attention multiplies the scores by
`qk_head_dim ** -0.5` times a factor set by `mscale_all_dim`. Recompute `attention_scores` from the
queries and keys with `self_attn.scaling`.

## The expert weights are not renormalized

`expert_weights` are the chosen experts' softmax probabilities times `routed_scaling_factor`, so on
the Lite models, where the factor is 1, a token's weights sum to less than one, and on the 236B
models they are the probabilities times 16. The 236B models also take a token's experts only from
its best `topk_group` groups, so the top `router_logits` are not always the chosen experts.
"""
