"""FlexOlmo: OLMo-2's post-norm block with a mixture of independently trained experts in place of the MLP, every expert on every token."""

MODEL_TYPE = "flex_olmo"
TITLE = "FlexOlmo"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/FlexOlmo-7x7B-1T"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-FlexOlmoForCausalLM"
CHECKPOINTS = [
    "allenai/FlexOlmo-7x7B-1T", "allenai/FlexOlmo-7x7B-1T-RT",
    "allenai/Flex-math-2x7B-1T", "allenai/Flex-news-2x7B-1T", "allenai/Flex-pes2o-2x7B-1T",
    "allenai/Flex-code-2x7B-1T", "allenai/Flex-creative-2x7B-1T", "allenai/Flex-reddit-2x7B-1T",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 107}
QUIRKS = ["post-norms", "qk-norm", "mixture-of-experts"]

#: Every checkpoint is 2x7B or 7x7B, so no real weights were run: shapes, identities and the routing edits
#: are from the pinned tiny checkpoint (8 experts, top 2; float32, CPU), the every-expert routing also from
#: the tiny loaded with num_experts_per_tok=8; sizes, expert counts and norm_topk_prob from the checkpoints'
#: configs; the experts' training data from the model cards.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = """
## The post-norm undoes any scale on the mixture

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

`attention_output` and `mlp_output` are the post-norms' outputs. Scaling `routed_output`, or every
`expert_weights` entry of a token together, leaves `mlp_output` unchanged; the routing sets the mix
of the experts' directions, not the size of the term. Scale `mlp_output` to scale what the block adds.

## Every expert runs on every token

`num_experts_per_tok` equals `num_experts` on every released checkpoint, so `expert_weights` are the
whole softmax and `expert_indices` lists every expert, ordered by weight. Counting how often an expert
is chosen tells nothing; compare its weights. Zeroing an expert's weight and setting its
`router_logits` entry to `-inf` give the same `mlp_output`.

## The tokenizer adds no beginning-of-sequence token

The sink that trained models form at position 0 falls on the prompt's first token. Leave position 0
out of activation statistics and steering-vector norms.
"""
