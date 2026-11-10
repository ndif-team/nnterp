"""JetMoE: Llama's pre-norm block in which the attention is itself a mixture of experts (each expert a query
and an output projection over shared keys and values) and the MLP a mixture of experts with a bias."""

MODEL_TYPE = "jetmoe"
TITLE = "JetMoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "jetmoe/jetmoe-8b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-JetMoeForCausalLM"
#: jetmoe/jetmoe-8b-chat is left out: its config names its sizes n_layer, n_head and n_embd, which
#: transformers' JetMoeConfig does not read, so it builds with the default 12 blocks, not 24.
CHECKPOINTS = ["jetmoe/jetmoe-8b", "jetmoe/jetmoe-8b-sft"]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 133}
QUIRKS = ["attention-experts", "mixture-of-experts"]

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
            "detail": "{num_experts_per_tok} of {num_local_experts} experts × {num_kv_heads} heads",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_indices", "expert_weights", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## A query head is a routing slot, not an expert

Each chosen attention expert projects the token's queries for `num_kv_heads` heads, so query head
`h` belongs to slot `h // num_kv_heads`, and the expert behind it changes from token to token.
nnterp serves no routing values for the attention. Its router logits are
`self_attn.experts.router.layer.output`, flat over tokens, and their top `top_k` give each slot's
expert in order. To ablate attention expert `e`, zero `attention_head_outputs` in the slots that
chose it.

## `attention_keys` holds one copy of the keys per slot

The shared keys and values are tiled to every query head, so `attention_keys` and
`attention_values` have `num_heads` heads, `top_k` copies of `num_kv_heads`. An edit to one head
reaches only its slot's queries. Edit key/value head `j` at every copy (`j`, `j + num_kv_heads`), or
at `kv_proj.output`, to reach every query that reads it.

## Both mixtures add a bias after the experts

`mlp_output` is `routed_output` plus the mixture's `bias`, and `attention_output` is the experts'
gated sum plus `self_attn.experts.bias`. Ablating every expert leaves the bias in the stream.
"""
