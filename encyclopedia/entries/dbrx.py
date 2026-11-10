"""DBRX: a pre-norm block whose attention and both norms sit in one wrapper module, a fused and clamped
Wqkv, LayerNorms without bias, and a mixture of 16 experts on every block."""

MODEL_TYPE = "dbrx"
TITLE = "DBRX"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Databricks' own repositories are not on the Hub; the page reads a re-upload's config.
REFERENCE = "Undi95/dbrx-base"
#: The Hub org the index files the family under: the copy is Undi95's; the model is Databricks'.
ORG = "databricks"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/dbrx-tiny256-random"
CHECKPOINTS = [
    "databricks/dbrx-base", "databricks/dbrx-instruct",
    "Undi95/dbrx-base", "alpindale/dbrx-instruct",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 90}
QUIRKS = ["layernorm", "mixture-of-experts"]

#: The real checkpoints are 132B parameters: nothing here ran on real weights. Every identity and shape in
#: the notes, and every snippet, ran on the pinned tiny checkpoint in float32 (2 blocks, 16 experts, top 4);
#: the sizes are the re-uploads' configs.


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device from the checkpoint's config with the pinned checkpoint's
    tokenizer. DBRX's checkpoints ship their tokenizer as remote code over ``tiktoken``
    (``TiktokenTokenizerWrapper``), which does not load without ``trust_remote_code`` and ``tiktoken``; the
    pinned checkpoint's is a ``GPT2Tokenizer`` over the same vocabulary, and the page reads no tokens."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    return StandardizedTransformer(checkpoint, tokenizer=AutoTokenizer.from_pretrained(PINNED), **kwargs)


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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv, Wqkv",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## A default load runs the rotary at the wrong base

The released configs give the trained base, 500,000, as `attn_config.rope_theta`, which transformers
does not read. It builds the rotary from `rope_parameters`, whose default base is 10,000, so a
default load rotates queries and keys faster than the trained model did. Load with
`rope_parameters={"rope_theta": 500000.0, "rope_type": "default"}`.

## `expert_outputs` is unavailable

The experts run as one module that loops over them, so no tensor holds one slot's output. To
measure one expert's term, zero its slot of `expert_weights` where `expert_indices` picks it and
compare `mlp_output`, which changes only on the tokens routed to it.
"""
