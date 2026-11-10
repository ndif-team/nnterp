"""Persimmon: Adept's Persimmon-8B, base and chat."""

MODEL_TYPE = "persimmon"
TITLE = "Persimmon"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "adept/persimmon-8b-base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-PersimmonForCausalLM"
CHECKPOINTS = ["adept/persimmon-8b-base", "adept/persimmon-8b-chat"]


def load(checkpoint, **kwargs):
    """The page's model with the pinned checkpoint's tokenizer. Adept's checkpoints ship only a
    sentencepiece ``tokenizer.model``, which does not load without ``sentencepiece``; the pinned
    checkpoint ships the same file (same sha256) converted to ``tokenizer.json``, and the page reads no tokens."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    return StandardizedTransformer(checkpoint, tokenizer=AutoTokenizer.from_pretrained(PINNED), **kwargs)


#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 73}
QUIRKS = ["qkv-bias", "qk-norm", "partial-rotary", "squared-relu", "layernorm"]

#: What the visualization draws: a Llama-shaped sequential block with LayerNorms.
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
            "detail": "{num_heads} heads × {head_dim}, fused qkv, q/k norm",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## `query_key_value` is laid out head by head

The fused projection's output is `[num_heads, 3, head_dim]` on its last axis: head `h`'s query, key
and value are three consecutive `head_dim` blocks, not thirds of the width, and the weight's rows
follow the same layout. Edit a head at `attention_queries`, `attention_keys` or `attention_values`,
which are read after the split.
"""
