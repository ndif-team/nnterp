"""OLMo 2: a post-norm block, and queries and keys RMS-normed across all heads."""

MODEL_TYPE = "olmo2"
TITLE = "OLMo 2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/OLMo-2-0425-1B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Olmo2ForCausalLM"
CHECKPOINTS = [
    "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT", "allenai/OLMo-2-0425-1B-DPO",
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B", "allenai/OLMo-2-1124-13B-SFT", "allenai/OLMo-2-1124-13B-DPO",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B", "allenai/OLMo-2-0325-32B-SFT", "allenai/OLMo-2-0325-32B-DPO",
    "allenai/OLMo-2-0325-32B-Instruct",
]

#: Checkpoints that ship their tokenizer as `vocab.json` and `merges.txt` only, with no `tokenizer.json`.
GPT2_FILES_ONLY = {"allenai/OLMo-2-0325-32B-SFT"}


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device. ``allenai/OLMo-2-0325-32B-SFT`` has no ``tokenizer.json``,
    and transformers' ``AutoTokenizer`` maps ``olmo2`` to the generic ``TokenizersBackend``, which cannot
    build from ``vocab.json`` and ``merges.txt``; the GPT-2 tokenizer its ``tokenizer_config.json`` names
    can, so it is loaded by type and handed over."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    if checkpoint in GPT2_FILES_ONLY:
        kwargs["tokenizer"] = AutoTokenizer.from_pretrained(checkpoint, tokenizer_type="gpt2")
    return StandardizedTransformer(checkpoint, **kwargs)


#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 98}
QUIRKS = ["post-norms", "qk-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, q_norm and k_norm",
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. RMSNorm is scale-invariant, so halving `mlp.output` does nothing to the stream. Scale,
steer and decompose through `attention_output` and `mlp_output`.

## The attention ignores the scale of the stream

`q_norm` and `k_norm` take the scale out of the queries and keys, and `post_attention_layernorm` takes
it out of the output, so `attention_output` depends only on the direction of the stream the attention
reads. A steering vector reaches the attention only by turning that direction; the MLP has no such
invariance.

## `q_norm` and `k_norm` span every head

Each takes one RMS over the whole projection before the heads are split, so an edit to one head's
slice of `q_proj.output` or `k_proj.output` reaches the other heads. Edit one head at
`attention_queries` or `attention_keys`, which no norm follows.
"""
