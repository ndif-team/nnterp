"""CodeGen: Salesforce's code models, the -nl, -multi and -mono checkpoints that load as CodeGenForCausalLM."""

MODEL_TYPE = "codegen"
TITLE = "CodeGen"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Salesforce/codegen-350M-mono"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-CodeGenForCausalLM"
CHECKPOINTS = [
    "Salesforce/codegen-350M-mono", "Salesforce/codegen-350M-multi", "Salesforce/codegen-350M-nl",
    "Salesforce/codegen-2B-mono", "Salesforce/codegen-2B-multi", "Salesforce/codegen-2B-nl",
    "Salesforce/codegen-6B-mono", "Salesforce/codegen-6B-multi", "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-16B-mono", "Salesforce/codegen-16B-multi", "Salesforce/codegen-16B-nl",
]

#: Set by hues.py (lineage: GPT-J).
PALETTE = {"hue": 57}
QUIRKS = ["parallel-blocks", "partial-rotary", "interleaved-rotary", "layernorm"]

#: What the visualization draws: GPT-J's parallel block. One ln_1 feeds both sublayers; the
#: schema draws a pre-norm per sublayer, so it appears twice, as one node.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "ln_1",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, rotary on {rotary_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "ln_1",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "head": "lm_head has a bias.",
}

NOTES = """
## The MLP does not read the attention

```
h   = ln_1(x)
out = x + self_attn(h) + mlp(h)
```

Both sublayers read `ln_1`'s output, so ablating or patching a block's attention leaves that
block's MLP unchanged. The first MLP to read block `i`'s attention is block `i + 1`'s.

## On `-multi` and `-mono`, a run of spaces is one token

Their tokenizers add a token for each run of 2 to 31 spaces and of 2 to 9 tabs, and the word after
the run loses its leading space: `"    return x"` is `'    '`, `'return'`, `' x'`. A lens or
patching target inside indented code is the bare word. The `-nl` tokenizer is GPT-2's and splits
the same code differently.

## CodeGen2's 1B and 3.7B load as this family and run wrong

They set `model_type` `codegen`, so transformers builds its own class around them without a
warning, and that class cuts `qkv_proj` into 4 groups of `[q | v | k]` where their weights hold 8.
Regroup each `qkv_proj.weight` from 8 groups to 4 before tracing. CodeGen2's 7B and 16B use 4 groups.
"""
