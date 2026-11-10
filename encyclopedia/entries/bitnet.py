"""BitNet b1.58: ternary-weight projections with 8-bit inputs, and an RMSNorm before each output projection."""

MODEL_TYPE = "bitnet"
TITLE = "BitNet b1.58"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/bitnet-b1.58-2B-4T"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-BitNetForCausalLM"
CHECKPOINTS = ["microsoft/bitnet-b1.58-2B-4T", "microsoft/bitnet-b1.58-2B-4T-bf16"]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 253}
QUIRKS = ["quantized-linears", "sub-norms", "squared-relu"]

#: Real values in the notes were measured on bitnet-b1.58-2B-4T-bf16 in bfloat16 on a GPU (its weights load as
#: bfloat16 and are ternarized in every forward); shapes, identities and read orders also ran on the pinned tiny
#: checkpoint, whose projections are plain nn.Linear and whose hidden_act is gelu.

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
            "detail": "{num_heads}/{num_kv_heads} heads, attn_sub_norm, o_proj",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act} gate, ffn_sub_norm",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The sub-norms tie the heads together

```
h   = x + o_proj(attn_sub_norm(heads))       # heads: the attention over input_layernorm(x)
out = h + down_proj(ffn_sub_norm(gated))     # gated: relu(gate_proj(z))² · up_proj(z)
```

`attn_sub_norm` divides by one RMS over every head's output together, and `attention_head_outputs`
is read before it. Zeroing or scaling one head changes that RMS and so moves the other heads' part
of `attention_output`, and a head's effect is not a separate term of it. The MLP's neurons share
`ffn_sub_norm` the same way. Ablate or scale a whole sublayer at `attention_output` or `mlp_output`.
"""
