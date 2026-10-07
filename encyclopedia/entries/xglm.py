"""XGLM: Meta's multilingual XGLM-564M to XGLM-7.5B, OPT's block with a scaled embedding and sinusoidal positions."""

MODEL_TYPE = "xglm"
TITLE = "XGLM"
SUBTITLE = (
    "OPT's pre-norm block with no MLP module, so fc2.output is the block's second term; the embedding is "
    "scaled by sqrt(d_model), sinusoidal positions are added after it, and the attention is its own bmm arithmetic."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "facebook/xglm-564M"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-XGLMForCausalLM"
CHECKPOINTS = [
    "facebook/xglm-564M", "facebook/xglm-1.7B", "facebook/xglm-2.9B", "facebook/xglm-4.5B", "facebook/xglm-7.5B",
]

#: OPT's kin (opt is 336).
PALETTE = {"hue": 346}
VLLM = False
QUIRKS = ["no-mlp", "scaled-embeddings", "position-embeddings", "own-attention-arithmetic", "layernorm", "qkv-bias"]

#: What the visualization draws: the attention and its pre-norm, then the MLP path, which has no module:
#: final_layer_norm, fc1, activation_fn and fc2 sit on the block, so the sublayer is hosted on the native fc2.
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
            "detail": "{num_heads} heads × {head_dim}, own bmm",
            "pre_norm_note": "Native name self_attn_layer_norm, a LayerNorm with a bias.",
        },
        {
            "host": "fc2",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "final_layer_norm",
            "reads": "fc1",
            "pre_norm_note": "The block's own final_layer_norm, a LayerNorm with a bias; the decoder's last norm is "
                             "layer_norm, model.norm.",
            "contribution": "fc2.output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
            "host_note": ("No mlp module: final_layer_norm, fc1, activation_fn and fc2 sit on the block, and fc2.output, "
                          "[batch, seq, hidden], is what the path adds."),
        },
    ],
    "identity": "layers[i].input + self_attn.attention_output + fc2.output == layer_output",
    "identity_note": "fc2.output is the MLP path's term: the block has no mlp module, and fc1 and fc2 sit on the block. "
                     "Exact; the suite checks it on this family.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_tokens multiplies its lookup by sqrt(d_model), so token_embeddings is the scaled lookup. "
             "The sinusoidal positions, model.embed_positions, are added after it: "
             "layers[0].input = token_embeddings + embed_positions.output.",
    "layers": "Each block runs the attention, then final_layer_norm, fc1, activation_fn and fc2, all on the block: "
              "there is no mlp module, and layers[i].fc2.output is the block's second term.",
    "norm": "layer_norm is a LayerNorm with a bias; project_on_vocab applies it.",
    "head": "lm_head has no bias and shares its weight with embed_tokens. Its output is logits, unchanged.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(self_attn_layer_norm(x))     # self_attn_layer_norm = input_layernorm
out = h + fc2(act(fc1(final_layer_norm(h))))    # act: gelu, relu on xglm-4.5B
```

The block returns a bare tensor. `attention_output` is the attention module's output after `out_proj`.
The block's own `final_layer_norm` is the MLP path's input norm and keeps its native name; the
decoder's last norm is `layer_norm`, which is `model.norm`. Every norm is a `LayerNorm` with a weight
and a bias, and every projection has a bias.

## The MLP is fc1 and fc2 on the block

No block has an `mlp` module, so `mlp`, `mlp_output` and their kin do not exist on XGLM and
`support()` lists no `mlp.*` key. What the MLP path adds is `layers[i].fc2.output`, `[batch, seq,
hidden]`, and the neurons are `layers[i].activation_fn.output`, `[batch, seq, ffn_dim]`:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    neurons = model.layers[1].activation_fn.output.save()
    mlp_out = model.layers[1].fc2.output.save()
    out = model.layers[1].layer_output.save()

torch.equal(x + attn + mlp_out, out)              # True
```

`intermediate_size` is the config's `ffn_dim`: four times the width on every size but `xglm-4.5B`,
whose MLP is eight times the width (16384 on 2048) and uses ReLU where the others use GELU.

## The embedding is scaled, the positions are sinusoidal

`embed_tokens` is an `XGLMScaledWordEmbedding`: it multiplies its lookup by `sqrt(d_model)` (32 on
`xglm-564M`), and `token_embeddings` is the product. `model.embed_positions` is a fixed sinusoidal
table with no parameters, kept in the buffer `weights`; position `t` reads row `t + 2`, and row 1,
the padding index, is zeros. Its output is added after the scaled lookup:

```python
with model.trace(prompt):
    ids = model.input_ids.save()
    tokens = model.token_embeddings.save()
    positions = model.model.embed_positions.output.save()
    x0 = model.layers[0].input.save()

lookup = model.embed_tokens.weight[ids]
torch.allclose(tokens, lookup * model.config.d_model ** 0.5)   # True
W = model.model.embed_positions._module.weights
torch.equal(tokens + positions, x0)                            # True
torch.equal(positions[0], W[2 : 2 + x0.shape[1]])              # True
```

On `xglm-564M` the raw lookup has a median norm of 4.1 per token, the scaled one 132, and every
position's row 22.6. The table is rebuilt longer when a prompt runs past it, so no length raises. A
left-padded prompt gives the same logits as the prompt alone (largest difference 1.3e-7 on the
pinned checkpoint).

## The attention is its own arithmetic

XGLM has one attention implementation, so the interior values need no eager load and carry no `⚠`.
The queries are scaled by `1/sqrt(head_dim)` as they are projected, so `attention_queries` is
`q_proj`'s output times that scale and `attention_scores` is `attention_queries @
attention_keys.transpose(-1, -2)` plus the mask. Every value is a view of the `[batch * heads, ...]`
tensor the forward holds, served heads first, so an in-place edit reaches the model. Every head has its
own keys and values.

## Block 0 writes a large shared vector, and position 0 is a sink

On `xglm-564M` block 0's two terms add a vector of norm about 860 to every position, almost the same
at each one and largest in coordinates 35, 148 and 874: `layer_output` of block 0 has a median norm of
858, and its mean over positions has a norm of 858, against 132 entering the block. The tokenizer
prepends `</s>` (id 2), and the stream at that position grows to a norm of 13,000 from block 8 to
block 16, mostly in coordinates 660 and 524, while the median elsewhere is 900 to 1,100. From block 7
to block 21, 59% to 82% of the attention pattern, averaged over heads and later queries, lands on key
0. Subtract the mean over positions, or leave position 0 out, before taking norms or activation
statistics.

## The readout: LayerNorm with a bias, tied weights

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` is `lm_head(layer_norm(x))`,
so a logit lens at the last block equals `logits`. `lm_head` has no bias and its weight is
`embed_tokens.weight`, the same tensor; the embedding's scale is applied in the forward, so the
unembedding is the unscaled matrix.

## The family's checkpoints

Blocks × width: 24 × 1024 (`xglm-564M`, 16 heads of 64), 24 × 2048 (`xglm-1.7B`), 48 × 2048
(`xglm-2.9B`, `xglm-4.5B`), 32 × 4096 (`xglm-7.5B`, 32 heads); `head_dim` is 128 from 1.7B on. Every
size has 2048 trained positions and the same 256008-token multilingual vocabulary.
"""
