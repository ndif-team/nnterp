"""Helium: Kyutai's Llama-shaped block whose rotary turns adjacent pairs of dimensions."""

MODEL_TYPE = "helium"
TITLE = "Helium"
SUBTITLE = (
    "Llama's block, with one key/value head per query head and a rotary embedding that turns adjacent pairs of "
    "dimensions (2i, 2i + 1) rather than i with i + head_dim / 2."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "kyutai/helium-1-preview-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HeliumForCausalLM"
#: The one public checkpoint whose config says model_type helium; Helium 1 (kyutai/helium-1-2b and its
#: per-domain variants) is model_type llama and loads as the llama family.
CHECKPOINTS = ["kyutai/helium-1-preview-2b"]

#: Llama-shaped, but the Llama lineage's hues (139 to 167) are full; 124 sits between phimoe's 120 and
#: gemma4_unified_text's 127, one of the largest free gaps.
PALETTE = {"hue": 124}
VLLM = False
QUIRKS = ["interleaved-rotary"]

#: Real values in the notes were measured on helium-1-preview-2b on a GPU, in bfloat16 unless the note says
#: float32; shapes, identities and the rotary recomputed exactly ran on the pinned tiny checkpoint.

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
            "detail": "{num_heads} heads × {head_dim}, interleaved rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends <s> (id 1), "
             "though its bos_token is None.",
    "norm": "RMSNorm whose gain is norm.weight as stored, applied in float32; eps is 1e-8.",
    "head": "lm_head has its own weight (tie_word_embeddings is false). logits is lm_head.output, with no cap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
mlp(z) = down_proj(silu(gate_proj(z)) * up_proj(z))
```

Llama's block, names and SwiGLU MLP. The RMSNorms have `eps` 1e-8 and multiply by their gain in
float32 before casting back to the input's dtype.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream
with nothing in between. The identity holds exactly in float32 on the pinned checkpoint:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Rotary turns adjacent pairs

Helium's rotary pairs dimension `2i` with `2i + 1`: its `rotate_half` takes the even and odd
dimensions, and each frequency appears twice in a row in the cos/sin table. Llama pairs `i` with
`i + head_dim / 2`. It turns the whole head, with `rope_theta` 100,000. `attention_queries` and
`attention_keys` are read after the rotation; on the pinned checkpoint they equal the interleaved
rotation of `q_proj.output` and `k_proj.output` exactly, and differ from the `rotate_half` one.
Code written for a `rotate_half` family's queries needs the head reordered, evens then odds:

```python
hd = model.head_dim
perm = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])

with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.save()

q_half = q[..., perm]          # in rotate_half's order
```

## One key/value head per query head

helium-1-preview-2b has 20 query heads and 20 key/value heads, each 128 wide, in a 2560-wide stream:
`attention_keys` and `attention_values` are `[batch, 20, seq, 128]`, and an edit to head `j`
reaches query head `j` alone. No projection has a bias.

## The default load runs the same attention as eager

The six attention interior values need `attn_implementation="eager"`. There is no softcap,
window or sink: on helium-1-preview-2b in float32 the default `sdpa` load's logits differ from
the eager load's by at most `2e-5`.

## `<s>` at position 0 is the sink

The tokenizer prepends `<s>` (id 1) though `tokenizer.bos_token` is `None`, so code that tests
`bos_token` to decide whether a BOS is present gets the wrong answer. On helium-1-preview-2b the
stream at position 0 has a norm of 930 after block 10, against 9 to 12 at the other positions of a
test prompt, and the heads of block 10 put on average 0.84 of their attention on it. Drop
`[:, 1:]` before averaging over positions.

## Target tokens carry the word marker

The tokenizer is SentencePiece: `"Paris"` encodes to `▁Paris` (2291), and `" Paris"` to two
tokens, `▁` and `▁Paris`. After `"The Eiffel Tower is in the city of"` the model puts 0.986 on
`▁Paris`. Take the target id from the word without its leading space:

```python
ids = model.tokenizer("Paris", add_special_tokens=False).input_ids   # [2291]
```

## The readout

`model.logits` is `model.lm_head.output`: no cap and no scale, and `project_on_vocab` is
`lm_head(norm(hidden))`. `lm_head` and `embed_tokens` are separate weights.

## What loads as this family

Only `kyutai/helium-1-preview-2b` is `model_type` `helium`. Helium 1 (`kyutai/helium-1-2b` and its
per-domain variants `-books`, `-wiki`, `-science` and the others) and `Sequential_Helium_6B` are
`model_type` `llama` and load as the Llama family, with Llama's `rotate_half` rotary;
`Helium1-VL-2B` is a remote-code architecture.
"""
