"""Persimmon: Adept's Persimmon-8B, base and chat."""

MODEL_TYPE = "persimmon"
TITLE = "Persimmon"
SUBTITLE = (
    "Llama's sequential block with LayerNorms, queries, keys and values cut per head from one fused, biased "
    "query_key_value, a LayerNorm on each query and key head, rotary on half of each head, and a squared-ReLU MLP."
)

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


#: No kin among the entries; set in the largest free gap of the hue table.
PALETTE = {"hue": 171}
VLLM = False
QUIRKS = ["fused-qkv", "qkv-bias", "qk-norm", "partial-rotary", "squared-relu", "layernorm"]

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
            "pre_norm_note": "A LayerNorm with a weight and a bias.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "The MLP's input norm, as on Llama: a LayerNorm with a weight and a bias.",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_tokens is a plain lookup: no scale and no position embedding (positions enter through rotary "
             "in each attention), so token_embeddings equals layers[0].input.",
    "norm": "final_layernorm is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight (tie_word_embeddings is false) and no bias. Nothing follows it: "
            "logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + dropout(mlp(post_attention_layernorm(h)))     # dropout is the identity in eval
```

The block returns a bare tensor, and nothing but the eval-mode dropout sits between a sublayer and the
stream: `attention_output` is the attention's output after `dense`, `mlp_output` is the MLP's output
after `dense_4h_to_h`, and `layers[i].input + attention_output + mlp_output == layer_output` holds
exactly. Every norm is a `LayerNorm` with a weight and a bias, and every projection has a bias.

## Queries, keys and values are cut per head from one projection

`self_attn.query_key_value` is one biased `nn.Linear` from the width to three times the width. Its
output is laid out head by head, `[heads, 3, head_dim]` on the last axis: head `h`'s query, key and
value are three consecutive `head_dim` blocks, not three blocks of the whole width. The rows of the
weight follow the same layout:

```python
attn = model.layers[1].self_attn
H, D = model.num_heads, model.head_dim
W = attn.query_key_value.weight.view(H, 3, D, -1)    # [heads, q/k/v, head_dim, hidden]
b = attn.query_key_value.bias.view(H, 3, D)

with model.trace(prompt):
    h = model.layers[1].self_attn.input.save()
    v = model.layers[1].self_attn.attention_values.save()

v_again = torch.einsum("bsx,hdx->bhsd", h, W[:, 2]) + b[:, 2, None]
torch.testing.assert_close(v_again, v)
```

Every head has its own keys and values (`num_kv_heads == num_heads`, 64 heads of 64 on the 8B
checkpoints). The attention runs through the shared interface: the interior values need
`attn_implementation="eager"`, and the scores are scaled by `head_dim ** -0.5` (0.125) inside it.

## Queries and keys are normed per head, then half of each is turned

Under `qk_layernorm` (set on both checkpoints) `q_layernorm` and `k_layernorm` are `LayerNorm`s over
`head_dim`, one weight shared by every head, applied to the split queries and keys before rotary.
Rotary then turns the first `partial_rotary_factor` of each head (0.5: 32 of 64 dimensions, with
`rope_theta` 25000); the other half carries no position. `attention_queries` and `attention_keys` are
read after both, so their last half is the normed projection unchanged and their first half is
rotated:

```python
rot = int(D * model.config.partial_rotary_factor)
with model.trace(prompt):
    h = model.layers[1].self_attn.input.save()
    q = model.layers[1].self_attn.attention_queries.save()

q_raw = torch.einsum("bsx,hdx->bhsd", h, W[:, 0]) + b[:, 0, None]
q_normed = attn.q_layernorm._module(q_raw.transpose(1, 2)).transpose(1, 2)
torch.testing.assert_close(q_normed[..., rot:], q[..., rot:])
```

An edit to `q_proj`'s share of `query_key_value.output` passes through the norm, so it is rescaled
and re-centred per head before it reaches the scores; an edit to `attention_queries` is not.

## The MLP is squared ReLU with no gate

`mlp` is `dense_h_to_4h`, the activation and `dense_4h_to_h`, `4096 → 16384 → 4096`
on the 8B checkpoints, with no gate projection. The checkpoints set `hidden_act` to `relu2`,
`relu(x) ** 2`: a neuron, `mlp.act.output[..., n]`, is exactly zero wherever its pre-activation is
negative. The pinned tiny checkpoint uses GELU, so a check of that property needs the real weights.

## The readout: LayerNorm with a bias, untied weights

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` is
`lm_head(final_layernorm(x))`, so a logit lens at the last block equals `logits`. `lm_head` has its
own weight and no bias. The tokenizer prepends `|ENDOFTEXT|`, its BOS token, to every prompt.

## The family's checkpoints

`persimmon-8b-base` and `persimmon-8b-chat` share one shape: 36 blocks of width 4096, 64 heads of
64, an MLP of 16384, a 262144-token vocabulary and 16384 positions, stored in bfloat16. Fuyu-8B's
language model is a Persimmon, but nnterp does not bind the `fuyu` wrapper.
"""
