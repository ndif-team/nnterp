"""Phi-3, Phi-3.5 and Phi-4: Microsoft's models that load as Phi3ForCausalLM."""

MODEL_TYPE = "phi3"
TITLE = "Phi-3 / Phi-3.5 / Phi-4"
SUBTITLE = (
    "Llama's block with two fused projections: qkv_proj yields queries, keys and values as three slices, and "
    "gate_up_proj the gate and the up projection as two halves; the 128k checkpoints rescale rotary by sequence "
    "length, and the 4k ones attend over a 2047-token window."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/Phi-3-mini-4k-instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Phi3ForCausalLM"
CHECKPOINTS = [
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-medium-4k-instruct", "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning",
    "microsoft/phi-4", "microsoft/Phi-4-reasoning", "microsoft/Phi-4-reasoning-plus",
]

#: Set by hues.py (lineage: Phi).
PALETTE = {"hue": 140}
VLLM = True
QUIRKS = ["fused-qkv", "sliding-window", "partial-rotary"]

#: No checkpoint of this family is 3B parameters or fewer, so nothing here ran on real weights: shapes, splits
#: and identities are from the pinned tiny checkpoint, the rotary from transformers' rotary module built on each
#: checkpoint's config, and every size from the configs.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, fused qkv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, as on Llama: it reads the stream after the attention's add.",
            "contribution": "mlp_output",
            "detail": "fused gate_up: {hidden_size} → 2 × {intermediate_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. No tokenizer of this family adds a BOS.",
    "norm": "An RMSNorm whose gain is norm.weight; project_on_vocab applies it.",
    "head": "lm_head has no bias. Its weight is embed_tokens' on Phi-4-mini (tie_word_embeddings) and its own on "
            "every other checkpoint. Nothing follows it: logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # one qkv_proj, then o_proj
out = h + mlp(post_attention_layernorm(h))    # one gate_up_proj, then down_proj
```

Llama's pre-norm block and Llama's names, with RMSNorms and no biases anywhere in the block.
`attention_output` is `o_proj`'s output and `mlp_output` is `down_proj`'s; the identity is the
plain sum, exact in float32 on the pinned checkpoint. `resid_attn_dropout` and `resid_mlp_dropout`
sit between each sublayer and its add and are inert in eval. The block returns a bare tensor, so
`model.layers[i].output` is `layer_output`.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## `qkv_proj` holds the queries, the keys and the values as three slices

`self_attn.qkv_proj` is one `Linear` without a bias, from `hidden_size` to
`(num_heads + 2 * num_kv_heads) * head_dim`. Its output is `[Q | K | V]` in three contiguous
slices, queries first and `num_heads * head_dim` wide, then keys and values `num_kv_heads * head_dim`
wide each; within a slice the heads follow one another. On Phi-3-mini that is 3072 + 3072 + 3072
columns, on Phi-4-mini 3072 + 1024 + 1024, on Phi-3-medium and phi-4 5120 + 1280 + 1280. No module
holds the queries alone, so there is no `q_proj` to hook; `attention_queries`, `attention_keys` and
`attention_values` are the three split into heads, the first two after the rotary. To read a slice
before the rotary, split the projection's output in the same trace:

```python
with model.trace(prompt):
    qkv = model.layers[1].self_attn.source.self_qkv_proj_0.output.save()
    v = model.layers[1].self_attn.attention_values.save()

H, KV, D = model.num_heads, model.num_kv_heads, model.head_dim
q_pre, k_pre, v_flat = qkv.split([H * D, KV * D, KV * D], dim=-1)
b, s, _ = qkv.shape
assert torch.equal(v_flat.view(b, s, KV, D).transpose(1, 2), v)
```

The weight's rows are in the same order: `qkv_proj.weight[:H * D]` are the queries' rows. An edit to
one head belongs at `attention_queries`, `attention_keys` or `attention_values`, which are per head.

## `gate_up_proj` holds the gate and the up projection as two halves

`mlp.gate_up_proj` is one `Linear` from `hidden_size` to `2 * intermediate_size`, without a bias.
The forward cuts its output with `chunk(2)`: the first half is the gate, the second the up
projection, and `down_proj` reads `up * silu(gate)`. A neuron's activation is `down_proj`'s input:

```python
with model.trace(prompt):
    gu = model.layers[1].mlp.gate_up_proj.output.save()
    act = model.layers[1].mlp.down_proj.input.save()

gate, up = gu.chunk(2, dim=-1)
assert torch.equal(up * torch.nn.functional.silu(gate), act)
```

Gate neuron `j` and up neuron `j` are columns `j` and `intermediate_size + j` of the output.

## The 128k checkpoints rescale the rotary by sequence length

Phi-3-mini-128k, Phi-3.5-mini and Phi-4-mini set `rope_scaling` type `longrope` (Phi-3-medium-128k
spells it `su`, the same scheme): two lists of per-frequency factors, `short_factor` and
`long_factor`, over an `original_max_position_embeddings` of 4096. The rotary multiplies every cos
and sin by `sqrt(1 + ln(32) / ln(4096))`, 1.19, at every length, so a rotated query's norm is not
the projection's. A sequence of up to 4096 tokens is rotated with `short_factor`; a longer one
with `long_factor`, at every position. Position 5 of a 4097-token prompt is rotated differently
from position 5 of a 4096-token prompt, so `attention_queries` and `attention_keys` at the same
position move when the prompt crosses 4096 tokens. The 4k checkpoints and phi-4 have no scaling:
plain rotary at `rope_theta` 10000 (phi-4: 250000).

## Rotary covers whole heads, except on Phi-4-mini

Phi-3, Phi-3.5 and phi-4 rotate all of each head (`partial_rotary_factor` 1.0): 96 dimensions on
mini, 128 on medium and phi-4. Phi-4-mini and Phi-4-mini-reasoning set 0.75, so 96 of each
128-dimensional head turn, as `rotate_half` over those 96, and the last 32 carry no position. The
scale is `head_dim ** -0.5` on every checkpoint.

## The 4k checkpoints attend over a sliding window

Phi-3-mini-4k and Phi-3-medium-4k set `sliding_window` 2047, on every block: a query sees itself
and the 2046 positions before it, and `attention_probabilities` are exactly zero further back. A
prompt shorter than 2048 tokens is not affected. The 128k checkpoints set a window at or past
their 131072 positions (262144 on mini and Phi-4-mini, 131072 on medium), which never masks
anything, and phi-4 sets none. The window holds under the default `sdpa` load and under
`attn_implementation="eager"`, which the attention interior needs.

## Grouped-query attention on medium, phi-4 and Phi-4-mini

Phi-3-mini and Phi-3.5-mini have 32 key/value heads for 32 query heads, so no grouping. Phi-3-medium
and phi-4 have 10 for 40, and Phi-4-mini 8 for 24. `attention_keys` and `attention_values` are
served before `repeat_kv`, `num_kv_heads` wide, so an edit to key/value head `j` reaches query heads
`4j` to `4j + 3` on medium and phi-4, `3j` to `3j + 2` on Phi-4-mini.

## Three tokenizers, none of which adds a BOS

- Phi-3 and Phi-3.5: Llama's SentencePiece vocabulary, 32011 tokens with the chat tokens, in 32064
  embedding rows. `<s>` (id 1) is the `bos_token`, and the tokenizer does not prepend it. A word
  after a space splits as `' Paris'` → `['▁', 'Paris']`; encode `"Paris"` (id 3681, `▁Paris`) to get one
  token.
- phi-4 and its reasoning checkpoints: 100352 tokens in 100352 rows, a GPT-style byte-level BPE.
- Phi-4-mini: 200029 tokens in 200064 rows, also byte-level BPE.

Position 0 is the prompt's own first token on every checkpoint.

## The checkpoints

The sizes, from the configs:

- Phi-3-mini-4k and -128k, Phi-3.5-mini (3.8B): 32 blocks, hidden 3072, 32 heads × 96, MLP 8192
- Phi-3-medium-4k and -128k (14B): 40 blocks, hidden 5120, 40 heads × 128 over 10, MLP 17920
- Phi-4-mini-instruct and -reasoning (3.8B): 32 blocks, hidden 3072, 24 heads × 128 over 8, MLP 8192,
  tied embeddings
- phi-4, Phi-4-reasoning, -reasoning-plus (14B): 40 blocks, hidden 5120, 40 heads × 128 over 10,
  MLP 17920, a 16384-token context on phi-4 and 32768 on the reasoning checkpoints (`rope_theta`
  500000 there)

Phi-3-small (`microsoft/Phi-3-small-8k-instruct`, `-128k-instruct`) is another architecture,
`model_type` `phi3small`, and so is Phi-4-mini-flash-reasoning (`phi4flash`): neither loads as this family.
"""
