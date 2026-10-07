"""MPT: MosaicML's MPT-7B and MPT-30B and the checkpoints that load as transformers' own MptForCausalLM."""

MODEL_TYPE = "mpt"
TITLE = "MPT"
SUBTITLE = (
    "The block adds the attention to the stream, but the MLP takes the residual and adds it inside the module, "
    "so mlp_output is the tensor before that add; ALiBi biases on the scores stand in for rotary, queries, keys "
    "and values are thirds of one Wqkv, and no layer has a bias."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ybelkada/mpt-7b-bf16-sharded"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MptForCausalLM"
#: mosaicml's own repositories are gone from the Hub; the copies below carry MPT-7B's and MPT-30B's configs.
CHECKPOINTS = [
    "ybelkada/mpt-7b-bf16-sharded",
    "anas-awadalla/mpt-7b",
    "gl198976/mpt-7b-instruct",
    "Alchan/mpt-7b-chat",
    "eluzhnica/mpt-30b-peft-compatible",
    "team-lucid/mptk-1b",
    "mosaicml/mpt-7b",
    "ethzanalytics/mpt-7b-storywriter-sharded",
]

PALETTE = {"hue": 350}
VLLM = True
QUIRKS = ["residual-inside-module", "tuple-blocks", "own-attention-arithmetic", "fused-qkv", "layernorm"]

#: What the visualization draws: a sequential block with two bias-free LayerNorm pre-norms; the block adds the
#: attention, the MLP module adds the residual itself.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Native name norm_1: a LayerNorm with no bias.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, Wqkv, ALiBi",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native name norm_2: a LayerNorm with no bias. The MLP module (ffn) takes the stream it "
                             "normalizes as a second argument and adds it to its own output.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
    "identity_note": "Exact in every dtype: the block adds attention_output to the input, and the MLP module adds "
                     "mlp_output to that, the same order as the sum.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "wte is a plain lookup: no scale and no position embedding (transformers' MptModel does not read "
             "learned_pos_emb; position enters as ALiBi biases on the scores), so token_embeddings equals layers[0].input.",
    "norm": "norm_f is a LayerNorm with no bias; project_on_vocab applies it.",
    "head": "lm_head is tied to wte and has no bias. Nothing follows it: logits equals lm_head.output.",
}

NOTES = """
## The MLP adds the residual inside the module

```
h   = x + self_attn(norm_1(x))                 # the block adds the attention
out = ffn(norm_2(h), residual=h)               # returns h + mlp_output
```

The attention returns its `out_proj` output and the block adds it, so `attention_output` is the
module's output. The MLP takes the stream as a second argument and ends in `output + residual`,
so `mlp.output` is the block's output, equal to `layer_output`, and `mlp_output` is the tensor
before that add: `down_proj`'s output after its dropout. The sum is exact in every dtype.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()
with model.trace(prompt):
    raw = model.layers[1].mlp.output.save()

assert torch.equal(raw, x + attn + mlp)       # the module returns the stream
assert torch.equal(x + attn + mlp, out)
```

Zeroing `mlp.output` zeroes the stream that leaves the block, not the MLP's contribution;
ablate, steer or patch `mlp_output`. `mlp_output` is an operation inside the module, so in one
trace read it before `mlp.output`: the other order raises `OutOfOrderError` naming
`ffn.source.F_dropout_0.output`.

## Queries, keys and values are thirds of one projection

`Wqkv` maps `hidden_size` to `3 * hidden_size` with no bias, and `chunk(3)` cuts its output into
queries, keys and values, each then split into heads: the first third is every head's query, the
second every key, the third every value. No grouping: keys and values have `num_heads` heads.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    h = model.layers[1].input_layernorm.output.save()
    q = attn.attention_queries.save()

fused = h @ attn.Wqkv.weight.T                                   # [batch, seq, 3 * hidden]
q_part, k_part, v_part = fused.chunk(3, dim=-1)
heads = q_part.view(*q_part.shape[:2], model.num_heads, model.head_dim).transpose(1, 2)
torch.testing.assert_close(heads, q)
```

The three are views of the outputs of one `chunk`, and torch refuses an in-place edit of them
(`RuntimeError: Output 0 of Select is a view and is being modified inplace`). Assign instead:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    attn.attention_queries = attn.attention_queries * 0
    logits = model.logits.save()
```

## ALiBi biases replace rotary

There is no rotary embedding and no position embedding: queries and keys carry no position, and
the same token at two positions gives the same `attention_queries` at block 0. Position enters as
a bias on the scores: `attention_scores` is `q @ kᵀ / sqrt(head_dim)` plus
`slope_h * (j - (seq - 1))` for key position `j`, then the mask. The bias is 0 on the last key,
negative before it, and the same on every query row; after the softmax it is a penalty of
`slope_h` per token of distance. Head `h` has the slope `2 ** (-8 * (h + 1) / heads)`: on MPT-7B,
with 32 heads, 0.84 for head 0 down to 0.0039 for head 31. transformers builds the bias with
`alibi_bias_max` 8 whatever the config says. On a float32 load the scores recompute to rounding:

```python
from transformers.models.mpt.modeling_mpt import build_mpt_alibi_tensor

attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    scores = attn.attention_scores.save()

seq = q.shape[2]
alibi = build_mpt_alibi_tensor(model.num_heads, seq)                  # [heads, 1, seq]
again = q.float() @ k.float().transpose(-1, -2) / model.head_dim ** 0.5 + alibi
causal = torch.ones(seq, seq, dtype=torch.bool).tril()
torch.testing.assert_close(again[..., causal], scores[..., causal])
```

## The attention needs no flag, and its scores are float32

transformers has one attention for MPT, its own arithmetic, so a load without
`attn_implementation` serves all six interior values: `support()` reports none missing. The
scores are read where the softmax takes them, after an upcast, so `attention_scores` is float32
whatever the model's dtype; the masked entries hold the model dtype's minimum (`-3.39e38` under
bfloat16). The pattern is cast back to the values' dtype. The head outputs are the second
`matmul`, heads first in the forward, served `[batch, seq, heads, head_dim]`.

## What transformers' MptForCausalLM reads from the config

The class is transformers' own; the checkpoints' `auto_map` code is not run. It takes the sizes,
`attn_config.softmax_scale` and `attn_config.clip_qkv`, and builds every `Linear` without a bias
and every LayerNorm without one. The MLP is `4 * hidden_size` wide; every listed checkpoint sets
`expansion_ratio` to 4. `ethzanalytics/mpt-7b-storywriter-sharded` sets `clip_qkv` to the integer
6, which transformers' `MptAttentionConfig` refuses, so its config does not load.

## The readout

`lm_head` is `wte`'s matrix (tied) with no bias, and `logits` equals `lm_head.output`; `norm_f`
has no bias, and `project_on_vocab` on the last block's `layer_output` equals `logits`.
"""
