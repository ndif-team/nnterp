"""DBRX: a pre-norm block whose attention and both norms sit in one wrapper module, a fused and clamped
Wqkv, LayerNorms without bias, and a mixture of 16 experts on every block."""

MODEL_TYPE = "dbrx"
TITLE = "DBRX"
SUBTITLE = (
    "A pre-norm block whose attention sits with both norms inside norm_attn_norm, queries, keys and "
    "values from one clamped Wqkv projection, and a mixture of 16 experts, 4 per token, on every block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Databricks' own repositories are not on the Hub; the page reads a re-upload's config.
REFERENCE = "Undi95/dbrx-base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/dbrx-tiny256-random"
CHECKPOINTS = [
    "databricks/dbrx-base", "databricks/dbrx-instruct",
    "Undi95/dbrx-base", "alpindale/dbrx-instruct",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 90}
VLLM = False
QUIRKS = ["fused-qkv", "layernorm", "mixture-of-experts"]

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
            "pre_norm_note": "norm_attn_norm.norm_1: a LayerNorm without bias, inside the module that also holds "
                             "the attention and the FFN's norm.",
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
            "pre_norm_note": "norm_attn_norm.norm_2: the FFN's input norm. It sits inside norm_attn_norm, after the "
                             "attention's residual add, and the module returns the stream and this norm's output.",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup (transformer.wte): token_embeddings equals layers[0].input.",
    "norm": "transformer.norm_f, a LayerNorm without bias: it subtracts the mean, divides by the standard "
            "deviation and multiplies by norm.weight. project_on_vocab applies it.",
    "head": "lm_head has its own weight (tie_word_embeddings is false).",
}

NOTES = """
## The block, in order

```
h, a = norm_attn_norm(x)    # a = norm_2(h), h = x + attn(norm_1(x))
out  = h + ffn(a)           # 4 of 16 experts per token
```

The native tree is `transformer.blocks[i].{norm_attn_norm.{norm_1, attn, norm_2}, ffn}`, and nnterp
names them `input_layernorm`, `self_attn`, `post_attention_layernorm` and `mlp` on the block, so a
recipe reads DBRX like Llama. `norm_attn_norm` norms, runs the attention, adds the residual, norms
again, and returns both the stream after the attention and the FFN's input; the block adds the
FFN's output to the first. The attention module's own output is its contribution, the block
returns a tensor, and the identity is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mid = model.layers[1].norm_attn_norm.output.save()   # (h, a)
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(mid[0], x + attn)
torch.testing.assert_close(x + attn + mlp, out)
```

## One projection gives queries, keys and values, clamped

`self_attn.Wqkv` maps the normed stream to `[queries | keys | values]`, `hidden_size`,
`num_kv_heads × head_dim` and `num_kv_heads × head_dim` wide (6144, 1024, 1024: 48 query heads and
8 key/value heads of 128). The forward clamps the whole output to `[-clip_qkv, clip_qkv]`
(`attn_config.clip_qkv`, 8) and then splits it, so `Wqkv.output` is before the clamp and
`attention_queries`, `attention_keys` and `attention_values` after it (queries and keys also after
rotary, `rope_theta` 500,000). Splitting the clamped output by those widths gives the values exactly:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    qkv = attn.Wqkv.output.save()           # before the clamp
with model.trace(prompt):
    v = attn.attention_values.save()        # [batch, kv_heads, seq, head_dim]

H, KV, D = model.num_heads, model.num_kv_heads, model.head_dim
clip = model.config.attn_config.clip_qkv
q, k, val = qkv.clamp(-clip, clip).split([H * D, KV * D, KV * D], dim=-1)
torch.testing.assert_close(val.view(*val.shape[:2], KV, D).transpose(1, 2), v)
```

An edit to a slice of `Wqkv.output` passes through the clamp, so a value written beyond ±8 arrives
as ±8; edit `attention_queries`, `attention_keys` or `attention_values` to set a head's tensor as
it is used. Query head `h` reads key/value head `h // 6`: an edit to one key/value head reaches 6
query heads. The output projection is `out_proj`, and no projection has a bias.

## Load with eager for the attention interior only

The attention runs on transformers' shared interface: the six interior values need
`attn_implementation="eager"`, and the default `sdpa` load computes the same attention without
them. The attention has no softcap, sink or window.

## The norms are LayerNorms without bias

`input_layernorm`, `post_attention_layernorm` and `norm` are `nn.LayerNorm` with `bias=False`: each
subtracts the mean over the hidden axis, divides by the standard deviation and multiplies by its
`weight`. A shift of the stream along the all-ones direction reaches no sublayer and no logit.

## The router renormalizes its top 4 to sum to one

The router is a bias-free linear map, `mlp.router.layer`, whose output is `router_logits`. The FFN
takes a softmax over all 16 logits, keeps the 4 largest, and divides them by their sum (an L1 norm,
`ffn_config.moe_normalize_expert_weights` 1), so `expert_weights` sum to one at every token. There is
no shared expert, so `routed_output` is `mlp_output`.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts]
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

top = logits.softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values / top.values.sum(-1, keepdim=True), w)
assert torch.equal(top.indices, idx)
```

The experts are one module, `mlp.experts`, whose GLU weights `w1` (gate), `v1` (up) and `w2`
(down) are stacked parameters on `experts.mlp`, `[num_experts × ffn_hidden_size, hidden]` each,
with SiLU. It loops over the experts in its own forward, so no tensor holds a slot's output and
`expert_outputs` is unavailable. `intermediate_size` is one expert's width,
`ffn_config.ffn_hidden_size` (10752); the config has no dense width. `moe_jitter_eps` scales the
router's input by noise in training only.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other 3 weights as they were, so the token's weights then sum to less than one.
`mlp_output` changes on exactly the tokens that chose `e`:

```python
moe, e = model.layers[1].mlp, 3
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

changed = (ablated != clean).any(-1)                       # [batch, seq]
assert torch.equal(changed, (idx == e).any(-1))            # only the tokens that chose e
```

## The readout and the checkpoints

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `databricks/dbrx-base` and `databricks/dbrx-instruct` are not on
the Hub; the page reads the configs of `Undi95/dbrx-base` and `alpindale/dbrx-instruct`, whose
sizes agree. Their tokenizer is remote code over `tiktoken`: load it with
`trust_remote_code=True` and `tiktoken` installed, and hand it to `StandardizedTransformer` as
`tokenizer=`.
"""
