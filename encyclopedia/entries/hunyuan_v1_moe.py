"""Hunyuan MoE V1 (Hunyuan-A13B): Hunyuan's attention and a mixture of 64 experts plus a shared expert on every block."""

MODEL_TYPE = "hunyuan_v1_moe"
TITLE = "Hunyuan-A13B"
SUBTITLE = (
    "Hunyuan's block, query and key heads RMS-normed after the rotary embedding, with a mixture on "
    "every block: a shared expert that runs first, plus 8 of 64 routed experts whose weights are "
    "renormalized to sum to one."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tencent/Hunyuan-A13B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HunYuanMoEV1ForCausalLM"
#: Hunyuan-A13B-Pretrain's config says ``model_type`` ``hunyuan`` and needs its own remote code: listed, greyed.
CHECKPOINTS = ["tencent/Hunyuan-A13B-Instruct", "tencent/Hunyuan-A13B-Pretrain"]

#: Hunyuan lineage: hunyuan_v1_dense is 29; the hash of hunyuan_v1_moe (316) sits among the Qwen MoE hues.
PALETTE = {"hue": 33}
VLLM = False
QUIRKS = ["qk-norm", "mixture-of-experts"]

#: No checkpoint of this family is small enough to run here (A13B has 80B parameters): the shapes, identities,
#: dtypes and read orders in the notes were checked on the pinned tiny checkpoint (8 experts, top 2) in
#: float32 and bfloat16; the sizes come from A13B's config and its stored weights' metadata.

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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the mixture's input norm.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, 1 shared",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))           # query_layernorm, key_layernorm inside
n   = post_attention_layernorm(h)
out = h + shared_mlp(n) + experts(n, gate(n))     # one shared expert, then 8 of 64 routed
```

Llama's pre-norm block and Llama's names, `post_attention_layernorm` included (the mixture's input
norm). Every block's `mlp` is a mixture; none is dense. The block adds the attention's output and
the mixture's, so the identity is the plain sum, exact in bfloat16:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## The shared expert runs first, and mlp_output includes it

The mixture runs `shared_mlp` (`shared_experts` in nnterp's names) on every token before the router,
then adds the routed sum to it: `mlp_output` is `routed_output + shared_expert_output`, exactly. Read
`shared_expert_output` before `router_logits` in a trace; the other order fails as out of order.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    shared = moe.shared_expert_output.save()     # first: it runs before the router
    logits = moe.router_logits.save()
    routed = moe.routed_output.save()
    mlp = moe.mlp_output.save()

torch.testing.assert_close(routed + shared, mlp)
```

The shared expert and each routed expert are SwiGLU MLPs `intermediate_size` wide, 3072 on A13B.
The config also carries `moe_intermediate_size` (3072 on each block) and `num_shared_expert` (1 on
each block); the module reads neither, and builds one shared expert `intermediate_size` wide.
Zeroing `shared_expert_output` leaves `mlp_output` equal to `routed_output`; the routing does not
depend on it.

## The router runs in float32 and its 8 weights sum to one

The router `gate` projects with a child `nn.Linear`, `wg`, whose weight is stored in float32 and
stays float32 in a bfloat16 load; `router_logits` is that projection's output, `[batch, seq, 64]`.
The router takes a softmax over all 64 logits in float32, keeps the 8 largest and divides them by
their sum, whatever the config's `norm_topk_prob` says. `expert_weights` are those renormalized
weights, slot 0 the largest, and they and `expert_outputs` are float32 in a bfloat16 model;
`routed_output` and `mlp_output` are in the model's dtype.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts], float32
    w = moe.expert_weights.save()           # [batch, seq, top_k], float32
    idx = moe.expert_indices.save()

top = logits.softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values / top.values.sum(-1, keepdim=True), w)
assert torch.equal(top.indices, idx)
```

`moe_topk` is a list in the config, one entry per block, and each block's router reads its own;
on A13B every entry is 8. `top_k` is the block's value.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other weights as they were: they are not renormalized again, so the token's routed terms
then sum with weights below one, while its shared expert term is unchanged. `mlp_output` changes on
exactly the tokens that chose `e`:

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

## Attention: per-head query and key norms, grouped heads

The attention is Hunyuan's dense one: `query_layernorm` and `key_layernorm` are RMSNorms 128 wide,
applied to each head's rotated query and key, and `attention_queries` and `attention_keys` are read
after them, so scale a head's query at `attention_queries`, not at `q_proj.output`, where the norm
removes it. A13B has 32 query heads over 8 key/value heads, `head_dim` 128 and a query scale of
`head_dim ** -0.5`; an edit to key/value head `j` reaches query heads `4j` to `4j + 3`. The rotary
base is `rope_theta * alpha ** (head_dim / (head_dim - 2))`, 1.116e7 with `alpha` 1000. The interior
needs `attn_implementation="eager"`.

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. A13B stores no `lm_head` weight: the head is `embed_tokens`
(`tie_word_embeddings`), so an edit to `embed_tokens.weight` is an edit to the unembedding. The
tokenizer prepends nothing to a plain prompt; the chat template starts with `<|startoftext|>`.

## What this family module covers

The module serves `model_type` `hunyuan_v1_moe`, which is `tencent/Hunyuan-A13B-Instruct` (80B
parameters, 13B active per token, 32 blocks). `Hunyuan-A13B-Pretrain` names the same
architecture class but says `model_type` `hunyuan`, which loads only with its own remote code.
The dense Hunyuan models are `hunyuan_v1_dense`.
"""
