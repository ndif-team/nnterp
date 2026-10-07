"""JetMoE: Llama's pre-norm block in which the attention is itself a mixture of experts (each expert a query
and an output projection over shared keys and values) and the MLP a mixture of experts with a bias."""

MODEL_TYPE = "jetmoe"
TITLE = "JetMoE"
SUBTITLE = (
    "Llama's pre-norm block in which the attention is a mixture too: per token a router picks 2 of 8 "
    "attention experts, each with its own query and output projections over shared keys and values."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "jetmoe/jetmoe-8b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-JetMoeForCausalLM"
#: jetmoe/jetmoe-8b-chat is left out: its config names its sizes n_layer, n_head and n_embd, which
#: transformers' JetMoeConfig does not read, so it builds with the default 12 blocks, not 24.
CHECKPOINTS = ["jetmoe/jetmoe-8b", "jetmoe/jetmoe-8b-sft"]

#: No kin among the entries; the hash of jetmoe (0) sits on deepseek_v3's 2. 26 is the middle of the gap
#: between mamba (17) and gpt2 (36).
PALETTE = {"hue": 26}
VLLM = False
QUIRKS = ["mixture-of-experts"]

#: The real checkpoints are 8B parameters: nothing here ran on real weights. Every identity and shape in
#: the notes, and every snippet, ran on the pinned tiny checkpoint (2 blocks, 4 experts, top 2, 2 key/value
#: heads); the sizes are jetmoe-8b's config.

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
            "detail": "MoA: top {num_experts_per_tok} of {num_local_experts} × {num_kv_heads} heads",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_indices", "expert_weights", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends <s> (id 1).",
    "head": "On jetmoe-8b lm_head shares its weight with embed_tokens (tie_word_embeddings).",
}

NOTES = """
## The block, in order

```
h   = x + self_attention(input_layernorm(x))   # 2 of 8 attention experts per token
out = h + mlp(post_attention_layernorm(h))     # 2 of 8 MLP experts per token, + bias
```

Llama's pre-norm block, with the attention under `self_attention`, aliased `self_attn`. Both
sublayers are mixtures with routers of their own. The attention returns
`(attn_output, attn_weights, router_logits)` and `attention_output` is the first element; the
block returns a tensor, and the identity is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## The attention is a mixture of experts

`self_attn.experts` holds 8 attention experts, each a query projection (`experts.input_linear`,
`hidden → num_kv_heads × head_dim`) and an output projection (`experts.output_linear`, back to
`hidden`); there is no `q_proj` or `o_proj`. Per token, `experts.router` takes the top 2 of 8
logits and a softmax over those 2. Each chosen expert projects the token's queries for 16 heads,
so a token has 2 × 16 = 32 query heads (`num_heads`), and slot `s` holds heads `16 s` to `16 s + 15`.
Keys and values come from one shared `kv_proj`, 16 heads of 128, and are tiled (`repeat`, not
interleaved) to all 32 heads: head `h` is slot `h // 16` over key/value head `h % 16`.

`attention_output` is the gated sum of the chosen experts' output projections, each over its
slot's 16 head outputs, plus the mixture's bias `experts.bias`. nnterp serves no standard value
for the attention's routing; its logits are `experts.router.layer.output`, flat over tokens
(`[batch × seq, num_local_experts]`), and the third element of `self_attn.output`:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    logits = attn.experts.router.layer.output.save()   # [batch * seq, experts]
with model.trace(prompt):
    heads = attn.attention_head_outputs.save()         # [batch, seq, num_heads, head_dim]
    out = attn.attention_output.save()

k, kv = model.config.num_experts_per_tok, model.num_kv_heads
top = logits.topk(k, dim=-1)
gates, idx = top.values.softmax(-1), top.indices      # [batch * seq, k]
experts = attn._module.experts
W = experts.output_linear.weight[idx]                 # each slot's expert
slots = heads.reshape(-1, k, kv * model.head_dim, 1)  # each slot's 16 heads
terms = (W @ slots).squeeze(-1)                       # [batch * seq, k, hidden]
manual = (gates[..., None] * terms).sum(1) + experts.bias
torch.testing.assert_close(manual.view_as(out), out)
```

`attention_keys` and `attention_values` have `num_heads` heads, two copies of the 16. The copies
are separate tensors, so an edit to head `h` of `attention_keys` reaches slot `h // 16`'s queries
only; edit key/value head `j` at heads `j` and `j + 16`, or edit `kv_proj.output`, to reach every
query that reads it. The expert behind a query head changes from token to token, so a query head
index names a slot, not an expert.

## Ablating an attention expert

Zeroing the head outputs of the slots that chose expert `e` removes exactly that expert's term
from `attention_output` at those tokens, since its output projection has no bias; the other
slot's term and `experts.bias` stay. `attention_output` changes on exactly the tokens whose
attention router chose `e`:

```python
attn, e = model.layers[1].self_attn, 3
k, kv = model.config.num_experts_per_tok, model.num_kv_heads
with model.trace(prompt):
    logits = attn.experts.router.layer.output.save()
    clean = attn.attention_output.save()
idx = logits.topk(k, dim=-1).indices.view(*clean.shape[:2], k)   # [batch, seq, k]
with model.trace(prompt):
    for s in range(k):
        attn.attention_head_outputs[:, :, s * kv:(s + 1) * kv][idx[:, :, s] == e] = 0
    ablated = attn.attention_output.save()

changed = (ablated != clean).any(-1)
assert torch.equal(changed, (idx == e).any(-1))
```

## Loading: eager for the attention interior

The attention runs on transformers' shared interface: the six interior values need
`attn_implementation="eager"`, and the default `sdpa` load computes the same attention without
them. The head width is `kv_channels` (128), so `head_dim` is 128 while `hidden_size / num_heads`
is 64: the 32 query heads are 4096 wide in all, twice the stream.

## The MLP: a top-2 softmax router, a bias, and a read order

The MLP's router (`mlp.router`) takes the top 2 of its 8 logits (`mlp.router.layer.output`) and a
softmax over those 2, so `expert_weights` sum to one. It then sorts the slots by expert and the
mixture runs `input_linear` (gate and up, `2 × intermediate_size` wide, the first half through
SiLU) and `output_linear` over the sorted slots, so `expert_outputs` is unavailable.
`expert_indices` and `expert_weights` are read before the sort, in token order, and
`routed_output` is the routed sum before the mixture adds its bias:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    routed = moe.routed_output.save()
    out = moe.mlp_output.save()

torch.testing.assert_close(out - routed, moe._module.bias.expand_as(out))
```

The router computes the indices before the weights, so a trace reads `router_logits`, then
`expert_indices`, then `expert_weights`. An ablation reads the indices first:

```python
moe, e = model.layers[1].mlp, 3
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    hit = moe.expert_indices == e                              # indices first
    moe.expert_weights = moe.expert_weights.masked_fill(hit, 0)
    ablated = moe.mlp_output.save()

changed = (ablated != clean).any(-1)
assert torch.equal(changed, (idx == e).any(-1))            # only the tokens that chose e
```

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. On jetmoe-8b `lm_head` and `embed_tokens` share one weight. The
tokenizer prepends `<s>` (id 1), so `model.input_ids` starts with it.
"""
