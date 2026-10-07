"""Qwen2-MoE: Qwen1.5-MoE and Qwen2-57B-A14B, Qwen2's attention with a mixture and a gated shared expert."""

MODEL_TYPE = "qwen2_moe"
TITLE = "Qwen1.5-MoE / Qwen2-MoE"
SUBTITLE = (
    "Qwen2's attention, with a bias on the query, key and value projections, and a mixture of experts on "
    "every block whose top-k softmax weights are not renormalized, beside a shared expert scaled by a "
    "sigmoid gate of its own."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen1.5-MoE-A2.7B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
CHECKPOINTS = [
    "Qwen/Qwen1.5-MoE-A2.7B", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "Qwen/Qwen2-57B-A14B", "Qwen/Qwen2-57B-A14B-Instruct",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 285}
VLLM = True
QUIRKS = ["qkv-bias", "mixture-of-experts", "unnormalized-routing"]

#: Every number in the notes comes from a checkpoint's config or tokenizer; the shapes, identities, read
#: orders and snippets were run on the pinned tiny checkpoint (4 experts, top 2, 2 blocks), in float32 and
#: bfloat16. No real weights were run: the smallest checkpoint has 14B parameters.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k, v biased",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "part_order": ["shared", "router", "experts"],
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. No BOS is prepended, "
             "so position 0 holds the text's first token.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is "
            "lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h      = x + self_attn(input_layernorm(x))       # q, k, v projections biased
y      = post_attention_layernorm(h)
shared = shared_expert(y)                        # runs first
routed = experts(y, router(y))                   # 4 of 60 experts per token (1.5-MoE)
out    = h + routed + sigmoid(shared_expert_gate(y)) * shared
```

Qwen2's pre-norm block with Llama's names; the mixture's output is the routed sum plus the gated
shared expert, and nothing norms a sublayer's output. The identity is the plain sum, and inside
the mixture `routed_output + shared_expert_output` is `mlp_output`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    mlp = moe.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(routed + shared, mlp)
torch.testing.assert_close(x + attn + mlp, out)
```

## The shared expert is gated by a sigmoid

`shared_experts` (native `shared_expert`) is a SwiGLU MLP `shared_expert_intermediate_size` wide
(5632 on 1.5-MoE, 20480 on 57B-A14B) that every token runs through. `shared_expert_gate` is a
linear map from the stream to one number per token; the mixture multiplies the shared expert's
output by its sigmoid. `shared_expert_output` is that product, what the mixture adds;
`moe.shared_experts.output` is the ungated output:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    ungated = moe.shared_experts.output.save()
    gate = moe.shared_expert_gate.output.save()      # [batch * seq, 1], before the sigmoid
    shared = moe.shared_expert_output.save()

torch.testing.assert_close(torch.sigmoid(gate).view(*shared.shape[:-1], 1) * ungated, shared)
```

Zeroing `shared_expert_output` removes the shared expert and leaves `mlp_output` equal to
`routed_output`; zeroing `shared_experts.output` does the same through the product.

## Read order: the shared expert runs before the router

The mixture's forward calls `shared_expert` first, then the router and the experts, and computes
the gated product last. In one trace `moe.shared_experts.output` comes before `router_logits`,
and `shared_expert_output` after `routed_output`; the other orders raise `OutOfOrderError`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    ungated = moe.shared_experts.output.save()
    logits = moe.router_logits.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
```

## The routed weights are the softmax's top-k, not renormalized

The router (`gate`, aliased `router`) takes a softmax over every expert's logit in float32 and
keeps the largest `num_experts_per_tok`; `norm_topk_prob` is false on every checkpoint here, so
`expert_weights` are those probabilities as they are, cast back to the model's dtype, and a
token's weights sum to less than one:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts], before the softmax
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

top = logits.float().softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values.to(w.dtype), w)    # the softmax's own entries
assert torch.equal(top.indices, idx)                     # slots in descending weight
```

1.5-MoE has 60 experts with 4 per token, each `moe_intermediate_size` 1408 wide; 57B-A14B has
64 with 8 per token, 2560 wide. The shared expert's gate does not depend on the routed weights,
so the balance between the routed sum and the shared expert moves with how much softmax mass
the router puts on the experts it chose.

## Ablating an expert

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term. `mlp_output`
changes on exactly the tokens that chose `e`, and an expert no token chose has no effect:

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

This holds on the pinned tiny checkpoint for every expert, in float32 and in bfloat16.
`expert_outputs` is read inside transformers' `grouped_mm` (the default) and `batched_mm` experts
forwards; under `experts_implementation="eager"` `support()` reports it missing and names the kwarg.

## Every block is a mixture

A block's `mlp` is a mixture where its index is not in `mlp_only_layers` and
`(i + 1) % decoder_sparse_step == 0`. No checkpoint here sets `mlp_only_layers` and every one has
`decoder_sparse_step` 1, so all 24 blocks of 1.5-MoE and 28 of 57B-A14B are mixtures. The config's
`intermediate_size` (5632, 18944) is the width of a dense MLP no block has;
`model.layers[i].mlp.intermediate_size` reads the experts' `moe_intermediate_size`.

## Attention

`q_proj`, `k_proj` and `v_proj` add a bias (`qkv_bias` is true by default and no checkpoint sets
it); `o_proj` does not. The bias is added before the rotary embedding, so it turns with the
position. 1.5-MoE has 16 query heads and 16 key/value heads, no grouping;
57B-A14B has 28 over 4, so an edit to key/value head `j` reaches query heads `7j` to `7j + 6`.
`head_dim` is 128 on both. Every checkpoint sets `use_sliding_window` false, so every entry of
`layer_types` is `full_attention` and no block has a window, whatever `sliding_window` says. The
attention interior needs `attn_implementation="eager"`.

## Tokenizer and chat template

The tokenizer prepends nothing (`bos_token` is `None`), so position 0 is the text's first token.
Every checkpoint ships a ChatML template that writes a system turn when given none: `You are a
helpful assistant` on 1.5-MoE-A2.7B, the same with a full stop on the Chat and Qwen2 models.

## The readout

`model.logits` equals `model.lm_head.output`, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `lm_head` and `embed_tokens` are separate weights on every
checkpoint here.
"""
