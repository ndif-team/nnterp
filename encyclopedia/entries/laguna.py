"""Laguna: each block with its own number of query heads, an attention gated per head after the interface, and a
mixture with a shared expert after a dense first block."""

MODEL_TYPE = "laguna"
TITLE = "Laguna XS / Laguna S / Laguna M"
SUBTITLE = (
    "Blocks with their own number of query heads, an attention whose head outputs are scaled by a softplus gate "
    "before o_proj, and a shared expert beside a sigmoid-routed mixture whose routed sum is scaled."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "poolside/Laguna-XS-2.1"
#: The tiny checkpoint the test suite builds the page from (two heads on both blocks; the suite's second copy,
#: with two and four, is what the per-block notes ran on).
PINNED = "hf-tiny-v2/tiny-random-LagunaForCausalLM"
#: The FP8, INT4 and NVFP4 copies and the DFlash drafters (DFlashLagunaForCausalLM) are left out.
CHECKPOINTS = [
    "poolside/Laguna-XS-2.1", "poolside/Laguna-XS.2", "poolside/Laguna-S-2.1",
    "poolside/Laguna-M.1", "poolside/Laguna-M.1-base",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 194}
VLLM = False
QUIRKS = [
    "per-block-sizes", "qk-norm", "output-gate", "sliding-window", "partial-rotary", "mixture-of-experts",
    "dense-first-blocks",
]

#: The smallest checkpoint has 33B parameters: nothing here ran on real weights. Every identity, shape and snippet
#: ran on tests/families/test_laguna.py's copy of the pinned tiny checkpoint with 2 and 4 query heads (block 0 full
#: and dense, block 1 sliding and a mixture of 8 experts, top 2; float32, eager); the sizes, block layouts and rotary
#: settings are the Hub configs', read on meta builds.

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on the first blocks, a mixture after).
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
            "detail": "{num_kv_heads} kv × {head_dim}, q/k normed, gated",
            "variants": {
                "full_attention": "{num_heads} heads × {head_dim}, full, gated",
                "sliding_attention": "{num_kv_heads} kv, window {sliding_window}, gated",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "part_order": ["shared", "router", "experts"],
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "shared_expert_output", "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}, 1 shared",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends id 2, 〈|EOS|〉.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # q/k norms, a softplus gate
out = h + mlp(post_attention_layernorm(h))    # dense first, then shared + 8 of 256
```

Llama's pre-norm block and Llama's names. `mlp_layer_types` makes the first block dense on XS and S
and the first three on M.1; the rest are mixtures. The block adds each sublayer's output to the
stream and returns a tensor, so the identity is the plain sum on both block kinds:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Each block has its own number of query heads

`num_attention_heads_per_layer` sets each block's query heads: 48 on the full blocks and 64 on the
sliding ones on XS-2.1 and XS.2, 48 and 72 on S-2.1, 64 on every block of M.1. Every block has 8
key/value heads of 128 dimensions. The root's `num_heads` is the config's `num_attention_heads`, the
full blocks' count; each block's own is on its attention:

```python
model.num_heads                         # 48 on XS-2.1
model.layers[1].self_attn.num_heads     # 64: block 1 slides
model.layers[0].self_attn.num_heads     # 48: block 0 is full
```

`attention_queries`, the pattern and `attention_head_outputs` have the block's own head count, so
the pattern is `[batch, 64, seq, seq]` on a sliding block of XS-2.1 and `[batch, 48, seq, seq]` on
a full one: stack them per block, not across blocks. A key/value head serves 6 query heads on a full
block and 8 (XS) or 9 (S) on a sliding one.

## Sliding and full blocks differ in window and rotary

On XS and S, `layer_types` makes every fourth block full (0, 4, 8, ...) and the others sliding, with
a window of 512 positions; every block of M.1 is full. A full block's rotary is YaRN with base
500,000 over the first half of each head (`partial_rotary_factor` 0.5; the whole head on M.1); a
sliding block's is plain rotary with base 10,000 over the whole head. The YaRN factor is 32 on
XS-2.1, 64 on XS.2 and M.1 and 128 on S-2.1. `q_norm` and `k_norm` are RMSNorms over each head
(`head_dim` wide), applied before the rotary; `attention_queries` and `attention_keys` are read after
both, so an edit to one head's slice of `q_proj.output` reaches that head only.

## The head outputs are gated before o_proj

After the interface, the attention multiplies each head's output by `softplus(g_proj(x))`, where `x`
is the attention's input. On XS-2.1, XS.2 and S-2.1 the gate has one value per head per token
(`gating` is `per-head` or true); on M.1 one per channel (`per-element`). `attention_head_outputs` is the ungated
interface output; `o_proj.input` is the gated one, and `attention_output` is `o_proj` of it:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    heads = attn.attention_head_outputs.save()  # [batch, seq, heads, head_dim], ungated
    g = attn.g_proj.output.save()               # [batch, seq, heads] when per-head
    gated = attn.o_proj.input.save()

gate = torch.nn.functional.softplus(g.float()).to(heads.dtype)
torch.testing.assert_close((heads * gate.unsqueeze(-1)).flatten(2), gated)
```

A head's term in `attention_output` is its gate times `o_proj`'s slice applied to its output, so
zeroing a head at `attention_head_outputs` removes the term, and the gate is a positive per-token
weight on each head.

## Load with eager for the attention interior

The attention interior needs `attn_implementation="eager"`; the default load runs `sdpa`. There is no
softcap or sink, and on the tiny checkpoint the two loads' logits differ by 1e-7 in float32.

## The mixture: a shared expert first, then sigmoid routing with a selection bias

Each mixture runs its shared expert (`mlp.shared_experts`, `shared_expert_intermediate_size` wide)
first, on the same input as the router. `router_logits` is the router's projection in float32 (before
the tanh softcap `moe_router_logit_softcapping`, which is 0, off, on every checkpoint). The router adds
its selection bias, `mlp.router.e_score_correction_bias`, to the sigmoids to choose; `expert_weights`
are the chosen experts' sigmoids without the bias, renormalized to sum to one. The experts' weighted
sum is then multiplied by `moe_routed_scaling_factor` (2.5 on XS and S, 1.0 on M.1): that product is
`routed_output`, and `mlp_output` is `routed_output + shared_expert_output`. Read
`shared_expert_output` first in a trace:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    shared = moe.shared_expert_output.save()    # first: it runs before the router
    logits = moe.router_logits.save()           # [batch, seq, num_experts], float32
    w = moe.expert_weights.save()               # [batch, seq, top_k]
    idx = moe.expert_indices.save()
    each = moe.expert_outputs.save()            # [batch, seq, top_k, hidden]
    routed = moe.routed_output.save()
    out = moe.mlp_output.save()

chosen = logits.sigmoid().gather(-1, idx)
scale = model.config.moe_routed_scaling_factor
torch.testing.assert_close(w, (chosen / chosen.sum(-1, keepdim=True)).to(w.dtype))
torch.testing.assert_close(each.sum(2) * scale, routed)
torch.testing.assert_close(routed + shared, out)
```

`expert_outputs` are the weighted outputs before the scale, so on XS and S a slot adds 2.5 times its
`expert_outputs` entry to the stream. A written `router_logits` moves both the choice and the
weights; a written bias moves only the choice. The shared expert is an `Mlp` whose own `mlp_output`
is unavailable: what the block adds is the mixture's.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other weights as they were. `mlp_output` changes on exactly the tokens that chose `e`, and the
shared expert's term stays:

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

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. The final norm is a plain RMSNorm whose gain is `norm.weight`.
The tokenizer prepends id 2, `〈|EOS|〉`, which is also an end token, so `model.input_ids[:, 0]` is 2;
the configs' end tokens are ids 2 and 24 (`</assistant>`).
"""
