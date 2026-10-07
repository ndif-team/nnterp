"""GraniteMoE SWA: Granite Swash's attention in GraniteMoE's scaled block, with a shared expert beside the mixture."""

MODEL_TYPE = "granitemoe_swa"
TITLE = "Granite Swash MoE"
SUBTITLE = (
    "Granite Swash's sliding-window attention with its sink in GraniteMoE's scaled block, where a shared expert "
    "runs beside the mixture of experts and mlp_output carries the mixture's term alone."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-swash-3b-a600m"
#: The tiny checkpoint the test suite builds the page from (it has no shared expert: shared_intermediate_size 0).
PINNED = "hf-tiny-v2/tiny-random-GraniteMoeSWAForCausalLM"
CHECKPOINTS = ["ibm-granite/granite-swash-3b-a600m"]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 214}
VLLM = False
QUIRKS = ["scaled-residual-adds", "sliding-window", "sink-scaled-heads", "mixture-of-experts", "embedding-multiplier", "scaled-logits"]

#: Numbers in the notes come from granite-swash-3b-a600m's weights in bfloat16 on an RTX A6000 unless they say
#: float32; the snippets also ran on the pinned tiny checkpoint and on a copy of it with Granite's multipliers away
#: from 1.0 and a shared expert (shared_intermediate_size 16).
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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}, sink",
            "variants": {
                "sliding_attention": "window of {sliding_window} tokens, sink",
                "full_attention": "full causal attention, sink",
            },
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the mixture's input norm, "
                             "and the router, every expert and the block's shared_mlp read its output.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
    "identity": "layers[i].input + self_attn.attention_output + mlp.mlp_output + shared_mlp.output * residual_multiplier "
                "== layer_output",
    "identity_note": "attention_output and mlp_output are the modules' outputs times residual_multiplier; the block's "
                     "shared_mlp is in neither, so its scaled output is the third term. Exact in float32 on the pinned "
                     "checkpoint given a shared expert; without one (shared_intermediate_size 0) the term drops out.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier (12) before block 0, "
             "so layers[0].input is token_embeddings · 12.",
    "layers": "Each block adds its attention's output and its mixture's plus its shared expert's output, each times "
              "residual_multiplier (0.26).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "logits = lm_head.output / logits_scaling (5). project_on_vocab divides too.",
}

NOTES = """
## The block, in order

```
x      = embed_tokens(ids) * embedding_multiplier
h      = x + self_attn(input_layernorm(x)) * residual_multiplier
n      = post_attention_layernorm(h)
out    = h + (block_sparse_moe(n) + shared_mlp(n)) * residual_multiplier
logits = lm_head(norm(out)) / logits_scaling
```

The attention is Granite Swash's: of granite-swash-3b-a600m's 28 blocks, 8 attend over the whole
prefix (0, 3, 7, 11, 15, 19, 23 and 27) and 20 over the last 128 tokens, every block applies
rotary embeddings with base 10000, and each head carries a sink that scales its output. The
feed-forward is a mixture of 48 experts, 512 wide, with the top 4 routed per token, and beside it
`shared_mlp`, a SwiGLU 1280 wide that runs on every token. The multipliers:
`embedding_multiplier` 12, `residual_multiplier` 0.26, `attention_multiplier` 1/64 and
`logits_scaling` 5.

## mlp_output leaves the shared expert out

The mixture's native name, `block_sparse_moe`, is `mlp`, and `mlp_output` is its output times
`residual_multiplier`. The block adds the shared expert's output times the multiplier too, and no
standard value carries that term: `shared_expert_output` is unavailable ("this mixture has no
shared expert"), and `layers[i].shared_mlp` keeps its native name. The block's sum therefore has a
third term:

```python
layer = model.layers[1]
with model.trace(prompt):
    x = layer.input.save()
    attn = layer.self_attn.attention_output.save()
    mlp = layer.mlp.mlp_output.save()
    shared = layer.shared_mlp.output.save()
    out = layer.layer_output.save()

m = model.config.residual_multiplier                 # 0.26
torch.testing.assert_close(x + attn + mlp + shared * m, out)
```

The term is not small. Over the positions after the first, the shared expert's scaled output has
a mean norm of 3.5 in block 1 against 3.4 for `mlp_output`, and 2.4 against 2.0 in block 14;
`x + attn + mlp` misses `layer_output` by up to 0.53 in block 1 and 8.1 in block 14. Decompose the
stream, attribute or ablate the feed-forward with both terms: a write to `mlp_output` replaces the
mixture's term and leaves the shared expert's, so zeroing the whole feed-forward means zeroing
`mlp_output` and `shared_mlp.output`.

## The router: top 4 of the logits, then a softmax over those 4

The router is one linear map without a bias. It takes the top 4 of its 48 logits and softmaxes
those 4 alone, so a token's `expert_weights` sum to one, and `expert_outputs` (each slot's
weighted expert output) sum to `routed_output`, which is `mlp.output`, unscaled:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()        # [batch, seq, 48]
    w = moe.expert_weights.save()            # [batch, seq, 4]
    idx = moe.expert_indices.save()
    slots = moe.expert_outputs.save()        # [batch, seq, 4, hidden]
    routed = moe.routed_output.save()

top, chosen = logits.topk(moe.top_k, dim=-1)
torch.testing.assert_close(chosen, idx)
torch.testing.assert_close(top.float().softmax(-1).to(w.dtype), w)
torch.testing.assert_close(slots.sum(2), routed)
```

The experts are stored fused: `experts.gate_up_proj` is `[48, 1024, 1280]`, each expert's gate
rows then its up rows, and `experts.down_proj` is `[48, 1280, 512]`. The default
`experts_implementation`, `grouped_mm`, serves `expert_outputs`.

## The sink scales the head outputs

As on Granite Swash, each head's output is multiplied after the softmax by
`sigmoid(logsumexp(scores) - sink)`, with the sink in `self_attn._module.sinks`, so
`attention_probabilities` rows sum to one and `attention_head_outputs` are read after the scale.
On a code prompt the median of that share over heads and queries is 0.84 in block 0, and 0.14 to
0.25 in blocks 1, 3 and 27. A pattern read as what a head moves is `probs * share[..., None]`; to
set what a head writes, edit `attention_head_outputs`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    scores = attn.attention_scores.save()
    probs = attn.attention_probabilities.save()

share = torch.sigmoid(scores.logsumexp(-1) - attn._module.sinks.view(1, -1, 1))
torch.testing.assert_close(scores.softmax(-1), probs)
```

transformers has no `sdpa` path for this attention, so a default load runs eager and the six
interior values are available. The query scale is `attention_multiplier`, 1/64, which is
`1 / head_dim`. There are 20 query heads over 4 key/value heads of 64: an edit to key/value head
`j` reaches query heads `5j` to `5j + 4`.

## A write moves the other positions in bfloat16

A write to `attention_output` or `mlp_output` reaches the model as the whole edited copy divided
by `residual_multiplier`. With 0.26 that round trip is exact for all but 2 of the 65280 finite
bfloat16 products `x * 0.26`, but the mixtures add a second source: an edit changes the edited
token's routing in later blocks, and from then on the grouped expert kernels compute the other
tokens' outputs with different rounding. Adding a unit vector at the last position of block 1, to
`mlp_output` or as `mlp.output[:, -1] += v / m`, moved the earlier positions' logits by up to 0.62
(a KL of 0.004) in bfloat16, against a KL of 9e-6 at the edited position. In float32 the earlier
positions stay bit-identical. Load in float32 for any edit whose effect is small.

```python
m = model.config.residual_multiplier
with model.trace(prompt):
    model.layers[1].mlp.output[:, -1] += v / m       # bfloat16: earlier positions move by rounding
```

## Logits are divided by 5, and the embeddings multiplied by 12

`model.logits` is `model.lm_head.output / logits_scaling` exactly, and `project_on_vocab` divides
too. A softmax over `lm_head.output` is at temperature 1/5: after "The capital of the United
Kingdom is", ` London` has probability 0.70 from `logits` and 1.00 from `lm_head.output`.
`token_embeddings` is the plain lookup and `layers[0].input` is it times 12; to set what block 0
reads, edit `layers[0].input`. `lm_head` and `embed_tokens` are one parameter.

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()
    first = model.layers[0].input.save()
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

torch.testing.assert_close(first, emb * model.config.embedding_multiplier)
torch.testing.assert_close(logits, raw / model.config.logits_scaling)
torch.testing.assert_close(model.project_on_vocab(resid), logits)
```

The vocabulary is Granite 4's, 100352 tokens; no beginning-of-sequence token is prepended, and
`" London"` (7295) and `" Paris"` (12366) are single tokens.

## What this family module covers

Every checkpoint whose config says `granitemoe_swa`: granite-swash-3b-a600m. `granite_swa` is the
dense Swash model, the same attention with an MLP. `granitemoeshared` is GraniteMoE's block with a
shared expert, whose `mlp_output` carries both experts' sum.
"""
