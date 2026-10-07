"""GraniteMoE-Shared: GraniteMoE's scaled block with a dense shared expert beside the mixture."""

MODEL_TYPE = "granitemoeshared"
TITLE = "GraniteMoE-Shared"
SUBTITLE = (
    "GraniteMoE's block with a dense shared expert beside the mixture: the block adds the sum of the two times "
    "residual_multiplier, and mlp_output is that scaled sum, while mlp.output is the mixture's output alone."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-research/moe-7b-1b-active-shared-experts"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GraniteMoeSharedForCausalLM"
CHECKPOINTS = ["ibm-research/moe-7b-1b-active-shared-experts"]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 217}
VLLM = False
QUIRKS = ["scaled-residual-adds", "mixture-of-experts", "embedding-multiplier", "scaled-logits"]

#: No public checkpoint of this family is small enough to run here (the one release is 7B), so the notes' shapes,
#: identities and snippets come from the pinned tiny checkpoint and a copy of it with Granite's multipliers away
#: from 1.0, and their numbers from the reference's config.
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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the feed-forward's input "
                             "norm, and the router, every expert and the shared expert read its output.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, + shared",
        },
    ],
    "identity_note": "The terms are already scaled: attention_output is the attention's output and mlp_output the "
                     "mixture's plus the shared expert's, each times residual_multiplier (0.22), so the plain sum is "
                     "exact. routed_output + shared_expert_output == mlp_output / residual_multiplier.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier (12) before block 0, "
             "so layers[0].input is token_embeddings · 12.",
    "layers": "Each block adds its attention's output and the sum of its mixture and shared expert, each times "
              "residual_multiplier (0.22).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "logits = lm_head.output / logits_scaling (6). project_on_vocab divides too.",
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

GraniteMoE's block with a dense expert beside the mixture on every block. The mixture's native
name, `block_sparse_moe`, is `mlp`; the shared expert keeps its native name, `shared_mlp`, on the
block. The released checkpoint, moe-7b-1b-active-shared-experts, has 40 blocks, each with 62
experts 512 wide, the top 6 routed per token, and a shared expert 1024 wide
(`shared_intermediate_size`), a SwiGLU like the experts. Its multipliers are Granite's:
`embedding_multiplier` 12, `residual_multiplier` 0.22, `attention_multiplier` 1/128 and
`logits_scaling` 6.

## mlp_output is both experts' sum, scaled

No module returns what the block adds after the attention, so `mlp_output` is the block's own
binding of `block_sparse_moe(n) + shared_mlp(n)`, times `residual_multiplier`. The mixture's values
are read before the multiplier: `routed_output` is `mlp.output`, the mixture's output alone, and
`shared_expert_output` is `layers[i].shared_mlp.output`. The routed experts run first, so read
`routed_output` before `shared_expert_output`, and both before `mlp_output`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    mlp = moe.mlp_output.save()
    out = model.layers[1].layer_output.save()

m = model.config.residual_multiplier       # 0.22
torch.testing.assert_close((routed + shared) * m, mlp)
torch.testing.assert_close(x + attn + mlp, out)
```

Attribute to single experts with `expert_outputs * m` and to the shared expert with
`shared_expert_output * m`: those are their terms in the stream.

## A write to mlp_output replaces both experts

A write to `mlp_output` reaches the model on the sum: the block's binding is replaced by the
edited copy divided by `residual_multiplier`, so `mlp_output[:] = 0` removes the mixture and the
shared expert together and leaves `layer_output` equal to `input + attention_output`. To ablate one
of the two, edit its module's output: `mlp.output` for the mixture, `layers[i].shared_mlp.output`
for the shared expert, scaled by `1 / residual_multiplier` to land as a given vector on the
stream.

```python
m = model.config.residual_multiplier
with model.trace(prompt):
    model.layers[1].shared_mlp.output[:, -1] += v / m    # the shared expert's term moves by v
```

Like every Granite write to a scaled copy, this divides the whole copy back, so positions you did
not edit are rewritten too. With 0.22 the round trip is not exact in bfloat16: for 502 of the
65280 finite products `x * 0.22`, `/ 0.22 * 0.22` does not give the product back. Load in float32
for fine-grained edits.

## The router: top 6 of the logits, then a softmax over those 6

GraniteMoE's routing: one linear map without a bias, the top `num_experts_per_tok` of its logits,
and a softmax over those alone, so a token's `expert_weights` sum to one. `expert_outputs` (each
slot's weighted output) sum to `routed_output`. The shared expert has no weight: it is added to
every token as it is.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()        # [batch, seq, num_experts]
    w = moe.expert_weights.save()            # [batch, seq, top_k]
    idx = moe.expert_indices.save()
    slots = moe.expert_outputs.save()
    routed = moe.routed_output.save()

top, chosen = logits.topk(moe.top_k, dim=-1)
torch.testing.assert_close(chosen, idx)
torch.testing.assert_close(top.float().softmax(-1).to(w.dtype), w)
torch.testing.assert_close(slots.sum(2), routed)
```

`intermediate_size` and `mlp.intermediate_size` are the routed experts' width, 512; the shared
expert's is `config.shared_intermediate_size`. A config with `shared_intermediate_size` 0 builds no
shared expert: `shared_mlp` is `None`, `shared_expert_output` is unavailable, and `mlp_output` is
the mixture's output times the multiplier.

## Attention, logits and embeddings are Granite's

The attention scales `q · kᵀ` by `attention_multiplier`, 1/128, which is `1 / head_dim`. There are
12 query heads over 4 key/value heads of 128, so an edit to key/value head `j` reaches query heads
`3j` to `3j + 2`. The six interior values need `attn_implementation="eager"`.

`model.logits` is `model.lm_head.output / logits_scaling` and `project_on_vocab` divides too;
`token_embeddings` is the plain lookup and `layers[0].input` is it times 12. `lm_head` and
`embed_tokens` are one parameter.

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

The tokenizer is Granite 3's, 49152 tokens (the embedding has 50257 rows), and prepends nothing.
It splits ` Paris` into `ĠPar` and `is`; ` London` is one token.

## What this family module covers

Every checkpoint whose config says `granitemoeshared`: ibm-research's
moe-7b-1b-active-shared-experts. `granitemoe` is the same block without the shared expert;
`granitemoehybrid` (Granite 4.0) has the same feed-forward, with Mamba-2 mixers on most blocks;
`granitemoe_swa` has a shared expert too, which its `mlp_output` leaves out.
"""
