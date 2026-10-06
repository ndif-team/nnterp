"""GraniteMoE: Granite's four multipliers around a block whose MLP is a mixture of experts."""

MODEL_TYPE = "granitemoe"
TITLE = "GraniteMoE"
SUBTITLE = (
    "Granite's block with a mixture of experts for its MLP: the embeddings are multiplied before block 0, "
    "each sublayer's output is multiplied by residual_multiplier before it is added, and the logits are "
    "divided by logits_scaling, so mlp_output is the mixture's output times 0.22."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-3.0-1b-a400m-base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GraniteMoeForCausalLM"
CHECKPOINTS = [
    "ibm-granite/granite-3.0-1b-a400m-base", "ibm-granite/granite-3.0-1b-a400m-instruct",
    "ibm-granite/granite-3.0-3b-a800m-base", "ibm-granite/granite-3.0-3b-a800m-instruct",
    "ibm-granite/granite-3.1-1b-a400m-base", "ibm-granite/granite-3.1-1b-a400m-instruct",
    "ibm-granite/granite-3.1-3b-a800m-base", "ibm-granite/granite-3.1-3b-a800m-instruct",
    "ibm-granite/granite-guardian-3.2-3b-a800m", "ibm-research/PowerMoE-3b",
]

#: Granite lineage (205, set by granite); GraniteMoE sits beside it.
PALETTE = {"hue": 215}
VLLM = False
QUIRKS = ["scaled-residual-adds", "mixture-of-experts", "embedding-multiplier", "scaled-logits"]

#: Numbers in the notes come from the reference's weights (float32 on an RTX A6000) unless they name
#: granite-3.1-3b-a800m-base; the snippets ran on both and on the pinned tiny checkpoint (block 1 for 5 and 20).
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
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the mixture's input norm, "
                             "and the router and every expert read its output.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
    "identity_note": "The terms are already scaled: attention_output and mlp_output are the modules' outputs times "
                     "residual_multiplier (0.22), so the plain sum is exact. routed_output and expert_outputs are not "
                     "scaled: mlp_output == routed_output · 0.22.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier (12.0) "
             "before block 0, so layers[0].input is token_embeddings · 12.",
    "layers": "Each block adds its attention's and its mixture's output times residual_multiplier (0.22).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings) on every checkpoint.",
    "logits": "logits = lm_head.output / logits_scaling (6.0). project_on_vocab divides too.",
}

NOTES = """
## The block, in order

```
x      = embed_tokens(ids) * embedding_multiplier
h      = x + self_attn(input_layernorm(x)) * residual_multiplier
out    = h + block_sparse_moe(post_attention_layernorm(h)) * residual_multiplier
logits = lm_head(norm(out)) / logits_scaling
```

Granite's block with the MLP replaced by a mixture of experts on every block; the mixture's
native name, `block_sparse_moe`, is `mlp`. Every released checkpoint has the same four scalars:
`embedding_multiplier` 12, `residual_multiplier` 0.22, `attention_multiplier` 1/64 and
`logits_scaling` 6. The 1B-A400M has 24 blocks of 32 experts, the 3B-A800M 32 blocks of 40;
both route each token to 8.

## The contributions are scaled; the mixture's own values are not

`mlp_output` is the mixture's output times `residual_multiplier`, and `attention_output` the
attention's: the tensors the block adds, so the plain identity holds bit for bit, in float32 and
bfloat16. The mixture's values are read inside it, before the multiplier: `expert_outputs` (each
slot's weighted expert output) sum to `routed_output`, which is `mlp.output`, 1/0.22 ≈ 4.5 times
what reaches the stream.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    slots = moe.expert_outputs.save()        # [batch, seq, 8, hidden], unscaled
    routed = moe.routed_output.save()
    mlp = moe.mlp_output.save()
    out = model.layers[1].layer_output.save()

m = model.config.residual_multiplier         # 0.22
torch.testing.assert_close(slots.sum(2), routed)
torch.testing.assert_close(mlp, routed * m)
torch.testing.assert_close(x + attn + mlp, out)
```

Attribute to single experts with `expert_outputs * m`: that is each slot's term in the stream.

## The router: top 8 of the logits, then a softmax over those 8

The router is one linear map without a bias. It takes the top 8 of its logits and softmaxes
those 8 alone, in float32, so a token's `expert_weights` sum to one and the other experts'
logits play no part in them. `router_logits` is the linear map's output, in the model's dtype.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()        # [batch, seq, 32]
    w = moe.expert_weights.save()            # [batch, seq, 8]
    idx = moe.expert_indices.save()

top, chosen = logits.topk(moe.top_k, dim=-1)
torch.testing.assert_close(chosen, idx)
torch.testing.assert_close(top.float().softmax(-1).to(w.dtype), w)
```

The slots come in the router's order, highest logit first. The experts are stored fused:
`experts.gate_up_proj` is `[32, 1024, 1024]`, each expert's gate rows then its up rows, and
`experts.down_proj` is `[32, 1024, 512]`; the checkpoint files name them `input_linear` and
`output_linear`, and the router `router.layer`, and transformers renames them on load. The
default `experts_implementation`, `grouped_mm`, serves `expert_outputs`; under `"eager"` it is
unavailable.

## One expert writes a vector of norm 1650 to 2010 on the first token

The tokenizer adds no beginning-of-sequence token, and the first token, whatever it is, is
routed at block 5 to expert 4 with a weight of 0.99 or more (nine prompts, starting with a word,
a digit, a comma, a newline or code). That one slot writes a vector of norm 1650 to 2010, 99% of
its square in dimension 1018, while the stream at the other positions has a norm of 2 to 4. It
stays on the stream until block 23, whose mixture takes most of it back out (the first token's
norm drops to between 80 and 180). Heads attend to it: on
average over heads, 0.54 to 0.66 of the attention in block 10 and 0.81 to 0.91 in block 20.

```python
moe = model.layers[5].mlp
with model.trace(prompt):
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()
    contribution = moe.mlp_output.save()

idx[0, 0, 0], w[0, 0, 0]          # expert 4, 0.99
contribution[0].norm(dim=-1)      # 1650 to 2010 at position 0, under 1 elsewhere
```

On granite-3.1-3b-a800m-base the same happens at block 3: expert 11 (weight 0.87) and expert 32
(0.10) write a vector of norm about 3600 across dimensions 192, 1282 and 350. Slice `[:, 1:]`
before averaging activations, counting expert usage or sweeping ablations.

## Ablate an expert at the positions after the first

Zeroing the weight of every slot that chose expert `e` changes the mixture's output exactly at
the tokens that chose it: on the test prompt below the other positions' `mlp_output` is
bit-identical. The ablation does not renormalize the token's other weights. Leave position 0 out
of a sweep, or the first token's expert dominates it: ablating expert 4 of block 5 everywhere
lowers log p(` London`) after "The capital of the United Kingdom is" by 9.4 nats, by removing the
first token's vector.

```python
moe, e = model.layers[20].mlp, 28
with model.trace(prompt):
    w, idx = moe.expert_weights, moe.expert_indices
    w[:, 1:] = w[:, 1:].masked_fill(idx[:, 1:] == e, 0)     # in place: position 0 keeps its routing
    ablated = model.logits[0, -1].float().log_softmax(-1)[target].save()
```

On that prompt in float32, with position 0 left out, the largest effects over all 24 × 32
experts are −1.50 nats (block 20, expert 28) and −0.81 (block 21, expert 21); the median nonzero
effect is 0.024 nats, and 222 of the 768 experts were chosen by no token after the first, so
their effect is exactly zero.

## Some experts are chosen by every token

Over 174 tokens of English prose, Python and French (position 0 left out), block 1's expert 8
and block 2's expert 24 were chosen by every token, block 13's expert 28 by 98% and block 5's
expert 24 by 96%; elsewhere the most used expert takes 44% to 81% of the tokens. On
granite-3.1-3b-a800m-base, blocks 1, 2 and 3 each have an expert every token chooses (9, 37 and
35), and block 1 left 14 of its 40 experts unused. Ablating such an expert removes a part of the
block every token runs through, as a dense MLP's would.

```python
moe = model.layers[1].mlp
with model.trace(text):
    idx = moe.expert_indices.save()

counts = torch.bincount(idx[0, 1:].flatten(), minlength=moe.num_experts)
share = counts / (idx.shape[1] - 1)      # the fraction of tokens that chose each expert
share.argmax(), share.max()              # expert 8, 1.0
```

## A write rounds the other positions, twice over

A write to `mlp_output` or `attention_output` reaches the model as Granite's do: the whole edited
copy is divided by `residual_multiplier` and the block multiplies it again, so the positions you
did not edit round too. Adding a unit vector at the last position of the last block moved the
earlier positions' logits by 0.06 in bfloat16 (3e-6 in float32); the same edit on the module's
output, scaled the other way, leaves them bit-identical:

```python
m = model.config.residual_multiplier
with model.trace(prompt):
    model.layers[-1].mlp.mlp_output[:, -1] += v    # every position rounds

with model.trace(prompt):
    model.layers[-1].mlp.output[:, -1] += v / m    # position -1 only
```

An edit before a later mixture rounds the other positions a second way. It changes the edited
token's routing in a later block, and from then on the mixtures compute the other tokens'
expert outputs with different rounding. From block 1 of a 25-token prompt, both forms of the edit
moved the earlier positions' logits by up to 0.81 in bfloat16 (a KL of 0.005), more than the
edit's own effect at the edited position (a KL of 0.002 in float32), and by 2e-5 in float32;
`experts_implementation="eager"` gives the same. Load in float32 for any edit whose effect is
small. The 3.0 base checkpoints and PowerMoE-3B are stored in float32, the others in bfloat16.

## The query scale is attention_multiplier, 1/64

The attention multiplies `q · kᵀ` by `config.attention_multiplier`, `1 / head_dim`, in place of
`head_dim ** -0.5`: an eighth of Llama's scale at `head_dim` 64. The 1B has 16 query heads over 8
key/value heads, the 3B 24 over 8, so an edit to key/value head `j` reaches query heads `2j` and
`2j + 1` on 1B and `3j` to `3j + 2` on 3B.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    scores = attn.attention_scores.save()

k = k.repeat_interleave(model.num_heads // model.num_kv_heads, dim=1)
scale = model.config.attention_multiplier    # not head_dim ** -0.5
qk = q @ k.transpose(-1, -2) * scale
causal = torch.ones_like(qk[0, 0], dtype=torch.bool).tril()
torch.testing.assert_close(qk[..., causal], scores[..., causal])
```

The six interior values need `attn_implementation="eager"`. The default load runs `sdpa`; its
logits equal eager's bit for bit in float32 and differ by up to 0.70 in bfloat16, with the same
top token.

## The logits are divided by 6, and the embeddings multiplied by 12

`model.logits` is `model.lm_head.output / logits_scaling` exactly, and `project_on_vocab`
divides too. A softmax over `lm_head.output` is at temperature 1/6: after "The capital of the
United Kingdom is", ` London` has probability 0.47 from `logits` and the top token 1.00 from
`lm_head.output`. `token_embeddings` is the plain lookup and `layers[0].input` is it times 12
(row norms 0.5 to 0.8 against 6 to 10); embeddings passed as `inputs_embeds` are multiplied
too, so to set what block 0 reads, edit `layers[0].input`.

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

`lm_head` and `embed_tokens` are one parameter, so an edit to `embed_tokens.weight` is an edit
to the unembedding.

## Target tokens

The tokenizer has 49152 tokens and splits many common words: ` Paris` is `ĠPar` and `is`,
` Rome` `ĠR` and `ome`, ` Berlin` `ĠBer` and `lin`, while ` London` and ` France` are whole
tokens. Check that a target is one token before reading its probability.

## What this family module covers

Every checkpoint whose config says `granitemoe`: Granite 3.0 and 3.1 at 1B-A400M and 3B-A800M,
base and instruct, Granite Guardian 3.2 3B-A800M and PowerMoE-3B. `granitemoeshared` is the
same block with a shared expert beside the mixture. `granitemoehybrid` is all of Granite 4.0:
Mamba-2 and attention blocks (attention only on `granite-4.0-micro`) with a shared MLP and, on
the mixture sizes, routed experts beside it. `granitemoe_swa` is the Swash mixture,
GraniteMoE's block with Granite SWA's attention.
"""
