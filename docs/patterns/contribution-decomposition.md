---
title: Contribution Decomposition
one_liner: "Direct logit attribution over the standard contributions: `model.lm_head(attention_output)` and `model.lm_head(mlp_output)` per block sum to the head of the final stream, the final norm frozen at the real stream puts them on the logits' scale, `attention_head_outputs` with the output projection's weight splits attention per head, and Gemma-4, Doge, ZAYA and DeepSeek-V4 weight each term before it sums."
tags: [patterns, attribution, residual-stream, heads, logit-lens]
related: [docs/usage/residual-stream.md, docs/usage/root-values.md, docs/patterns/logit-lens.md, docs/patterns/ablation.md, docs/patterns/attention-patterns.md, docs/reference/families.md]
sources: [nnterp/standardized.py, nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/mlp.py, nnterp/families/gpt2.py, nnterp/families/gemma4_text.py, nnterp/families/granite.py]
---

# Contribution Decomposition

## What this is for

The residual stream is a sum: the block input, plus what each attention sublayer
adds, plus what each MLP adds. Direct logit attribution (DLA) pushes each term
through the unembedding on its own and asks how much it moved a target logit.

nnterp defines the terms so that `layers[i].input + attention_output + mlp_output ==
layer_output`, checked per family by the test suite, whether the block is sequential,
parallel, sandwich-normed or a DeltaNet hybrid. Four families scale the stream itself
between blocks, and there the terms have to be weighted before they sum: Gemma-4 (a
per-block scalar), Doge and ZAYA (per-channel gates) and DeepSeek-V4 (parallel streams);
see [Families where the plain sum does not hold](#families-where-the-plain-sum-does-not-hold).
This page verifies the sum end to end, then decomposes it: by block, by sublayer and by
head.

## Canonical pattern

Read the base and every contribution in forward order; their running sum is the
last block's output:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or paris is not the word
paris = ids[0]

parts = {}                                          # made outside the trace
with model.trace(prompt):
    base = model.layers[0].input.save()             # the stream entering block 0
    for i, layer in enumerate(model.layers):        # block i's attention, then its MLP: forward order
        parts["attn", i] = layer.self_attn.attention_output.save()
        parts["mlp", i] = layer.mlp.mlp_output.save()
    final = model.layers[-1].layer_output.save()

total = base + sum(parts.values())
torch.testing.assert_close(total, final, rtol=1e-4, atol=1e-3)   # the stream is the sum of its contributions
# a float32 sum of 2 * num_layers terms lands within ~1e-4 of the stream on a real model; the tiny
# test checkpoints are exact, so a tolerance that passes there can fail on GPT-2
```

`add_special_tokens=False` keeps the BOS token out of the target (`tokenizer.encode(" Paris")[0]`
is the BOS id on Llama and Gemma). The assertion
catches a word that is more than one token: Mistral's sentencepiece tokenizer gives
`['▁', '▁Paris']` and Granite's `['ĠPar', 'is']`. Then try the word without the leading
space (`"Paris"` is `['▁Paris']` on Mistral), or pick a target word that is one token.

Then the linear DLA: `model.lm_head(x)` inside a trace runs the unembedding on any
`[..., hidden]` tensor, stood down from the model's own call, so each term's logits
come from the same trace:

```python
dla = {}
with model.trace(prompt):
    dla["base"] = model.lm_head(model.layers[0].input)[0, -1].save()             # [vocab]
    for i, layer in enumerate(model.layers):
        dla["attn", i] = model.lm_head(layer.self_attn.attention_output)[0, -1].save()
        dla["mlp", i] = model.lm_head(layer.mlp.mlp_output)[0, -1].save()
    head_final = model.lm_head(model.layers[-1].layer_output)[0, -1].save()

assert torch.allclose(sum(dla.values()), head_final, atol=1e-4)                  # linear: the attributions sum

for i in range(model.num_layers):
    print(f"block {i:2d}   attn {dla['attn', i][paris]:+.3f}   mlp {dla['mlp', i][paris]:+.3f}")
```

`lm_head` has no bias on tied-embedding checkpoints (GPT-2, Llama); on one that has
a bias, it is added once per call, so subtract it `2 * num_layers` times from the
sum before comparing.

## The norm is nonlinear

`head_final` above is `lm_head` applied to the *unnormed* final stream, which is
not the model's logits: the model applies `model.norm` first (LayerNorm on GPT-2,
RMSNorm on Llama), then `lm_head`, then any step after the head. A norm rescales each
position by its own statistics, so it does not distribute over the sum, and there are
two honest ways to read a contribution through it:

- **DLA through the frozen norm** (next section): take the norm's per-position scale
  from the real final stream and hold it fixed. The norm is then a linear map, the
  attributions sum exactly to the head's output, and they are on the model's scale.
- **The normed lens**, `model.project_on_vocab(contribution)`: the full final norm,
  `lm_head` and the step after the head applied to the contribution *alone*, as in
  [logit-lens](logit-lens.md). It answers "what would this contribution predict by
  itself", and it does not sum to the logits:

```python
lens = {}
with model.trace(prompt):
    for i, layer in enumerate(model.layers):
        lens["attn", i] = model.project_on_vocab(layer.self_attn.attention_output)[0, -1].save()
        lens["mlp", i] = model.project_on_vocab(layer.mlp.mlp_output)[0, -1].save()
    logits = model.logits[0, -1].save()

sum(lens.values())      # not close to `logits`: the norm was applied per term
```

Use the frozen norm for attribution shares, the normed lens for "what does this term
say"; say which you used.

## Through the final norm

Four pieces of family knowledge sit between a contribution and a logit:

| piece | families | what it does to an attribution |
| --- | --- | --- |
| LayerNorm's centering and bias | GPT-2, GPT-NeoX/Pythia, GPT-J, CodeGen, Phi, OPT, BLOOM, Falcon, StableLM, ... | each term is centered before it is scaled; the norm's bias is added once, as a term of its own |
| the gain `1 + w` | Gemma, Gemma-2, Gemma-3, VaultGemma, Qwen3-Next, Qwen3.5 (their RMSNorm multiplies by `1 + weight`) | the gain is not `norm.weight` |
| an `lm_head` bias | GPT-J, CodeGen, Phi | added once |
| a step after the head | Granite and its relatives divide by `logits_scaling`; Cohere multiplies by `logit_scale`, Falcon-H1 by `lm_head_multiplier`, HyperCLOVA X by `logits_scaling`; Gemma-2/3/4 softcap | a linear step scales every attribution by the same constant; a softcap is not linear, so the attributions sum to the head's output, not to the capped logit |

The recipe below reads the first two off the norm module itself (its output on a zero
vector is its bias, on a constant vector tells whether it centers, on a `±1` vector gives
its gain), so it needs no table of families. It continues the canonical pattern:

```python
norm, head = model.norm._module, model.lm_head._module
x = final[0, -1]                                          # the stream the final norm reads at this position
with torch.no_grad():
    shift = norm(torch.zeros_like(x))                     # a LayerNorm's bias; zeros on an RMSNorm
    centers = torch.allclose(norm(torch.ones_like(x)), shift)          # a LayerNorm subtracts the mean first
    center = (lambda t: t - t.mean(-1, keepdim=True)) if centers else (lambda t: t)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0))
    probe = torch.ones_like(x); probe[1::2] = -1          # mean 0, mean square 1
    gain = (norm(probe) - shift) / probe * (1 + eps) ** 0.5            # w, or 1 + w on Gemma's RMSNorm
    scale = (center(x).pow(2).mean() + eps).rsqrt()       # the norm's one nonlinear step, frozen at the real stream
    frozen = lambda t: gain * center(t[0, -1]) * scale    # the final norm as a linear map at this position
    row = head.weight[paris]
    normed = {key: float(frozen(part) @ row) for key, part in parts.items()}
    normed["base"] = float(frozen(base) @ row)
    constant = float(shift @ row) + (float(head.bias[paris]) if head.bias is not None else 0.0)

with model.trace(prompt):
    raw = model.lm_head.output[0, -1, paris].save()       # the head's output, before any step after it

print(sum(normed.values()) + constant, float(raw))        # equal, to float error
```

On GPT-2 the sum and `raw` agree to four decimals (-92.0873), with the LayerNorm's bias
contributing -8.70 of it; the same lines agree on Llama-3.2-1B and Gemma-3-270m (RMSNorm,
`1 + w`), Pythia-70m (LayerNorm), CodeGen-350M (an `lm_head` bias) and
granite-3.0-1b-a400m (`raw` is 126.57, and the model's logit is that divided by
`logits_scaling`, 21.10). Read `normed` for the shares: a block's attention and MLP are
on the scale of the logit they move.

## The base: `layers[0].input`, not `token_embeddings`

`model.token_embeddings` is the embedding module's output: before positional embeddings
or an embedding norm the model applies afterwards, and including any scale the module
applies itself (Gemma's `sqrt(hidden_size)`). On GPT-2 the block input is `wte + wpe`
(after the embedding dropout), so `token_embeddings` is *not* the base of the sum
and the running sum from it misses the positional term; on Granite the model multiplies
it by `embedding_multiplier` before block 0. On Llama the two are equal.
`layers[0].input` is what enters block 0 on every family, so it is the base to use;
`token_embeddings` is the value to read when you want the token's own vector.

## Per head

`attention_head_outputs` is each head's output before concatenation and the output
projection, `[batch, seq, heads, head_dim]`. The projection is linear, so head `h`'s
contribution is its slice times the projection weight's matching rows. The projection
module keeps its native name (`o_proj` on Llama-style families, `c_proj` on GPT-2,
`dense` on GPT-NeoX, `out_proj` on GPT-Neo, GPT-J and CodeGen) and its weight layout
differs: `torch.nn.Linear` stores `[out, in]`, GPT-2's `Conv1D` stores `[in, out]`.

```python
attention_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
LAYER = attention_blocks[len(attention_blocks) // 2]          # on a hybrid, num_layers // 2 can be a linear block
attention = model.layers[LAYER].self_attn

projection = None                          # `or` over envoys calls __len__ on the module; test `is not None`
for name in ("o_proj", "c_proj", "dense", "out_proj"):
    candidate = getattr(attention, name, None)
    if candidate is not None:
        projection = candidate
        break

with model.trace(prompt):
    heads = attention.attention_head_outputs.save()          # [batch, seq, heads, head_dim]
    contribution = attention.attention_output.save()         # [batch, seq, hidden]

W = projection._module.weight
W_in_out = W.t() if isinstance(projection._module, torch.nn.Linear) else W     # [heads * head_dim, hidden]
H, D = heads.shape[2], heads.shape[3]
per_head = torch.einsum("bshd,hdo->bsho", heads, W_in_out.reshape(H, D, -1))    # [batch, seq, heads, hidden]

bias = projection._module.bias
total = per_head.sum(2) + (bias if bias is not None else 0)
torch.testing.assert_close(total, contribution, rtol=1e-4, atol=1e-3)          # the heads sum to the contribution

head_dla = per_head[0, -1] @ model.lm_head._module.weight.t()                    # [heads, vocab], linear
print(head_dla[:, paris])
```

The assertion holds where `attention_output` is the projection's output, as on GPT-2,
Llama, GPT-Neo and CodeGen. It does not on these families, where something sits after
the heads:

| families | what `per_head.sum(2)` equals |
| --- | --- |
| Granite, GraniteMoE(-Shared, -Hybrid), Granite-SWA, GraniteMoE-SWA | `projection.output`; `attention_output` is that times `config.residual_multiplier` (0.22 on granite-3.0-1b-a400m) |
| the post-norm families: Gemma-2/3/4, OLMo-2/3, EXAONE-4, FlexOlmo, GLM-4, OLMo-Hybrid's attention blocks; HyperCLOVA X (its post-norm, then `residual_multiplier`) | `projection.output`, the pre-norm output; the norm is not linear, so the split is of that, not of the contribution |
| Qwen3-Next, Qwen3.5, Qwen3.5-MoE (a sigmoid gate from `q_proj`), Laguna (a softplus gate, `g_proj`), AFMoE (a sigmoid gate, `gate_proj`, then a post-norm) | neither: the gate multiplies the heads after `attention_head_outputs`. Split `projection.input` instead, `projection.input.unflatten(-1, (H, D))`, the gated heads the projection reads; its per-head sum is `projection.output` |
| BitNet (`attn_sub_norm` before `o_proj`) | neither: the norm mixes the heads, so `projection.input` split by head is a split of the normed tensor |

The same numbers come from the model itself: zero every head but `h` in
`attention_head_outputs` inside a trace and read `attention_output`; that equals
`per_head[..., h, :]` plus the bias where the assertion holds. Under grouped-query
attention the head axis of `attention_head_outputs` is still `num_heads` (query heads),
so the slicing is unchanged.

## Families where the plain sum does not hold

### Gemma-4: a per-block scalar

Each Gemma-4 block ends `hidden_states *= layer_scalar`, so a term added in block `i`
reaches the final stream multiplied by the scalars of blocks `i` through the last, and
the base by all of them. On gemma-4-E2B that weight is 1.9e-13 for block 0 and 0.16 for
the last block; the plain sum is off by about 64 times the stream's own norm, and a naive
DLA ranks block 0's attention first. Weight each term by its suffix product:

```python
model = StandardizedTransformer("google/gemma-4-E2B", dispatch=True, dtype=torch.float32)
paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]      # this tokenizer's id (one token here)

scalars = torch.cat([layer._module.layer_scalar.float() for layer in model.layers])   # [num_layers]
weights = scalars.flip(0).cumprod(0).flip(0)        # weights[i]: the scalars of blocks i..last multiplied

parts = {}
with model.trace(prompt):
    base = model.layers[0].input.save()
    for i, layer in enumerate(model.layers):
        parts["attn", i] = layer.self_attn.attention_output.save()
        parts["mlp", i] = layer.mlp.mlp_output.save()
        parts["ple", i] = layer.per_layer_output.save()       # E2B and E4B only: the per-layer embedding term
    final = model.layers[-1].layer_output.save()

total = weights[0] * base + sum(weights[i] * part for (_, i), part in parts.items())
torch.testing.assert_close(total, final, rtol=1e-4, atol=1e-3)

row = model.lm_head._module.weight[paris]
dla = {key: float(weights[key[1]] * part[0, -1] @ row) for key, part in parts.items()}
```

The weighted `dla` plus the base's term sums to `final[0, -1] @ row`; its largest terms on
E2B are the last blocks' MLPs and per-layer terms (block 34's MLP, -0.27), where the
unweighted one names block 0's attention (+22.0). The final-norm recipe above applies to
`weights[i] * part` unchanged.

Steering through a contribution differs from `steer` by the scalar: `steer(i, v)` adds
`v` to the stream after the scalar, so the same move through a contribution is `v /
layer_scalar`, added to the block's *last* term (`per_layer_output` on E2B and E4B,
`mlp_output` on the others). An edit of an earlier term also moves what the rest of the
block reads: on E2B, `mlp_output += v / layer_scalar` misses `steer` by 0.15 in the logits,
`per_layer_output += v / layer_scalar` by 2e-5.

### Doge and ZAYA: per-channel gates on the stream

Doge's block is `h = input_residual * input + attention_output`, `layer_output =
post_attention_residual * h + mlp_output`, with `[hidden]` gates (`layers[i]._module.input_residual`,
`.post_attention_residual`). A term's weight in the final stream is the per-channel product of
every gate after it: block `i`'s `attention_output` is multiplied by its own
`post_attention_residual` and by both gates of every later block, its `mlp_output` by both gates
of every later block. Multiply each term by that vector before the unembedding. ZAYA's merges
are the same with `residual_scale`, plus a `residual_bias` added to the stream before each scale,
which enters the sum as one more term per merge ([residual-stream](../usage/residual-stream.md#where-the-families-differ)).

### DeepSeek-V4: parallel streams

`layer_output` is `[batch, seq, streams, hidden]` and each sublayer's output is weighted into
each stream and the streams mixed; the identity is the block's stream formula
([residual-stream](../usage/residual-stream.md#where-the-families-differ)). The plain
`layers[i].input + attention_output + mlp_output` broadcasts `[batch, seq, hidden]` against
the stream axis: it raises a shape error at most prompt lengths, and silently gives a wrong
tensor when the prompt is one token or exactly `hc_mult` (4) tokens long.

## Gotchas

- Read in forward order: block `i`'s `attention_output` then its `mlp_output`, then
  block `i + 1`. Two list comprehensions, one over all attentions and one over all
  MLPs, are out of order (`OutOfOrderError`).
- `model.token_embeddings` must be read before `model.layers[0].input` in one
  trace, and it is not the base of the sum on a family with positional
  embeddings added after it (GPT-2's `wpe`), an embedding norm (BLOOM) or a multiplier
  (Granite's `embedding_multiplier`).
- `model.lm_head(x)` inside a trace is a stood-down call on your tensor;
  `model.lm_head.output` is the model's own projection of the *normed* final
  stream, and `model.logits` adds the softcap on Gemma-2. Three different things.
- Gemma-4, Doge, ZAYA and DeepSeek-V4 are not plain sums; see
  [above](#families-where-the-plain-sum-does-not-hold).
- A softcapped model's logits are not a sum of anything; `project_on_vocab` applies
  the cap, the linear DLA does not.
- `getattr(a, "x", None) or getattr(b, ...)` on envoys raises on a module without
  `__len__`; chain with `is not None`.
- The match holds to the checkpoint's dtype: within ~1e-4 relative in float32, and
  a few percent on a bf16 load (a sum of 60 bf16 terms on SmolLM2 lands about 2% off
  logits in the hundreds). Use a relative tolerance, or load with
  `dtype=torch.float32` for the check.
- A name bound inside the trace does not survive it; `parts`, `dla`, `lens`, `normed`
  are made outside.

## Related

- [logit-lens](logit-lens.md): `project_on_vocab` on the stream itself.
- [ablation](ablation.md): the causal counterpart to an attribution share.
- [attention-patterns](attention-patterns.md): what the heads whose contributions
  you just ranked attend to.
- [../usage/residual-stream.md](../usage/residual-stream.md): the identity and
  where each family binds its contributions.
- [../usage/root-values.md](../usage/root-values.md): `logits`, `token_embeddings`.
- [../reference/families.md](../reference/families.md): the scaled and gated residual families.
- nnsight `docs/usage/access-and-modify.md`: calling modules inside a trace.
- Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits".
