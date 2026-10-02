---
title: Layouts
one_liner: "Every standard value has one axis layout on every family (one exception: `layer_output` is `Streams` on DeepSeek-V4), one of thirty-one named `jaxtyping` types defined beside the envoy that serves them (`Residual`, `Pattern`, `Keys`, ... from `nnterp.components`) you can read (`value.dims`), check (`isinstance(t, value.layout)`) and annotate your own values with."
tags: [usage, layouts, shapes, jaxtyping, dims, heads, kv_heads, Residual, Pattern, Streams, experts, top_k]
related: [docs/usage/root-values.md, docs/usage/residual-stream.md, docs/reference/families.md, docs/usage/availability.md, docs/extending/custom-values.md]
sources: [nnterp/components/eproperty.py, nnterp/components/moe.py, nnterp/components/layer.py, nnterp/families/deepseek_v4.py, nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/standardized.py, nnterp/components/__init__.py]
---

# Layouts

## What this is for

A value's shape is part of what it means. Each standard value is annotated with one of
thirty-one named layouts, each defined in the file of the envoy that serves it (`Residual`, `Streams`, `StreamWeights`, `StreamMixing` in `nnterp/components/layer.py`; `Queries`, `Keys`, `Values`, `Pattern`, `HeadOutputs` in `nnterp/components/attention.py`; `LinearQK`, `LinearV`, `Gates` in `nnterp/components/linear_attention.py`; `ScanQK`, `ScanValues`, `ScanSteps`, `ScanDecays`, `ScanState`, `ScanStates` in `nnterp/components/selective_scan.py`; `SSDQueries`, `SSDKeys`, `SSDValues`, `SSDHeadOutputs` in `nnterp/components/state_space.py`; `State`, `States` in `nnterp/components/recurrent.py`; `RouterLogits`, `ExpertWeights`, `ExpertIndices`, `ExpertOutputs` in `nnterp/components/moe.py`; `Logits`, `NextTokenProbs`, `Tokens` beside the root values in `nnterp/standardized.py`); `nnterp.components`
re-exports the twenty-eight envoy-level names, and the root's three come from `nnterp.standardized`.
They are `jaxtyping` types such as `Residual = Float[Tensor, "batch seq hidden"]` and
`Pattern = Float[Tensor, "batch heads query key"]`. `value.layout` returns that alias itself
and `value.dims` names its axes. Layouts differ between values, not between families:
`attention_probabilities` is a `Pattern`, `[batch, heads, query, key]`, on GPT-2, Llama and
BLOOM alike, and the per-family suite checks every value's axes against the model's sizes.
One layout per value on every family, except `layer_output` on the hyper-connection
families (DeepSeek-V4), whose residual is several parallel streams: there it is `Streams`,
`[batch, seq, streams, hidden]` ([below](#streams-on-deepseek-v4)).

## Canonical pattern

```python
import torch
from nnterp import Attention, Layer, StandardizedTransformer
from nnterp.components import Queries, Residual
from nnterp.standardized import Logits

Attention.attention_queries.dims        # ('batch', 'heads', 'seq', 'qk_head_dim')
Attention.attention_queries.layout      # jaxtyping.Float[Tensor, 'batch heads seq qk_head_dim']
Attention.attention_queries.layout is Queries   # True: the alias itself, not a copy of it
Layer.layer_output.layout is Residual   # True
StandardizedTransformer.logits.layout is Logits   # True
Layer.layer_output.dims                 # ('batch', 'seq', 'hidden')
StandardizedTransformer.logits.dims     # ('batch', 'seq', 'vocab')

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True, attn_implementation="eager")

with model.trace("The Eiffel Tower is in"):
    q = model.layers[0].self_attn.attention_queries.save()

isinstance(q, Attention.attention_queries.layout)          # True: rank 4, floating dtype
isinstance(q, Queries)                                     # the same check, by name
isinstance(q[0], Attention.attention_queries.layout)       # False: rank 3
isinstance(q.long(), Attention.attention_queries.layout)   # False: not a float
q.shape                                                    # (1, num_heads, seq, head_dim)
```

Read the layout off the class (`Attention.attention_queries`) or off the instance's type
(`type(model.layers[0].self_attn).attention_queries`); the family's subclass inherits the
annotation unless it redefines the value, and a redefinition is annotated with the same
name (`nnterp.families.falcon.Attention.attention_keys.layout is Keys`), so it cannot drift
from the base.

## The layouts

The thirty-one names, their axes, and the values that carry each:

| layout | axes | values |
| --- | --- | --- |
| `Residual` | `batch seq hidden` | `layer_output`, `attention_output`, `mlp_output`, `token_embeddings` (and the plain `self_attn.input`, `mlp.input`) |
| `Streams` | `batch seq streams hidden` | `layer_output` (and `layers[i].input`) on a hyper-connection family (DeepSeek-V4), in place of `Residual` |
| `StreamWeights` | `batch seq streams` | a hyper-connection family's `attention_post`, `mlp_post` |
| `StreamMixing` | `batch seq streams streams` | a hyper-connection family's `attention_comb`, `mlp_comb` |
| `Logits` | `batch seq vocab` | `logits` |
| `NextTokenProbs` | `batch vocab` | `next_token_probs` |
| `Tokens` | `batch seq` (`Int`) | `input_ids`, `attention_mask` |
| `Queries` | `batch heads seq qk_head_dim` | `attention_queries` |
| `Keys` | `batch kv_heads seq qk_head_dim` | `attention_keys` |
| `Values` | `batch kv_heads seq head_dim` | `attention_values` |
| `Pattern` | `batch heads query key` | `attention_scores`, `attention_probabilities` |
| `HeadOutputs` | `batch seq heads head_dim` | `attention_head_outputs` |
| `LinearQK` | `batch seq heads key_dim` | `linear_attn.attention_queries`, `attention_keys` |
| `LinearV` | `batch seq heads value_dim` | `linear_attn.attention_values`, `attention_head_outputs` |
| `Gates` | `batch seq heads` | `linear_attn.decays`, `betas` |
| `State` | `batch heads key_dim value_dim` | `state_input`, `state_output`, `state` |
| `States` | `batch seq heads key_dim value_dim` | `states` |
| `ScanQK` | `batch seq groups state_dim` | a Mamba-1 `linear_attn.attention_queries` (`C`), `attention_keys` (`B`) |
| `ScanValues` | `batch seq channels` | a Mamba-1 `linear_attn.attention_values` (`x`), `attention_head_outputs` (`y`) |
| `ScanSteps` | `batch seq channels` | a Mamba-1 `linear_attn.betas` (`dt`) |
| `ScanDecays` | `batch seq channels state_dim` | a Mamba-1 `linear_attn.decays` (`dt * A`) |
| `ScanState` | `batch channels state_dim` | a Mamba-1 `state_input`, `state_output`, `state` |
| `ScanStates` | `batch seq channels state_dim` | a Mamba-1 `states` |
| `SSDQueries` / `SSDKeys` | `batch seq groups state_dim` | a Mamba-2 `linear_attn.attention_queries` (`C`) / `attention_keys` (`B`) |
| `SSDValues` | `batch seq heads head_dim` | a Mamba-2 `linear_attn.attention_values` (`x`) |
| `SSDHeadOutputs` | `batch seq heads head_dim` | a Mamba-2 `linear_attn.attention_head_outputs` (`y`) |
| `RouterLogits` | `batch seq experts` | a mixture's `router_logits` (`experts` is `num_experts + 1` on ZAYA: the skip class) |
| `ExpertWeights` | `batch seq top_k` | a mixture's `expert_weights` |
| `ExpertIndices` | `batch seq top_k` (`Int`) | a mixture's `expert_indices` |
| `ExpertOutputs` | `batch seq top_k hidden` | a mixture's `expert_outputs` |

An axis name means the same thing on every layout: `batch` is axis 0 everywhere, `seq`
the token axis, `heads` the query heads and `kv_heads` the key/value heads, `head_dim`
the width of a head's values and outputs and `qk_head_dim` that of its queries and keys
(the same number outside latent attention), `query` and `key` the two token axes of a
pattern, `streams` the parallel copies of a hyper-connection residual (`hc_mult`), on a
gated DeltaNet mixer `key_dim` / `value_dim` the state's two sides, and on a mixture of
experts `experts` the router's classes and `top_k` the routing slots per token
([mixture-of-experts](mixture-of-experts.md)). A mixture routes tensors flat over tokens,
`[batch * seq, ...]`; its values are served in these layouts, one invoke's rows each. The
comments above each alias in its defining file state the same.

`isinstance` checks rank and dtype only; the axis *names* are documentation plus what the
suite asserts against `model.num_heads`, `model.num_kv_heads`, `model.head_dim`,
`model.qk_head_dim`, `model.hidden_size` and `model.vocab_size` ([root-values](root-values.md)).

## Annotating with a name

A value you define is annotated with the name, so its layout is the same object as the
base's and reads back through `.layout` and `.dims` like any standard value:

```python
from nnterp import EProperty
from nnterp.components import Pattern, interface_reason
from nnterp.families import gpt2


class Attention(gpt2.Attention):
    @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.output", description="The softmax output before the dropout, [batch, heads, query, key]", unavailable=interface_reason)
    def attention_softmax(self, value) -> Pattern:
        return value


Attention.attention_softmax.layout is Pattern                          # True
Attention.attention_softmax.layout is Attention.attention_probabilities.layout   # True
```

A family that redefines a standard value writes the base's name (`-> Keys`, `-> Residual`),
never an inline string, which is what keeps a redefinition from drifting; a value of your
own takes the name where one fits, and an inline `Float[Tensor, "..."]` with the same axis
names where none does (a per-row entropy, `batch heads query`). See
[custom-values](../extending/custom-values.md) and
[overriding-values](../extending/overriding-values.md).

## Where the sequence axis is

Batch is axis 0 everywhere. The sequence axis is 1 on every value that has one, with one
exception: softmax attention's `attention_queries`, `attention_keys` and
`attention_values`, where it is 2, the `[batch, heads, seq, head_dim]` layout transformers
hands its attention interface. `attention_head_outputs` is back to sequence-first,
`[batch, seq, heads, head_dim]`, on every family (a family whose own arithmetic keeps heads
first serves a transposed view). `attention_scores` and `attention_probabilities` have two
sequence axes, `query` then `key`.

So "the last token" is `value[:, -1]` for a residual-stream value, `value[:, :, -1]` for the
queries, keys and values, and `pattern[:, :, -1, :]` for the last query row of the pattern.

## `kv_heads` under grouped-query attention

Keys and values are read before `repeat_kv`, so their head axis is `num_kv_heads` wide, not
`num_heads`:

```python
model = StandardizedTransformer("Qwen/Qwen3-8B", dispatch=True, attn_implementation="eager")
model.num_heads, model.num_kv_heads, model.head_dim     # e.g. (4, 2, 128) on the tiny checkpoint

with model.trace(prompt):
    k = model.layers[0].self_attn.attention_keys.save()
with model.trace(prompt):
    q = model.layers[0].self_attn.attention_queries.save()
k.shape, q.shape       # (1, 2, seq, 128), (1, 4, seq, 128)
```

Two families expand the key/value heads before the interface, so `kv_heads` reads as
`num_heads` there: Falcon's 40B layout (`num_kv_heads` 8 in the config, 128 heads on the
tensor) and DeepSeek's latent attention.

## `qk_head_dim` on DeepSeek

Multi-head latent attention gives queries and keys a different width from values:
`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`, and `head_dim` is `v_head_dim`. The
root publishes both:

```python
model = StandardizedTransformer("deepseek-ai/DeepSeek-V3", attn_implementation="eager")
model.head_dim, model.qk_head_dim        # (128, 192)

# one value per trace: the interface serves them at one point of the forward
with model.trace(prompt):
    q = model.layers[0].self_attn.attention_queries.save()       # (1, heads, seq, 192)
with model.trace(prompt):
    k = model.layers[0].self_attn.attention_keys.save()          # (1, heads, seq, 192)
with model.trace(prompt):
    v = model.layers[0].self_attn.attention_values.save()        # (1, heads, seq, 128)
with model.trace(prompt):
    h = model.layers[0].self_attn.attention_head_outputs.save()  # (1, seq, heads, 128)
```

The keys share the queries' width, so their last axis is `qk_head_dim` too; the two sizes
coincide on every family without latent attention.

## `Streams` on DeepSeek-V4

DeepSeek-V4 carries `hc_mult` parallel copies of the residual stream between blocks, and
`layer_output` is the block's own tensor, so its layout is `Streams`, not `Residual`. The
contributions stay `Residual`: each sublayer reads one weighted collapse of the streams and
returns `[batch, seq, hidden]`, which the block writes into every stream with the weights
`attention_post` / `mlp_post` (`StreamWeights`) after mixing the streams with
`attention_comb` / `mlp_comb` (`StreamMixing`):

```python
import torch
from nnterp import StandardizedTransformer
from nnterp.components import Residual, Streams

model = StandardizedTransformer("deepseek-ai/DeepSeek-V4-Flash", dispatch=True, attn_implementation="eager")
type(model.layers[0]).layer_output.layout is Streams    # True

with model.trace(prompt):
    attn = model.layers[0].self_attn.attention_output.save()
    post = model.layers[0].mlp_post.save()
    out = model.layers[0].layer_output.save()
out.shape, attn.shape, post.shape    # (1, seq, hc_mult, hidden), (1, seq, hidden), (1, seq, hc_mult)
isinstance(out, Streams), isinstance(out, Residual)    # (True, False): rank 4
```

On the pinned tiny checkpoint (`yujiepan/deepseek-v4-bf16-tiny-random`, loaded with
`dtype=torch.float32`) `hc_mult` is 4 and `hidden` 8. The last token of a stream value is
`out[:, -1]`, `[batch, streams, hidden]`; one stream is `out[:, :, k]`, `[batch, seq, hidden]`.
Code written for `Residual` runs on a `Streams` value without an error and answers per
stream, so check `value.layout` (or `out.dim()`) where a recipe must run across families.

The same checkpoint's compressed-attention blocks append the compressor's entries after
the token keys, so there the `seq` axis of `attention_keys` / `attention_values` and the
`key` axis of the pattern are longer than the prompt once it reaches the block's
compression rate (4 tokens on `compressed_sparse_attention`, 128 on
`heavily_compressed_attention`).

## `heads` on DeltaNet values

On a hybrid's `linear_attn`, `heads` is the mixer's value-head count (`num_v_heads`), which
is what the keys are repeated to before the delta rule; `key_dim` and `value_dim` are the
mixer's `head_k_dim` and `head_v_dim`, not the root's `head_dim`:

```python
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
mix = model.layers[0].linear_attn

with model.trace(prompt):
    q = mix.attention_queries.save()        # (1, seq, num_v_heads, head_k_dim)   bf16
with model.trace(prompt):
    g = mix.decays.save()                   # (1, seq, num_v_heads)               float32
with model.trace(prompt):
    s = mix.state_output.save()             # (1, num_v_heads, head_k_dim, head_v_dim)  float32
```

On the tiny checkpoint the root says `num_heads=8`, `num_kv_heads=4`, and the mixer has
`num_v_heads=8`, `num_k_heads=4`, `head_k_dim=32`, `head_v_dim=32`; the values come out
`(1, 6, 8, 32)`, `(1, 6, 8)` and `(1, 8, 32, 32)` for a 6-token prompt. `state_input` is
`None` on a fresh prompt (no state enters), so `isinstance` on it is meaningless there.
`decays` and the states are float32 whatever the model's dtype; the rest follow the model.

## Gotchas

- **`isinstance(t, value.layout)` checks rank and dtype, not axis sizes.** Compare sizes
  against the root's sizes yourself, or trust the suite.
- **Sequence is axis 2 on `attention_queries` / `attention_keys` / `attention_values`**, axis
  1 everywhere else.
- **`kv_heads` is `num_kv_heads` except where the family expands first** (Falcon 40B layout,
  DeepSeek), where it is `num_heads`.
- **A `.layout` is `None` for a value with no tensor annotation**; `dims` is then `None` too.
- **`layer_output` is rank 4 on DeepSeek-V4** (`Streams`); read the layout off the family's
  class, `type(model.layers[0]).layer_output.layout`, not off the base `Layer`.
- **The names live in `nnterp.components`, not `nnterp`**: `from nnterp.components import Residual`;
  the root's three (`Logits`, `NextTokenProbs`, `Tokens`) only in `nnterp.standardized`.
- **Read one interior value per trace when in doubt.** The five interior values bind at
  different points of the forward on the off-interface families (Falcon: values before
  queries and keys).

## Related

- [root-values](root-values.md): the sizes each axis is checked against.
- [residual-stream](residual-stream.md): the `Residual` (`batch seq hidden`) values, and the `Streams` ones.
- [availability](availability.md): a value has a layout whether or not this checkpoint has it.
- [custom-values](../extending/custom-values.md): annotating a value of your own with a name.
