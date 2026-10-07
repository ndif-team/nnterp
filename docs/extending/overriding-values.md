---
title: Overriding Values
one_liner: How a family redefines a standard value when the base does not hold — an `EProperty` keyed on a path (`../norm.output`, `source.<op>.input`, `source.<call>.inputs` with `select`), `unavailable` markers and predicates, `off_interface`, `seq_first`, a clone with a transform, `postprocess` for writes; and a root size, which is a plain function in the family module, not a descriptor.
tags: [extending, families, eproperty, source, availability]
related: [docs/extending/adding-a-family.md, docs/extending/custom-values.md, docs/extending/finding-source-ops.md]
sources: [nnterp/components/eproperty.py, nnterp/components/attention.py, nnterp/components/layer.py, nnterp/components/mlp.py, nnterp/components/standard.py, nnterp/families/gemma2.py, nnterp/families/olmo2.py, nnterp/families/bloom.py, nnterp/families/mpt.py, nnterp/families/gptj.py, nnterp/families/falcon.py, nnterp/families/gpt2.py, nnterp/families/gpt_oss.py, nnterp/families/llama4_text.py, nnterp/families/deepseek_v2.py, nnterp/families/deepseek_v3.py, nnterp/standardized.py]
---

# Overriding Values

## What this is for

A standard value means the same thing on every family: `attention_output` is what the
attention adds to the residual stream, `attention_probabilities` the pattern the values
are mixed with. The base `Layer`, `Attention` and `Mlp` locate those values where
Llama's forward puts them. When a family's forward puts a value somewhere else, its
subclass redefines the descriptor under the same name, pointing at the right place,
and the name keeps its meaning. The pointer is the `EProperty`'s key, a path from the
host envoy: the host's own `output`, a module named relative to it
(`../post_attention_layernorm.output`), or an operation inside a forward
(`source.dropout_add_0.input`). This page lists every shape of override the shipped
families use, with the real snippet and the reason. The rule for all of them: keep the
name, keep the layout name in the annotation (`-> Residual`, the base's, imported from
`..components` with the envoys), keep the description, change only the path.

## Canonical pattern

Gemma-2's block is a sandwich: `x + post_attention_layernorm(attn(input_layernorm(x)))`.
What the block adds is the sibling norm's output, not the attention module's, so the
value points there:

```python
# nnterp/families/gemma2.py
from ..components import Attention, EProperty, Layer, Mlp, Residual


class Attention(Attention):
    """Gemma-2's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value
```

Every read, write and in-place edit of `layers[i].self_attn.attention_output` goes to
`layers[i].post_attention_layernorm.output`, and the contribution identity the suite checks
holds. Gemma-3 (text), OLMo-2, OLMo-3 and EXAONE-4 are the same override; OLMo-2/3 have
only the post-norms, so `self_attn.input` is the block input there.

## The path

A key is dotted segments ending in `output`, `input` or `inputs`, walked from the host
envoy, or from the model's root when it starts with `/`:

- `"output"` is the host's own output, the same location as `.output`.
- A leading `/` anchors the path at the root (`envoy.root`, the model envoy) and walks
  down from there the way a path below the host walks, aliases and `source` included:
  `"/inputs"` is the model's inputs, read from the vision tower (`vision.image_token_mask`;
  `vision.image_features` is read at the scatter in the wrapper model's forward,
  `"/model.source.inputs_embeds_masked_scatter_0.inputs"`, from a key function), and
  `"/norm.output"` is the final norm's from any block. Use it for a value that lives far
  from its host; a sibling is `../name` (below).
- A leading `../` steps to the parent module, as many times as written. After it, every
  segment is a *native* name: above the host the path is joined onto the native path as a
  string, so an alias does not resolve there. `"../post_attention_layernorm.output"` on
  Gemma-2's attention reaches its sibling norm (that is the native name); on GPT-2's
  attention it names a module that does not exist and the read fails with
  `OutOfOrderError`, while `"../ln_2.output"` works. A path that does not go up walks child
  modules of the host, aliases included: `"embed_tokens.output"` on the root is
  `token_embeddings`.
- `source` drills into the current module's forward (or, after an operation, into that
  call's), instrumenting it for this run; the segment after it names an operation.
  `"source.dropout_add_0.input"` is an operation of the host's own forward,
  `"source.attention_interface_1.source.nn_functional_dropout_0.output"` one inside a
  call the forward makes, `"../source.hidden_states_view_0.output"` one in the parent's.
- `input` is the call's first argument, `inputs` the `(args, kwargs)` pair, of whatever
  the path ends on: a module or an operation.
- The key may be a function of the host returning a path, decided at read time inside
  the trace: Falcon's `by_alibi(without, with_alibi, attribute="output")` reads
  `config.alibi`, and a `RecurrentMixer`'s `kernel("inputs")` reads the
  bindings the forward branches on and names whichever kernel fires on this call
  (`LinearAttention`'s chunked or token-by-token delta rule)
  ([finding-source-ops.md](finding-source-ops.md)).
- On a `RecurrentMixer` a value that moves is declared at the kernel call; the kernel
  constants (`CHUNK_KERNEL`, `RECURRENT_KERNEL`, `STATE_OP`) are the base's to use, and a
  family that needs another kernel name sets the constant rather than redeclaring values
  ([../developing/recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md)).

The value is served at that location by nnsight, whatever the path, so in-place edits
reach the model without anything more, and `postprocess` and `transform` work on every
value.

## A value another module produces

An `input` path serves the first argument of that module's call, nnsight's own `.input`
rule, so a stub on one receives the tensor:

```python
class Layer(gpt2.Layer):
    @EProperty("post_attention_layernorm.input", description="The residual stream after the attention sublayer")
    def mid_stream(self, value) -> Residual:
        return value
```

On GPT-2 tiny this reads `layer.input + attention_output` exactly, and assigning it swaps
in the first argument with the call's other arguments intact. The pair, for a stub that
needs a keyword argument, is an `inputs` path: the root's `input_ids` is
`EProperty(key="inputs")` with a stub that takes `kwargs["input_ids"]` and a postprocess
that puts it back.

The path can reach an operation in the parent's forward. Llama 4's mixture of experts
returns its output flattened to `[batch * seq, hidden]` and the block views it back
(`residual + hidden_states.view(residual.shape)`), so its `mlp_output` is that view:

```python
# nnterp/families/llama4_text.py
class Mlp(Mlp):
    @EProperty("../source.hidden_states_view_0.output", description="...", unavailable=_not_a_block_feed_forward)
    def mlp_output(self, value) -> Residual:
        return value
```

An operation is only served on a call whose forward was instrumented before the call
began, and this one is read after the block has started (after its attention), when the
drill a read performs is too late. The path does not say so; the family does, on the envoy
that owns the forward:

```python
# nnterp/families/llama4_text.py
class Layer(Layer):
    """Llama 4's decoder block; returns a bare tensor, so the base holds.

    `Mlp.mlp_output` is an operation in this forward, read after the block
    has started (its attention has returned), so the forward is instrumented
    at build.
    """

    sourced = True
```

`Standard.sourced` is `False` by default; `True` makes `__init__` and `_update` touch the
envoy's `.source`, so the block's forward is instrumented when it is built and again when
real weights replace the meta ones a lazy load starts from. Without the flag the first
trace that reads `attention_output` then `mlp_output` ends with an `OutOfOrderError` on
the view ([../developing/eproperty-internals.md](../developing/eproperty-internals.md#standard-the-sourced-flag)).

## A value at an operation inside the forward

BLOOM's sublayers take the residual as an argument and add it inside the module
(`dropout_add(x, residual, ...)`), so the module's output is a residual-stream state, not
a contribution. The contribution is the first argument of that call:

```python
# nnterp/families/bloom.py
class Attention(Attention):
    @EProperty(
        "source.dropout_add_0.input",
        description="What the attention adds to the residual stream: the tensor entering dropout_add",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    @EProperty("source.dropout_add_0.input", description="What the MLP adds to the residual stream: the tensor entering dropout_add")
    def mlp_output(self, value) -> Residual:
        return value
```

MPT's MLP adds the residual inside too; its contribution is the dropout's output, the
tensor just before the add: `EProperty("source.F_dropout_0.output", description=...)`.

The segment after `source` is the operation's name under the module's `.source`, with
another `source` between a call and an operation inside it;
[finding-source-ops.md](finding-source-ops.md) is how to find the name. The descriptor
walks the path before every read or write, so the operation is instrumented for the
current run, then reads or writes the operation's `output` / `input` / `inputs` through
nnsight. An operation the run does not have raises `SourceNotAvailable` naming what
exists.

### `select`

- `input` is the call's first argument; assigning swaps in the first argument and keeps
  the rest (BLOOM above).
- `inputs` with `select=n` is the n-th positional argument, `select="name"` a keyword
  argument. A write repacks that one element into the call's `(args, kwargs)`, so
  assigning changes just that argument. The base `Attention` reads the queries, keys and
  values this way off the interface call:

  ```python
  # nnterp/components/attention.py
  @EProperty(f"source.{INTERFACE}.inputs", select=1, description="The queries entering attention", unavailable=interface_reason)
  def attention_queries(self, value: torch.Tensor) -> Queries:
      return value
  ```

  GPT-J does its arithmetic in its own `_attn` method, so its family reads the same
  three values off that call: `EProperty("source.self__attn_0.inputs", select=0, ...)`
  for the queries, `select=1` keys, `select=2` values.
- `output` with `select=i` is one element of a returned tuple. BLOOM's `_reshape`
  returns `(query, key, value)`, so its family reads
  `EProperty("source.self__reshape_0.output", select=0)` for the queries and
  `select=1`, `select=2` for the rest; Falcon's queries and keys are
  `apply_rotary_pos_emb_0`'s two returns, and its values the `value_layer_0` binding just
  before it.

The element handed back is the object the call holds, so in-place edits reach the model;
a transform is there for a value whose stub returns a copy.

### Reuse the base description and the base layout name

A redefined value keeps its meaning, so it keeps its description and its layout name:

```python
from ..components import Queries


@EProperty("source.self__reshape_0.output", select=0, description=Attention.attention_queries.description)
def attention_queries(self, value) -> Queries:
    return value
```

`Attention.attention_queries` on the class is the descriptor itself (`__get__` with no
instance returns it), so `.description` is the base's text; `-> Queries` is the base's
annotation, so `layout` is the same alias on both (`bloom.Attention.attention_queries.layout
is Attention.attention_queries.layout`) and the redefinition cannot drift from it.

### The pattern: the dropout after the softmax

The base reads `attention_probabilities` at
`source.attention_interface_1.source.nn_functional_dropout_0.output` and
`attention_scores` at the softmax's input. A family whose attention does its own
arithmetic points both at its own operations: GPT-J at
`source.self__attn_0.source.self_attn_dropout_0.output`, MPT at its
`source.nn_functional_dropout_0.output`, BLOOM at `source.self_attention_dropout_0.output`.
Falcon without alibi has no dropout after its softmax, so its pattern is `F_softmax_0`'s
output; with alibi it is `self_attention_dropout_0`, after the second softmax, and
`by_alibi("F_softmax_0", "self_attention_dropout_0")` is the key function that picks per
checkpoint (its third argument is the attribute, `"input"` for the scores). Read the
pattern after the dropout wherever one exists: that is the tensor the values are mixed
with, in the model's dtype and, on a sink model, with the sink column dropped.

GPT-OSS is on the shared interface but its softmax takes one extra column (the sink), so
its family reads `attention_scores` one step earlier, at the masked scores bound just
before the sink joins them, and flags the sink for the suite:

```python
# nnterp/families/gpt_oss.py
class Attention(Attention):
    SINK = True

    @EProperty(f"source.{INTERFACE}.source.attn_weights_1.output", description=Attention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value
```

## Availability

### `unavailable(...)`: a value the family does not have

A marker in the class body takes the place of the inherited descriptor, keeps the name in
the tree and the repr (`(attention_scores): Unavailable: <reason>`), makes `support()`
report the reason, and makes any access raise `nnterp.Unavailable` with it before the model
runs:

```python
from nnterp.components import NOT_ON_INTERFACE, unavailable


class Attention(Attention):
    attention_scores = unavailable(NOT_ON_INTERFACE)
```

`NOT_ON_INTERFACE` is the reason a family gives for an interface value it has not mapped
onto its own arithmetic. A value that cannot exist takes its own reason
(`unavailable("no softmax: the attention is linear")`).

### `off_interface()`: one decision for every interface value

The base `Attention`'s interface values all take `unavailable=interface_reason`, which
calls `self.off_interface()`; the default is `needs_eager` (the model runs `sdpa` or
another fused kernel). A family with another reason overrides the method rather than
each value:

```python
# nnterp/families/gpt2.py
class Attention(Attention):
    def off_interface(self):
        if self._module.config.reorder_and_upcast_attn:
            return "this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"
        return super().off_interface()
```

### A per-instance `unavailable=` callable

`unavailable=` takes a function of the envoy returning a reason or `None`, evaluated on
the instance so the checkpoint's config decides. `needs_eager` is the one every value read
inside the eager attention forward uses, and it is nothing more than such a function:

```python
# nnterp/components/attention.py
def needs_eager(envoy: Envoy) -> str | None:
    implementation = envoy._module.config._attn_implementation
    if implementation != "eager":
        return f"read inside the eager attention forward, but this model runs {implementation!r}; load with attn_implementation='eager'"
    return None
```

Falcon's six interior values pass it as is (`unavailable=needs_eager`), on both of its
attention branches. A family with a config flag of its own writes a predicate of the same
shape; one that combines two reasons reads `needs_eager(self) or <its own check>`, so the
eager reason wins when both apply; and when the flag decides for every interface value at
once, `off_interface()` above is the place.

Families whose attention ignores `attn_implementation` (BLOOM, MPT) pass no
`unavailable=` on their own operations: the pattern needs no eager load there.

## Layout: `seq_first` views

`attention_head_outputs` is `[batch, seq, heads, head_dim]`, what the shared interface
returns. GPT-J, MPT and Falcon keep heads first at their own operation, so the family
serves a transposed view on read and transposes back on write. `seq_first` is its own
inverse, so both callbacks are the same function:

```python
# nnterp/families/mpt.py
@EProperty("source.torch_matmul_1.output", description=Attention.attention_head_outputs.description)
def attention_head_outputs(self, value) -> HeadOutputs:
    return seq_first(value)

@attention_head_outputs.postprocess
def attention_head_outputs(self, value):
    return seq_first(value)
```

A view keeps in-place edits landing on the model's tensor. BLOOM's `bmm` result is
`[batch * heads, seq, head_dim]`, so its family views and transposes on read and
reverses both on write in the `postprocess`.

## A clone carried back by a transform

Falcon's parallel block adds the attention output *into the MLP's output tensor in
place*. A plain `mlp_output` would be a live tensor the block later mutates, so a saved
read would silently become `mlp + attn`. The value reads a clone; a clone is invisible
to the model, so in-place edits to it would be lost, and an `eproperty` transform hands
the edited copy back to be swapped in once the block is done with the read:

```python
# nnterp/families/falcon.py
class Mlp(Mlp):
    @EProperty(key="output", description="What the MLP adds to the residual stream (a copy, since the block adds the attention into the live tensor in place)")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value).clone()

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        # Fires on the model side, after the read. The module returns a bare
        # tensor, so ``raw`` needs no rebuilding around the edited copy.
        return value.clone()
```

The transform's second clone keeps the user's tensor clean when the block then adds into
the swapped-in one; `test_falcon.py` checks both that `mlp.output == mlp_output +
attention_output` and that `mlp_output[:] = 0` moves the logits while the saved copy stays
zero. A transform is needed only when the preprocess returns something other than the
served object (a clone, a reshaped copy) *and* in-place edits must still reach the model.
GPT-2's MLP output is not mutated later, so the base holds and `mlp_output[:] = 0` reaches
the model with no clone and no transform. `transform(self, view, raw)` receives the raw
served value so a module that returns a tuple can be rebuilt around the edited element
(`(edited.clone(), *raw[1:])`); see nnsight docs/developing/extending-envoy.md. A
transform works on any path, an operation inside a forward included.

## `postprocess`: writes on a `key="output"` value

The base boundary values are `EProperty(key="output")` with `first_tensor` as the
preprocess and `rewrap` as the postprocess, so a tuple module (GPT-J's block, GPT-2's
attention returning `(attn_output, attn_weights)`) reads as a tensor and an assignment
puts the tensor back in its tuple with the other elements unchanged:

```python
# nnterp/components/layer.py
@EProperty(key="output", description="The residual stream leaving the block, a tensor even when the block returns a tuple")
def layer_output(self, value: Any) -> Residual:
    return first_tensor(value)

@layer_output.postprocess
def layer_output(self, value: torch.Tensor) -> Any:
    return rewrap(self, value)
```

A family that redefines a value on `key="output"` (Falcon's `Mlp` above) keeps both
halves. A value redefined on a path that serves a bare tensor (a sibling norm's output,
an operation's) needs neither.

## A size: a function in the family module

The root's sizes are not value descriptors. Each is a `StandardizedProperty` on
`StandardizedTransformer` (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`,
`num_kv_heads`, `head_dim`, `qk_head_dim`, `intermediate_size`): no location, nothing
served inside a trace, no `support()` entry, only a plain rule over the config, and
read-only (an assignment raises `AttributeError` pointing at `def <name>(model)`). A family
whose config spells a size its own way overrides it with a module-level function of the
same name taking the model, which the descriptor calls instead of its rule. DeepSeek-V2's
latent attention gives values and queries different widths, and the config's own
`head_dim` key is the latent width that no served value has:

```python
# nnterp/families/deepseek_v2.py

def head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's values and outputs: ``v_head_dim`` (the config's ``head_dim`` is the latent width, which no served value has)."""
    return model.config.v_head_dim


def qk_head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's queries and keys: the non-rotary part plus the rotary part."""
    return model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
```

DeepSeek-V3 has the same attention and imports both (`from .deepseek_v2 import head_dim,
qk_head_dim`), as do DeepSeek-V3.2, GLM-5, GLM-4.7-Flash and Youtu. Twenty-five shipped
families define a size this way (`intermediate_size` on GPT-2, OPT, BLOOM, Falcon, DBRX,
Llama 4, ...; `num_kv_heads` on Falcon, GPT-BigCode and Gemma-4); the full list with what
each reads is in [../reference/families.md](../reference/families.md#logits-scales-and-sizes), and the recipe in
[adding-a-family.md](adding-a-family.md#sizes). Keep the docstring in the same voice as
the root's: what the width is, and which config key says so.

## Gotchas

- **Keep the name.** An override under another name adds a value instead of redefining
  one; the inherited descriptor stays, and `support()` keeps reporting it.
- **Keep the layout name.** Annotate the redefinition with the base's name from
  `..components` (`-> Keys`, `-> Residual`), never an inline
  `Float[Tensor, "..."]`: `layout` and `dims` are read off the annotation, the name
  keeps them identical to the base's, and the suite checks every value's tensor against
  it ([custom-values.md](custom-values.md), [../usage/layouts.md](../usage/layouts.md)).
- **`input` is the first argument, `inputs` the pair.** A stub on an `input` path
  receives the tensor and an assignment keeps the other arguments; a stub that needs a
  keyword argument takes an `inputs` path and a postprocess that repacks the pair.
- **Reads in one trace follow forward order.** Falcon's values bind before the rotary that
  produces its queries and keys, so `attention_values` must be read first; the suite
  reads interior values one per trace for this reason.
- **An `unavailable=` predicate must not raise `AttributeError`.** A descriptor's
  `AttributeError` falls through to `Envoy.__getattr__` as "no attribute"; `EProperty`
  re-raises it as a `RuntimeError` naming the check, so a wrong config attribute in a
  predicate reports itself on a read. `support()` calls the predicate directly and lets
  the raw `AttributeError` through (`'GPT2MLP' object has no attribute 'config'`): an
  MLP module carries no `config`, so a per-checkpoint predicate on an `Mlp` reaches it
  another way.
- **A value read after its forward has started needs that forward instrumented before
  the trace**, and only the envoy that owns the forward can say so: `sourced = True` on
  its class (Llama 4's `Layer`). A `../source.` path on a child declares the location and
  nothing about instrumentation; without the flag the read is an `OutOfOrderError`.
- **A class attribute like `SINK` is for tooling**, not availability; `support()` reports
  only descriptors.
- **A size override goes on the module, not on a class.** `StandardizedProperty` looks
  for `model.family.<name>`; a `head_dim` on the family's `Attention` class is an
  ordinary attribute the root never reads.

## Related

- [adding-a-family.md](adding-a-family.md): where these subclasses go.
- [finding-source-ops.md](finding-source-ops.md): the operation names a `source.` path takes.
- [custom-values.md](custom-values.md): adding a value rather than redefining one.
- nnsight docs/developing/extending-envoy.md: `eproperty`, `postprocess` and `transform` in full.
