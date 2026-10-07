---
title: Residual stream and contributions
one_liner: "`layer_output`, `attention_output` and `mlp_output` are tensors on every family, defined by `layers[i].input + attention_output + mlp_output == layer_output`."
tags: [usage, residual-stream, layer_output, attention_output, mlp_output, contributions, hyper-connections, Streams]
related: [docs/usage/vocabulary.md, docs/usage/methods.md, docs/usage/root-values.md, docs/usage/availability.md]
sources: [nnterp/components/layer.py, nnterp/families/deepseek_v4.py, nnterp/components/attention.py, nnterp/components/mlp.py, nnterp/components/standard.py, nnterp/components/eproperty.py, nnterp/families/gemma2.py, nnterp/families/gemma4_text.py, nnterp/families/bloom.py, nnterp/families/mpt.py, nnterp/families/falcon.py]
---

# Residual stream and contributions

## What this is for

Three standard values give every decoder block the same three tensors:

| value | what it is |
| --- | --- |
| `model.layers[i].layer_output` | the residual stream leaving the block |
| `model.layers[i].self_attn.attention_output` | what the attention sublayer adds to the residual stream |
| `model.layers[i].mlp.mlp_output` | what the MLP sublayer adds to the residual stream |

All three are `[batch, seq, hidden]`, the `Residual` layout (`Layer.layer_output.layout is
nnterp.components.Residual`; see [layouts](layouts.md)), except `layer_output` on DeepSeek-V4,
whose residual is several parallel streams, `[batch, seq, streams, hidden]`
([below](#where-the-families-differ)).

`attention_output` and `mlp_output` are *contributions*, defined by one identity that
holds on a sequential block and a parallel block alike:

```
layers[i].input + attention_output + mlp_output == layers[i].layer_output
```

Raw nnsight `.output` on the same modules does not have that meaning everywhere: a block
may return a tuple, an attention module returns `(attn_output, attn_weights)`, a sandwich
block adds a norm's output rather than the module's, and some modules add the residual
inside. The three values put the same thing at the same name on every family, and the
per-family test suite checks the identity on each.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)   # the identity, in the block's dtype
```

Run on the GPT-2 and Gemma-2 tiny checkpoints this gives a maximum absolute difference of
`0.0` in float32 and bfloat16 alike; Falcon's bf16 block sums in another order and lands
within a few bf16 ulps (`torch.testing.assert_close` with `rtol=atol=8 * eps` of the
block's dtype passes, as the suite checks).

Read the four in forward order within one trace: the block's `input`, then the attention,
then the MLP, then the block's output.

On a vision-language wrapper the tower's blocks carry the same three values over the patch
stream (`model.vision.layers[i].layer_output`, `[images, patches, vision_hidden]`), and the
text blocks' values hold the image positions too: `layer_output[vision.image_token_mask]`
is the image rows, `layer_output[~vision.image_token_mask]` the text rows
([vision.md](vision.md#image-positions-and-text-positions)).

## Tensor blocks and tuple blocks

A Llama, GPT-2 or GPT-NeoX block returns `hidden_states` alone. A GPT-J, GPT-Neo, CodeGen,
GPT-NeoX-Japanese, BLOOM, MPT or Falcon block returns a tuple with it first. `layer_output` is the tensor either way, the
same object the block returned:

```python
model = StandardizedTransformer("bigscience/bloom-560m", dispatch=True)

with model.trace(prompt):
    raw = model.layers[2].output.save()          # a tuple on BLOOM
    out = model.layers[2].layer_output.save()    # the tensor

isinstance(raw, tuple), torch.equal(raw[0], out)   # (True, True)
```

Assigning `layer_output` on a tuple block replaces the first element and leaves the others
as they were, so the next block receives a well-formed tuple:

```python
with model.trace(prompt):
    model.layers[2].layer_output = model.layers[2].layer_output * 0
    after = model.layers[2].output.save()        # (zeros, <the other element, unchanged>)
    nxt = model.layers[3].input.save()           # zeros
```

The same holds for `attention_output` on an attention module that returns
`(attn_output, attn_weights)`: the value is the first tensor, and an assignment rewraps it.

## Reading, editing in place, assigning

All three values go through the interleaver the way `.output` does. In-place edits reach
the model because the value is the live tensor; assignment replaces it:

```python
with model.trace(prompt):
    model.layers[5].self_attn.attention_output[:, -1] = 0     # ablate attention at the last position
    model.layers[5].mlp.mlp_output[:] = 0                     # ablate the MLP everywhere
    model.layers[5].layer_output[:, -1] += direction          # steer the stream leaving the block
    logits = model.logits.save()
```

```python
with model.trace(prompt):
    resid = model.layers[3].layer_output.save()
    model.layers[4].layer_output = resid * 2                  # assign a new tensor
```

`model.layers[i].input` is the residual stream entering the block, a tensor on every
family (the block's first argument is `hidden_states` throughout the registered families).

## Where the families differ

Three shapes of block put the contribution somewhere other than the module's own output,
and the family's `Attention` or `Mlp` subclass points the value at the right place, so the
name means the same thing everywhere. Contrast each with the raw `.output`:

**Sandwich norms (Gemma-2, Gemma-3, Gemma-4, OLMo-2, OLMo-3, EXAONE-4, FlexOlmo).** The block adds
`post_attention_layernorm(attn(...))` and `post_feedforward_layernorm(mlp(...))`. What
reaches the residual stream is the post-norm's output, so `attention_output` is
`post_attention_layernorm.output` and `mlp_output` is `post_feedforward_layernorm.output`
(an `EProperty` keyed `"../post_attention_layernorm.output"`, the sibling norm):

```python
model = StandardizedTransformer("google/gemma-2-2b", dispatch=True)

with model.trace(prompt):
    raw_attn = model.layers[0].self_attn.output.save()                      # (tensor, weights): before the post-norm
    post = model.layers[0].post_attention_layernorm.output.save()
with model.trace(prompt):
    attn = model.layers[0].self_attn.attention_output.save()

torch.equal(attn, post), torch.equal(attn, raw_attn[0])                     # (True, False)
```

**Residual added inside the module (BLOOM both sublayers, MPT's MLP; DBRX adds outside
the attention module, so its base holds).** The module's output is already a
residual-stream state: BLOOM's `self_attention.output[0]` equals
`layers[i].input + attention_output`. The contribution is the tensor entering the add: the
first argument of `dropout_add` on BLOOM, the MLP dropout's output on MPT (an
`EProperty` keyed inside the forward, `"source.dropout_add_0.input"`):

```python
model = StandardizedTransformer("bigscience/bloom-560m", dispatch=True)

with model.trace(prompt):
    x = model.layers[2].input.save()
    attn = model.layers[2].self_attn.attention_output.save()
with model.trace(prompt):
    raw_attn = model.layers[2].self_attn.output.save()

torch.allclose(raw_attn[0], x + attn)                                        # True: the module returns x + contribution
```

Because the contribution is an operation inside the module, read it before the module's
own `.output` in one trace, or in a trace of its own as above.

**Falcon's copy.** Falcon's block sums `x + attn + mlp` by adding the attention output
*into the MLP's output tensor in place*, so by the time the trace ends the live
`mlp.output` no longer holds the MLP's contribution. `mlp_output` reads a copy taken as the
MLP returns, and an `eproperty` transform carries in-place edits to that copy back into the
model, so both editing forms still land:

```python
model = StandardizedTransformer("tiiuae/falcon-7b", dispatch=True)

with model.trace(prompt):
    model.layers[0].mlp.mlp_output[:] = 0                # reaches the logits
with model.trace(prompt):
    model.layers[0].mlp.mlp_output = model.layers[0].mlp.mlp_output * 0   # also reaches the logits
```

A raw `mlp.output.save()` on Falcon is the live tensor, and it reads as `mlp + attn` after
the block has run; `mlp_output` is the MLP's contribution.

**GPT-NeoX-Japanese's bias.** The attention's output projection has no bias; on the last
block the attention returns a separate `dense_bias` as its third element and the block adds
it (`residual + dropout(attn + bias)`). `attention_output` is that sum, the module's output
plus the bias, computed as it is read; an `eproperty` transform subtracts the bias from an
edit and hands the rest back, so in-place edits and assignment both land, and a read with no
edit leaves the forward bit-identical. On the other blocks `dense_bias` is `None` and
`attention_output` is the module's own tensor.

```python
model = StandardizedTransformer("abeja/gpt-neox-japanese-2.7b", dispatch=True)

last = model.layers[-1]
with model.trace(prompt):
    raw = last.self_attn.output.save()                   # (attn_output, attn_weights, dense_bias)
    attn = last.self_attn.attention_output.save()

torch.allclose(attn, raw[0] + raw[2])                     # True
```

**Granite and its relatives: scaled copies.** Granite, GraniteMoE(-Shared, -Hybrid),
Granite-SWA, GraniteMoE-SWA and HyperCLOVA X add each sublayer's output times
`residual_multiplier` (0.22 on granite-3.0-1b-a400m), so `attention_output` and
`mlp_output` are that product: a computed copy, which an assignment or an in-place edit
reaches the model through by dividing the whole copy back into the module's output. The
division rounds, so a write at one position changes the others by rounding: about 1e-7
relative in float32, one or two units in the last place in bf16. That can be a visible
fraction of a small edit's effect: on granite-3.0-1b-a400m in bf16, a small edit at one
position moved the logits at earlier positions, which the edit cannot reach causally,
by 0.17, against 0.19 at the edited position. That depends on the multiplier: 0.22 (Granite 3.x,
granite-4.1-3b/8b, granite-4.0-1b) does not survive a bf16 divide-and-multiply exactly, while
0.28, 0.263, 0.246 and 0.175 do and 1.0 (Granite 4.2) is exact. Load in float32 for fine-grained edits.
ZAYA's contributions are computed copies the same way. `self_attn.output` and
`mlp.output` stay the unscaled module outputs ([families](../reference/families.md#scaled-residual-adds)).

**Gemma-4: a third add, and a scaled sum.** Gemma-4's block is Gemma-3's sandwich, then on
the checkpoints with per-layer embeddings (E2B, E4B) a third add, then the whole sum times
`layer_scalar`, a per-block buffer, in place:

```
x1  = x  + post_attention_layernorm(attn(...))           # attention_output
x2  = x1 + post_feedforward_layernorm(mlp(...) [+ experts]) # mlp_output
x3  = x2 + post_per_layer_input_norm(...)                 # layers[i].per_layer_output (E2B, E4B)
out = x3 * layer_scalar                                   # layer_output
```

The third term is `layers[i].per_layer_output`, a value on Gemma-4's `Layer` only
(unavailable, with a reason, on 26B-A4B and 31B, which have no per-layer embeddings). The
contributions are served unscaled, so the identity carries the scalar:

```python
model = StandardizedTransformer("google/gemma-4-E2B", dispatch=True)

layer = model.layers[3]                                   # bound outside: a name bound in the trace does not survive it
with model.trace(prompt):
    x = layer.input.save()
    attn = layer.self_attn.attention_output.save()
    mlp = layer.mlp.mlp_output.save()
    ple = layer.per_layer_output.save()
    out = layer.layer_output.save()

scalar = layer._module.layer_scalar                       # a [1] buffer
torch.testing.assert_close((x + attn + mlp + ple) * scalar, out)
```

`layer_scalar` is far from one on the released weights (0.005 to 0.99; the first block's is
between 0.018 and 0.11 on every size), so a contribution reaches a later block's stream
shrunk by its own block's scalar and by every later one. Compare contributions across
blocks with that in mind; nnterp computes nothing for it. On a mixture-of-experts block
(26B-A4B) `mlp_output` is the dense MLP and the experts together: the block norms each and
then their sum, and `mlp_output` is that last norm's output, while `mlp.output` is the dense
MLP alone.

**Doge and ZAYA: the stream is rescaled too.** Two families scale the residual the block
carries, not only what it adds, so the identity carries the block's parameters as Gemma-4's
carries its scalar. Doge gates the stream per channel and adds the modules' outputs
unscaled: `h = input_residual * x + attention_output`, `out = post_attention_residual * h +
mlp_output`, the gates being `layers[i]._module.input_residual` and `.post_attention_residual`
(both start at one). ZAYA merges each sublayer's output `o` into the stream `r` with a
`ZayaResidualScaling`, `(o + hidden_states_bias) * hidden_states_scale + (r + residual_bias) *
residual_scale`; `attention_output` and `mlp_output` are the first term, a computed copy
that assignment and in-place edits carry back into the module's output as on Granite, and
the stream's term uses `layers[i].post_attention_residual_scale._module` and
`.post_mlp_residual_scale._module`. On both, the plain identity holds only at the
parameters' initial values; `tests/families/test_doge.py` and `test_zaya.py` check the
gated forms on copies with the parameters moved.

**DeepSeek-V4: parallel streams.** The residual between DeepSeek-V4's blocks is `hc_mult`
parallel copies of the stream, `[batch, seq, streams, hidden]` (the model copies the
embedding into each before block 0). `layer_output` and `layers[i].input` are that tensor,
the block's own (layout `Streams`), and writes to them land natively. Each sublayer reads one
weighted collapse of the streams and returns `[batch, seq, hidden]`; a hyper-connection
(`attn_hc`, `ffn_hc`) weights that output into each stream and mixes the streams it is added
to. `attention_output` and `mlp_output` are the sublayers' own outputs, unscaled, as on
Gemma-4, and the weights are four values on the family's `Layer`:

| value | layout | what it is |
| --- | --- | --- |
| `layers[i].attention_post`, `layers[i].mlp_post` | `StreamWeights`, `[batch, seq, streams]` | how much of the sublayer's output each stream receives, in (0, 2) |
| `layers[i].attention_comb`, `layers[i].mlp_comb` | `StreamMixing`, `[batch, seq, streams, streams]` | a doubly stochastic matrix mixing the streams, applied transposed: stream `k` receives `sum_j comb[j, k] * stream_j` |

The four are float32 whatever the model's dtype, and writes to them land. The identity is
the block's own formula, not a sum:

```
h   = attention_combᵀ · input + attention_post ⊗ attention_output
out = mlp_combᵀ · h + mlp_post ⊗ mlp_output                       # layer_output
```

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("deepseek-ai/DeepSeek-V4-Flash", dispatch=True)

layer = model.layers[2]
with model.trace(prompt):
    x = layer.input.save()
    post_a, comb_a = layer.attention_post.save(), layer.attention_comb.save()
    attn = layer.self_attn.attention_output.save()
    post_f, comb_f = layer.mlp_post.save(), layer.mlp_comb.save()
    mlp = layer.mlp.mlp_output.save()
    out = layer.layer_output.save()

h = comb_a.transpose(-1, -2) @ x + post_a.unsqueeze(-1) * attn.unsqueeze(-2)
torch.testing.assert_close(comb_f.transpose(-1, -2) @ h + post_f.unsqueeze(-1) * mlp.unsqueeze(-2), out)

# The stream mean is additive, up to the Sinkhorn projection's residual (about 2e-6 relative in float32):
mean = x.mean(2) + post_a.mean(-1, keepdim=True) * attn + post_f.mean(-1, keepdim=True) * mlp
torch.testing.assert_close(mean, out.mean(2), rtol=1e-5, atol=1e-5)
```

On the pinned tiny checkpoint (`yujiepan/deepseek-v4-bf16-tiny-random`, `dtype=torch.float32`)
the stream form is exact (difference `0.0`) on every block. The attention's output reaches
`layer_output` mixed by `mlp_comb` as well, so what a sublayer "adds" is not one tensor in
stream space. The plain `input + attention_output + mlp_output` is not the output: it
adds `[batch, seq, hidden]` to `[batch, seq, streams, hidden]`, which raises a shape error
at most prompt lengths and broadcasts silently into a wrong tensor when the prompt is one
token or exactly `hc_mult` tokens long.

## Gotchas

- **Forward order within one trace.** `layers[i].input`, then `self_attn.attention_output`,
  then `mlp.mlp_output`, then `layer_output`. On BLOOM and MPT a contribution is an
  operation inside the module, so it comes before that module's `.output`; on Gemma-2/3
  and OLMo-2/3 it is the post-norm's output, so it comes after `self_attn.output`. Reading
  the raw and the standard value of one module in one trace forces you to know which; a
  second trace does not. On GPT-Neo `self_attn` is the inner `attn.attention`, so its
  `attention_output` comes before the `attn` wrapper's `.output`.
- **A tuple block's `.output` is a tuple; `layer_output` is the tensor.** Skip a block with
  `Layer.skip_with` ([methods](methods.md)) rather than `.skip(tensor)`, which would hand a
  bare tensor where a tuple is expected.
- **`mlp_output` does not exist on OPT or XGLM** (no MLP module; `layers[i].fc2.output` is what the block adds); `support()` says so
  ([availability](availability.md)).
- **The identity is exact in float32 on a sequential block** (GPT-2: difference 0.0). A
  parallel block sums `x + attn + mlp` in another order, so there it holds to rounding even
  in float32 (within 2e-5 absolute on Pythia-70m and CodeGen-350M), and within a few ulps
  in bf16 (Falcon). Compare with a tolerance in the block's dtype.
- **`layer_output` is rank 4 on DeepSeek-V4.** `resid[:, -1]` is `[batch, streams, hidden]`
  there, and anything written for a `[batch, seq, hidden]` stream (a probe, a hand-written
  lens) runs and answers per stream. Check `out.dim()` or the family's layout where a recipe
  crosses families; `model.project_on_vocab` collapses the streams the way the model does.
- **Nothing bound inside a trace survives without `.save()`**, including the value you read
  to compute a difference; save each operand.

## Related

- [vocabulary](vocabulary.md): `self_attn.input`, `mlp.input`.
- [methods](methods.md): `skip_layers`, `steer` over `layer_output`.
- [root-values](root-values.md): `token_embeddings`, the stream before block 0.
- [availability](availability.md): which blocks have which value.
- nnsight docs/usage/access-and-modify.md: `.output`, in-place edits and assignment.
