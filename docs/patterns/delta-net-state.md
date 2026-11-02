---
title: DeltaNet State
one_liner: "On a Qwen3-Next / Qwen3.5 / Qwen3.5-MoE hybrid, read `state_output` and `state_input`, patch the recurrent state between prompts (`state_input` on a decode step, or `set_state_after(t, value)` inside a prompt after `route_kernels`), and track the state's norm token by token with `states`."
tags: [patterns, hybrids, deltanet, state, generation]
related: [docs/usage/availability.md, docs/usage/generation.md, docs/patterns/activation-patching.md, docs/patterns/ablation.md, docs/patterns/cross-family-sweep.md]
sources: [nnterp/components/linear_attention.py, nnterp/components/eproperty.py, nnterp/families/qwen3_5_text.py, tests/families/test_qwen3_5_text.py]
---

# DeltaNet State

## What this is for

A gated DeltaNet block has no attention pattern. Each head keeps a recurrent state,
`[key_dim, value_dim]`: every token decays it by a learned gate, writes its key/value
pair in scaled by a beta, and the query reads against it. Everything the block
remembers about the prompt is in that matrix, so the experiments that ask "what is
carried forward" are experiments on the state: read it, swap it between prompts,
watch how it grows.

`model.layers[i].linear_attn` on Qwen3-Next, Qwen3.5 (text) and Qwen3.5-MoE is a
`LinearAttention` with the state as standard values, read at the delta-rule kernel
call: `state_input` and `state_output` on any call, and the state *after every
token* (`state`, `states`, `state_after`, `set_state_after`) once the family's
prompts are routed through transformers' token-by-token kernel. The three hybrid
families share the values, so this page is one recipe.

## Canonical pattern

The state leaving a prompt, and the one entering it (`None` on a fresh prompt):

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
linear_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
mix = model.layers[linear_blocks[0]].linear_attn        # decided outside the trace
prompt = "The Eiffel Tower is in the city of"

entering = []                                            # a None bound inside the trace would not survive it
with model.trace(prompt):
    state_in = mix.state_input
    entering.append(state_in.save() if state_in is not None else None)
    state_out = mix.state_output.save()                  # [batch, heads, key_dim, value_dim]
    clean = model.next_token_probs.save()

print(entering[0], state_out.shape, float(state_out.norm()))
```

`state_input` is a copy of the cache's buffer, because the cache overwrites that
buffer in place with the step's new state; `state_output` is what the next decode
step starts from.

## Patching the state on a decode step

Under `generate`, the prompt is one kernel call and each new token another, and
`state_input` on step `k` is `state_output` of step `k - 1`. Assigning it makes the
step start from another prompt's memory:

```python
other = "My favourite food is pizza with"
with model.trace(other):
    other_state = mix.state_output.save()

N = 3
with model.generate(prompt, max_new_tokens=N, do_sample=False) as tracer:
    ids = tracer.result.save()

with model.generate(prompt, max_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        mix.state_input = other_state                    # decode step 1 continues from the other prompt's state
    ids_patched = tracer.result.save()

print(model.tokenizer.decode(ids[0]), "|", model.tokenizer.decode(ids_patched[0]))
```

Step 0 is the prompt, whose `state_input` is `None`; steps 1 and beyond are one
token each with the cached state entering. The write lands: step 1's `state_output`
differs and its logits move (by up to 0.43 on Qwen3.5-0.8B). But one block's memory is
one of many linear blocks' (18 on Qwen3.5-0.8B) and every attention block still reads the whole prompt, so the
greedy tokens need not change: on Qwen3.5-0.8B both runs continue `" Paris, France"`.
Patching every linear block's state does change them:

```python
mixes = [model.layers[i].linear_attn for i in linear_blocks]
donors = []
with model.trace(other):
    for m in mixes:
        donors.append(m.state_output.save())

with model.generate(prompt, max_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        for m, donor in zip(mixes, donors):
            m.state_input = donor
    ids_all = tracer.result.save()

print(model.tokenizer.decode(ids_all[0]))       # the continuation after step 1 differs
```

## The state after every token

A prompt normally runs the chunked kernel, which carries the state between
64-token chunks and never materializes it per token. Route the family through
transformers' token-by-token kernel first, then every position is a value:

```python
from nnterp import route_kernels

route_kernels(model.family, "torch")                    # process-wide; before the first trace of that layer
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
mix = model.layers[linear_blocks[0]].linear_attn

with model.trace(prompt):
    states = mix.states.save()                           # [batch, seq, heads, key_dim, value_dim]
    final = mix.state_output.save()
    probs = model.next_token_probs.save()

assert torch.equal(states[:, -1], final)                 # the last token's state is the one the prompt leaves
norms = states.flatten(2).norm(dim=-1)[0]                # [seq]: how the memory grows along the prompt
tokens = [model.tokenizer.decode(t) for t in model.tokenizer(prompt).input_ids]
for token, norm in zip(tokens, norms):
    print(f"{token!r:12} {float(norm):.3f}")
```

The two kernels compute the same rule; `probs` equals the chunked run's to float
error. The cost is the slower kernel, the same trade as `attn_implementation="eager"`.
`route_kernels(model.family, "default")` restores the default. Without the
routing, `states` raises `nnterp.Unavailable` with that instruction and `support()`
reports it under `linear_attn.states`.

`state` is the same value one token at a time, walked with nnsight's own iteration:

```python
n = len(model.tokenizer(prompt).input_ids)
walked = []
with model.trace(prompt) as tracer:
    for t in tracer.iter[:n]:
        walked.append(mix.state.save())                  # walked[t] == states[:, t]
```

## Patching the state inside a prompt

`set_state_after(t, value)` writes the state after token `t`; tokens `t + 1` onward
continue from it. Take the donor from the other prompt at the same token:

```python
T = 3
with model.trace(other):
    donor = mix.state_after(T).save()

with model.trace(prompt):
    before = mix.state_after(T - 1).save()               # positions before the write: read first
    mix.set_state_after(T, donor)
    after = mix.state_after(T + 1).save()                # positions after it: read after
    patched = model.next_token_probs.save()

assert torch.equal(before, states[:, T - 1]) and not torch.equal(after, states[:, T + 1])
```

The write also changes token `T`'s own output (its query reads the state after `T`),
and it is not all the block carries forward: the short convolution in front of the
kernel (width 4) still mixes tokens `T - 2` to `T` into tokens `T + 1` to `T + 3`. So a
state patch is a patch of the recurrent memory, not of everything the prompt left.

The same write as an assignment inside `tracer.iter`, with the same result:

```python
with model.trace(prompt) as tracer:
    for t in tracer.iter[T]:
        mix.state = donor
    patched = model.next_token_probs.save()
```

## The gate and the write strength

```python
with model.trace(prompt):
    decays = mix.decays.save()                           # [batch, seq, heads], log decay, <= 0
    betas = mix.betas.save()                             # [batch, seq, heads], in (0, 1)

keep = decays.exp()                                      # the fraction of the state each token keeps, per head
```

A head with `keep` near one is a long memory; one with small `keep` forgets the
prompt within a few tokens, and its state norm curve above stays flat.

## Gotchas

- Reads follow the forward. In one trace, positions before a write are read before
  it and positions after it afterwards; `states` reads every position, so it goes
  in a trace of its own or before any write.
- `route_kernels` is process-wide and must run before the layer's forward is
  instrumented: call it before loading, or before the first trace that touches the
  layer. A forward already instrumented keeps the kernel it was compiled with.
- `states` is read-only, a stack of copies; write one position with
  `set_state_after` or `state` under `tracer.iter`.
- `state_input` is `None` on a fresh prompt; do not bind that `None` to a name inside
  the trace and expect it after (make a list outside, as above).
- Under `generate`, `states`, `state_after` and `set_state_after` count from the
  current call's own first token: on step 0 the prompt's tokens, on a decode step
  the single token it processes. The per-token view under `generate` is an inner
  `tracer.iter[:n]` on step 0 and `state_output` on each later step.
- The kernels must be transformers' pure-torch ones. With `flash-linear-attention`
  or `causal-conv1d` installed the kernel has no Python source, every interior value
  is unavailable, and `support()` says so; `attention_output` stays available.
- A block has either `self_attn` or `linear_attn`; on the attention block
  (`config.layer_types`) the `linear_attn` values are reported missing. Decide the
  blocks outside the trace.
- The state layout is `[batch, heads, key_dim, value_dim]` with `heads` the mixer's
  `num_v_heads`, not the model's `num_heads`.

## Related

- [activation-patching](activation-patching.md): patching the stream and the
  contributions; the state is the hybrid's third site.
- [ablation](ablation.md): zeroing a mixer's `attention_output`.
- [cross-family-sweep](cross-family-sweep.md): hybrids inside a multi-checkpoint
  loop.
- [../usage/availability.md](../usage/availability.md): the `route_kernels` and
  kernel reasons in `support()`.
- [../usage/generation.md](../usage/generation.md): `generate`, `tracer.iter`, the
  prefill and decode steps.
- nnsight `docs/usage/iter-all-next.md`, `docs/usage/source.md`.
- Yang et al. (2024), "Gated Delta Networks".
