# DeltaNet state

A gated DeltaNet block keeps one `[key_dim, value_dim]` matrix per head in place of an attention pattern, and that state, with a short convolution window, is all the block carries forward about the prompt. The recipe reads the state entering and leaving a block, patches it between two prompts on a decode step and inside a prompt, and tracks its norm token by token, on Qwen3.5-0.8B, a hybrid with three DeltaNet blocks in every four ([Gated DeltaNet](../guide/architectures/delta-net.md)). nnsight.net's [State-Space Models](https://nnsight.net/tutorials/tutorials/architectures/state_space_models/) tutorial covers the recurrent view of models that carry such a state.

## The state leaving a prompt

```python
import torch
import nnterp
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3.5-0.8B", dispatch=True, device="cpu")
linear = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
print(linear)   # [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
mixes = [model.layers[i].linear_attn for i in linear]                  # decided outside the trace
prompt = "The Eiffel Tower is in the city of"
other = "My favourite food is pizza with"

entering = []                                                          # a None bound inside the trace would not survive it
with torch.no_grad(), model.trace(prompt):
    state_in = mixes[0].state_input
    entering.append(state_in.save() if state_in is not None else None)
    state_out = mixes[0].state_output.save()
    clean = model.next_token_probs.save()
print(entering[0], state_out.shape, f"{float(state_out.norm()):.3f}")  # None torch.Size([1, 16, 128, 128]) 8.596
print(repr(model.tokenizer.decode(clean[0].argmax())), f"{float(clean[0].max()):.3f}")   # ' Paris' 0.590
```

A block has either `self_attn` or `linear_attn`, so the list of recurrent blocks is decided outside the trace. `state_input` is `None` on a fresh prompt and the cached state on a decode step, served as a copy because the cache overwrites its buffer in place with the step's new state. `state_output` is the state after the call's last token, `[batch, heads, key_dim, value_dim]` with `heads` the mixer's value heads (16 here, against the model's 8 attention heads), and it is what the next decode step starts from.

## Patching the state on a decode step

```python
donors = []
with torch.no_grad(), model.trace(other):
    for mix in mixes:
        donors.append(mix.state_output.save())

n = len(model.tokenizer(prompt).input_ids)
with model.generate(prompt, max_new_tokens=4, do_sample=False) as tracer:
    ids = tracer.result.save()
with model.generate(prompt, max_new_tokens=4, do_sample=False) as tracer:
    for step in tracer.iter[1]:                                        # decode step 1 starts from the other prompt's memory
        mixes[0].state_input = donors[0]
    ids_one = tracer.result.save()
with model.generate(prompt, max_new_tokens=4, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        for mix, donor in zip(mixes, donors):
            mix.state_input = donor
    ids_all = tracer.result.save()
print(repr(model.tokenizer.decode(ids[0, n:])), repr(model.tokenizer.decode(ids_one[0, n:])), repr(model.tokenizer.decode(ids_all[0, n:])))
# ' Paris, France.' ' Paris, France.' ' Paris.\nThe'
```

Under `generate`, step 0 is the prompt and each later step one token with the cached state entering. `state_input` on step 1 is `state_output` of step 0, and assigning it makes the step continue from another prompt's memory. One block's memory changes nothing visible, since the other 17 recurrent blocks and the six attention blocks still hold the prompt. Swapping every recurrent block's state does change the continuation, from the second new token on.

## The state after every token

```python
print(model.support()["linear_attn.states"][0])
# the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through carries it between chunks. Call nnterp.route_kernels(model.family, 'torch') before tracing this layer (slower, like attn_implementation='eager')
nnterp.route_kernels(model.family, "torch")                            # process-wide; before the layer's first trace
model = StandardizedTransformer("Qwen/Qwen3.5-0.8B", dispatch=True, device="cpu")
mixes = [model.layers[i].linear_attn for i in linear]

with torch.no_grad(), model.trace(prompt):
    states = mixes[0].states.save()                                    # [batch, seq, heads, key_dim, value_dim]
    final = mixes[0].state_output.save()
    probs = model.next_token_probs.save()
assert torch.equal(states[:, -1], final)
norms = states.flatten(2).norm(dim=-1)[0]
for token, norm in zip(model.tokenizer.convert_ids_to_tokens(model.tokenizer(prompt).input_ids), norms):
    print(f"{token!r:10} {float(norm):.3f}")
# 'The'      2.802
# 'ĠEiff'    3.674
# 'el'       4.180
# 'ĠTower'   4.912
# 'Ġis'      5.792
# 'Ġin'      6.647
# 'Ġthe'     7.837
# 'Ġcity'    8.107
# 'Ġof'      8.596
print(f"{float((probs - clean).abs().max()):.4f}")   # 0.0117
```

A prompt normally runs the chunked kernel, which carries the state between chunks and never materializes it per token, and `states` says so until the family is routed through transformers' token-by-token kernel. `route_kernels` is process-wide and must run before the layer's forward is instrumented, which is why the model is loaded again after it. `route_kernels(model.family, "default")` restores the chunked kernel. The two kernels compute the same rule, and the next-token probabilities agree to bfloat16 rounding. The memory grows monotonically along this prompt, and the last token's state is the one the prompt leaves.

## Patching the state inside a prompt

```python
T = 3
patched = []
with torch.no_grad(), model.trace(other):
    for mix in mixes:
        patched.append(mix.state_after(T).save())
with torch.no_grad(), model.trace(prompt):
    before = mixes[0].state_after(T - 1).save()                        # positions before the write: read first
    mixes[0].set_state_after(T, patched[0])
    after = mixes[0].state_after(T + 1).save()                         # positions after it: read after
    for mix, donor in zip(mixes[1:], patched[1:]):                     # then the later blocks, in forward order
        mix.set_state_after(T, donor)
    probs_patched = model.next_token_probs.save()
print(torch.equal(before, states[:, T - 1]), torch.equal(after, states[:, T + 1]))   # True False
print(repr(model.tokenizer.decode(probs_patched[0].argmax())), f"{float(probs_patched[0].max()):.3f}", f"{float(probs_patched[0, clean[0].argmax()]):.3f}")
# ' the' 0.055 0.043
```

`set_state_after(t, value)` writes the state after token `t`, and tokens `t + 1` onward continue from it. Reads follow the forward: in one trace, a position before the write is read before it, a position after it afterwards, and a block's reads come before any write to a later block, which is why block 0's `before` and `after` bracket its own write and the later blocks are written afterwards. With every recurrent block's memory replaced after `Tower`, `Paris` falls from 0.590 to 0.043. The state is not all a block carries forward. A width-4 convolution in front of the kernel still mixes tokens `T - 2` to `T` into the next three, and the patch replaces the recurrent memory, not everything the prompt left.

## On another family

Qwen3-Next, Qwen3.5-MoE and OLMo-Hybrid serve the same values on their `linear_attn` blocks, and Kimi-Linear's mixer is the same component with a decay per key channel, so the recipe changes only in the checkpoint. The kernels have to be transformers' torch ones: With `flash-linear-attention` or `causal-conv1d` installed, the kernel has no Python source, every interior value reports unavailable in `support()`, and `attention_output` stays. Under `generate`, `states`, `state_after` and `set_state_after` count from the current call's own first token, and on a decode step they address that step's single token.
