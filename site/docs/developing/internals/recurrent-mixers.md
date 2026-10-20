# Recurrent mixers

A recurrent mixer's values are the arguments and results of a kernel call its forward makes, and `RecurrentMixer` in `nnterp/components/recurrent.py` holds everything about reaching that call that does not depend on which mixer it is: which of the two kernels fires, decided once per call by the forward's own test; the requirement that the kernels be transformers' pure-torch ones, so `.source` has a body to read inside; `route_kernels`, which rebinds a family's kernels process-wide; and the state after every token, read by occurrence arithmetic. `LinearAttention` (gated DeltaNet), `SelectiveScan` (Mamba-1) and `StateSpace` (Mamba-2) name their kernels in class constants and declare their values.

```python
import torch
import nnterp
from nnterp import StandardizedTransformer, route_kernels

model = StandardizedTransformer("Qwen/Qwen3.5-0.8B", dispatch=True, device="cpu")
mix = model.layers[0].linear_attn
cls = type(mix)
print(cls.__name__, cls.CHUNK_KERNEL, cls.RECURRENT_KERNEL, cls.STATE_OP)   # LinearAttention torch_chunk_gated_delta_rule_0 torch_recurrent_gated_delta_rule_0 last_recurrent_state_3
print(cls.attention_queries.key, cls.state.key, cls.states.key)            # <kernel.inputs> <_token_state_op> <_states>
print(model.support()["linear_attn.states"][0])
# the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through carries it between chunks. Call nnterp.route_kernels(model.family, 'torch') before tracing this layer (slower, like attn_implementation='eager')

route_kernels(model.family, "torch")   # before the first trace that touches a mixer of this family
prompt = "Hello world there"
n = len(model.tokenizer(prompt).input_ids)
with model.trace(prompt):
    q = mix.attention_queries.save()
    prompt_states = mix.states.save()
    out = mix.state_output.save()
print(n, tuple(q.shape), tuple(prompt_states.shape), torch.equal(prompt_states[:, -1], out))  # 3 (1, 3, 16, 128) (1, 3, 16, 128, 128) True

firsts, seqs, stacks, outs = [], [], [], []          # bound outside: a list first bound in the block does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:]:
        op, first, seq = mix._token_op()              # (the state op, the occurrence its first token is, this call's length)
        seqs.append(seq)
        firsts.append(first)
        stacks.append(mix.states.save())
        outs.append(mix.state_output.save())
print(seqs, firsts)                                   # [3, 1, 1] [0, 0, 1]
print(torch.equal(stacks[0], prompt_states), torch.equal(stacks[2][:, 0], outs[2]), torch.equal(stacks[2][:, 0], outs[1]))  # True True False
route_kernels(model.family, "default")
```

Qwen3.5 is a hybrid, and block 0 holds a `linear_attn`. `support()` lists the value as a dict because every fourth block has a `self_attn` instead. The open `tracer.iter[:]` ends with nnsight's "never reached" warning, which is how an open slice ends. Without the `first` offset, step 2 would ask for occurrence 0 of the recurrent kernel's state op, which step 1 consumed.

## Two kernels, one test

The mixer's forward binds a flag before it branches and then calls the token-by-token kernel when it has a cached state and one token, the chunked kernel otherwise. A subclass names the pieces in class constants.

| constant | meaning | `LinearAttention` |
|---|---|---|
| `BRANCH` | the binding made before the branch, `True` when the call continues a cached state | `"use_precomputed_states_0"`, the base's default |
| `SEQ_OP` | the forward's masking of its input, once per call, with the call's length on axis 1 | `"apply_mask_to_padding_states_0"`, the base's default |
| `CHUNK_KERNEL` | the call a prompt runs through | `"torch_chunk_gated_delta_rule_0"` |
| `RECURRENT_KERNEL` | the call a decode step runs through | `"torch_recurrent_gated_delta_rule_0"` |
| `STATE_OP` | inside the token-by-token kernel, the binding of the state after each token's update, or `None` | `"last_recurrent_state_3"` |
| `STEP_STATE_OP` | set when the decode kernel updates the state once instead of looping: the binding of the new state inside it | `None` |

`KERNEL` is one `staticmethod` on the base. It reads the `BRANCH` and `SEQ_OP` outputs in the order the forward binds them, sorted by each operation's source line, and returns `RECURRENT_KERNEL` when the call continues a cached state with one token, `CHUNK_KERNEL` otherwise, which means that several tokens over a cached state read the prompt's kernel. `kernel(attribute)` is the key function the values are declared on and returns `source.<KERNEL(envoy)>.<attribute>`: `attention_queries` is `kernel("inputs")` with `select=0`, `state_output` is `kernel("output")` with `select=1`. `attention_output` is the module's own output, unwrapped from a tuple and rewrapped on write, as on `Attention`.

## Once per call

The two bindings are served once per module call and every value of the call depends on the kernel they pick, and `KERNEL` makes its choice once, inside `per_call(envoy, "kernel", choose)`. `per_call` keeps one record per `(envoy.path, key)` on the worker greenlet running the intervention code and recomputes when the record's call is not the current one. The current call is `mediator.iteration or mediator.occurrence(f"{envoy.path}.output")`. A read pinned by `tracer.iter` to step k is served at the k-th occurrence of its location, and a mixer's kernel fires once per call, so the read is in call k. Relaxed, on step 0, or outside `tracer.iter`, the call is how many times the module's output has been passed, which nnsight counts whether or not anything read it ([interleaver](https://nnsight.net/documentation/intervention/interleaver/)). Every run of every invoke has its own worker, so two invokes reading one mixer never share a record and nothing stays on the envoy. The records are the kernel choice, `_token_op`'s `(op, first, seq)` under `"call"`, and `StateSpace`'s kernel arguments under `"arguments"`.

## Pure-torch kernels and routing

transformers decorates each kernel with its hub dispatcher, which binds the module-level name to a closure holding `torch_function`, the pure-torch body, and `implementation`, the optimized kernel when its package is installed. `_dispatch` reads that closure with `inspect.getclosurevars`, and `needs_torch_kernels`, the `unavailable` predicate of every kernel value, refuses when `implementation is not torch_function`, naming the package: a compiled kernel has no Python source to drill into.

`route_kernels(family, "torch")` rebinds the family's two kernel names in its modeling module to their pure-torch functions, stashing the originals in the module's `_nnterp_kernels`, from which `"default"` restores them. On a mixer with a `STATE_OP` the prompt's name is bound to the loop kernel too, which on a DeltaNet is the decode kernel: after routing both names run the token-by-token rule, the operation keeps its label `torch_chunk_gated_delta_rule_0`, and `STATE_OP` under it fires once per token. The routing is process-wide and must come before the first trace that touches a mixer of that family, because nnsight builds an instrumented forward over a copy of the module's globals on the first drill. A mixer traced before routing keeps calling the kernel it was compiled with, and a read of `states` on it fails as nnsight's "no attribute `states`" while an untouched mixer of the same model serves it. `route_delta_rule(family, "recurrent" | "chunked")` is the DeltaNet spelling.

```python
mamba = StandardizedTransformer("state-spaces/mamba-130m-hf", dispatch=True, device="cpu")
scan = mamba.layers[0].linear_attn
print(type(scan).__name__, type(scan).STATE_OP, type(scan).STEP_STATE_OP)   # SelectiveScan ssm_state_3 ssm_state_0
print(mamba.support()["linear_attn.attention_queries"][0])
# read inside transformers' pure-torch mamba_selective_scan, but this process dispatches it to an optimized kernel (mamba_ssm) with no Python source; uninstall it, or call nnterp.route_kernels(model.family, 'torch'), to read these
route_kernels(mamba.family, "torch")
with mamba.trace(prompt):
    states = scan.states.save()
    out = mamba.layers[0].layer_output.save()
print(mamba.support()["linear_attn.states"], tuple(states.shape), tuple(out.shape))   # None (1, 3, 1536, 16) (1, 3, 768)
route_kernels(mamba.family, "default")
```

With `mamba_ssm` installed, a Mamba-1 or Mamba-2 model needs the routing before any trace on a CPU, including one that reads only `layer_output`: the compiled kernel refuses a CPU tensor (`Expected u.is_cuda() to be true`).

## The state after every token

`state` is an `EProperty` with the key function `_token_state_op`, which returns `source.<kernel>.source.<STATE_OP>.output`, a location with one occurrence per token, so nnsight's own `tracer.iter` walks it: `for t in tracer.iter[:n]: mix.state` is the state after every prompt token, and an assignment there is a write the following tokens continue from ([iterator](https://nnsight.net/documentation/intervention/iterator/)). Under `tracer.iter` the pin at the read is a token index while the bindings `KERNEL` reads and the kernel call's `.fn` fire once per forward. `_token_state_op` decides the kernel and drills into its call inside `with pinned(None)`, which relaxes the pin and restores it on exit, and the pinned read that follows asks for that token's occurrence. `pinned(n)` sets the worker mediator's `iteration` the way `tracer.iter[n]` does, and one read relaxes it: a `with` holds one read.

`states`, `state_after(t)` and `set_state_after(t, value)` address the same location by index and need `_token_op`'s three numbers, computed once per call: the state op under the drilled kernel, its sequence length from `_seq()` (the queries' axis 1 on a DeltaNet), and `first`, the op's occurrence count at the moment the kernel's arguments are read. Reading an argument parks the worker at the call's start, the one moment that count is the number of tokens earlier calls put through the op. Occurrences are counted per location since the op was drilled this run, so on step k of a `generate` the recurrent op's count is k minus 1, and the prompt's tokens, under the chunk kernel's label, do not offset it. `_states` then loops `with pinned(first + t): op.output` and stacks on axis 1. It is a `DerivedEProperty`, read-only.

`needs_recurrent_routing` guards `state` and `states`, in this order. A mixer with no `STATE_OP` says its kernels do not materialize the state per token. An optimized kernel gives `needs_torch_kernels`' reason. A `CHUNK_KERNEL` name still bound to the chunked function says to route. It reads the live binding, so `support()` follows the routing, as the first block shows.

## Mamba-1 and Mamba-2

`SelectiveScan`, in `nnterp/components/selective_scan.py`, differs from the DeltaNet at the points the base leaves open to a subclass. Its forward binds the branch flag before the mask, which `KERNEL`'s line ordering handles. Its pure-torch prompt kernel is itself the token loop (`STATE_OP = "ssm_state_3"`) while the decode kernel updates once (`STEP_STATE_OP = "ssm_state_0"`): `route_kernels` binds each name to its own function, and `_state_op(kernel)` picks the binding for the kernel that fires. Its arguments sit at different positions in the two kernels, and `select` is a function looking the name up in `ARGUMENTS` by whether the call decodes. Its tensors are channel-first, and each preprocess returns a tokens-first view that the postprocess lays back. `betas` and `decays` are `DerivedEProperty` values computed from `dt`, `dt_bias` and `A` the way the kernel computes them.

`StateSpace`, in `nnterp/components/state_space.py`, has `STATE_OP = None`, since neither Mamba-2 kernel binds the state once per token. `state` and `set_state_after` are `unavailable(...)` with reasons, and `states` is a `DerivedEProperty` over the chunk scan's `new_states_0` binding, the state at every chunk boundary. `chunk_per_token(model)` sets `chunk_size` to 1 on each `StateSpace` module of one model, which makes every token a boundary. It is per model, not process-wide, and `needs_per_token_chunks` reports the chunk size until then. The two kernels take their arguments in different places (`CHUNK_ARGUMENTS`, `RECURRENT_ARGUMENTS`, through `argument(name)`). The update has no sequence axis, and a decode step's values gain one on read and lose it on write. `betas` and `decays` are the kernel's `dt` seen through `dt_bias`, the softplus and `A`, with an inverse postprocess, and an assignment reaches the kernel. Values that need another argument read the call's `(args, kwargs)` once per call through `_arguments`.

```python
mamba2 = StandardizedTransformer("AntonV/mamba2-130m-hf", dispatch=True, device="cpu")
ssd = mamba2.layers[0].linear_attn
route_kernels(mamba2.family, "torch")
print(mamba2.support()["linear_attn.states"][0])
# the chunk scan materializes the state only at chunk boundaries, every 256 tokens (chunk_size=256); call nnterp.chunk_per_token(model) to set every mixer's chunk_size to 1, so every token is a boundary (slower on long prompts)
nnterp.chunk_per_token(mamba2)
with mamba2.trace(prompt):
    states = ssd.states.save()
    out = ssd.state_output.save()
print(mamba2.support()["linear_attn.states"], tuple(states.shape), torch.equal(states[:, -1], out))   # None (1, 3, 24, 128, 64) True
nnterp.chunk_per_token(mamba2, enabled=False)
route_kernels(mamba2.family, "default")
```

## Adding a mixer

A new mixer subclasses `RecurrentMixer`, sets `CHUNK_KERNEL` and `RECURRENT_KERNEL` (and `BRANCH`, `SEQ_OP` or `KERNEL` itself, as a `staticmethod`, when its forward spells the test otherwise), sets `STATE_OP` only when a kernel binds the state once per token, and declares its values at `kernel("inputs")` and `kernel("output")` with `unavailable=needs_torch_kernels`. A kernel with tensors not laid out `[batch, seq, ...]` overrides `_seq`. `tests/test_base.py` holds the base's own tests, and a family's file mixes in `SelectiveScanSuite` or the tests in `tests/families/ssd.py` ([Testing](../testing.md)).
