---
title: Remote (NDIF)
one_liner: `remote=True` on a `StandardizedTransformer`: the server deploys a plain `TransformersModel`, your block re-runs against your envoy tree, and nnterp must be installed server-side at the same version.
tags: [usage, remote, ndif, serialization]
related: [docs/usage/availability.md, docs/usage/loading.md, docs/usage/attention-interior.md, docs/usage/delta-net.md, docs/usage/activations.md, docs/usage/prompt-utils.md]
sources: [nnterp/standardized.py, nnterp/components/eproperty.py]
---

# Remote (NDIF)

## What this is for

`remote=True` on a `StandardizedTransformer` is nnsight's remote trace with
nnterp's names and values in the block (nnsight docs/remote/remote-trace.md).
The model builds on the meta device, nothing downloads, and the block runs on
NDIF against a model the server deployed. This page is the contract nnterp
adds to that: which model the request reaches, what travels with it, what the
server needs installed, and which values depend on the server's own stack.
The mechanics are verified on the in-process simulation of the remote path
(`remote="local"`, which serializes, deserializes and executes the block the
way the server does); the statements about a live server follow from the
design and are marked as such.

## Canonical pattern

```python
from nnsight import CONFIG
from nnterp import StandardizedTransformer

CONFIG.set_default_api_key("YOUR_KEY")
model = StandardizedTransformer("meta-llama/Llama-3.1-70B", attn_implementation="eager")   # meta device

with model.trace("The Eiffel Tower is in the city of", remote=True):
    resid = model.layers[40].layer_output[:, -1].save()                     # [batch, hidden]
    pattern = model.layers[40].self_attn.attention_probabilities[0, :, -1].save()   # [heads, key]
    model.layers[41].mlp.mlp_output[:] = 0
    probs = model.next_token_probs.save()                                   # [batch, vocab]
```

Save small things: every `.save()` is downloaded. Slice inside the block.

## Which model the request reaches

A remote request carries a model key the server matches deployments by. A
`StandardizedTransformer`'s key names `TransformersModel`, not its own class:

```python
model.to_model_key()
# 'nnsight.modeling.transformers.TransformersModel:{"repo_id": "meta-llama/Llama-3.1-70B", "revision": null}'
```

`_remoteable_class` returns `TransformersModel` because that is what a server
deploys: a plain model of that repo id. A key naming `StandardizedTransformer`
would match nothing. So every checkpoint NDIF serves is reachable through a
`StandardizedTransformer` without the server deploying anything nnterp-specific.

## What travels with the block

The block is re-run on the server against the *client's* envoy tree, which
carries the family's aliases (`model.layers[i].self_attn`) and the family's
envoy classes (`nnterp.families.llama.Layer`, `Attention`, `Mlp`) by
reference. The server therefore imports `nnterp` when it deserializes the
request, and needs it installed at the same version as the client: a value is
a descriptor on those classes, and the operation names it reads inside a
forward are what releases change.

Do not ship nnterp by value. `nnsight.register(nnterp)` (nnsight
docs/remote/register-local-modules.md) is for local helper modules; an
installed package is pickled by reference anyway, and an `eproperty` cannot be
pickled by value, so registering nnterp fails rather than helping. Register your
own helper module if the block calls one.

## What works remotely

Everything in the block is what runs locally, so every standard value works
the way it does in a local trace, on the in-process simulation:

- the boundary values (`layer_output`, `attention_output`, `mlp_output`,
  `token_embeddings`), read, edited in place or assigned;
- the root values (`logits`, `next_token_probs`, `input_ids`);
- the interior values, `attention_probabilities` included: nnterp drills
  `.source` on the server the way it does locally, and an in-place edit of the
  pattern moves the remote logits;
- `skip_layers`, `steer` and `project_on_vocab`, which are plain methods over
  the values.

Reads follow forward order inside the block, exactly as locally; an
out-of-order read is an `OutOfOrderError` from the server's run.

## What depends on the server

- **Source-located values resolve against the server's transformers.** Names
  like `attention_interface_1` and `nn_functional_dropout_0` are read at
  runtime from the server's forward. A server on another transformers release
  can spell them differently; the read then fails server-side with
  `SourceNotAvailable` naming what is there, not with `Unavailable` on the
  client.
- **Availability is checked against the client's config.** `support()` and the
  `Unavailable` check run on your meta model: `attn_implementation="eager"` at
  load makes the interior *declared* available, but whether the eager forward
  runs is the deployment's choice. On a deployment running `sdpa` the read
  fails server-side as above.
- **The DeltaNet per-token state is a process-wide switch.**
  `route_kernels` rebinds the kernel in *your* process; the server's is not
  routed by a request. Treat `state`, `states`, `state_after` and
  `set_state_after` as unavailable remotely; the call-level values
  (`state_output`, `decays`, `betas`, ...) are ordinary source-located values
  and follow the rule above ([delta-net.md](delta-net.md)).
- **Server-side version.** A client and server on different nnterp versions may
  disagree on a value's definition; the error is whatever the mismatch
  produces on the server.

## Try it without a server

`remote="local"` runs the serialize, deserialize and execute path in-process
against a dispatched model, with no server and no key: the way to check that a
block, its aliases and its values survive the round trip before submitting it.

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
with model.trace("The Eiffel Tower is in the city of", remote="local"):
    pattern = model.layers[3].self_attn.attention_probabilities.save()
```

## Gotchas

- **Load the client model with the same `attn_implementation` the deployment
  runs** if you read the interior; the client check cannot see the server.
- **A name bound in the block is gone after it** unless `.save()`d, remotely as
  locally; a `support()` call inside the block is a plain dict and needs no
  save, but it describes the client's model.
- **The model key names `TransformersModel`**; a server that deploys a
  checkpoint under any other class does not match.
- **Batch remote work into one `session`** (nnsight docs/remote/remote-session.md);
  `collect_last_token_activations_session` and `run_prompts(..., remote=True)`
  do this for their loops ([activations.md](activations.md), [prompt-utils.md](prompt-utils.md)).

## Related

- [availability.md](availability.md), `support()` and `Unavailable`, which are client-side checks.
- [loading.md](loading.md), `attn_implementation` and `tokenizer_kwargs` at load.
- [attention-interior.md](attention-interior.md) and [delta-net.md](delta-net.md), the source-located values this page qualifies.
- [activations.md](activations.md) and [prompt-utils.md](prompt-utils.md), helpers that take `remote=`.
- nnsight docs/remote/remote-trace.md, docs/remote/remote-session.md, docs/remote/register-local-modules.md.
