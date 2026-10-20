# Remote

`remote=True` on a `StandardizedTransformer` trace is nnsight's remote trace ([nnsight: remote execution](https://nnsight.net/features/15_remote_execution/)) with nnterp's names and values in the block. The server runs a plain `TransformersModel` of the same checkpoint. Your block is re-run there against your envoy tree, which carries the family's aliases and envoy classes by reference, so nnterp must be installed on the server, at the same version as yours. It is never shipped by value.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")      # no dispatch: the architecture on the meta device, no weights
print(model.to_model_key())
# nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null, "task": "text-generation"}
```

The key a request carries names `TransformersModel`, because that is what a server deploys, so every checkpoint a server serves is reachable through a `StandardizedTransformer` without the server deploying anything nnterp-specific. The block itself is what you would run locally, and every `.save()` is downloaded, so slice inside it:

```python
# not run for this page: needs an NDIF key and a deployed model
from nnsight import CONFIG
from nnterp import StandardizedTransformer

CONFIG.set_default_api_key("YOUR_KEY")
model = StandardizedTransformer("meta-llama/Llama-3.1-70B", attn_implementation="eager")

with model.trace("The Eiffel Tower is in the city of", remote=True):
    resid = model.layers[40].layer_output[:, -1].save()                              # [batch, hidden_size]
    pattern = model.layers[40].self_attn.attention_probabilities[0, :, -1].save()   # [num_heads, key]
    model.layers[41].mlp.mlp_output[:] = 0
    probs = model.next_token_probs.save()                                            # [batch, vocab_size]
```

## The request's contents

The request carries the block's source, the names it saves, and the envoy tree by reference: `model.layers[i].self_attn` resolves on the server to the same module it does on your machine, and a value such as `layer_output` is a descriptor on the family's envoy classes, which the server imports from its own nnterp. A different version there can define a value differently or spell a source-located operation another way, and the error is whatever the mismatch produces server-side. Shipping nnterp by value does not work around that: `nnsight.register(nnterp)` fails with `TypeError: cannot pickle 'EProperty' object`. Register your own helper module if the block calls one. An installed package travels by reference either way.

## A dry run without a server

`remote="local"` serializes the block, deserializes it with your local modules hidden, and runs it against a dispatched model, with no server and no key. A block that survives it, aliases and values included, is one that ships:

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu", attn_implementation="eager")

with model.trace("The Eiffel Tower is in the city of", remote="local"):
    pattern = model.layers[3].self_attn.attention_probabilities[0, :, -1].save()
    model.layers[7].mlp.mlp_output[:] = 0
    probs = model.next_token_probs.save()

print(tuple(pattern.shape), model.probs_to_dict(probs[0], k=2))
# (12, 10) {' Paris': 0.09203002601861954, ' London': 0.0368831530213356}
```

## Decided by the deployment

Availability is checked against your model's config, not the deployment's: `attn_implementation="eager"` at load declares the attention interior available, but whether the deployed forward runs eager is the server's choice, and on an `sdpa` deployment the read fails server-side rather than with `Unavailable` on the client. Source-located values, such as `attention_probabilities`, `attention_queries` and a mixture's `expert_outputs`, resolve against the server's transformers release, and a release that spells the operation differently fails the same way. A recurrent mixer's per-token state needs `nnterp.route_kernels` in the process that runs the model, which a request does not reach, so treat `state` and `states` as unavailable remotely ([Gated DeltaNet](architectures/delta-net.md)). The helpers batch their remote loops into one `model.session`: `collect_token_activations_batched(..., remote=True)` and `run_prompts(..., remote=True)` are one request each ([Helpers](helpers.md)).
