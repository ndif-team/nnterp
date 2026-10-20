# Descriptors

Every value is an `EProperty`, nnsight's [`eproperty`](https://nnsight.net/documentation/intervention/eproperty/) with two additions: its key is a path from the host envoy, resolved into a served location on every access, and an `unavailable=` reason is checked before anything runs. `select` picks one element of what is served, a transform carries edits back from a reshaped view, and `DerivedEProperty` computes a read-only value from other served ones. The class is in `nnterp/components/eproperty.py` ([eproperty](../../documentation/components/eproperty.md)).

```python
import torch
import nnterp
from nnterp import StandardizedTransformer, Unavailable
from nnterp.components import DerivedEProperty, EProperty, Pattern, Residual, interface_reason, unavailable
from nnterp.families import gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


def entropy(attn):
    p = attn.attention_probabilities
    return -(p * (p + 1e-12).log()).sum(-1)


class Attention(gpt2.Attention):
    @EProperty("source.attention_interface_1.inputs", select=4, description="The mask the interface is called with", unavailable=interface_reason)
    def mask(self, value: torch.Tensor) -> Pattern:
        return value

    @EProperty("source.attention_interface_1.source.torch_matmul_0.output", description="q @ k^T before the scaling", unavailable=interface_reason)
    def raw_scores(self, value: torch.Tensor) -> Pattern:
        return value

    @EProperty("../ln_2.output", description="The MLP's pre-norm output, read from the attention")
    def sibling_norm(self, value: torch.Tensor) -> Residual:
        return value

    pattern_entropy = DerivedEProperty(entropy, description="Entropy of each row of the pattern")
    attention_head_outputs = unavailable("dropped on purpose for this example")


model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager", device="cpu", envoys={GPT2Attention: Attention})
attn = model.layers[0].self_attn

print(Attention.sibling_norm.key, Attention.sibling_norm.inside_forward(), Attention.raw_scores.inside_forward())  # ../ln_2.output False True
print(Attention.raw_scores.layout is Pattern, Attention.raw_scores.dims)   # True ('batch', 'num_heads', 'query', 'key')
print(str(Attention.raw_scores))                                           # (raw_scores) -> Pattern [batch num_heads query key]: q @ k^T before the scaling
print(attn.support()["attention_head_outputs"])                            # dropped on purpose for this example

with model.trace("The Eiffel Tower is in"):
    mask = attn.mask.save()
    raw = attn.raw_scores.save()
    scores = attn.attention_scores.save()
    ent = attn.pattern_entropy.save()
    normed = attn.sibling_norm.save()
    ln2 = model.layers[0].ln_2.output.save()
print(torch.equal(normed, ln2), torch.allclose(raw * attn._module.scaling + mask, scores), tuple(ent.shape))  # True True (1, 12, 7)
```

The reads are in forward order: the interface's arguments, then an operation inside it, then its softmax, then the pattern, then the block's second norm. Passing the subclass through `envoys=` keyed on `GPT2Attention` replaces the family's entry, and every block's attention carries the five new names. [Custom values](../custom-values.md) is the user-facing version of this page.

## The key is a path

A key is dotted segments ending in `output`, `input` or `inputs`, walked from the host, or from the model's root after a leading `/`. A function of the host is also a key: It runs at read time, inside the trace, and returns such a path, which is how a forward that branches names the call that fires (`RecurrentMixer.KERNEL`) and how Falcon picks an operation by `config.alibi`. `None` means the attribute's own name, which is what a bare `unavailable("...")` marker gets.

| segment | meaning | example |
|---|---|---|
| `output`, last | the current node's output | `"output"`: `Layer.layer_output`, a view over the same location as `.output` |
| `input`, last | the first argument of the current node's call | `"source.dropout_add_0.input"`: BLOOM's contribution |
| `inputs`, last | the `(args, kwargs)` pair, or one element of it with `select` | `"source.attention_interface_1.inputs"` with `select=1`: `attention_queries` |
| `/`, leading | the model's root, then a relative path from there, aliases included | `"/inputs"`: `Vision.image_token_mask` |
| `../`, leading, repeatable | the parent module, by native name | `"../post_attention_layernorm.output"`: Gemma-2's `attention_output` |
| a name | a child module, aliases included, or under `source` an operation | `"embed_tokens.output"`: `token_embeddings` |
| `source` | the current module's or operation's forward, instrumented for this run | `"source.attention_interface_1.source.nn_functional_dropout_0.output"`: `attention_probabilities` |

Operation names are nnsight's labels, `{callable}_{occurrence}`, and `print(envoy.source)` lists them ([Finding source ops](../finding-source-ops.md)). `inside_forward()` answers whether a key has a `source` segment or is a function, which the suite uses to find the values a transformers release can move.

## Availability

`unavailable=` is a reason string or a predicate `f(envoy) -> str | None`. `reason(obj)` evaluates it on the instance, which lets a checkpoint's config decide (`needs_eager` reads `_attn_implementation`) and a hybrid answer per block. `_check` runs before every read and write and raises `Unavailable`, a `RuntimeError`, naming the host path, the value and the reason. A predicate that itself raises `AttributeError` is re-raised as a `RuntimeError` saying the availability check failed, because an `AttributeError` escaping a descriptor falls through to nnsight's `Envoy.__getattr__` and comes back as "no attribute", hiding the predicate's own bug. For the same reason `Unavailable` is not an `AttributeError`, which is why `hasattr(envoy, name)` and `getattr(envoy, name, None)` raise it instead of answering `False` or `None`. Decide what a checkpoint has with `support()`, outside the trace ([Availability](../../guide/values/availability.md)).

A marker such as `attention_head_outputs = unavailable("...")` is never called on a stub, so nnsight's `eproperty.__call__` never fills its `name` and `key`. `EProperty.__set_name__` fills them from the class body, which is what makes the marker show in `support()` and the repr with its reason.

## Resolving a path at run time

`_resolve(obj, key)` turns the path into the location string that nnsight's `Mediator` serves, and it runs before every read and write rather than once, because an operation inside a called function exists only after a drill into that call in the current run. nnsight resolves the callee from the live value and clears what it built at the start of every run ([source](https://nnsight.net/documentation/intervention/source/)). A leading `/` swaps the host for `obj.root` and drops the slash, and a `../` right after it is refused, since nothing is above the root. Leading `../` segments are counted and stripped. Below the host every remaining segment but the last is handed to `obj.get(...)`, and each is an attribute: a child module (an alias works), `source`, or an operation of the `Source` a drill returned. The result is `f"{node.path}.{attribute}"`, with `inputs` folded into `input`, since the pair and its first argument are one served location and picking the argument is `_pick`'s job. An `AttributeError` anywhere on the walk becomes `SourceNotAvailable` naming the value, the path and what nnsight found there.

```python
class Probe(Attention):
    @EProperty("source.attention_interface_1.source.no_such_op_0.output", unavailable=interface_reason)
    def broken(self, value):
        return value


probe = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager", device="cpu", envoys={GPT2Attention: Probe})
try:
    with probe.trace("The Eiffel Tower is in"):
        probe.layers[0].self_attn.broken.save()
except Exception as e:
    print(type(e).__name__, str(e)[:261])
# SourceNotAvailable model.transformer.h.0.attn.broken reads 'source.attention_interface_1.source.no_such_op_0.output', which this run does not have: 'model.transformer.h.0.attn.source.attention_interface_1.source' has no operation 'no_such_op_0'; available: query_size_0, scaling_0

try:
    with model.trace("The Eiffel Tower is in"):
        normed = attn.sibling_norm.save()
        raw = attn.raw_scores.save()
except Exception as e:
    print(type(e).__name__, e)
# OutOfOrderError 'model.transformer.h.0.attn.source.attention_interface_1.fn.i0' was requested but the model already ran past it
```

The second failure is the drill's own ordering. Drilling into a call parks the worker on that call's `.fn`, a location the forward passes once, just before the call runs. A value inside the call read after the module has returned asks for a location the model passed already, and the error names the call's `.fn`, not the value. On the module's own `source` nothing is asked of the model, and an operation already drilled this run is reused, so one trace that reads several interior values in forward order drills once.

A function key is evaluated once per access and its last segment passed down, never re-derived: `RecurrentMixer.KERNEL` reads the forward's branch variable as a step's first, pinned read, and a second evaluation would run with the pin relaxed ([Recurrent mixers](recurrent-mixers.md)).

## Above the host

A path that starts with `../` is not walked through envoys. `_resolve` drops as many trailing segments from the host's own path as there are `../`, appends the rest of the key, and returns the string, so the parent and the sibling are named natively: `"../ln_2.output"` on `model.transformer.h.0.attn` is `model.transformer.h.0.ln_2.output`, which is why the example above spells `ln_2` and not `post_attention_layernorm`. Nothing is drilled. A `source` segment above the host names an operation of the parent's own forward, served by its string because the family instrumented that forward at build. A second `source` on such a path would need the parent drilled at read time, and `_resolve` refuses it with a `ValueError`. A value that reads something far from its host by standard name anchors its key at `/` instead.

Instrumenting at build is `Standard.sourced`, in `nnterp/components/standard.py`. A value inside a forward is served only when that forward was instrumented before the call began. Drilling at read time is enough when the read comes before the module runs, and too late when the worker is already parked inside it, as it is when a block's `mlp_output` is an operation of the block's forward read after the block's `attention_output` (Llama 4). The family sets `sourced = True` on the envoy that owns the forward, and `Standard.__init__` and `_update` then touch `self.source` at build and again when real weights replace meta ones, which reinstalls the plain forward. Nothing infers `sourced` from a path. `inside_forward()` says where a value is, not when its forward has to be instrumented.

## `select` and writes

`_pick(attribute, value, select)` takes one element of the served value and `_put` puts one back. `input` is `first_input(args, kwargs)` on read and `replace_first_input` on write, nnsight's own rule for `.input`, so a stub on an `input` path receives a tensor and the call's other arguments stay intact. `inputs` with an int `select` is `args[n]`, with a str `kwargs[name]`, and with none the pair itself (`input_ids` takes `kwargs["input_ids"]` in its stub). `output` with an int indexes the returned tuple (`attention_head_outputs` is `select=0`), and a module that returns a bare tensor takes no `select`. `select` may also be a function of the host, evaluated before the read, for a value that sits at different positions in the calls a forward branches to (`StateSpace`'s two kernels).

`__get__` is `_check`, one evaluation of the key, `_resolve`, `Mediator.value`, `_pick`, then the preprocess. `__set__` is `_check`, the postprocess, `_resolve`, and when the key selects or ends in `input` a re-read of the current value with the element put back, then `Mediator.swap` of the whole. Assigning `attn.mask` above swaps the interface's fifth argument and keeps the other four.

## Layouts and the repr

`layout` is the return annotation of the stub, resolved with `typing.get_type_hints`, and `-> Pattern` written in a family module gives the identical alias object the base uses, and `Attention.attention_keys.layout is Keys` holds across families. A `State | None` annotation is a union, and the non-`None` member is taken. `dims` splits the alias's `dim_str`, and `__str__` renders the repr line as `(name) -> Layout [axes]: description`, which is what `print(model)` shows ([Layouts](../../guide/values/layouts.md)).

## Transforms

A value that serves a copy or a reshaped view carries edits back with a transform, nnsight's write-back of `(view, raw)` to the served value. `__get__` binds a `WriteBack` with the raw served value riding along, nnsight flushes it when the worker next moves on, and a repeated read in between is answered with the view it holds. It fires whether or not the view was edited, so a transform must be an identity on an unedited view. Falcon's `mlp_output` uses one to carry in-place edits on a clone back into the model.

## `DerivedEProperty`

`DerivedEProperty(compute)` has no location. `__get__` runs `_check` and then `compute(obj)`, which may read any number of served values in forward order. `__set__` raises `AttributeError` saying the value is derived and read-only, which is the one place an `AttributeError` is right, since it is the assignment that fails. `compute` is stored as `_preprocess` only so that `layout` can read its annotation. `pattern_entropy` above and `RecurrentMixer.states` are instances.

```python
try:
    with model.trace("The Eiffel Tower is in"):
        attn.pattern_entropy = torch.zeros(1)
except AttributeError as e:
    print(e)   # pattern_entropy is derived and read-only
```

The function keys, `pinned` and `per_call` that a forward with two kernels needs are the next page, [Recurrent mixers](recurrent-mixers.md).
