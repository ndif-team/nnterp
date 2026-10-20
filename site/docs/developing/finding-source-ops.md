# Finding source ops

A `source.` path names an operation inside a module's forward by the label nnsight gives it, and the only correct list of labels is the one printed from the transformers version you run: `print(envoy.source)` outside a trace for the module's own forward, `<call>.source` inside a trace for a function the forward calls. nnsight's [intermediate operations](https://nnsight.net/features/11_source/) page is the full account of `.source`. This is what a path needs from it. The listings below are transformers 5.17's.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
print(model.layers[0].self_attn.source)
```

```
                                             * def forward(
 ...
 self_c_attn_1                           -> 40         query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
 split_1                                 ->  +         ...
 ...
 query_states_view_0                     -> 46     query_states = query_states.view(shape_q).transpose(1, 2)
 transpose_4                             ->  +     ...
 query_states_1                          ->  +     ...
 ...
 using_eager_0                           -> 56     using_eager = self.config._attn_implementation == "eager"
 ALL_ATTENTION_FUNCTIONS_get_interface_0 -> 57     attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
 attention_interface_0                   ->  +     ...
                                            61     if using_eager and self.reorder_and_upcast_attn:
 self__upcast_and_reordered_attn_0       -> 62         attn_output, attn_weights = self._upcast_and_reordered_attn(
                                            65     else:
 attention_interface_1                   -> 66         attn_output, attn_weights = attention_interface(
                                            67             self,
                                            68             query_states,
                                            69             key_states,
                                            70             value_states,
                                            71             attention_mask,
 ...
 self_c_proj_0                           -> 78     attn_output = self.c_proj(attn_output)
 attn_output_1                           ->  +     ...
 self_resid_dropout_0                    -> 79     attn_output = self.resid_dropout(attn_output)
 attn_output_2                           ->  +     ...
                                            81     return attn_output, attn_weights
```

The labels on the left are the operation names, trimmed here to the lines a path uses. A call is `<callee>_<n>`, the attribute chain joined with `_`: `self.c_proj(...)` is `self_c_proj_0`, and `self._upcast_and_reordered_attn(...)` is `self__upcast_and_reordered_attn_0`, the underscore of the method name kept. A binding is an operation too, `<name>_<n>` for the n-th assignment to that name in the forward, and the `+ ...` lines under a call are the bindings of its result. Calls and bindings share one counter per name, so `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)` is `attention_interface_0` and the call `attention_interface(...)` is `attention_interface_1`, the string `nnterp.components.INTERFACE` holds. The counter follows the order in the source, not the order the forward runs, and the listing includes branches this forward never takes: GPT-2's cross-attention path owns `self_c_attn_0`, and the live call is `self_c_attn_1`. `.output` is what a call returned or a binding bound, `.input` a call's first argument, `.inputs` its `(args, kwargs)` pair.

## Inside a call

`attention_interface_1` calls a function, transformers' `eager_attention_forward`. Its operations exist only under `attention_interface_1.source`, and that drill works inside a trace only, because the callee is a local bound at run time: nnsight resolves it from the live call the first time a trace drills into it, and clears what it built when the next run starts.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
attn = model.layers[0].self_attn
with model.trace("The Eiffel Tower is in"):
    args = attn.source.attention_interface_1.inputs.save()   # the call's arguments: served first
    inner = attn.source.attention_interface_1.source          # the drill into the callee, inside the trace only
    print(inner)
    probs = inner.nn_functional_dropout_0.output.save()      # then an operation inside it
    pattern = attn.attention_probabilities.save()
print(args[0][1].shape, probs.shape, torch.equal(probs, pattern))
# torch.Size([1, 12, 7, 64]) torch.Size([1, 12, 7, 7]) True
```

```
                             * def eager_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
 ...
 key_transpose_0         ->  3     attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
 torch_matmul_0          ->  +     ...
 attn_weights_0          ->  +     ...
                             5     if attention_mask is not None:
 attn_weights_1          ->  6         attn_weights = attn_weights + attention_mask
 nn_functional_softmax_0 ->  8     attn_weights = nn.functional.softmax(attn_weights, dim=-1)
 attn_weights_2          ->  +     ...
 attn_weights_type_0     -> 11     attn_weights = attn_weights.type(value.dtype)
 attn_weights_3          ->  +     ...
 nn_functional_dropout_0 -> 12     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
 attn_weights_4          ->  +     ...
 torch_matmul_1          -> 14     attn_output = torch.matmul(attn_weights, value)
 attn_output_0           ->  +     ...
 attn_output_transpose_0 -> 15     attn_output = attn_output.transpose(1, 2)
 attn_output_1           ->  +     ...
                            17     return attn_output, attn_weights
```

This listing is where the base `Attention`'s values come from: The queries, keys and values are the call's arguments 1 to 3, `attention_scores` is `nn_functional_softmax_0`'s input, `attention_probabilities` is `nn_functional_dropout_0`'s output, and `attention_head_outputs` is element 0 of the call's own output. An `EProperty` does the drill at every read, with a `source` segment between the call and the operation inside it: `"source.attention_interface_1.source.nn_functional_dropout_0.output"`. Outside a trace, `attn.source.attention_interface_1.source` raises `SourceNotAvailable: recursive .source is only available inside a trace`.

## The operation to pick

The pattern is the dropout after the softmax wherever one exists, because that is the tensor the values are mixed with, cast back to the model's dtype and, on an attention-sink model, with the sink column dropped. In eval the two tensors are equal, and the rule is about what the location means. A contribution added inside the module is the tensor entering the add, BLOOM's `dropout_add_0` input. Queries, keys and values are what the attention arithmetic receives, after the rotary embedding where there is one and before `repeat_kv`, so the head axis is `num_kv_heads` wide.

## Errors from a wrong name

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
model.layers[0].self_attn.source.nope_0
# AttributeError: 'model.transformer.h.0.attn.source' has no operation 'nope_0'; available: is_cross_attention_0, isinstance_0, ...
```

An `EProperty` on a missing operation raises `SourceNotAvailable` with the value's path, its key and that list: "model.transformer.h.0.attn.broken reads 'source.no_such_op_0.output', which this run does not have: ...". A label that is in the listing but under a branch this forward does not take fails differently: `attn.source.self__upcast_and_reordered_attn_0.output` raises `OutOfOrderError: '...self__upcast_and_reordered_attn_0.output.i0' was requested but the model already ran past it`, because the model ran to the end without reaching it. A call to a submodule is not drilled: `attn.source.self_c_proj_0.source` raises `SourceNotAvailable: 'self_c_proj_0' calls a submodule; call .source on that submodule directly`, which is `attn.c_proj.source`.

Requests are served in the forward's order, and for a call and its inside that order is the call's `.inputs`, then the drill, then the operations inside, then the call's `.output`. The block above reads them that way. Reading `attention_interface_1.inputs` after an operation inside the call raises `OutOfOrderError` on `...attention_interface_1.input.i0`. Two values read in one trace follow the same rule, `attention_queries` (the call's inputs) before `attention_probabilities` (inside the call), and the family suite reads interior values one per trace for this reason.

The labels are what transformers releases change: A renamed local renames a binding, and a refactor that routes a family onto the shared interface replaces its own softmax with `attention_interface_1`. Print the source for the version you run rather than copying a label from another family or from this page, and let `test_every_source_value_resolves_on_every_layer` and `test_written_pattern_moves_the_logits` in the family suite be the guard: every source-located value must resolve on every attention block, and a written pattern must move the logits, since a label can read a perfectly good tensor that nothing downstream uses. The descriptors that take these labels are on [Overriding a value](overriding-values.md).
