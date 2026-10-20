# transformers compatibility

`pyproject.toml` requires `transformers>=5.17`, `nnsight>=0.8` and `jaxtyping`, and the pages on this site ran on transformers 5.17.0 and nnsight 0.8.0. The module names and classes a family binds to (`RENAME`, `ENVOYS`) rarely move between releases. The operation names inside a forward that every `source.` path pins are what a release renames, so this page lists them, says how the suite catches a move, and gives the procedure.

```python
import nnterp
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager", device="cpu")
attn = model.layers[0].self_attn

print([op.name for op in attn.source][-12:])   # outside a trace: the module's own forward
# ['using_eager_0', 'ALL_ATTENTION_FUNCTIONS_get_interface_0', 'attention_interface_0', 'self__upcast_and_reordered_attn_0', 'attention_interface_1', 'attn_output_reshape_0', 'contiguous_0', 'attn_output_0', 'self_c_proj_0', 'attn_output_1', 'self_resid_dropout_0', 'attn_output_2']
with model.trace("Hello world there"):
    inner = [op.name for op in attn.source.attention_interface_1.source].save()   # inside: drilled from the live callee
print(inner)
# ['query_size_0', 'scaling_0', 'key_transpose_0', 'torch_matmul_0', 'attn_weights_0', 'attn_weights_1', 'nn_functional_softmax_0', 'attn_weights_2', 'attn_weights_type_0', 'attn_weights_3', 'nn_functional_dropout_0', 'attn_weights_4', 'torch_matmul_1', 'attn_output_0', 'attn_output_transpose_0', 'attn_output_1']
print({name: attr.key for name, attr in type(attn).values().items() if attr.inside_forward()})
# {'attention_queries': 'source.attention_interface_1.inputs', 'attention_keys': 'source.attention_interface_1.inputs', 'attention_values': 'source.attention_interface_1.inputs', 'attention_scores': 'source.attention_interface_1.source.nn_functional_softmax_0.input', 'attention_probabilities': 'source.attention_interface_1.source.nn_functional_dropout_0.output', 'attention_head_outputs': 'source.attention_interface_1.output'}
```

`print(attn.source)` renders the forward with every operation labeled at its line, and that listing is what to diff across releases ([Finding source ops](finding-source-ops.md)).

## What a release can move

An operation name is `{callable}_{occurrence}`: the dotted name in the source joined with `_`, and a count over the calls and bindings of that name in the order they appear. Three things move a name: the call is renamed, a binding of the same name is added or removed before it, or the arithmetic moves onto or off transformers' shared attention interface. The occurrence counts bindings as well as calls, which is why `attention_interface_1` is the call: `attention_interface_0` is the binding `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)` two lines above it. nnsight numbers every call in the source, executed or not, so GPT-2's `self__upcast_and_reordered_attn_0`, a branch that does not run, still takes a label.

### The shared interface

`INTERFACE = "attention_interface_1"` in `nnterp/components/attention.py` is the call every family on transformers' shared attention path makes, and inside `eager_attention_forward` the pattern is `nn.functional.softmax(...)` then `nn.functional.dropout(...)`. The base `Attention` reads the interface's arguments 1, 2 and 3 as the queries, keys and values, `nn_functional_softmax_0.input` as the scores, `nn_functional_dropout_0.output` as the pattern, and `output[0]` as the head outputs. GPT-OSS keeps the interface but concatenates a sink column before its softmax, so `nnterp/families/gpt_oss.py` reads the scores at `attn_weights_1`, the masked scores bound just before the column joins. That binding exists only when a mask is passed, as one is on every prompt.

### Families with their own attention arithmetic

| family | value to operation, in `nnterp/families/<family>.py` |
|---|---|
| BLOOM | queries, keys, values `self__reshape_0.output[0..2]`; scores `F_softmax_0.input`; head outputs `torch_bmm_0.output`; pattern `self_attention_dropout_0.output`; both contributions at `dropout_add_0.input` |
| MPT | queries, keys, values the bindings `query_states_0`, `key_states_0`, `value_states_0`; scores `nn_functional_softmax_0.input`; head outputs `torch_matmul_1.output` |
| Falcon | each key chosen by `config.alibi` through `by_alibi(without, with_alibi)`: queries and keys `apply_rotary_pos_emb_0.output` or the bindings `query_layer_0`, `key_layer_0`; values `value_layer_0`; scores `F_softmax_0.input` or `F_softmax_1.input`; head outputs `attn_output_1` or `flatten_0`; pattern `F_softmax_0` or `self_attention_dropout_0` |
| GPT-J, GPT-Neo, GPT-NeoX-Japanese | around the `self._attn(...)` call: `self__attn_0.inputs[0..2]`, `self__attn_0.source.nn_functional_softmax_0.input`, `self__attn_0.output[0]`, the dropout inside `self__attn_0.source` |
| CodeGen | GPT-J's operations, except the scores at `self__attn_0.source.call_0.input`, the call of the `nn.Softmax` instance built on the same line |
| XGLM | queries, keys, values at `*_states_reshape_0.output`; scores at `nn_functional_softmax_0` or `nn_functional_softmax_1` by the weights' dtype, a function key; head outputs `torch_bmm_1.output` |

### Recurrent mixers

`torch_chunk_gated_delta_rule_0` and `torch_recurrent_gated_delta_rule_0` are the two DeltaNet kernel calls, `use_precomputed_states_0` the branch binding that picks between them, `apply_mask_to_padding_states_0` the masking that gives a call's length, and `last_recurrent_state_3` the per-token state binding inside the recurrent loop, where the `_3` counts the `last_recurrent_state = ...` bindings before the one after a token's update. A release that adds or removes one binding in that loop moves it. The kernels' argument names `g`, `beta` and `initial_state` are the `select` keys of `decays`, `betas` and `state_input`. Mamba-1 pins `mamba_selective_scan_0`, `mamba_selective_state_update_0`, `ssm_state_3`, `ssm_state_0`, `ssm_state_to_0`, `scan_output_5` and `out_1`. Mamba-2 pins `mamba2_chunk_scan_0`, `mamba2_selective_state_update_0`, `new_states_0` and `ssm_states_0`, with each argument's position at the two call sites. The hub dispatcher's closure variables `torch_function` and `implementation` are what `route_kernels` and `needs_torch_kernels` read ([Recurrent mixers](internals/recurrent-mixers.md)).

### Names, not operations

A `RENAME` key that resolves nowhere is skipped silently, so a renamed module is caught by `test_standard_names_alias_native_envoys`, not at load. Where a release has renamed a module both spellings are keys, as with GPT-NeoX's `embed_out` and `lm_head`, and nnsight binds whichever resolves.

## How the suite catches a move

`test_every_source_value_resolves_on_every_layer` reads every available value on the family's `Attention` with `inside_forward()` on every attention block, one trace per value. A renamed operation fails there with `SourceNotAvailable` naming the value, the path it pinned and the operations the forward has now. `test_written_pattern_moves_the_logits` catches an operation that still resolves but is no longer what the values are mixed with, since a read can be causally inert. `test_interior_shapes` checks `softmax(scores) == pattern`, which catches a `_0` to `_1` slip onto another tensor of the same shape. `test_contribution_identity` catches a moved contribution operation on BLOOM, MPT and Falcon, and the DeltaNet and Mamba tests catch the kernel names, the branch binding, the argument names and the state bindings ([Testing](testing.md)). `support()` cannot know an operation moved, since it does not run the forward, and a `SourceNotAvailable` is raised only inside a trace.

## Upgrading transformers

1. Upgrade in a scratch environment and run `HF_HUB_OFFLINE=1 python -m pytest -q`. Each `SourceNotAvailable` names the value, the operation it pinned and the operations that exist now.
2. For each failing family, `print(model.layers[0].self_attn.source)` on the old and new versions, and inside a trace `print(attn.source.attention_interface_1.source)` for the interface, and diff the labeled listings. Decide whether the value moved (a renamed call, a binding added before it) or the family moved onto or off the shared interface.
3. Update the operation string in the family module, or delete the override when the family now uses the shared interface. Keep the standard name's meaning fixed: the pattern is the tensor the values are mixed with, the scores are the softmax's input, the contribution is what is added to the stream.
4. Re-run the family's file, then the whole suite. Families get no version conditionals: one module targets the version nnterp is developed on, and this procedure moves it.
5. A checkpoint with a config that no longer parses is a test-data problem, not a family problem: patch the config the way `tests/families/test_olmo3.py` does, into a copy of the snapshot.

Releases also move behavior no operation name pins. From 5.18 `generate` calls the model with no `attention_mask` when the batch has no padding, so `attention_mask` installs the all-ones mask it stands for into the call on read, and `generate` encodes a wrapper's images before the first forward, so under `generate` the tower's values come before `vision.image_token_mask`.

## What nnterp needs from nnsight

Every envoy keeps a link to the envoy that holds it, as `parent`, and `root` follows the links to the model envoy; a value reads its block through `self.parent` and the model through `self.root`, and a key with a leading `/` is walked from `root`. `envoys=` chooses the envoy class per module at construction, by type first and path suffix second. `eproperty.transform` takes `(view, raw)`, the edited view and the value as served. Beyond the documented surface nnterp also relies on `Mediator.current`, `Mediator.iteration`, `Mediator.occurrence`, `Interleaver.sourced` and the iterator's pin, exactly as [Descriptors](internals/descriptors.md) and [Recurrent mixers](internals/recurrent-mixers.md) describe. nnsight's `rename` must take a dotted alias (`"attn.c_proj": "attn.o_proj"` binds on the attention), which GPT-2's two `c_proj` modules need.

Import nnterp, or nnsight, before any `transformers.models...` module in a script and in every test process. `tests/conftest.py` is the one line that enforces it under pytest.
