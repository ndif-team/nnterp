# Names

Every family's tree also answers to Llama's names: `model.embed_tokens`, `model.layers[i]` with its `self_attn` and `mlp`, `model.norm` and `model.lm_head`. The names are nnsight aliases, a second attribute on the same envoy, so the native name keeps working and both reach one object.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")

with model.trace("The Eiffel Tower is in"):
    emb = model.embed_tokens.output.save()
    attn_in = model.layers[5].self_attn.input.save()
    mlp_in = model.layers[5].mlp.input.save()
    normed = model.norm.output.save()
    logits = model.lm_head.output.save()

print(model.layers[5] is model.transformer.h[5])
print(model.layers[5].self_attn is model.transformer.h[5].attn)
print(model.layers[5].self_attn.path, model.get("layers.5.self_attn").path)
# True
# True
# model.transformer.h.5.attn model.transformer.h.5.attn
```

The same trace body runs on `HuggingFaceTB/SmolLM2-135M` (a Llama) and `EleutherAI/pythia-70m-deduped` (a GPT-NeoX). `envoy.path` is always the native path, and `model.get()` resolves a dotted string through the aliases.

## The vocabulary

| name | GPT-2 | Llama | GPT-NeoX |
| --- | --- | --- | --- |
| `model.embed_tokens` | `transformer.wte` | `model.embed_tokens` | `gpt_neox.embed_in` |
| `model.layers[i]` | `transformer.h[i]` | `model.layers[i]` | `gpt_neox.layers[i]` |
| `model.layers[i].self_attn` | `...h[i].attn` | same | `...layers[i].attention` |
| `model.layers[i].mlp` | same | same | same |
| `model.norm` | `transformer.ln_f` | `model.norm` | `gpt_neox.final_layer_norm` |
| `model.lm_head` | same | same | same |

The containers are lifted out of the inner model, so it is `model.layers` on every family, never `model.model.layers`. Inside the sublayers the output projection, the activation and the down projection are `o_proj`, `act_fn` and `down_proj` (GPT-2's `c_proj`, `act` and `c_proj`). A module with no standard role keeps its native name only, such as GPT-2's `wpe`. Each family's own spellings are on its page under the Documentation tab, with [llama](../documentation/families/llama.md) as the reference.

## Key forms

A family's `RENAME` is nnsight's `rename=` ([Module Access](https://nnsight.net/features/6_modules/)). A dotted key such as `transformer.h` resolves from the root and mounts there, which is what lifts `layers` to `model.layers`. A single-component key such as `attn` binds on every envoy that has a child of that name, so one entry renames the attention in every block. A key that resolves nowhere is skipped, which is how `embed_out` covers the older spelling of GPT-NeoX's head alongside `lm_head`. Because a single-component key binds everywhere, no family aliases a block-level name that also exists outside the blocks: OPT's block norm `final_layer_norm` keeps its native name, since an alias would also bind on the decoder's `final_layer_norm`, which is `model.norm`.

## The norms

`input_layernorm` and `post_attention_layernorm` bind as aliases where a family spells them otherwise (GPT-2's `ln_1` and `ln_2`), but they are not part of the vocabulary, because their meaning differs between families. On a sequential block (Llama, GPT-2) the first feeds the attention and the second the MLP. On Gemma-2 and Gemma-3 `post_attention_layernorm` norms the attention's output, and the MLP's pre-norm is `pre_feedforward_layernorm`. On a parallel block (GPT-J, Phi, Falcon-7B) one `input_layernorm` feeds both sublayers, and OLMo-2 has only post-norms, so no `input_layernorm` exists to alias.

What matters about a norm is its output, and that is `self_attn.input` and `mlp.input` on every family:

```python
import torch

with model.trace("The Eiffel Tower is in"):
    norm1 = model.layers[5].input_layernorm.output.save()
    attn_in = model.layers[5].self_attn.input.save()
    norm2 = model.layers[5].post_attention_layernorm.output.save()
    mlp_in = model.layers[5].mlp.input.save()
print(torch.equal(norm1, attn_in), torch.equal(norm2, mlp_in))
# True True
```

On GPT-2 the two agree. The accessor form reads the right tensor on Gemma-2 and OLMo-2 as well, where the norm aliases would not.

## Blocks without a module

A hybrid (Qwen3-Next, Qwen3.5, OLMo-Hybrid) has `linear_attn` on most blocks and `self_attn` on a few, never both, and OPT has no `mlp` module at all. Decide which blocks have which outside the trace, with `getattr(layer, "self_attn", None)`. [Availability](values/availability.md) shows how `support()` reports them.

## `print(model)`

The repr prints a standard name beside its native one as `alias/native`, and each standard value with its layout and description, so `print(model.layers[i])` is the reference for what a block of this family has.

```python
print(model.layers[0].self_attn)
# GPT2Attention(
#   (c_attn): Conv1D()
#   (o_proj/c_proj): Conv1D()
#   (attn_dropout): Dropout(p=0.1, inplace=False)
#   (resid_dropout): Dropout(p=0.1, inplace=False)
#   (attention_queries) -> Queries [batch num_heads seq qk_head_dim]: The queries entering attention
#   (attention_keys) -> Keys [batch num_kv_heads seq qk_head_dim]: The keys entering attention
#   (attention_values) -> Values [batch num_kv_heads seq head_dim]: The values entering attention
#   (attention_scores) -> Pattern [batch num_heads query key]: The attention scores entering the softmax, masked
#   (attention_premix) -> HeadOutputs [batch seq num_heads head_dim]: Every head's output side by side, as the output projection takes it
#   (attention_output) -> Residual [batch seq hidden_size]: What the attention adds to the residual stream
#   (attention_probabilities) -> Pattern [batch num_heads query key]: The attention pattern the values are mixed with
#   (attention_head_outputs) -> HeadOutputs [batch seq num_heads head_dim]: The per-head outputs before the output projection
# )
```

A value is listed whether or not this checkpoint has it; `support()` is what says. The [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/) draws each family's block under these names, with the native name beside each. The values themselves start with the [residual stream](values/residual-stream.md).
