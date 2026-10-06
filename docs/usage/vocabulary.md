---
title: The standard vocabulary
one_liner: "One set of module names on every family — `embed_tokens`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head` — as nnsight aliases beside the native names."
tags: [usage, vocabulary, rename, aliases, families]
related: [docs/usage/loading.md, docs/usage/residual-stream.md, docs/usage/availability.md]
sources: [nnterp/families/__init__.py, nnterp/families/gpt2.py, nnterp/families/llama.py, nnterp/families/gpt_neox.py, nnterp/families/bloom.py, nnterp/families/mpt.py, nnterp/families/falcon.py, nnterp/families/dbrx.py, nnterp/families/opt.py, nnterp/standardized.py]
---

# The standard vocabulary

## What this is for

Every family spells its tree differently (`transformer.h`, `model.layers`,
`gpt_neox.layers`). nnterp gives them all Llama's block names, with the containers lifted
out of the inner `.model`, so a trace body written once runs on any registered
checkpoint. The names are nnsight `rename` aliases: an extra attribute on the same envoy,
not a replacement, so the native name keeps working and the two reach the same object.

## Canonical pattern

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    emb = model.embed_tokens.output.save()
    attn_in = model.layers[5].self_attn.input.save()      # what enters the attention
    mlp_in = model.layers[5].mlp.input.save()             # what enters the MLP
    normed = model.norm.output.save()
    logits = model.lm_head.output.save()

model.layers[5].self_attn is model.transformer.h[5].attn   # True: one envoy, two names
model.get("layers.0.self_attn").path                       # 'model.transformer.h.0.attn'
```

The same body runs on `"meta-llama/Llama-3.1-8B"` and `"EleutherAI/pythia-70m-deduped"`.

## The names

| standard name | GPT-2 | Llama | GPT-NeoX |
| --- | --- | --- | --- |
| `model.embed_tokens` | `model.transformer.wte` | `model.model.embed_tokens` | `model.gpt_neox.embed_in` |
| `model.layers[i]` | `model.transformer.h[i]` | `model.model.layers[i]` | `model.gpt_neox.layers[i]` |
| `model.layers[i].self_attn` | `...h[i].attn` | same | `...layers[i].attention` |
| `model.layers[i].mlp` | same | same | same |
| `model.norm` | `model.transformer.ln_f` | `model.model.norm` | `model.gpt_neox.final_layer_norm` |
| `model.lm_head` | same | same | same (`embed_out` on old releases) |

Two derived accessors belong to the vocabulary as well, because they are what a norm is
for:

| accessor | meaning |
| --- | --- |
| `model.layers[i].self_attn.input` | what enters the attention: the normed stream, or the block input where the block has no pre-norm (OLMo-2/3) |
| `model.layers[i].mlp.input` | what enters the MLP: the pre-MLP norm's output, whatever the family calls that norm |

On a vision-language checkpoint loaded with `task="image-text-to-text"`, the vision side has
names too: `model.vision` (the tower), `model.vision.layers[i]`, `model.vision.patch_embed`,
`model.vision.norm` and `model.projector`, on Gemma 3 and Llava 1.5 today ([vision.md](vision.md)).

Where other families differ, their `RENAME` says how. The container keys change per
family; the block keys only where a block spells a sublayer otherwise:

| family | containers | block-level aliases |
| --- | --- | --- |
| Llama, Mistral, Qwen2/3, Gemma-1/2/3, Phi-3, OLMo, DeepSeek, GPT-OSS, hybrids | `model.{embed_tokens, layers, norm}` | none needed |
| Llama 4 (text) | `model.{embed_tokens, layers, norm}`; `language_model.model.{embed_tokens, layers, norm}` and `language_model.lm_head` on a `Llama4ForConditionalGeneration` module | `feed_forward` -> `mlp` |
| Gemma-3, Gemma-4 | `model.{embed_tokens, layers, norm}`; `model.language_model.{embed_tokens, layers, norm}` on a `Gemma3ForConditionalGeneration` / `Gemma4ForConditionalGeneration` (`lm_head` stays at the root) | none needed |
| a multimodal wrapper of Llama, Qwen2/3/3.5, Mistral, Ministral 3, Gemma, Cohere 2, EXAONE 4 (loaded with `task="image-text-to-text"`) | `model.language_model.{embed_tokens, layers, norm}` (Idefics 3 / SmolVLM: `model.text_model.*`); `lm_head` stays at the root | none needed |
| GPT-2 | `transformer.{wte, h, ln_f}` | `attn` -> `self_attn`, `ln_1` -> `input_layernorm`, `ln_2` -> `post_attention_layernorm` |
| GPT-J | `transformer.{wte, h, ln_f}` | `attn` -> `self_attn`, `ln_1` -> `input_layernorm` |
| GPT-Neo | `transformer.{wte, h, ln_f}` | `attn.attention` -> `self_attn` (the module inside the `attn` wrapper), `ln_1` -> `input_layernorm`, `ln_2` -> `post_attention_layernorm` |
| GPT-NeoX | `gpt_neox.{embed_in, layers, final_layer_norm}` | `attention` -> `self_attn`; `embed_out` -> `lm_head` |
| BLOOM | `transformer.{word_embeddings, h, ln_f}` | `self_attention` -> `self_attn` |
| Falcon | `transformer.{word_embeddings, h, ln_f}` | `self_attention` -> `self_attn` |
| MPT | `transformer.{wte, blocks, norm_f}` | `attn` -> `self_attn`, `ffn` -> `mlp`, `norm_1`/`norm_2` -> the two norm aliases |
| DBRX | `transformer.{wte, blocks, norm_f}` | `norm_attn_norm.attn` -> `self_attn`, `norm_attn_norm.norm_1`/`norm_2` -> the two norm aliases, `ffn` -> `mlp` |
| OPT | `model.decoder.{embed_tokens, layers, final_layer_norm}` | `self_attn_layer_norm` -> `input_layernorm`; no `mlp` |
| Phi | `model.{embed_tokens, layers, final_layernorm}` | none |

Modules outside the vocabulary keep their native names only: GPT-2's `wpe` and `drop`,
BLOOM's `word_embeddings_layernorm`, OPT's `embed_positions`, `fc1`, `fc2`.

## How a key binds

A `RENAME` key with several components (`transformer.h`, `model.decoder.layers`,
`norm_attn_norm.attn`) binds where it resolves from: the root. That is what lifts `layers`
to `model.layers` on every family rather than `model.model.layers` on some. A
single-component key (`attn`, `attention`, `ffn`) binds on every envoy that has a child of
that name, so one entry renames the attention in every block. A key that resolves nowhere
is skipped, which is how `embed_out` -> `lm_head` covers both spellings of GPT-NeoX's head.
The rules are nnsight's (nnsight docs/usage/rename-modules.md); nnterp only chooses the keys.

Because a single-component key binds everywhere it resolves, OPT's block-level
`final_layer_norm` (its pre-MLP norm) keeps its native name: an alias for it would also
bind on the decoder's `final_layer_norm`, which is `model.norm`.

## Norm names are not part of the vocabulary

`input_layernorm` and `post_attention_layernorm` bind as aliases where a family spells them
otherwise (GPT-2's `ln_1`/`ln_2`, MPT's `norm_1`/`norm_2`), but they are best-effort, because
their *meaning* varies:

- On a sequential block (Llama, GPT-2) `input_layernorm` feeds the attention and
  `post_attention_layernorm` feeds the MLP.
- On Gemma-2/3/4 `post_attention_layernorm` follows the attention (it norms the attention's
  output), and the pre-MLP norm is `pre_feedforward_layernorm`.
- On a parallel block (GPT-NeoX with `use_parallel_residual`, Phi, GPT-J, StableLM-2,
  Falcon) one norm feeds both sublayers; GPT-J, Phi and Falcon have no
  `post_attention_layernorm` at all.
- OLMo-2/3 have only post-norms; there is no `input_layernorm` to alias.

What a user wants from a norm is what it produces, and that is `self_attn.input` and
`mlp.input` on every family. The test suite checks them per family against the family's
own norm. Use the accessors, not the norm names:

```python
with model.trace(prompt):
    attn_in = model.layers[i].self_attn.input.save()   # not input_layernorm.output
    mlp_in = model.layers[i].mlp.input.save()          # not post_attention_layernorm.output
```

## Blocks without one of the modules

A hybrid (Qwen3-Next, Qwen3.5, Qwen3.5-MoE text, OLMo-Hybrid) has `linear_attn` on three blocks in
four and `self_attn` on the fourth, never both. Decide which blocks have which outside the
trace:

```python
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
attn_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
linear_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]

with model.trace(prompt):
    mix = model.layers[linear_blocks[0]].linear_attn.attention_output.save()
    attn = model.layers[attn_blocks[0]].self_attn.attention_output.save()
```

OPT has no MLP module: `fc1` and `fc2` sit on the block, so `model.layers[i].mlp` does not
exist and `support()` lists no `mlp.*` key (a module no block has is not listed). See
[availability](availability.md).

## Same name, different meaning

`linear_attn` is one name for three recurrent mixers, and their values share names where
the roles match (a query reads the state, a key says where a token writes, a value is what
it writes). The roles match; the tensors do not. Code written against one mixer runs on
another and computes something else:

| | gated DeltaNet ([delta-net](delta-net.md)) | Mamba-1 ([selective-scan](selective-scan.md)) | Mamba-2 ([state-space](state-space.md)) |
| --- | --- | --- | --- |
| state per | head | channel | head |
| state layout | `[batch, heads, key_dim, value_dim]`, key side first | `[batch, channels, state_dim]`, value side first | `[batch, heads, state_dim, head_dim]`, key (`B`) side first, the transpose of the cache |
| update | delta rule: decay, then write what the state lacks at the key, `S += k ⊗ beta (v - Sᵀk)` | `h = exp(dt A) h + dt B x`, per channel | `h = exp(dt A) h + dt x Bᵀ`, per head |
| `attention_queries`, `attention_keys` | per head, before the kernel's l2-norm and `1/sqrt(key_dim)` | `C`, `B`: one group shared by every channel | `C`, `B`: per group of heads |
| `decays` | log decay per head; writable | `dt * A` per channel *and* state dimension; read-only | `dt * A` per head; writable, but it is `dt`, so writing it rewrites `betas` |
| `betas` | write strength in `(0, 1)` (`(0, 2)` on OLMo-Hybrid); writable on its own | the step `dt`; read-only | the step `dt`; writable, rounded on the way back |
| per-token state writes | `state` under `tracer.iter`, `set_state_after` | `state` under `tracer.iter`, `set_state_after` | none; write `state_input` |

On every recurrent mixer the sequence axis of `attention_queries`, `attention_keys` and
`attention_values` is 1 (`[batch, seq, ...]`); on softmax attention (`self_attn`) it is 2
(`[batch, heads, seq, head_dim]`), the layout transformers hands its attention interface.
An index such as `q[:, -1]` is the last token on `linear_attn` and the last head on
`self_attn`.

## Gotchas

- **Aliases are not paths.** `envoy.path` is always the native path
  (`model.transformer.h.0.attn`), `model.get("layers.0.self_attn")` resolves through the
  alias, and `envoys=` keys must be native paths ([loading](loading.md)).
- **A single-component alias binds in every block**, which is why no family aliases a name
  that also exists outside the blocks (OPT's `final_layer_norm`).
- **The norm aliases mean different things on different families.** Read
  `self_attn.input` and `mlp.input` instead.
- **`getattr(layer, "self_attn", None)` inside a trace can trip a served value.** Decide
  which blocks have which mixer before the trace, as above.
- **`model.layers` is the native `ModuleList` envoy**, so it indexes and iterates like
  one; a tuple or tensor it returns per block is what [residual-stream](residual-stream.md)
  normalizes.

## Related

- [loading](loading.md): merging your own `rename=` over the family's.
- [residual-stream](residual-stream.md): the values on `layers[i]`, `self_attn`, `mlp`.
- [availability](availability.md): a block that lacks a module.
- nnsight docs/usage/rename-modules.md: alias mechanics.
