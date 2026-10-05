---
title: Root values and sizes
one_liner: "The model answers for the whole run — `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`, `input_size` — and for its sizes from the config; each block's own sizes are on its attention and MLP."
tags: [usage, logits, token_embeddings, next_token_probs, input_ids, sizes, config]
related: [docs/usage/residual-stream.md, docs/usage/methods.md, docs/usage/layouts.md, docs/usage/availability.md, docs/extending/adding-a-family.md]
sources: [nnterp/standardized.py, nnterp/components/eproperty.py, nnterp/components/attention.py, nnterp/components/mlp.py, nnterp/families/falcon.py, nnterp/families/deepseek_v2.py, nnterp/families/gpt2.py]
---

# Root values and sizes

## What this is for

Inside a trace the root envoy carries the values that belong to the whole model rather
than to one block: the final logits, the embeddings entering block 0, the next-token
distribution, and the ids and mask the model was called with. Outside a trace it answers
for the sizes an experiment needs (`num_layers`, `hidden_size`, `num_heads`, ...), read off
the config by one plain rule, or by the family's own spelling where its config differs.
Both are the same on every family.

## Canonical pattern

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("google/gemma-2-2b", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    ids = model.input_ids.save()               # [batch, seq]
    emb = model.token_embeddings.save()        # [batch, seq, hidden]
    raw = model.lm_head.output.save()          # the raw projection
    logits = model.logits.save()               # [batch, seq, vocab], after any softcap or scale past the head
    probs = model.next_token_probs.save()      # [batch, vocab]

model.num_layers, model.hidden_size, model.vocab_size, model.num_heads, model.num_kv_heads
```

Read them in forward order: `input_ids` / `attention_mask` / `input_size` first (they are
the model's input), `token_embeddings` next, `lm_head.output`, `logits` and
`next_token_probs` last.

## `logits`

The `.logits` of the model's output, `[batch, seq, vocab]`. On a family that softcaps after
`lm_head` (Gemma-2, `config.final_logit_softcapping`) it differs from `lm_head.output`:

```python
cap = model.config.final_logit_softcapping        # 30.0 on Gemma-2
torch.equal(logits, raw)                          # False on Gemma-2, True on GPT-2
torch.allclose(logits, cap * torch.tanh(raw / cap))   # True
```

`model.project_on_vocab(hidden)` ends with that step on any family: the softcap by default,
and the family's own where the model does something else after the head (Cohere multiplies
by `logit_scale`, Granite divides by `logits_scaling`), so `project_on_vocab` of the last
block's `layer_output` equals `logits` everywhere ([methods.md](methods.md#project_on_vocabhidden)).

Assigning replaces the logits in the model's output, so `tracer.result.logits` is what you
set:

```python
with model.trace(prompt) as tracer:
    model.logits = model.logits * 0
    result = tracer.result.logits.save()          # zeros
```

## `token_embeddings`

The embedding module's output, `[batch, seq, hidden]`: `model.embed_tokens.output` under
its standard name. It includes whatever the embedding module does itself (Gemma's
`sqrt(hidden_size)` scale is inside every Gemma family's `...TextScaledWordEmbedding`, XGLM's inside its
scaled embedding) and nothing the model applies afterwards: GPT-2's positional `wpe` is
added after it, BLOOM's `word_embeddings_layernorm` norms it, and Granite, HyperCLOVA X and
Falcon-H1 multiply it by `embedding_multiplier`. So it is what enters the first block only
where the model adds nothing; `layers[0].input` is that tensor on every family. Assign to
replace the module's output:

```python
with model.trace(prompt):
    model.token_embeddings = model.token_embeddings * 0
    logits = model.logits.save()                  # changed
```

## `next_token_probs`

`logits[:, -1].softmax(-1)`, `[batch, vocab]`, derived from the output. Read-only: there is
no inverse, and assigning raises

```
AttributeError: next_token_probs is derived from the logits and cannot be assigned; assign model.logits instead
```

Position `-1` is the last token of every row only under left padding, which is what a
causal model gets by default: nnsight sets the tokenizer's `padding_side` to `"left"` at
load, and its pad token to the EOS where it has none. A tokenizer set to right padding
(`tokenizer_kwargs={"padding_side": "right"}`) puts a pad token at a shorter prompt's
last position.

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True)
model.tokenizer.padding_side, model.tokenizer.pad_token   # ('left', '<|endoftext|>')
with model.trace(["Hi", "The Eiffel Tower is in"]):
    probs = model.next_token_probs.save()         # row 0 is the distribution after "Hi"
```

It is the softmax in the model's dtype: on a bf16 checkpoint a bf16 tensor (on
Llama-3.2-1B, 1e-4 from a float32 softmax of the same logits), and on a float16 one thousands of
entries underflow to exact zeros (17,583 of 50,304 on Pythia-70m). For a precise distribution or a KL, use
`model.logits[:, -1].float().softmax(-1)` or `.log_softmax(-1)`.

`nnterp.nnsight_utils.compute_next_token_probs(model, prompts)` is this read over a list of
prompts.

## `input_ids`, `attention_mask`, `input_size`

What the model was called with, `[batch, seq]` each; `input_size` is the ids' `torch.Size`.
`input_ids` and `attention_mask` are assignable, and the model then runs on what you set:

```python
with model.trace("Paris is the capital of"):
    other_ids = model.input_ids.save()
    other_logits = model.logits.save()

with model.trace("The Eiffel Tower is in"):
    model.input_ids = other_ids.clone()
    model.attention_mask = torch.ones_like(other_ids)
    logits = model.logits.save()

torch.allclose(logits, other_logits)              # True: the second trace ran on the first prompt's ids
```

`input_size` is read-only (`AttributeError: input_size is the ids' shape and cannot be
assigned; assign input_ids`). A `torch.Size` bound inside the block does not survive the
trace; save `torch.tensor(model.input_size)` if you need it outside.

The six print with the model. The `token_embeddings` description says "entering the first
block", which holds only where the model adds nothing after the embedding module (above):

```
  (logits): The logits the model returns, after anything it does past lm_head (a softcap, a scale), [batch, seq, vocab]
  (token_embeddings): The embedding module's output; layers[0].input is what enters the first block, [batch, seq, hidden]
  (next_token_probs): The next-token distribution at the last position, [batch, vocab]; derived, read-only
  (input_ids): The token ids the model was called with, [batch, seq]
  (attention_mask): The attention mask the model was called with, [batch, seq]; zeros are padding
  (input_size): [batch, seq] of the current call; read-only
```

## Sizes

Each size is a `StandardizedProperty` on the root, read-only, readable before any trace and
without `dispatch`. It reads the text config (a multimodal checkpoint's `text_config`, else the config itself) by the plain rule below unless the model's family module
defines a function of the same name (`def num_kv_heads(model): ...` in `falcon.py`), in
which case that function answers. The plain rule is what a Llama-style config needs; a
family whose config spells a size its own way keeps that spelling beside its names and
values, and anything the family does not define falls to the plain rule.

A root size is the config's value, equal to every block's on the families where the
blocks agree. Some sizes belong to one attention or MLP module, and on a family whose
blocks differ (Gemma-4, MiMo-V2-Flash) the root reports the config's top-level value
while each block's own is on `layers[i].self_attn` (`num_heads`, `num_kv_heads`,
`head_dim`, `qk_head_dim`) and `layers[i].mlp` (`intermediate_size`, one routed expert's
on a mixture of experts). Those are read off the module itself, outside or inside a
trace:

```python
model = StandardizedTransformer("hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM")
model.num_kv_heads                                         # 2: the config's
[layer.self_attn.num_kv_heads for layer in model.layers]   # [2, 4]: the sliding block has twice
[layer.mlp.intermediate_size for layer in model.layers]    # [64, 16]: the dense MLP, then one expert
```

| size | the plain rule |
| --- | --- |
| `num_layers` | `len(model.layers)` |
| `hidden_size` | `config.hidden_size` |
| `vocab_size` | `config.vocab_size` |
| `num_heads` | `config.num_attention_heads` |
| `num_kv_heads` | `config.num_key_value_heads`, else `num_heads` |
| `head_dim` | `config.head_dim` when the config says (Qwen3, Gemma), else `hidden_size // num_heads` |
| `qk_head_dim` | `head_dim` |
| `intermediate_size` | `config.intermediate_size` |

Some of the twenty-five families whose configs say it otherwise (all of them, with what each
reads, are in [families.md](../reference/families.md#logits-scales-and-sizes)):

| family | defines | what it reads |
| --- | --- | --- |
| `falcon` | `num_kv_heads` | `config.num_kv_heads` on the 40B layout (`new_decoder_architecture`); `1` under `multi_query`; else `num_heads` |
| `falcon` | `intermediate_size` | `config.ffn_hidden_size` |
| `deepseek_v2`, `deepseek_v3` | `head_dim` | `config.v_head_dim`, the width of one head's values and outputs. The config's own `head_dim` key is the latent width, which no served value has. |
| `deepseek_v2`, `deepseek_v3` | `qk_head_dim` | `config.qk_nope_head_dim + config.qk_rope_head_dim` |
| `gpt2`, `gptj`, `codegen` | `intermediate_size` | `config.n_inner`, `None` meaning `4 * hidden_size`. GPT-2's config also carries an `intermediate_size` key the model never reads. |
| `gpt_neo` | `intermediate_size` | `config.intermediate_size`, `None` meaning `4 * hidden_size` |
| `opt`, `xglm` | `intermediate_size` | `config.ffn_dim` |
| `gpt_neox_japanese` | `intermediate_size` | `hidden_size * config.intermediate_multiple_size` |
| `mpt` | `intermediate_size` | `config.expansion_ratio * hidden_size` |
| `bloom` | `intermediate_size` | `4 * hidden_size`; the config has no key for it |
| `gemma4_text`, `gemma4_unified_text` | `head_dim`, `num_kv_heads` | the config's top-level `head_dim` / `num_key_value_heads` as stored, the sliding blocks'. transformers marks both per-layer and refuses a plain `config.head_dim`; the full blocks' (512-wide heads, often fewer key/value heads) are on `layers[i].self_attn`. |

On the tiny checkpoints:

```python
StandardizedTransformer("hf-internal-testing/tiny-random-gpt2").intermediate_size      # 128: n_inner is None, so 4 * hidden_size (config.intermediate_size says 37)
StandardizedTransformer("Rocketknight1/tiny-random-falcon-7b").num_kv_heads           # 1: multi_query
StandardizedTransformer("Rocketknight1/tiny-random-falcon-40b").num_kv_heads          # 8: config.num_kv_heads, with num_heads 128
StandardizedTransformer("hf-internal-testing/tiny-random-OPTForCausalLM").intermediate_size   # 4: config.ffn_dim
model = StandardizedTransformer("hf-internal-testing/tiny-random-DeepseekV3ForCausalLM")
model.head_dim, model.qk_head_dim                                                      # (128, 192): v_head_dim, and nope + rope
```

`head_dim` is the config's on Qwen3 and Gemma, not `hidden // heads`, by the plain rule:
a Qwen3 checkpoint with `hidden_size=8`, `num_heads=4` and `head_dim=128` has 128-wide
heads. On DeepSeek-V3 `head_dim` (values, 128) and `qk_head_dim` (queries and keys, 192)
differ; see [layouts](layouts.md), which also names each root value's layout (`Logits`,
`Residual`, `NextTokenProbs`, `Tokens`). A mixture of experts' experts are
`config.moe_intermediate_size` wide, not `intermediate_size`; `layers[i].mlp.intermediate_size`
is the block's own.

A family of your own, shipped or passed to `nnterp.families.register()`, defines a size the
same way: [adding-a-family](../extending/adding-a-family.md#sizes).

## Gotchas

- **On Gemma-4 and MiMo-V2-Flash the root's sizes are the config's top-level ones.** A
  Gemma-4 full-attention block's `head_dim` and key/value heads differ, a MiMo-V2-Flash
  sliding block has twice the key/value heads, and Gemma-4 E2B's KV-sharing blocks have a
  double-width MLP: read `layers[i].self_attn.head_dim`, `.num_kv_heads` and
  `layers[i].mlp.intermediate_size`.
- **`logits` is not `lm_head.output` on Gemma-2, Gemma-4, Cohere or Granite.** Use `logits` for the
  model's prediction and `lm_head.output` only when you want the raw projection.
- **`next_token_probs` and `input_size` are read-only.** Assign `logits` or `input_ids`.
- **`next_token_probs` assumes the last position is the last token.** nnsight left-pads a
  causal model's batch by default; keep it that way.
- **`token_embeddings` is the embedding module's output**, not always what enters block 0
  (GPT-2's `wpe`, Granite's `embedding_multiplier` come after it); read `layers[0].input`
  for that.
- **Forward order.** `input_ids` and `token_embeddings` come before any block's value in
  the same trace; `logits` and `next_token_probs` after them all.
- **Assigning `input_ids` does not resize the mask.** Assign `attention_mask` to match when
  the new ids have another length.
- **`intermediate_size` is the dense MLP's width.** For an all-MoE family read
  `config.moe_intermediate_size`, or a block's `layers[i].mlp.intermediate_size`.
- **A size is read-only.** `model.hidden_size = 5` raises `AttributeError: hidden_size is
  read off the config; a family defines `def hidden_size(model)` to say it otherwise`. The
  family module is the one place a size is said.
- **A variant registered by spreading a shipped family's dicts drops its size functions.**
  Carry them over (`intermediate_size=gpt2.intermediate_size`), or the root's plain rule
  answers: see [registering](../extending/registering.md).

## Related

- [residual-stream](residual-stream.md): the per-block values.
- [methods](methods.md): `project_on_vocab`, which reproduces `logits` from a block's stream.
- [layouts](layouts.md): the axes of each value against these sizes.
- [availability](availability.md): the root values in `support()`.
