---
title: Attention Interior
one_liner: Read, edit and assign the queries, keys, values, scores, pattern and per-head outputs inside every family's attention, under `attn_implementation="eager"`, call the block's own attention function again on tensors of your own (`self_attn.interface`), and read each head's value and output weights (`value_weight`, `output_weight`, `attention_input_gain`).
tags: [usage, attention, interior, source, eager, heads]
related: [docs/usage/residual-stream.md, docs/usage/layouts.md, docs/usage/availability.md, docs/usage/loading.md, docs/usage/generation.md, docs/usage/delta-net.md, docs/usage/remote.md, docs/patterns/head-kv-patching.md]
sources: [nnterp/components/attention.py, nnterp/components/eproperty.py, nnterp/families/gpt2.py, nnterp/families/falcon.py, nnterp/families/gpt_oss.py, nnterp/families/deepseek_v4.py, nnterp/families/gptj.py, nnterp/families/codegen.py, nnterp/families/xglm.py, nnterp/families/gpt_neox_japanese.py, nnterp/families/bloom.py, nnterp/families/mpt.py, nnterp/families/deepseek_v2.py, nnterp/families/gemma4_text.py, nnterp/components/layer.py, nnterp/families/phi3.py, nnterp/families/gpt_neox.py, tests/families/suite.py, tests/families/test_llama.py]
---

# Attention Interior

## What this is for

`attention_output` is what the attention adds to the residual stream. The six
values on `model.layers[i].self_attn` below are what happens *inside* it: the
queries, keys and values the softmax attention receives, the scores entering
the softmax, the pattern the values are mixed with, and each head's output
before the heads are concatenated and projected. Head ablation, pattern
surgery, key-side steering and query/key patching are all reads and writes of
these. They are source-located values: nnterp reaches them through nnsight
`.source` (nnsight docs/usage/source.md) at transformers' shared eager
attention call, or at the family's own operations where it does its own
arithmetic, and presents them with one layout on every family.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", attn_implementation="eager")
attn = model.layers[10].self_attn

with model.trace("The Eiffel Tower is in the city of"):
    q = attn.attention_queries.save()          # [batch, num_heads, seq, head_dim]      after RoPE
    k = attn.attention_keys.save()             # [batch, num_kv_heads, seq, head_dim]   before repeat_kv
    v = attn.attention_values.save()           # [batch, num_kv_heads, seq, head_dim]
    scores = attn.attention_scores.save()      # [batch, num_heads, query, key]         masked and scaled
    pattern = attn.attention_probabilities.save()   # [batch, num_heads, query, key]    what the values are mixed with
    heads = attn.attention_head_outputs.save() # [batch, seq, num_heads, head_dim]      before concat and o_proj
    attn.attention_head_outputs[:, :, 7] = 0   # ablate head 7 of block 10
    logits = model.logits.save()
```

The six reads are in forward order, so they fit in one trace; a write follows
the same rule (write the head outputs after reading the scores, not before).
`model.order(i)` gives block `i`'s order on any family, Falcon's different one included,
and `model.rank(name, i)` sorts a list of reads and writes by it
([availability](availability.md#forward-order-order-and-rank)).
`softmax(scores)` equals `pattern` to the dtype cast, and the pattern's rows
sum to one and are lower-triangular.

## The six values

| value | what it is | layout |
| --- | --- | --- |
| `attention_queries` | the queries entering the attention interface, after the query projection and, where the family has one, after the rotary embedding | `Queries`: `batch num_heads seq qk_head_dim` |
| `attention_keys` | the keys entering the interface, after RoPE and before `repeat_kv`, so the head axis is `num_kv_heads` wide under grouped-query attention | `Keys`: `batch num_kv_heads seq qk_head_dim` |
| `attention_values` | the values entering the interface, before `repeat_kv` | `Values`: `batch num_kv_heads seq head_dim` |
| `attention_scores` | the scaled, masked scores that enter the softmax | `Pattern`: `batch num_heads query key` |
| `attention_probabilities` | the pattern the values are mixed with: the dropout's output after the softmax, in the model dtype, an attention sink's column already dropped | `Pattern`: `batch num_heads query key` |
| `attention_head_outputs` | what the interface returns, each head's mix of the values, before the reshape to `[batch, seq, hidden_size]` and the output projection | `HeadOutputs`: `batch seq num_heads head_dim` |

On a Llama-style attention `attention_head_outputs.flatten(-2)` is `o_proj.input`, the
concatenated head outputs, and `self_attn.o_proj.input` is readable under any
`attn_implementation` (checked equal on the tiny Gemma 3 under eager and sdpa). That is
where Gemma Scope 2's attention SAEs read, not `attention_output`.

Each block's sizes are on its attention, read off the module: `attn.num_heads`,
`attn.num_kv_heads`, `attn.head_dim` and `attn.qk_head_dim` (`head_dim` except
under latent attention and on MiMo-V2-Flash). The root's `model.num_heads`, ...
are the config's and equal every block's except on Gemma-4, MiMo-V2-Flash and Laguna,
whose blocks differ. Every value's layout is on the descriptor, one of the named
aliases in `nnterp.components`: `Attention.attention_keys.layout is Keys`, and
`Attention.attention_keys.dims` is `("batch", "num_kv_heads", "seq", "qk_head_dim")`.
The sequence axis is 2 on the queries, keys and values, the layout transformers
hands its interface, and 1 on the head outputs.

The pattern is read after the dropout, not at the softmax, because that is the
tensor the values are multiplied with on every family: cast back to the model
dtype, and on a sink model without the sink column. Under a sink the rows sum
to less than one (GPT-OSS below).

## `attention_premix`: the heads at the output projection

`model.layers[i].self_attn.attention_premix` is what the output projection receives,
`[batch, seq, num_heads, head_dim]` (`HeadOutputs`), split by head as a view of the projection's
flat input. It is a module boundary, not an operation inside the eager forward, so it is
served under any `attn_implementation`; on a plain attention it is the same tensor as
`attention_head_outputs`, and on one with a gate or a norm between the heads and the
projection (Qwen3-Next, Laguna) it is what comes after them. The projection is `o_proj`, a
standard child name: a family that spells it `c_proj`, `dense`, `out_proj` or `wo` aliases it in
its `RENAME`, so `model.layers[i].self_attn.o_proj` is the module on every family. In-place edits
reach the model; assignment flattens the heads back.

```python
with model.trace(prompt):
    model.layers[3].self_attn.attention_premix[:, :, 7] = 0     # head 7's output never reaches o_proj, under sdpa too
```

## Before the rotary embedding: `attention_queries_prerope`, `attention_keys_prerope`, `rotate`

`attention_queries` and `attention_keys` are rotated for the positions they run at. Two
values on `model.layers[i].self_attn` are the same queries and keys one step earlier, as
the block's rotary embedding receives them: after the projection and any query or key
norm (Qwen3's `q_norm` / `k_norm`, OLMo-2/3, Gemma-3/4), before the rotation. And
`attn.rotate(x, positions)` applies that block's own rotary embedding to any tensor laid
out like them, at the absolute positions you give it.

| value | what it is | layout |
| --- | --- | --- |
| `attention_queries_prerope` | the queries the rotary embedding receives | `Queries`: `batch num_heads seq qk_head_dim` |
| `attention_keys_prerope` | the keys the rotary embedding receives, before the cache and `repeat_kv`: the new tokens' only | `Keys`: `batch num_kv_heads seq qk_head_dim` (`SharedRopeKeys` under latent attention, below) |

`attn.rotate(attention_queries_prerope, positions)` is `attention_queries` bit for bit, and
the same for the keys, wherever the block does nothing more between its rotary and its
attention ([below](#where-a-familys-values-before-the-rotary-differ)): `rotate` calls the rotary embedding module the model computes its
cos and sin with (rope scaling included, and the block's own layer type where sliding and
full blocks rotate differently) and the apply function the block's forward calls (partial
and interleaved rotations included). The per-family suite checks both on every family.
`rotate` takes `x` as `[..., seq, qk_head_dim]` and `positions` as `[seq]` (a tensor, a
list or a `range`), or `[batch, seq]` for an `x` of `[batch, heads, seq, qk_head_dim]` whose
prompts sit at different positions; it needs no trace, and on a model split over several
devices with `device_map` it applies the embedding on `x`'s device. Both values are on
the attention module's own forward, not inside the eager interface, so they are served
under any `attn_implementation`, and they come before `attention_queries` in a trace
(`model.order(i)` ranks them first).

Use them instead of `attention_queries` / `attention_keys`:

- **For any transplant to another position.** A rotated query carries its position in its
  rotation; written at another position it is attended with the wrong one. Write the value
  before the rotary instead: the block rotates what you wrote for the position it lands
  at, so moving a query from one prompt or position to another is one assignment.
- **For any edit that must happen before the rotation**: a direction found on the
  unrotated queries or keys (a probe, a steering vector), which the rotary would turn by a
  different angle at every position.
- **To recompute attention with keys from elsewhere**: `rotate` the source keys at the
  destination positions, outside the trace, and run the block's attention on them with
  [`interface`](#calling-the-attention-function-again-interface).

And instead of `self_attn.q_proj.output`: nnterp promises nothing about a projection's
layout (Phi-3 and GPT-NeoX fuse the queries, keys and values into one projection, Phi-3
has no `q_proj`), and on Qwen3, OLMo-2/3 and Gemma a norm sits between the projection and
the rotary, which a hand-rotated `k_proj.output` misses.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3-8B", attn_implementation="eager")
attn = model.layers[1].self_attn

source, target = "The capital of France is Paris", "My favourite city in the whole world is Rome"
with model.trace(source):
    q_src = attn.attention_queries_prerope.save()       # [1, num_heads, seq, head_dim], not rotated yet
    k_src = attn.attention_keys_prerope.save()          # [1, num_kv_heads, seq, head_dim]
with model.trace(target):
    attn.attention_queries_prerope[:, :, -1] = q_src[:, :, -1]   # the source's last query, rotated for the target's last position
    moved = attn.attention_queries.save()

seq = k_src.shape[2]
assert torch.equal(moved[:, :, -1:], attn.rotate(q_src[:, :, -1:], [moved.shape[2] - 1]))   # what the block made of it, bit for bit
later = attn.rotate(k_src, range(5, 5 + seq))           # the source's keys as if five positions later
```

On a block without a rotary embedding both values are `Unavailable` and `rotate` raises
the same reason: GPT-2, GPT-BigCode, GPT-Neo, OPT and XGLM add absolute positions to the
stream, BLOOM, MPT and Falcon with `alibi` bias the scores with ALiBi, and the attention
blocks of Jamba, Nemotron-H and Kimi-Linear use no position encoding. Some families skip it
per block: the full-attention blocks of Cohere 2, AFMoE and a hybrid EXAONE 4, the blocks
SmolLM3's and Llama 4's `no_rope_layers` mark, a Granite-SWA block whose
`layer_rope_theta` is 0, and every block of a GraniteMoE-Hybrid or OLMo-Hybrid checkpoint
whose config gives no rotary (`position_embedding_type`, `rope_theta`). `support()` names
the reason per block. `StandardizedVLLM` serves both values and `rotate` too, with the
differences [below](#on-the-vllm-engine).

### Where a family's values before the rotary differ

- **A partial rotary** (GPT-NeoX, Phi, Phi-3, StableLM, Persimmon, GPT-NeoX-Japanese, GLM,
  MiniMax-M2, MiMo-V2-Flash, GPT-J and CodeGen with a `rotary_dim`): the values are the
  whole heads, `qk_head_dim` wide, though the rotary rotates only each head's leading
  channels; `rotate` rotates those and passes the rest, as the forward does.
- **GPT-J and CodeGen** rotate interleaved channel pairs, tokens before heads, with sin and
  cos from the module's own table; the values are served heads first as views, as
  everywhere else.
- **Latent attention** (DeepSeek-V2/V3, Kimi-K2, DeepSeek-V3.2, GLM-5, GLM-4.7-Flash, Youtu):
  only the trailing `qk_rope_head_dim` channels of a query head rotate, and one rope key
  per position that every head shares; the rest of each key comes out of the latent and
  carries no position. So `attention_queries_prerope` is whole heads (`Queries`) and
  `attention_keys_prerope` that shared key, `SharedRopeKeys`: `batch 1 seq qk_rope_head_dim`.
  `rotate` rotates a query head's trailing channels, or all of a rope key; every head's
  `attention_keys[..., -qk_rope_head_dim:]` is the rotated shared key.
- **Gemma-4**: a KV-sharing block rotates no keys of its own, so its
  `attention_keys_prerope` is `Unavailable` (read the source block's). A source block
  stores its rotated keys for its borrowers, so a write to its `attention_keys_prerope`
  reaches them too, as an edit of its `k_proj` does.
- **DeepSeek-V4** rotates each head's trailing rope slice of one tensor that is both its keys
  and its values: a write to `attention_keys_prerope` rewrites the values too. Its sliding
  and compressed blocks take the rotary's `main` and `compress` frequencies.
- **Steps between the rotary and the attention**: HunYuan (dense and MoE) norms the
  queries and keys after the rotary (`query_layernorm`, `key_layernorm`), and Llama 4 too
  (`qk_norm`, an L2 norm), so `attention_queries` is the norm of `rotate(...)` there;
  JetMoE tiles the keys to every routing slot (`repeat(1, top_k, 1, 1)`) after the rotary,
  so its `attention_keys_prerope` is `num_kv_heads` wide where `attention_keys` is not;
  Ministral 3 scales its queries after the rotary by a factor that is 1 below
  `original_max_position_embeddings` and grows past it. `rotate` applies the rotary alone.

Assignment reaches the model on every family. An in-place edit does too, except where the
value is a split view of one fused tensor and torch refuses it as it does GPT-2's queries
(`RuntimeError: Output 0 of Select is a view and is being modified inplace`): CodeGen's,
DBRX's and GPT-NeoX's queries and keys, JetMoE's keys, and latent attention's shared rope
key. Assign there, or edit under `torch.no_grad()`.

### `rotate` and the positions it is given

A rope scaling that adapts to the sequence (`dynamic`, Phi-3's `longrope`) picks its
frequencies from the largest position in the call: the model's from the whole prompt,
`rotate`'s from the positions you pass. Past `original_max_position_embeddings` the two
differ unless the positions you pass reach as far as the prompt did; rotate the whole
sequence and index the result to match a long prompt. Under `generate` the keys before the
rotary on decode step `k` of `tracer.iter` (step 0 is the prompt) are the new token's
only, at position `n + k - 1` for a prompt of `n` tokens: `attn.rotate(keys, [n + k - 1])`
is the last row of that step's `attention_keys`.

On a model split tensor-parallel the two values are gathered along their head axis as the
queries and keys are ([Across GPUs](#across-gpus)), so every rank reads and writes them
whole, and `rotate` runs on the whole tensor on every rank.

### On the vLLM engine

`StandardizedVLLM` serves both values as the arguments of the attention module's own
`self.rotary_emb(positions, q, k)` call, `[1, heads, tokens, head_dim]`, in these layouts
([vllm.md](vllm.md#before-the-rotary-embedding)). vLLM's rotary kernel rotates in place, so they
are private copies taken before it runs; a write is rotated for the position it lands at, as
here. Three things differ:

- **`rotate` runs inside a trace only.** It calls the block's vLLM rotary embedding, which lives
  in the engine's worker, on a copy of `x`; there `rotate(attention_queries_prerope,
  range(tokens))` is the `attention_queries` vLLM's attention layer receives, bit for bit, and
  outside a trace it raises `RuntimeError`. `positions` is `[tokens]`; on the M-RoPE rotary of the
  Qwen-VL text families a position is a text token's, the same on all three rows.
- **Not served on vLLM's transformers backend** (OLMo, OLMo 2, SmolLM3, StarCoder2): the rotary
  is a call inside a forward the backend rewrites at load, which nnsight cannot instrument.
  GPT-BigCode, the backend's fifth family, has no rotary to serve and gives its no-rotary reason.
- **Not served under latent attention** (DeepSeek-V2/V3, Kimi-K2, GLM-4.7-Flash): vLLM rotates
  only each query head's rope slice and the shared rope key, inside its latent-attention layer,
  where `StandardizedTransformer` serves whole-head queries.

The blocks without a rotary embedding are the ones above, with the same reasons.

## The weights per head: `value_weight`, `output_weight`, `attention_input_gain`

Three accessors read a head's weights off the modules, outside a trace or inside one:

| accessor | what it is | shape |
| --- | --- | --- |
| `self_attn.value_weight(kv)` | key/value head `kv`'s slice of the value projection: what maps the attention's input to that head's values | `[head_dim, hidden_size]` |
| `self_attn.output_weight(j)` | head `j`'s slice of the output projection: what maps its output (its slice of `attention_premix`) into the stream | `[hidden_size, head_dim]` |
| `layers[i].attention_input_gain` | the per-channel gain between the stream and what the attention reads: the pre-attention norm's gain, ones on a post-norm block | `[hidden_size]` |

Use them for anything that multiplies a head's weights rather than reading its activations:
an OV circuit (`output_weight(j) @ value_weight(kv)`), direct logit attribution per head
(`model.lm_head.weight @ output_weight(j)` against that head's `attention_premix`), a head's
write of a given stream direction; and to read an adapted model's weights without merging
the adapter. Use them instead of slicing `v_proj.weight` / `o_proj.weight` by hand: the rows
of a head are laid out differently on fused projections (below), a LoRA layer's `.weight` is
the base weight, and the norm's gain is `weight` on one family and `1 + weight` on another.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen2.5-0.5B-Instruct", dispatch=True)
layer = model.layers[10]
attn = layer.self_attn
group = attn.num_heads // attn.num_kv_heads     # query heads per key/value head: 7 of 14 here

paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(paris) == 1
h = model.embed_tokens.weight[paris[0]]
h = h / h.pow(2).mean().sqrt()                  # unit RMS, as the block's norm leaves the stream

j = 3                                            # query head 3 reads key/value head 3 // group
with torch.no_grad():
    ov = attn.output_weight(j) @ attn.value_weight(j // group) @ (layer.attention_input_gain * h)   # [hidden_size]
    top = model.project_on_vocab(ov).topk(5).indices    # what head 3 writes for " Paris", through the unembedding
```

`value_weight(kv) @ x`, plus that head's slice of the projection's bias, is
`attention_values[:, kv]` for what the attention is called with (`attention_norm_output`, or
the stream itself on a post-norm block); `sum_j output_weight(j) @ attention_premix[:, t, j]`,
plus the projection's bias, is what `o_proj` returns at `t`. The suite checks both on every
family, to the rounding of a matmul over a slice of rows (which sums in another order than the
whole projection's, so not bit for bit). Under grouped-query attention query head `j` reads
key/value head `j // (num_heads // num_kv_heads)`. The accessors serve weights only: read a
bias off the module, and note that DBRX, and MPT and OLMo where `clip_qkv` is set, clamp the
projections' outputs after them, and that MiMo-V2-Flash's `value_weight` carries the
`attention_value_scale` its forward multiplies the values by. They are in the weight's dtype
and on its device, views of the parameter where torch allows (an in-place edit of one is an
edit of the model), and the model must hold its weights: an undispatched model's are meta until
its first trace or `model.dispatch()`.

**The layouts.** The base slices head-major rows of `v_proj` and columns of `o_proj`. Where a
family fuses or transposes its projections, its `Attention` reads its own:

| layout | families |
| --- | --- |
| every query, every key, then every value | Phi-3 (`qkv_proj`), DBRX (`Wqkv`), MPT (`Wqkv`), GPT-BigCode under multi-query (`c_attn`) |
| each head's query, key and value, `[num_heads, 3, head_dim]` rows | GPT-NeoX, GPT-NeoX-Japanese, BLOOM, Persimmon (`query_key_value`), GPT-BigCode without multi-query, Falcon-RW without multi-query |
| per key/value head its group's queries, its key, its value | Falcon's 40B layout (and falcon-11B) |
| every query, then the one key and the one value | Falcon's multi-query 7B layout |
| four parts, each a quarter of the heads' queries, values, then keys | CodeGen (`qkv_proj`, the forward's `mp_num`) |
| `Conv1D`, `[in, out]`: columns of `c_attn`, rows of `c_proj`, transposed | GPT-2 |

**Under a PEFT adapter** the weight is the one the projection's LoRA layer applies: the base
layer's plus each active adapter's `get_delta_weight`, in the base weight's dtype, which is what
`merge()` leaves, bit for bit. Once an adapter is merged it is in the base weight (the forward
then runs the base layer alone); with the adapters disabled it is the base weight, without an
adapter still merged into it (the forward unmerges that first). The result is a new tensor, not a
view. A DoRA or aLoRA adapter, or a non-LoRA tuner, raises `NotImplementedError`: its forward is
not the base layer plus a delta.

**The gain** is measured by calling the norm `attention_norm_output` is read at on `+-256` in
alternating channels, which every norm here normalizes to `+-1`, so it is the gain whatever the
parameterization (`weight`, Gemma's `1 + weight`, ones on OLMo's weightless LayerNorm): exact on
a norm without a bias, to the rounding of the bias on a LayerNorm with one (which the accessor
leaves out, as the weights leave out theirs). The attention reads `gain * n(x)`, plus that bias,
of a stream `x` whose normalization `n` is `x / rms(x)` (centred first on a LayerNorm). On a
post-norm block (OLMo-2/3, FlexOlmo, EXAONE-4, OLMo-Hybrid's attention blocks, OPT-350m) the
attention reads the stream itself and the gain is ones, so a head's OV map there has no norm to
fold. It follows `attention_norm_output`: on a block where that is unavailable (DeepSeek-V4's
hyper-connection streams, Nemotron-H's MLP blocks) it raises the same reason, and on a Mamba
block it is the gain of the norm the mixer reads.

**Where they are refused**, `self_attn.no_head_weights()` gives the one reason both methods raise:
latent attention (DeepSeek-V2/V3/V3.2, Kimi-K2/K2.5, GLM-5, GLM-4.7-Flash, Youtu, Kimi-Linear's
attention blocks: the values come out of a low-rank latent), DeepSeek-V4 (keys and values one
normed projection, a low-rank output projection), Gemma-4 (`v_norm` normalizes each head's values
after the projection), JetMoE (a mixture of attention heads, each expert with its own output
projection), ZAYA (half of each value head is the previous token's projection), the default
family, a vision tower's attention, a model split tensor-parallel (each rank holds a shard), and
`StandardizedVLLM` (fused, sharded projections; `attention_input_gain` is refused there too).

## Requirement: eager attention

All six live inside the eager attention forward. A model loaded with `sdpa`
or `flash_attention_2` never runs it, and each value says so:

```python
model = StandardizedTransformer("openai-community/gpt2", attn_implementation="sdpa")
model.support(layer=0)["self_attn.attention_probabilities"]
# "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"
```

Reading one raises `nnterp.Unavailable` with the same reason, before the model
runs. `attention_output` does not depend on the implementation and stays
available. A load with no `attn_implementation` gets transformers' default,
`sdpa` on every family that supports it, so pass `attn_implementation="eager"`
at load for any of the six ([loading.md](loading.md)). GPT-J, GPT-Neo, CodeGen,
XGLM, GPT-NeoX-Japanese, BLOOM and MPT have no `sdpa` implementation in transformers
and load eager with no flag.

## Reading, editing in place, assigning

Each value is the tensor the forward holds, so an in-place edit reaches the
model without any further step, and an assignment replaces it:

```python
with model.trace(prompt):
    attn.attention_scores[:, :, :, 0] = float("-inf")      # no query may attend to key 0 ...
    pattern = attn.attention_probabilities.save()          # ... so the pattern's first column is zero
    attn.attention_probabilities = my_pattern              # or replace the pattern outright
    logits = model.logits.save()
```

Assigning one argument of the interface replaces just that argument. Setting
`attention_keys` leaves the queries and values as they were:

```python
with model.trace(prompt):
    attn.attention_keys = attn.attention_keys * 0          # only the keys change
    v = attn.attention_values.save()                       # untouched
```

A written pattern must be `[batch, num_heads, query, key]` in the model dtype; a
written argument must match the shape the interface expects for it. The
model, not nnterp, reports a mismatch, from inside the forward.

The scores already carry the causal mask: masked entries hold the dtype's minimum
(`-inf` on GPT-Neo). So an edit that overwrites or multiplies them lifts the mask:
`attn.attention_scores[:] = 0` or `attn.attention_scores * 0` lets every query attend to
the tokens after it (and is NaN on GPT-Neo). Additive edits keep it. To set the real
entries, keep the masked ones:

```python
with model.trace(prompt):
    scores = attn.attention_scores
    masked = scores <= torch.finfo(scores.dtype).min        # the causal and padding mask
    attn.attention_scores = torch.where(masked, scores, torch.zeros_like(scores))   # uniform over the past only
    logits = model.logits.save()
```

## Calling the attention function again: `interface`

`model.layers[i].self_attn.interface` is the attention function this block's forward called
(`eager_attention_forward`, or the family's own: Gemma-2's with its logit softcap, GPT-OSS's
with its sink column), bound to everything else the call received this time: the mask, the
scale, the dropout, the sliding window. Call it on queries, keys and values and it returns
`(head_outputs, pattern)`, computed the way the model computes its own:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen2.5-0.5B", dispatch=True, attn_implementation="eager")
attn = model.layers[14].self_attn
prompt = "The capital of France is"

with model.trace(prompt):
    q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
    heads, pattern = attn.interface(q, k, v)         # the function this call ran, run again on the same tensors
    probs = attn.attention_probabilities.save()      # read after the call: the pattern comes later in the forward
    served = attn.attention_head_outputs.save()
    heads, pattern = heads.save(), pattern.save()

torch.equal(heads, served), torch.equal(pattern, probs)   # (True, True): the same function on the same shapes
heads.shape, pattern.shape                                # [1, 5, 14, 64], [1, 14, 5, 5]
```

The tensors you pass take the layouts of `attention_queries`, `attention_keys` and
`attention_values`, keys and values `num_kv_heads` wide (the function repeats them for each
group itself), with the live call's query and key lengths, which its mask has. The result is
in the layouts of `attention_head_outputs` and `attention_probabilities`, and on the served
tensors it is those two values bit for bit; the per-family suite checks that on every
family, on the prompt, on a decode step and on a padded batch. The call computes and
returns: nothing reaches the model, the cache, the other heads or the other rows until you
write a result.

Use it for:

- **any edit of keys or values that must not reach the cache or the other heads of a
  group.** Writing `attention_keys` changes the keys every query head of the group reads, at
  every query row; under `generate` an in-place edit also stays in the cache for every later
  step ([Under `generate`](#under-generate)). Call `interface` on edited copies instead and
  write only the heads you chose.
- **any per-row patch**: compute with the edited tensors, then write the rows you want of
  `attention_head_outputs` (the last one, for the prediction).
- **a recompute that must match the model bit for bit.** A hand-written
  `softmax(q @ k.T * scale)` needs the module's scale, misses Gemma-2's softcap and a sink
  column, and rounds differently in bf16 when its matmuls have other shapes than the
  model's; the call runs the model's function on the model's shapes.

A key or query edited before the rotary goes in rotated for its position,
`interface(q, attn.rotate(k_prerope, positions), v)`, with `attention_keys_prerope` read
first, since the forward reaches it before the interface
([Before the rotary embedding](#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate)).

Reading `attn.interface` takes the call's arguments, so it is read where the queries, keys
and values are, in any order with them; the first call of the result takes the function, which
the forward hands over just before it runs it, so make the first call before reading or writing
anything later in the attention (the scores, the pattern, the head outputs). Later calls reuse
the function and can come after those reads. `model.order(i)` ranks `self_attn.interface`
between the values and the scores. Read it once per forward and call the result as often as you
need. A second `attn.interface` after a call, or a first call after those reads, asks for a
point the model has passed: it raises `OutOfOrderError` in a plain trace and in a `tracer.iter`
loop from step 0 (`iter[:3]`). In a loop that starts past step 0 (`iter[1:3]`) it lands on the
next step's call instead: a second read then fails at the call with a shape mismatch (this
step's tensors against the next step's mask), and a late first call ends the loop early with
nnsight's `was never reached` warning:

```python
with model.trace(prompt):
    interface = attn.interface                       # read once, with the queries, keys and values
    q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
    k2 = k.clone()                                   # a copy: k is what the model attends with (and, under generate, the cache)
    k2[:, 1, 0] = 0                                  # key/value head 1, position 0
    clean, _ = interface(q, k, v)
    edited, _ = interface(q, k2, v)                  # called again: same mask, same scale
    attn.attention_head_outputs[:, -1, 9] = edited[:, -1, 9]   # head 9, last row: the only write
    moved = edited.ne(clean).flatten(0, 1).any(0).any(-1).save()

moved.nonzero().flatten().tolist()   # [7, 8, 9, 10, 11, 12, 13]: the result moves for the whole group; only head 9 is written
```

Inside `tracer.iter` it is that step's call: on a decode step the queries have one row, the
keys and values are the whole cache, and the mask is the step's (none, on an unpadded
`generate`). Within one invoke, a padded batch included, it is bit for bit. Across invokes
the model runs every invoke's rows in one call and `interface` runs one invoke's rows, so
the two agree to rounding (about `1e-8` in float32 on the tiny checkpoints), not bit for
bit: put the prompts in one invoke (`model.trace([a, b])`) when that matters. On a model
split tensor-parallel ([Across GPUs](#across-gpus)) it takes the gathered queries, keys and
values and runs every head on every rank. In training mode with a dropout the call draws
its own mask.

Which call it makes follows from where the family reads its queries, keys, values and head
outputs: the one call that takes the first three and returns the last. That is
transformers' shared interface call on most families, and the `_attn` method on GPT-J,
GPT-Neo, CodeGen and GPT-NeoX-Japanese. `support()` reports `self_attn.interface`
unavailable where those values are (any `attn_implementation` but `"eager"`, with the same
reason) and where no such call exists: BLOOM, MPT, Falcon and XGLM do their own arithmetic,
DeepSeek-V4 rotates the interface's output back before its projection, the Qwen ViT and
RADIO make one call per image, and `StandardizedVLLM`'s attention is the engine's kernel.

### Recomputing by hand

To compute with other tensors, call `interface`. To check the served values against the
arithmetic, two facts matter. The softmax scale is the module's, `attn._module.scaling`, not
always `1/sqrt(head_dim)` (Granite-SWA's is `attention_multiplier`, 0.0078 on granite-swash-2b against 0.088;
Granite's and GraniteMoE's is `attention_multiplier` too, `1/head_dim`, 0.015625 on
granite-3.0-2b against `1/sqrt(64)` = 0.125, so `attention_scores` are an eighth of a
Llama-scaled product), and
under grouped-query attention query head `h` reads key/value head `h // groups`:
`repeat_interleave`, not `repeat`. With both, the served values reproduce the scores on
the causal entries and the head outputs:

```python
with model.trace(prompt):
    q, k, v = attn.attention_queries.save(), attn.attention_keys.save(), attn.attention_values.save()
    scores = attn.attention_scores.save()
    pattern = attn.attention_probabilities.save()
    heads = attn.attention_head_outputs.save()

groups = q.shape[1] // k.shape[1]
again = q @ k.repeat_interleave(groups, dim=1).transpose(-1, -2) * attn._module.scaling
causal = torch.ones(again.shape[-1], again.shape[-1], dtype=torch.bool).tril()
torch.testing.assert_close(again[..., causal], scores[..., causal])
torch.testing.assert_close((pattern @ v.repeat_interleave(groups, dim=1)).transpose(1, 2), heads)
```

The last line fails where something sits between the mixed values and the head outputs:
Granite-SWA's sink scales each head's output after the mix ([Attention sink](../reference/families.md#attention-sink)).

The per-family suite checks every one of these on every pinned family: the
six shapes, `softmax(scores) == pattern`, that zeroing any interior value
moves the logits, and that an in-place edit of the scores and of the head
outputs lands (`tests/families/suite.py`).

## Family caveats

### GPT-2 and MPT: queries, keys and values are split views

GPT-2's `c_attn` produces one tensor that `split` divides into queries, keys
and values, and MPT's `Wqkv` one that `chunk` divides. torch refuses to edit
such a view in place:

```python
with model.trace(prompt):
    attn.attention_queries[:, 0] = 0
# RuntimeError: Output 0 of Select is a view and is being modified inplace. This view is the output of a
# function that returns multiple views. ...
```

Assign instead: `attn.attention_queries = attn.attention_queries * 0` or any
tensor of the same shape. The scores, the pattern and the head outputs accept
in-place edits on both (GPT-2's head outputs are a transposed view, which
torch allows).

A GPT-2 checkpoint with `reorder_and_upcast_attn` set takes GPT-2's own upcast
attention path, where the shared interface never runs. All six values then
report `this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own
upcast attention path`; `attention_output` stays available.

### Grouped-query attention: two head widths

On a checkpoint with `num_key_value_heads < num_attention_heads` (Llama-3,
Mistral, Qwen), the keys and values are read *before* `repeat_kv`, so their
head axis is `model.num_kv_heads` wide while the queries, the scores, the
pattern and the head outputs are `model.num_heads` wide. A head-wise edit of
the keys touches the group of query heads that shares that key head.

### Falcon: two attention branches, by `config.alibi`

Falcon's attention does its own arithmetic, and `config.alibi` picks one of
two branches with different operations. Each of the six values names its
operation by that flag, and all six need only an eager load on either branch.
Without alibi the queries are the rotary embedding's first return; with alibi
there is no rotary and they are the `query_layer` binding. Either branch then
updates the cache itself, and where the call has a cache (every call but
`use_cache=False`) the keys and values are what the update returns, every
position so far, read after the queries:

```python
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()         # the cache update's return
    v = attn.attention_values.save()
```

Without a cache they are the new tokens' own: the rotary's second return and
the `value_layer` binding before it without alibi (read the values first
there), the `key_layer` and `value_layer` bindings with alibi. BLOOM, MPT and
XGLM update the cache in their own forwards too, and their keys and values are
the update's return the same way. Whether the call has a cache is read off the
call's arguments, so read the attention's standard values before its child
modules' inputs or outputs in one trace. The pattern is the softmax itself without alibi (no dropout follows)
and the dropout after the second softmax with it. The head outputs are the
`scores @ values` product on both branches, served `[batch, seq, heads,
head_dim]` as a view: heads-first without alibi, flattened over batch and
heads with it, where the view unflattens them and a write reshapes back. The
7B layout is multi-query: keys and values have one head.

### Falcon-40B layout: keys and values already expanded

The `new_decoder_architecture` layout broadcasts its key/value heads to every
query head before the rotary embedding, so `attention_keys` and
`attention_values` are `num_heads` wide there, although
`model.num_kv_heads` reports the config's `num_kv_heads`.

### GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT: the same six on their own operations

These seven do their own attention arithmetic too, and their families map the
same six values onto it: GPT-J's, GPT-Neo's, CodeGen's and GPT-NeoX-Japanese's around
their `_attn` calls, XGLM's on the flattened tensors its two `bmm` calls take, BLOOM's on
its `_reshape` split and `bmm`, MPT's on its `*_states` bindings and second `matmul`. The
head outputs are heads-first in those forwards and are served as a sequence-first view,
so an in-place edit still lands. All seven load eager by default. BLOOM's, MPT's,
CodeGen's, XGLM's and GPT-NeoX-Japanese's values do not check the implementation at all
(the model has no other); GPT-J's and GPT-Neo's carry the eager check like the interface
families'. BLOOM's queries, keys and values take in-place edits; MPT's are split views
(above).

GPT-Neo's scores are `q @ k^T` with no `1/sqrt(head_dim)` scaling, computed and
served in float32 whatever the model's dtype; the pattern is cast back to the
values' dtype. Its `local` layers (every other one, by `attention_layers`)
mask keys `window_size` or more tokens back, so their pattern is zero there
and their scores hold float32's minimum. The causal mask is `-inf` on every
layer, so `scores * 0` is NaN there, not zero. `self_attn` is the inner
`attn.attention` module, which the `attn` wrapper calls and returns unchanged.

The scale sits in a different place on each of the others. CodeGen's scores are
`(q @ k^T + mask) / sqrt(head_dim)`, in float32: the mask is added before the division.
Its rotary embedding multiplies by a float32 buffer, so on a half-precision model its
queries (and its keys, when no cache casts them back) are served in float32. XGLM scales
the queries by `1/sqrt(head_dim)` as it projects them, so `attention_queries` is already
scaled and the scores are `q @ k^T` plus the mask; each of its six values is a view of the
`[batch * num_heads, ...]` tensor the forward holds, so an in-place edit lands and an
assignment is reshaped back. GPT-NeoX-Japanese folds the scale into a `baddbmm` and adds
the mask after it.

### GPT-OSS: an attention sink

Each GPT-OSS head carries a learned sink logit that joins the softmax as one
extra key column and is dropped afterwards. So the pattern's rows sum to
*less* than one, and `attention_scores` is read at the masked scores just
before the sink column joins them, the same shape as the pattern. A plain
`softmax(scores)` does not reproduce the pattern there; the sink does:

```python
sinks = attn._module.sinks.to(scores.dtype)                          # [num_heads]
column = sinks.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
combined = torch.cat([scores, column], dim=-1)
combined = combined - combined.max(dim=-1, keepdim=True).values
pattern_again = combined.softmax(-1)[..., :-1]                        # equals attention_probabilities
```

Granite-SWA's sink is not in the pattern. Its rows sum to one and `softmax(scores)`
reproduces them; the sink scales each head's output after the values are mixed, so
`attention_head_outputs` carries it and forcing the pattern does not set what a head
writes. The scale is computed from the scores, which are read before it, so a write to
`attention_scores` moves the pattern and the scale together ([families.md](../reference/families.md#attention-sink)).

### DeepSeek: multi-head latent attention

DeepSeek-V2/V3 give queries and keys `qk_head_dim = qk_nope_head_dim +
qk_rope_head_dim` and values `v_head_dim`, so `attention_queries` and
`attention_keys` are `model.qk_head_dim` wide and `attention_values` and
`attention_head_outputs` are `model.head_dim` wide. The interface sees
`num_heads` key/value heads whatever `num_key_value_heads` says: the latent
projection produces keys and values for every head.

### DeepSeek-V4: one tensor for keys and values, compressed keys, rotated-back heads

DeepSeek-V4 is not latent attention: the plain sizes hold (one key/value head,
`head_dim` from the config for queries, keys and values). It has GPT-OSS's sink on every
block and reads `attention_scores` at the same binding. Three things differ:

- The attention passes **one tensor as both keys and values**, so `attention_keys` and
  `attention_values` are the same object: an in-place edit of either edits both, and an
  assignment to one gives it a tensor of its own.
- On a compressed block (`compressed_sparse_attention`, `heavily_compressed_attention` in
  `config.layer_types`) the compressor's entries are concatenated after the token keys once
  the prompt reaches the block's rate (`config.compress_rates`), so the keys' `seq` axis and
  the pattern's `key` axis are `seq + seq // rate` long there; the mask over the extra
  columns carries the compressor's causality and, on a sparse block, its indexer's
  selection.
- The interface's output keeps a rotation on each head's rotary slice (the values are the
  rotated keys), which the module rotates back before the grouped output projection;
  `attention_head_outputs` is that rotated-back tensor, what `o_a_proj` reads, not the
  interface's output.

### Gemma-4: borrowed keys and values, per-layer head sizes

Gemma-4 (`gemma4_text`, `gemma4_unified_text`) has three things the other
families do not, and the six values stay available through all of them:

- **KV sharing.** The last `num_kv_shared_layers` blocks (20 of 35 on E2B, 18
  of 42 on E4B) have no `k_proj` or `v_proj`. Each attends with the keys and
  values of the last earlier block of its kind (sliding or full) before the
  sharing starts. `attention_keys` and `attention_values` there are what the
  attention receives, served as a copy private to the block on every Gemma-4
  block: an edit, in place or by assignment, changes what that block attends
  with and nothing else. Its borrowers keep the source block's tensors as
  computed, and an edit on a borrowing block stays on that block:

  ```python
  source, borrower = model.layers[13].self_attn, model.layers[15].self_attn   # E2B: 15 borrows 13's sliding keys
  with model.trace(prompt):
      source.attention_keys[:, :, -1] = 0                    # in place: block 13 alone
  with model.trace(prompt):
      source.attention_keys = source.attention_keys * 0      # assigned: block 13 alone
  ```

  To change the shared keys or values for every borrower, edit each block
  that attends with them (the source and each borrower of its kind), or edit
  the source block's `k_proj` / `v_proj` output (`source.k_proj.output`),
  which the shared tensors are computed from. Which blocks borrow is
  `layers[i].self_attn._module.is_kv_shared_layer`; the source of a kind is
  the block with `store_full_length_kv`.
- **`attention_k_eq_v`** (26B-A4B, 31B, 12B). The full-attention blocks have no
  `v_proj`; their values are `v_norm(k_proj(x))`, the keys' projection before
  `k_norm` and the rotary embedding. `attention_values` is that tensor.
- **Per-layer sizes.** Sliding blocks have `head_dim` 256, full blocks 512, and
  on 26B-A4B, 31B and 12B (not on E2B or E4B) the two kinds have different
  `num_key_value_heads`. `model.head_dim` and `model.num_kv_heads` are the
  config's top-level values, the sliding blocks'; read a full block's widths off
  its tensors (`attention_keys.shape`) or `model.config.get_text_config().per_layer_config[i]`.

The queries and keys are served after `q_norm` / `k_norm` and the rotary
embedding, and the softmax scale is 1 (`scaling = 1.0`), so the scores are the
plain dot products of what is served.

## On a vision-language wrapper

The text blocks' pattern has the image tokens as ordinary positions, so
`attention_probabilities[0, :, -1, mask[0]].sum(-1)`, with `mask` the tower's
`vision.image_token_mask` read first in the trace, is each head's mass from the last token
onto the image. The tower's own blocks serve the same six values over the patches
(`model.vision.layers[i].self_attn.attention_probabilities`, `[images, heads, patches,
patches]`, no causal mask), with the per-tower caveats in [vision.md](vision.md#the-values).

## Across GPUs

On a model split tensor-parallel (`distributed_config=`, [loading.md](loading.md#across-gpus-distributed_config)),
each rank computes its share of the heads, so inside the attention forward it holds only
those. nnterp gathers the six values along the head axis (`num_heads`, `num_kv_heads` for
the keys and values) wherever a rank holds a piece, so each rank reads all of them, in the
order of a one-device run, and splits an edit back: `attention_head_outputs[:, :, 20] = 0`
ablates head 20 whichever rank computes it. A plan that runs the whole attention on every
rank (OLMoE's) serves the values whole, and nothing is gathered. `attention_premix` is the
output projection's input, which nnsight reassembles itself.

## Under `generate`

The values are per forward call. On a decode step the queries, the scores,
the pattern and the head outputs have one query position, and the keys and
values (and the pattern's key axis) are as long as the cache:
`attention_probabilities` is `[batch, num_heads, 1, prompt + step]`. See
[generation.md](generation.md).

The keys and values a decode step's attention receives are the cache's own tensors on
most families, so an in-place edit of `attention_keys` or `attention_values` writes into the
cache and every later step attends with it, while an assignment replaces the call's
argument for that one forward. Under latent attention (DeepSeek-V2/V3 and the families on
their attention) the cache holds the compressed latent and the keys and values are expanded
from it on every call, and Gemma-4 serves copies, so there an edit lasts one forward either
way. To change what chosen heads read without touching the cache, call
[`interface`](#calling-the-attention-function-again-interface) on edited copies.

## Gotchas

- **`attn_implementation="eager"` or nothing.** Under `sdpa` every interior
  value is `Unavailable` with the reason above; `support()` says so per block
  before any trace.
- **Reads follow the forward within one trace.** Queries, keys and values,
  then scores, then pattern, then head outputs; on Falcon, BLOOM and MPT the
  keys and values after the queries where the call has a cache (on Falcon without
  alibi and without one, the values first). An out-of-order read raises `OutOfOrderError`, and
  inside a `tracer.iter` body the error can name a later location than the one
  you misplaced.
- **The first trace that reads a child module's value and then an interior value fails.**
  On a fresh model `self_attn.q_proj.output` (or `k_proj.output`) followed by
  `attention_queries` (or `interface`) raises `OutOfOrderError` naming an
  `attention_interface` input, on the first such trace only: nnsight builds
  `self_attn.source` lazily, after the forward has started. Touch
  `model.layers[i].self_attn.source` outside a trace first, or read them in two traces.
- **Read `interface` once per forward, and make the first call before the scores.** Reading
  it again after a call, or making the first call after reading the scores, the pattern or the
  head outputs, asks for a point the forward has passed: `OutOfOrderError` in a plain trace and
  in a `tracer.iter` loop from step 0, the next step's call in a loop that starts later. Later
  calls of the same result can follow those reads.
- **GPT-2 and GPT-BigCode queries, MPT queries, keys and values: assign, do not edit in place** (or edit under `torch.no_grad()`). GPT-2's and GPT-BigCode's keys and values take an in-place edit.
- **`hasattr(attn, "attention_probabilities")` raises `Unavailable`** when the
  value is unavailable; use `attn.support()` or `model.support(layer=i)`.
- **A sink model's pattern rows sum to less than one**, and its
  `attention_scores` are one step before the softmax's own input.
- **Overwriting or multiplying `attention_scores` lifts the causal mask.** Add to them,
  or keep the masked entries with `torch.where` as above.
- **A padded prompt's pad query rows are not attention**: uniform over every key on
  GPT-2, all on key 0 on GPT-Neo. Mask them with `model.attention_mask` before a
  per-row metric.
- **Two invokes of one trace can both read and edit `attention_probabilities` and
  `attention_scores`.** At one location the invoke written first is served first: an
  empty `tracer.invoke()` after an invoke that edits the scores reads the edited pattern.
- **On a hybrid, three blocks in four have no `self_attn`.** Decide which blocks
  have one outside the trace; see [delta-net.md](delta-net.md).
- **Remote runs re-resolve these on the server**, against its transformers; see
  [remote.md](remote.md).

## Related

- [residual-stream.md](residual-stream.md), `attention_output`: what these six add up to.
- [layouts.md](layouts.md), every value's axes, and `value.dims`.
- [availability.md](availability.md), `support()` and `Unavailable`.
- [loading.md](loading.md), `attn_implementation` at load.
- [generation.md](generation.md), the same values per decode step.
- [delta-net.md](delta-net.md), the `linear_attn` counterpart on hybrids: queries, keys, values, gates and a recurrent state, no pattern.
- [remote.md](remote.md), what a source-located value needs from an NDIF server.
- [head-kv-patching](../patterns/head-kv-patching.md), chosen heads reading another run's keys and values through `interface`.
- nnsight docs/usage/source.md, the mechanism these values are built on.
