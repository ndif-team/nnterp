"""CodeGen: Salesforce's code models, the -nl, -multi and -mono checkpoints that load as CodeGenForCausalLM."""

MODEL_TYPE = "codegen"
TITLE = "CodeGen"
SUBTITLE = (
    "GPT-J's parallel block: one LayerNorm feeds attention and MLP and the block adds both at once; "
    "queries, values and keys come from one projection laid out as four groups of [q | v | k], rotary turns "
    "the leading rotary_dim channels of each head in adjacent pairs, and lm_head has a bias."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Salesforce/codegen-350M-mono"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-CodeGenForCausalLM"
CHECKPOINTS = [
    "Salesforce/codegen-350M-mono", "Salesforce/codegen-350M-multi", "Salesforce/codegen-350M-nl",
    "Salesforce/codegen-2B-mono", "Salesforce/codegen-2B-multi", "Salesforce/codegen-2B-nl",
    "Salesforce/codegen-6B-mono", "Salesforce/codegen-6B-multi", "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-16B-mono", "Salesforce/codegen-16B-multi", "Salesforce/codegen-16B-nl",
]

#: Set by hues.py (lineage: GPT-J).
PALETTE = {"hue": 57}
VLLM = False
QUIRKS = ["parallel-blocks", "fused-qkv", "partial-rotary", "interleaved-rotary", "own-attention-arithmetic",
          "layernorm", "tuple-blocks"]

#: What the visualization draws: GPT-J's parallel block. One ln_1 feeds both sublayers; the
#: schema draws a pre-norm per sublayer, so it appears twice, as one node.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "ln_1",
            "pre_norm_note": "One module, drawn on both rows: ln_1's output is what the attention and the MLP "
                             "both read, so self_attn.input equals mlp.input. The block has no second norm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, rotary on {rotary_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "ln_1",
            "pre_norm_note": "One module, drawn on both rows: ln_1's output is what the attention and the MLP "
                             "both read, so self_attn.input equals mlp.input. The block has no second norm.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "wte is a plain lookup: no scale and no position embedding (position enters through rotary in each "
             "attention), so token_embeddings equals layers[0].input.",
    "norm": "ln_f is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight, not tied to wte (tie_word_embeddings is false), and a bias. Nothing "
            "follows it: logits equals lm_head.output.",
}

NOTES = """
## The block is parallel, with one LayerNorm

```
h   = ln_1(x)
out = x + self_attn(h) + mlp(h)
```

`ln_1` is the block's only norm. Its output is both `self_attn.input` and `mlp.input`, the same
tensor; `input_layernorm` is its alias, and the block has no `post_attention_layernorm`.
`attention_output` is `out_proj`'s output and `mlp_output` is `fc_out`'s, and nothing normalizes
either before the add. Nothing connects the two sublayers of one block: zeroing `attention_output`
leaves that block's `mlp.input` bit-identical, so block `i`'s attention first reaches an MLP at
block `i + 1`.

```python
with model.trace(prompt):
    x = model.layers[2].input.save()
    attn = model.layers[2].self_attn.attention_output.save()
    mlp_in = model.layers[2].mlp.input.save()
    mlp = model.layers[2].mlp.mlp_output.save()
    out = model.layers[2].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)

with model.trace(prompt):
    model.layers[2].self_attn.attention_output[:] = 0
    mlp_in_ablated = model.layers[2].mlp.input.save()

assert torch.equal(mlp_in, mlp_in_ablated)
```

The block's `attn_weights` are returned on every forward, not only under `output_attentions`:
`model.layers[i].output[1]` is the block's attention pattern, equal to
`self_attn.attention_probabilities`.

## `qkv_proj` holds four groups of queries, values and keys

`self_attn.qkv_proj` is one `Linear` from `hidden_size` to `3 * hidden_size` without a bias
(`out_proj` has none either; `fc_in` and `fc_out` have one). The forward cuts its output into
`mp_num = 4` groups, a constant in the source, and splits each group as `[q | v | k]`: values in
the middle, keys last. Group `c` holds heads `c * num_heads / 4` onward, four heads per group on
350M. Neither a `[Q | K | V]` nor a `[Q | V | K]` split of the full width recovers a head. Keys and
values have `num_heads` heads; there is no grouping of key/value heads.

`attention_queries` and `attention_keys` are read after the rotary, `attention_values` as split.
To get queries and keys before the rotary, regroup the projection's output in the same trace:

```python
with model.trace(prompt):
    qkv = model.layers[2].self_attn.source.self_qkv_proj_0.output.save()
    q = model.layers[2].self_attn.attention_queries.save()
    v = model.layers[2].self_attn.attention_values.save()

b, s, _ = qkv.shape
H, D = model.num_heads, model.head_dim
groups = qkv.view(b, s, 4, 3, H // 4, D)            # 4 groups of [q | v | k]
q_pre, v_split, k_pre = (
    groups[:, :, :, j].reshape(b, s, H, D).transpose(1, 2) for j in range(3)
)
assert torch.equal(v_split, v)
rot = model.config.rotary_dim                       # 32 of 64 on 350M
assert torch.equal(q_pre[..., rot:], q[..., rot:])  # past rotary_dim: not rotated
```

The weight's rows are in the same order: `qkv_proj.weight.view(4, 3, num_heads // 4, head_dim,
hidden_size)[c, j, h]` is head `c * num_heads / 4 + h` of the queries (`j = 0`), values (`1`) or
keys (`2`). An in-place edit of `attention_queries`, `attention_keys` or `attention_values`
reaches the scores or the head outputs: the three are the tensors `_attn` computes with.

## Rotary turns adjacent pairs of the leading channels

`rotary_dim` is 32 of each head's 64 channels on 350M; the configs set 64 on every larger size, of
80 on 2B and of 256 on 6B and 16B. The rotation pairs adjacent channels `(2i, 2i + 1)` (GPT-J's
`rotate_every_two`), not the two halves `(i, i + rotary_dim / 2)` that `rotate_half` pairs on
Llama and GPT-NeoX; pair `i` turns by `position · 10000^(-2i / rotary_dim)`. A de-rotation or a
frequency analysis written for `rotate_half` pairs the wrong channels here. The channels past
`rotary_dim` are the projection's output unchanged and carry no position.

## The masked scores are the float32 minimum divided by the scale

`attn_implementation="sdpa"` fails at load; `_attn` is the only path. Because the mask is added
before the division by `sqrt(head_dim)`, a masked entry of `attention_scores` is
`finfo(float32).min / 8` on 350M (-4.25e37): finite, and not the float32 minimum a mask test may
look for. `softmax(attention_scores)` equals `attention_probabilities` exactly on a float32 load.

## The checkpoints are float16; the queries are not

The configs set `torch_dtype: float16` (2B-nl: `float32`), and a load without `dtype` keeps it,
on CPU too. The rotary multiplies by a float32 sin/cos buffer, so on that load `attention_queries`
is float32 while `attention_keys` is float16 (the cache casts the keys back, and a trace runs
with the cache on; under `use_cache=False` they stay float32). `attention_values` and
`attention_probabilities` are float16, `attention_scores` float32. `q @ k.transpose(-1, -2)` on
the two raises; cast first:

```python
with model.trace(prompt):
    q = model.layers[2].self_attn.attention_queries.save()     # float32
    k = model.layers[2].self_attn.attention_keys.save()        # float16
    scores = model.layers[2].self_attn.attention_scores.save()

recomputed = q @ k.float().transpose(-1, -2) / model.head_dim ** 0.5
```

`recomputed` equals `scores` wherever the mask lets a key through.

The load report lists `transformer.h.{0...19}.attn.causal_mask` as unexpected: the files store a
mask buffer that transformers' class does not have; the mask is built per call.

## The head has a bias, and a constant logit vector

`logits` equals `lm_head.output`, and `project_on_vocab` applied to the last block's
`layer_output` equals `logits` exactly, `ln_f`'s bias and `lm_head`'s included. A logit lens
therefore adds the same vector at every block, whatever the input:

```python
bias_logits = model.lm_head.weight @ model.norm.bias + model.lm_head.bias
```

On 350M-mono its standard deviation is 1.8 logits, against 3.9 for a whole lens readout at block 5
on a sample prompt, most of it from `ln_f.bias` (`lm_head.bias` alone: 0.11), and its top tokens
are `\\n`, `,`, `-`, `.`, `_`. Folding the norm into the unembedding keeps both biases as one term.
`wte` and `lm_head` are separate matrices. The vocabulary is padded: 51200 rows against 50295
tokenizer tokens on `-mono` (50257 on `-nl`). On 350M-mono the 905 extra rows are not zero
(biases between -0.38 and -0.34); on the sample prompt their logits are at most -4.9 and their
total probability 3e-10.

## On -multi and -mono, runs of whitespace are single tokens

The `-multi` and `-mono` tokenizers are GPT-2's 50257 tokens, at the same ids, plus 38: one token
for each run of 2 to 31 spaces (ids 50286 down to 50257) and of 2 to 9 tabs (50294 down to 50287).
The `-nl` tokenizers are GPT-2's alone, so the same code prompt has different tokens and positions
on `-nl` than on the other two. On `-multi` and `-mono` a run of two or more spaces becomes one
token and the word after it loses its leading space, so an indented statement starts with the
bare word:

```python
model.tokenizer.encode("    return x")   # [50284, 7783, 2124]: '    ', 'return', ' x'
```

GPT-2 encodes the same text as three `' '` and `' return'` (1441). A logit-lens or patching target
inside a block is the bare `return` (7783): after `"def add(a, b):\\n    "` 350M-mono puts 0.34 on
it. After the newline that opens a function body the model predicts the indent itself, the
four-space token, at 0.995 on a sample prompt.

The tokenizer adds no BOS: position 0 is the prompt's first token. `<|endoftext|>` (50256) is its
`bos_token`, `eos_token` and `unk_token`; the config's `bos_token_id` is 1, which is the token `"`.

## Three data variants at four sizes

Each size was released three times; the model cards and the paper (*A Conversational Paradigm for
Program Synthesis*) describe the data:

- `-nl`: trained on the Pile.
- `-multi`: initialized from `-nl` and trained further on BigQuery GitHub code in C, C++, Go,
  Java, JavaScript and Python (119.2B tokens).
- `-mono`: initialized from `-multi` and trained further on BigPython (71.7B tokens of Python).

The three share the architecture and the 51200-row embedding; the tokenizers differ (above). The
sizes, from the configs:

- 350M: 20 blocks, hidden 1024, 16 heads × 64, `rotary_dim` 32
- 2B: 32 blocks, hidden 2560, 32 heads × 80, `rotary_dim` 64
- 6B: 33 blocks, hidden 4096, 16 heads × 256, `rotary_dim` 64
- 16B: 34 blocks, hidden 6144, 24 heads × 256, `rotary_dim` 64

## CodeGen2 loads under this name, and its 1B and 3.7B run wrong

`Salesforce/codegen2-1B`, `codegen2-3_7B`, `codegen2-7B` and `codegen2-16B` set
`model_type: "codegen"` and ship their own modeling code (`auto_map`). Without
`trust_remote_code`, transformers builds its own `CodeGenForCausalLM`, nnterp wraps it, and
`support()` reports every value available. But the 1B and 3.7B code cuts `qkv_proj` into 8 groups
(`mp_num = 8`), not 4, so the class mixes columns of the queries, values and keys. On a 13-line
Python sample codegen2-1B's loss is 8.93 as loaded and 0.34 after regrouping each
`qkv_proj.weight` from 8 groups of `[q | v | k]` to 4. The 7B and 16B code uses 4 groups. With
`trust_remote_code=True` the remote code fails to import on transformers 5.17 (it imports
`transformers.onnx`). CodeGen2.5 (`Salesforce/codegen25-7b-mono`, ...) is a `LlamaForCausalLM`
and loads as Llama.
"""
