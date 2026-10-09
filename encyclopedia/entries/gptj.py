"""GPT-J: EleutherAI's GPT-J-6B and the fine-tunes that load as GPTJForCausalLM."""

MODEL_TYPE = "gptj"
TITLE = "GPT-J"
SUBTITLE = (
    "A parallel block with one LayerNorm: ln_1's output is the input of both the attention and the MLP, "
    "and the block adds both to the stream at once; the attention does its own arithmetic, and rotary "
    "turns the first quarter of each head in adjacent pairs."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "EleutherAI/gpt-j-6b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTJForCausalLM"
CHECKPOINTS = [
    "EleutherAI/gpt-j-6b",
    "togethercomputer/GPT-JT-6B-v1",
    "nomic-ai/gpt4all-j",
    "PygmalionAI/pygmalion-6b",
]

#: Set by hues.py (lineage: GPT-J).
PALETTE = {"hue": 54}
VLLM = True
QUIRKS = ["parallel-blocks", "tuple-blocks", "own-attention-arithmetic", "partial-rotary", "interleaved-rotary",
          "layernorm"]

#: What the visualization draws: one ln_1 whose output both sublayers read (drawn once per branch,
#: one node), no post-norms; the two contributions join the stream in one add.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One module, ln_1, drawn on both branches: the attention and the MLP read the same "
                             "output tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, rotary {rotary_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One module, ln_1, drawn on both branches: the attention and the MLP read the same "
                             "output tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "wte is a plain lookup with no scale and no position embedding (position enters through rotary in "
             "each attention), so token_embeddings equals layers[0].input. 50400 rows; the model card says only "
             "the 50257 GPT-2 BPE ids are used.",
    "norm": "ln_f is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight, not tied to wte (tie_word_embeddings is false), and a bias. Nothing "
            "follows it: logits equals lm_head.output.",
}

NOTES = """
## One LayerNorm feeds both sublayers

```
h   = ln_1(x)
out = self_attn(h) + mlp(h) + x
```

`ln_1` is `input_layernorm` and the block's only norm: there is no `post_attention_layernorm`.
Its one output tensor is passed to the attention and then to the MLP, so `self_attn.input` and
`mlp.input` are not two equal tensors but the same one. The block returns
`(hidden_states, attn_weights)` on every call: `model.layers[i].output[1]` is the attention
pattern, equal to `self_attn.attention_probabilities`, and `layer_output` is the first element.

## An in-place edit of `self_attn.input` reaches the MLP

Because both sublayers read one tensor, an in-place edit of `self_attn.input` also changes what
the MLP reads. An assignment replaces only the attention's argument and leaves `mlp.input` as it
was. An edit of `mlp.input`, in place or assigned, reaches only the MLP: the attention has already
run. To change what both read, assign `input_layernorm.output`.

```python
with model.trace(prompt):
    model.layers[4].self_attn.input[:, -1] = 0      # in place
    mlp_in = model.layers[4].mlp.input.save()       # its last row is zero too

with model.trace(prompt):
    # an assignment replaces only the attention's argument
    model.layers[4].self_attn.input = torch.zeros_like(model.layers[4].self_attn.input)
    mlp_in = model.layers[4].mlp.input.save()       # unchanged
```

The attention runs before the MLP, so inside one trace read and edit the attention's values
before `mlp.input`; asking for `mlp.input` first makes a later attention read fail as out of
order.

## The contributions, added in the block's order

`attention_output` is the attention module's output (`out_proj`, no bias) and `mlp_output` the
MLP's (`fc_out`, with a bias). The block computes `attn + mlp + x`, so that sum reproduces
`layer_output` bit for bit; `x + attn + mlp` differs from it by rounding (up to 0.5 at block 4
of 6B in bfloat16 on a sample prompt). Nothing inside a block connects the two sublayers:
zeroing `attention_output` leaves that block's `mlp.input` bit-identical, and the first MLP that
reads block `i`'s attention is block `i + 1`'s.

```python
with model.trace(prompt):
    x = model.layers[4].input.save()
    attn = model.layers[4].self_attn.attention_output.save()
    mlp = model.layers[4].mlp.mlp_output.save()
    out = model.layers[4].layer_output.save()

assert torch.equal(attn + mlp + x, out)                 # the block's own order
```

## The attention is eager by default and computes its scores in float32

transformers has no `sdpa` path for GPT-J: a load without `attn_implementation` runs the eager
`GPTJAttention`, and all six interior values are served with no flag. They carry the eager check,
so a `flash_attention_2` load reports them unavailable. The values are read around the module's
`_attn(query, key, value, mask)` call: its arguments are `attention_queries`, `attention_keys`
and `attention_values` (16 heads each on 6B, no grouping, in the model's dtype), and its first
return is `attention_head_outputs`.

`_attn` casts the queries and keys to float32, so `attention_scores` is float32 whatever the
model's dtype: `q @ kᵀ / 16` (`sqrt(head_dim)`, 256 on 6B) plus the causal mask, whose masked
entries hold the model dtype's minimum (`-3.39e38` under bfloat16). The pattern is cast back to
the model's dtype, and `softmax(attention_scores)` cast to it equals `attention_probabilities`
exactly. `q_proj`, `k_proj`, `v_proj` and `out_proj` are separate `Linear`s with no bias;
`attention_head_outputs.flatten(-2)` is `out_proj.input`.

```python
with model.trace(prompt):
    q = model.layers[4].self_attn.attention_queries.save()
    k = model.layers[4].self_attn.attention_keys.save()
    scores = model.layers[4].self_attn.attention_scores.save()

again = q.float() @ k.float().transpose(-1, -2) / model.head_dim ** 0.5
causal = torch.ones(q.shape[2], q.shape[2], dtype=torch.bool, device=q.device).tril()
assert torch.equal(again[..., causal], scores[..., causal])
```

## Rotary turns the first 64 dimensions, in adjacent pairs

`rotary_dim` is 64 of each head's 256 dimensions on 6B. `attention_queries` and
`attention_keys` are read after the rotary; their dimensions 64 to 255 are the projection's
output unchanged, and only the first 64 make a score depend on the distance between two tokens.
Within those 64, GPT-J rotates adjacent pairs: dimensions `2i` and `2i + 1` turn together at
frequency `10000 ** (-2i / 64)` (`rotate_every_two`). Llama and GPT-NeoX rotate dimension `i`
with `i + 32` (`rotate_half`). Query dimension 1 on GPT-J is the partner of dimension 0, so a
comparison of queries or keys across families, or rotary code written for `rotate_half`, needs
the rotated span reordered, evens then odds. Reordered, GPT-J's rotation is `rotate_half`'s with
the same frequencies; recomputed in float32 the two agree exactly.

```python
rot = model.config.rotary_dim                                        # 64
perm = torch.cat([torch.arange(0, rot, 2), torch.arange(1, rot, 2)])

with model.trace(prompt):
    q = model.layers[4].self_attn.attention_queries.save()

q_half = torch.cat([q[..., :rot][..., perm], q[..., rot:]], dim=-1)   # rotate_half's
```

The sine and cosine table is a float32 buffer, cast to the model's dtype before it multiplies,
so the queries and keys are served in the model's dtype.

## Loading: the main branch is float32

The `main` branch of `EleutherAI/gpt-j-6b` holds float32 weights (24 GB) and its config names no
dtype, so a load without `dtype` is float32. Pass a half dtype to fit a single GPU, or load the
`float16` branch, a 12 GB half-precision copy:

```python
model = StandardizedTransformer(
    "EleutherAI/gpt-j-6b", dtype=torch.bfloat16, device="cuda", dispatch=True
)
```

## The head has a bias, and so does the final norm

`logits` equals `lm_head.output`, and `project_on_vocab` applied to the last block's
`layer_output` equals `logits` exactly. `ln_f` is a LayerNorm with a bias and `lm_head` adds a
bias of its own, so a logit lens adds `lm_head.weight @ norm.bias + lm_head.bias` at every block,
whatever the input. On 6B that vector has a standard deviation of 0.40 logits, most of it from the
norm's bias (`lm_head.bias` alone: 0.034), and its top tokens are `,`, `\\n`, `.`, `-`, ` and`.
Folding the norm into the unembedding, or a direct logit attribution, keeps both as constant
terms. `wte` and `lm_head` are separate matrices: the unembedding is not the embedding's
transpose.

```python
bias_logits = model.lm_head.weight @ model.norm.bias + model.lm_head.bias
```

## The vocabulary is padded to 50400

The tokenizer is GPT-2's BPE, 50257 tokens, plus 143 added `<|extratoken_N|>` tokens (ids 50257
to 50399), so `len(model.tokenizer)` equals `vocab_size`, 50400. The model card states that only
the 50257 GPT-2 ids are used. Their unembedding rows are smaller (norm 0.32 against 1.31) but not
zero: on six sample prompts their logits stay below −3.7 while every position's top logit is
above 12. The tokenizer adds no BOS: position 0 is the prompt's first token, and
`<|endoftext|>` (50256) is the BOS, EOS and unknown token.

## Model editing and causal tracing

- ROME's GPT-J hyperparameters rewrite `transformer.h.5.mlp.fc_out`, which is
  `model.layers[5].mlp.fc_out.weight`. Its key is the input of that `Linear` at the subject's last
  token, `model.layers[5].mlp.fc_out.input`, and its value the output, which equals
  `model.layers[5].mlp.mlp_output`. MEMIT spreads the same update over blocks 3 to 8.
- ROME's causal tracing on `EleutherAI/gpt-j-6B` adds noise to the output of `transformer.wte`,
  `model.token_embeddings`, and restores the outputs of `transformer.h.{i}`, `.mlp` and `.attn`
  (first element), which are `layer_output`, `mlp_output` and `attention_output`.
- Function vectors (Todd et al.) on GPT-J read each head from the input of `attn.out_proj` split
  by head, which is `model.layers[i].self_attn.attention_head_outputs`, and add the vector to the
  output of `transformer.h.{i}`, `model.layers[i].layer_output`.
"""
