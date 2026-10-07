"""EXAONE 4.0: a post-norm block with per-head query/key norms, and on 32B sliding blocks beside full blocks without rotary."""

MODEL_TYPE = "exaone4"
TITLE = "EXAONE 4.0"
SUBTITLE = (
    "Llama's tree with the norms moved after the sublayers: attention and MLP read the raw residual stream, "
    "what reaches the stream is a post-norm's output, queries and keys are RMS-normed per head, and on 32B "
    "three sliding-window blocks alternate with one full-attention block that applies no rotary."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "LGAI-EXAONE/EXAONE-4.0-1.2B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Exaone4ForCausalLM"
CHECKPOINTS = [
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "LGAI-EXAONE/EXAONE-4.0-32B",
    "LGAI-EXAONE/EXAONE-4.0.1-32B",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 182}
VLLM = True
QUIRKS = ["post-norms", "qk-norm", "sliding-window", "nope-blocks"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads, head_dim {head_dim}; "
                      "q_norm and k_norm per head, before the rotary",
            "variants": {
                "sliding_attention": "sliding window, RoPE",
                "full_attention": "full causal; NoPE on 32B",
            },
            "post_norm_note": "On EXAONE 4.0 this norm follows the attention. On Llama the same name is the norm "
                              "before the MLP; EXAONE 4.0 has no norm before either sublayer.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "layers": "30 blocks on 1.2B, all full attention with the rotary. 64 on 32B: three sliding-window blocks with "
              "the rotary, then one full-attention block without it, repeated.",
    "norm": "A plain RMSNorm: the gain is norm.weight (not 1 + weight), eps 1e-5. project_on_vocab applies it.",
    "head": "On 1.2B lm_head shares its weight with embed_tokens (tie_word_embeddings); on 32B it has its own.",
}

NOTES = """
## The block, in order

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

Two RMSNorms on the block, both after a sublayer, none before. `post_attention_layernorm` has
Llama's name and the opposite place: on Llama it is the norm before the MLP, here it follows the
attention, and nothing norms the MLP's input. There is no `input_layernorm`, so code that hooks
`input_layernorm.output` to read what the attention sees has nothing to attach to.

## The sublayers read the raw stream

`self_attn.input` is `layers[i].input` itself and `mlp.input` is
`layers[i].input + attention_output`, unnormalized. A probe or dictionary trained on a
sublayer's input sees the stream at its own scale.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn_in = model.layers[1].self_attn.input.save()           # equals x
    attn = model.layers[1].self_attn.attention_output.save()
    mlp_in = model.layers[1].mlp.input.save()                  # equals x + attn
```

The attention's contribution depends almost only on the direction of its input: `q_norm` and
`k_norm` remove the scale from the queries and keys, the projections are linear with no bias,
and `post_attention_layernorm` removes it again. Doubling what the attention reads changes
`attention_output` by 1e-2 relative at block 2 of 1.2B, 9e-4 at block 15 and 1e-4 at block 28;
the residue is the norms' `eps` (with every `eps` set to 0 it is exactly 0). The MLP has no such invariance: doubling what it reads changes
`mlp_output` by 40 to 68% at the same blocks.

## The contributions are the post-norms' outputs

`attention_output` and `mlp_output` are what the block adds, the outputs of
`post_attention_layernorm` and `post_feedforward_layernorm`, not of the modules. The identity
holds exactly, in float32 and bfloat16 alike:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

**Scale the contribution, not the module, and do not count on the post-norm undoing a scale
in the first blocks.** The modules' outputs are small at the bottom of 1.2B: the MLP's median
RMS per token is 4e-4 at block 0 and 2e-3 at block 2, below `√eps` (3.2e-3, `eps` 1e-5), so
there the post-norm is close to a fixed multiplier rather than a rescaling. Halving
`mlp.output` at block 0 moves `layer_output` by 42% relative, as much as halving `mlp_output`
does; at block 2 the two are 19% and 25%; from block 15 up halving `mlp.output` moves it by
under 0.4% against 13 to 23% for `mlp_output`, and by 2e-5 at block 28. The attention is the
same in kind (10% against 45% at block 0, 0.1% against 17% at block 15). Only `mlp_output`
and `attention_output` scale what the block adds at every depth:

```python
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:] *= 0.5     # halves what the block adds

with model.trace(prompt):
    model.layers[1].mlp.output[:] *= 0.5         # its effect depends on the depth
```

The post-norms' gains set the size of each term in a decomposition. Across 1.2B's 30 blocks
their means rise from 0.02 at block 0 to 0.36 on the attention side and lie between 0.12 and
0.48 on the MLP side, largest in the last blocks.

## Queries and keys are normed per head

`q_norm` and `k_norm` are RMSNorms over `head_dim` (64 on 1.2B, 128 on 32B), applied after the
heads are split and before the rotary embedding; `attention_queries` and `attention_keys` are
read after both. Each head is normed on its own, so an edit to one head's slice of
`q_proj.output` reaches no other head, and scaling the slice reaches nothing: at block 15 of
1.2B, tripling head 0's slice changes head 0's queries by 4e-5 and head 1's by 0, while adding
head 1's slice to head 0's changes head 0 by 35%. To weaken or strengthen a head's attention,
edit `attention_queries` or `attention_keys`, which no norm follows. Zeroing a head's queries
makes its scores zero and its pattern uniform over the causal prefix:

```python
with model.trace(prompt):
    model.layers[1].self_attn.attention_queries[:, 0] = 0   # head 0: uniform
```

## Which blocks rotate

The forward applies the rotary when `config.sliding_window` is null or the block is a sliding
one, so the two sizes differ:

- **1.2B**: `sliding_window` is null and `layer_types` is `full_attention` on all 30 blocks;
  every block is full causal attention with the rotary (`llama3` scaling, factor 16,
  `rope_theta` 1e6).
- **32B** (and 4.0.1-32B): `sliding_window` is 4096 and `layer_types` repeats three
  `sliding_attention` blocks and one `full_attention` (`LLLG`): blocks 3, 7, …, 63 are full,
  16 of 64. The sliding blocks apply the rotary over a 4096-token window; the full blocks
  apply none (NoPE) and see the whole prefix.

`attention_queries` and `attention_keys` are read after the rotary, so on a 32B full block they
equal `q_norm.output` and `k_norm.output` and carry no position; on every other block they are
rotated. On a full NoPE block the only order is the causal mask.

```python
with model.trace(prompt):
    normed = model.layers[1].self_attn.q_norm.output.save()
    queries = model.layers[1].self_attn.attention_queries.save()   # = normed on NoPE
```

The block's kind is `model.layers[i].self_attn._module.is_sliding`, not `sliding_window`: every
attention module stores the config's window, 4096 on 32B's full blocks too. On a prompt
shorter than 4096 tokens the sliding and full masks coincide, and on 32B the two kinds differ
only in the rotary.

## Grouped-query attention

32 query heads share 8 key/value heads on 1.2B (four each), 40 share 8 on 32B (five each).
`attention_keys` and `attention_values` are served before `repeat_kv`, `[batch, 8, seq,
head_dim]`, so an edit to key/value head `j` reaches query heads `4j` to `4j + 3` on 1.2B (an
edit to head 1's keys at block 8 changes heads 4 to 7 and no other) and `5j` to `5j + 4` on
32B. The query scale is `head_dim ** -0.5`.

## Load with eager for the attention interior only

The attention interior needs `attn_implementation="eager"`. The default `sdpa` load computes
the same function: on 1.2B in float32 the two loads' logits differ by at most 5e-5 (the largest
logit is about 25). There is no softcap or sink.

## The readout is lm_head after a plain RMSNorm

`logits` is `lm_head.output`, with no cap or scale after it, and `project_on_vocab` is
`lm_head(norm(h))`, equal to `logits` on the last block's `layer_output`. The final norm's
gain is `model.norm._module.weight` itself, mean 1.79 on 1.2B, between -0.64 and 2.03. On
1.2B `lm_head.weight` is `embed_tokens.weight`, so an edit to the embedding is an edit to the
unembedding; 32B keeps them separate. `token_embeddings` is the unscaled lookup and equals
`layers[0].input`.

## Every checkpoint is a chat model, and the tokenizer adds no BOS

The released checkpoints are the post-trained models, with a non-reasoning and a reasoning
mode; there is no base checkpoint. The tokenizer prepends nothing (`add_bos_token` is false;
`[BOS]`, id 1, exists and the chat template does not use it either), and `eos_token` is
`[|endofturn|]` (id 361). The template writes `[|user|]\\n…[|endofturn|]\\n[|assistant|]\\n` and,
with `add_generation_prompt`, an empty `<think>\\n\\n</think>\\n\\n` block; `enable_thinking=True`
ends on an open `<think>\\n` instead. Every assistant turn already in the conversation is
rendered with an empty think block, its reasoning dropped, unless `skip_think=False` is passed:

```python
ids = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, enable_thinking=True, tokenize=False,
)
```

## Checkpoints and kin

`EXAONE-4.0.1-32B` is a patch release of 32B with the same config. EXAONE 3.5 and EXAONE Deep
are `model_type` `exaone`, a pre-norm block (`ln_1`, `ln_2`) loaded as remote code, with
no query/key norms; EXAONE 4.5 (`exaone4_5`) and K-EXAONE (`exaone_moe`) each have a
`model_type` of their own. None of them is this family.
"""
