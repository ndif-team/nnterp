"""OLMo 3: OLMo-2's post-norm block, with three sliding-window blocks for every full-attention block."""

MODEL_TYPE = "olmo3"
TITLE = "OLMo 3 / OLMo 3.1"
SUBTITLE = (
    "OLMo-2's post-norm block: attention and MLP read the raw stream and the stream receives the post-norms' "
    "outputs; three blocks in four attend over a 4096-token window with the plain rotary, and the fourth attends "
    "over the whole prefix with a YaRN-scaled rotary."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/Olmo-3-1025-7B"
#: The tiny checkpoint the test suite builds the page from (the test rewrites its config into a local copy).
PINNED = "yujiepan/olmo-3-tiny-random"
CHECKPOINTS = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-7B-Think-SFT", "allenai/Olmo-3-7B-Think",
    "allenai/Olmo-3-7B-RL-Zero-Math",
    "allenai/Olmo-3-1125-32B", "allenai/Olmo-3-32B-Think",
    "allenai/Olmo-3.1-32B-Instruct", "allenai/Olmo-3.1-32B-Think",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 101}
VLLM = True
QUIRKS = ["post-norms", "qk-norm", "sliding-window"]

#: Every checkpoint is 7B or 32B, so no real weights were run: shapes, identities, masks and read places are
#: from the pinned tiny checkpoint as tests/families/test_olmo3.py rewrites it (float32, CPU; the window
#: check loads it with sliding_window=3); the rotary scaling per block type from a meta build of
#: Olmo-3-1025-7B; sizes, layer_types and rope parameters from the checkpoints' configs.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "post_norm_note": "On OLMo-3 this norm follows the attention and its output is attention_output. "
                              "On Llama the same name is the norm before the MLP; OLMo-3 has no norm before "
                              "either sublayer.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, q_norm and k_norm",
            "variants": {
                "sliding_attention": "window {sliding_window}, plain rotary",
                "full_attention": "full causal, YaRN rotary",
            },
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

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "norm": "A plain RMSNorm: the gain is norm.weight (not 1 + weight), eps 1e-6. project_on_vocab applies it.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. Nothing follows it: "
            "logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

Two RMSNorms on the block, both after a sublayer, none before. `post_attention_layernorm` has
Llama's name and the opposite place: here it follows the attention, and nothing norms the MLP's
input. There is no `input_layernorm`. `self_attn.input` is `layers[i].input` itself and `mlp.input`
is `layers[i].input + attention_output`, both unnormalized, so a probe trained on a sublayer's
input sees the stream at its own scale.

## The contributions are the post-norms' outputs

`attention_output` and `mlp_output` are what the block adds, the outputs of
`post_attention_layernorm` and `post_feedforward_layernorm`, not of the modules. The identity is the
plain sum, exact in float32:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

Each post-norm divides by the RMS of what it reads, so a uniform scale of `mlp.output` or
`self_attn.output[0]` cancels in the contribution except through `eps` (1e-6). Zeroing a module's
output still removes its contribution, since the norm of zero is zero. Partial ablation and steering
of a sublayer's effect go through the contribution:

```python
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:] *= 0.5     # halves what the block adds
```

Direct logit attribution and any per-sublayer decomposition use the post-norms' outputs as the terms.

## Three sliding-window blocks for every full one

`config.layer_types` is `sliding_attention` on three blocks and `full_attention` on the fourth,
repeating: blocks 3, 7, 11, ... are full, 8 of 32 on 7B and 16 of 64 on 32B, and the last block is
full. On a sliding block a query attends to its own position and the 4095 before it
(`sliding_window` 4096); on a full block, to the whole prefix. On a prompt shorter than the window
the two masks coincide, and the two kinds of block are told apart by their rotary, below, not by
their patterns. Read a block's kind outside the trace:

```python
kinds = [model.layers[i].self_attn._module.sliding_window for i in range(model.num_layers)]
# 4096 on a sliding block, None on a full one
```

## The rotary differs by block kind

The config keeps one rotary per kind in `rope_parameters`. Sliding blocks use the plain rotary,
`rope_theta` 500000. Full blocks use YaRN, `factor` 8 over `original_max_position_embeddings` 8192,
with `attention_factor` 1.2079 multiplying the cosines and sines. `attention_queries` and
`attention_keys` are read after the rotary, so on a full block each query and key head is 1.2079
times the norm of its slice of `q_norm` or `k_norm`'s output, and the scores are about 1.459 times
what the same heads would give without the factor. Both kinds apply the same `head_dim ** -0.5`
query scale. The same holds on 7B and 32B and on every post-trained checkpoint.

## Queries and keys are normed across all heads

`q_norm` and `k_norm` are RMSNorms over the whole projection, `num_heads * head_dim` (4096 on 7B,
5120 on 32B) and `num_kv_heads * head_dim`, applied before the heads are split and before the
rotary. One RMS covers every head, so an edit to one head's slice of `q_proj.output` reaches the
others. Edit a single head at `attention_queries` or `attention_keys`, which no norm follows:

```python
with model.trace(prompt):
    model.layers[1].self_attn.attention_queries[:, 0] = 0   # head 0 only
```

7B has 32 key/value heads for 32 query heads. 32B has 40 query heads over 8 key/value heads, so
there an edit to key/value head `j` reaches query heads `5j` to `5j + 4`.

## Load with eager for the attention interior

The attention interior needs `attn_implementation="eager"`; the default load runs `sdpa` and
reports the six values unavailable. There is no softcap and no sink.

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale after it, and `project_on_vocab` on the last
block's `layer_output` equals `logits`. The final norm is a plain RMSNorm whose gain is
`model.norm._module.weight`, `eps` 1e-6. `lm_head` and `embed_tokens` are separate weights on every
checkpoint. The base tokenizer prepends nothing, so `model.input_ids` is the prompt's tokens alone;
`<|endoftext|>` (id 100257) is its end-of-text token.

## Intermediate checkpoints

The base repositories keep their training checkpoints as Hub branches: on Olmo-3-1025-7B, 1,421
`stage1-step…` branches from `stage1-step0` to `stage1-step1413814`, 52 `stage2-step…` and 13
`stage3-step…`; on Olmo-3-1125-32B, 655 `stage1-step…`, two `stage2-ingredient…` runs and
`stage3-step…`. Each loads with `revision=`:

```python
model = StandardizedTransformer("allenai/Olmo-3-1025-7B", revision="stage1-step1000")
```
"""
