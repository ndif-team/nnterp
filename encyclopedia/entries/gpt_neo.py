"""GPT-Neo: EleutherAI's GPT-Neo and the TinyStories models that load as GPTNeoForCausalLM."""

MODEL_TYPE = "gpt_neo"
TITLE = "GPT-Neo / TinyStories"
SUBTITLE = (
    "GPT-2's sequential block with a learned position embedding, whose attention sits one module down inside "
    "an attn wrapper and does its own arithmetic: the scores are not scaled, and every other block masks "
    "keys outside a local window."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "EleutherAI/gpt-neo-125m"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
CHECKPOINTS = [
    "EleutherAI/gpt-neo-125m", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
    "roneneldan/TinyStories-1M", "roneneldan/TinyStories-3M", "roneneldan/TinyStories-8M",
    "roneneldan/TinyStories-28M", "roneneldan/TinyStories-33M",
    "roneneldan/TinyStories-1Layer-21M", "roneneldan/TinyStories-2Layers-33M",
    "roneneldan/TinyStories-Instruct-33M",
]

#: GPT-J's kin (EleutherAI's line): GPT-J hashes to 222, CodeGen sits at 234.
PALETTE = {"hue": 228}
VLLM = False
QUIRKS = ["tuple-blocks", "own-attention-arithmetic", "sliding-window", "position-embeddings", "layernorm"]

#: What the visualization draws: GPT-2's sequential block, ln_1 before the attention, ln_2 before the MLP.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Native name ln_1: a LayerNorm with a bias. Its output is the input of the attn wrapper "
                             "and of attn.attention, the module self_attn names.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, unscaled",
            "variants": {
                "global": "{num_heads} heads × {head_dim}, unscaled",
                "local": "{num_heads} heads × {head_dim}, window {window_size}",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native name ln_2, the MLP's input norm: a LayerNorm with a bias.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is wte's lookup alone. The learned position embedding transformer.wpe is added "
             "after it, so layers[0].input = token_embeddings + transformer.wpe.output.",
    "norm": "ln_f is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has no bias and shares its weight with embed_tokens (wte). Its output is logits, unchanged.",
}

NOTES = """
## The block, in order

```
h   = x + attn(ln_1(x))      # attn returns attn.attention's output; ln_1 = input_layernorm
out = h + mlp(ln_2(h))       # ln_2 = post_attention_layernorm
```

`transformer.h[i].attn` is a `GPTNeoAttention` wrapper that calls `attn.attention`, a
`GPTNeoSelfAttention`, and returns its output unchanged. `self_attn` names the inner module: the
projections, the arithmetic and the six interior values are there, and `attention_output` is its
first output, after `out_proj`. Inside one trace it comes before the wrapper's `attn.output`.
The block returns `(hidden_states, attn_weights)` on every call, so `model.layers[i].output[1]`
is the pattern and `layer_output` is the first element. Nothing sits between a sublayer and the
stream, and `layers[i].input + attention_output + mlp_output == layer_output` is exact (difference
0.0 on the pinned checkpoint).

## Local and global blocks alternate

`config.attention_layers` names each block `global` or `local`; every released checkpoint starts
with `global` and alternates. A `local` block masks every key `window_size` or more tokens back
(256 on every released checkpoint), so a query attends to itself and the 255 tokens before it.
The window is part of the causal mask the module holds, so the pattern is zero outside it and the
scores there hold float32's minimum. A prompt of 256 tokens or fewer gives a `local` block the
same mask as a `global` one.

```python
window = model.config.window_size
local = model.config.attention_layers.index("local")
prompt = " ".join(["token"] * (window + 4))

with model.trace(prompt):
    pattern = model.layers[local].self_attn.attention_probabilities.save()

n = pattern.shape[-1]
outside = torch.ones(n, n, dtype=torch.bool, device=pattern.device).tril(-window)
assert pattern[..., outside].abs().max() == 0
```

## The scores are not scaled

The module's `_attn(query, key, value, mask)` computes `q @ kᵀ` in float32 with no
`1 / sqrt(head_dim)`, applies the causal (and local) mask with `torch.where`, adds the padding mask,
takes the softmax, casts it back to the values' dtype and multiplies the values. Its arguments are
`attention_queries`, `attention_keys` and `attention_values` (`num_heads` each, no grouping), the
softmax's input is `attention_scores` and its first return is `attention_head_outputs`.

```python
with model.trace(prompt):
    q = model.layers[0].self_attn.attention_queries.save()
    k = model.layers[0].self_attn.attention_keys.save()
    scores = model.layers[0].self_attn.attention_scores.save()

n = scores.shape[-1]
causal = torch.ones(n, n, dtype=torch.bool, device=scores.device).tril()
again = q.float() @ k.float().transpose(-1, -2)          # no division by sqrt(head_dim)
assert torch.equal(again[..., causal], scores[..., causal])
```

The scores are large: at block 6 of `gpt-neo-125m` their standard deviation over the unmasked
entries is 10.8 and their largest entry 38 on a 47-token prompt. Above the diagonal the scores are
`-inf`, not the dtype's minimum, so `attention_scores * 0` is NaN there; add to the scores, or
keep the masked entries.

## Loading: eager is the default

transformers has `eager` and `flash_attention_2` for GPT-Neo and no `sdpa`, so a load with no
`attn_implementation` runs eager and serves all six interior values. They carry the eager check,
and a `flash_attention_2` load reports them unavailable. The released configs name no dtype, so a
load with no `dtype` is float32.

## Position embeddings are added after embed_tokens

`token_embeddings` is `wte`'s lookup only. The learned position embedding `transformer.wpe`, one
row per position up to `max_position_embeddings` (2048), is added to it before block 0, so an
edit to `token_embeddings` leaves the position term in place:

```python
with model.trace(prompt):
    tokens = model.token_embeddings.save()
    positions = model.transformer.wpe.output.save()
    x0 = model.layers[0].input.save()

assert torch.equal(tokens + positions, x0)
```

## The first position takes most of the global blocks' attention

On `gpt-neo-125m`, on a 47-token prompt, the `global` blocks 6, 8 and 10 put 63% to 76% of their
pattern (averaged over heads and the later queries) on key 0, and the `local` blocks between them
23% to 35%. The stream at position 0 has a norm of about 9,100 to 9,500 from block 5's output
to block 9's, against medians of about 1,500 at the other positions. The tokenizer is GPT-2's and
adds no BOS, so the prompt's first token takes that role; leave position 0 out of norms and
activation statistics.

## The readout: LayerNorm with a bias, tied weights

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` applied to the last
block's `layer_output` equals `logits` exactly (difference 0.0 on `gpt-neo-125m`). `lm_head` has no
bias and its weight is `embed_tokens.weight`, the same tensor. `ln_f`'s bias adds
`lm_head.weight @ norm.bias` to every position's logits whatever the input: on `gpt-neo-125m`
that vector has a standard deviation of 1.34 logits and its top tokens are ` the`, `,`, `\\n`,
` and`, ` "`.

```python
bias_logits = model.lm_head.weight @ model.norm.bias
```

## The family's checkpoints

`gpt-neo-125m` is 12 blocks × 768 (12 heads of 64), `gpt-neo-1.3B` 24 × 2048 and `gpt-neo-2.7B`
32 × 2560 (16 and 20 heads of 128). Every one has `gelu_new`, an MLP four times the width
(`intermediate_size` is `None`, which means `4 * hidden_size`) and GPT-2's 50257-token vocabulary.
The TinyStories models have 1 to 8 blocks of 64 to 1024 wide, always 16 heads, so `head_dim` runs
from 4 (`TinyStories-1M`) to 64; `TinyStories-1Layer-21M`'s one block is `global`.
"""
