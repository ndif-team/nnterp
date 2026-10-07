"""Bamba: IBM's Mamba-2 hybrid (BambaForCausalLM), Llama's pre-norm block with a Mamba-2 mixer on most blocks."""

MODEL_TYPE = "bamba"
TITLE = "Bamba"
SUBTITLE = (
    "Llama's pre-norm block whose mixer is a Mamba-2 state-space scan on all but three blocks and, on those "
    "three, attention that rotates half of each head; the block returns a tuple."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-ai-platform/Bamba-9B-v2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-BambaForCausalLM"
#: Bamba-9B on the Hub redirects to Bamba-9B-v1. The -fp8 copies are left out: their compressed-tensors
#: config does not build without that package.
CHECKPOINTS = [
    "ibm-ai-platform/Bamba-9B-v2", "ibm-ai-platform/Bamba-9B-v1",
    "ibm-ai-platform/Bamba-9B-1.8T", "ibm-ai-platform/Bamba-9B-2T",
]

#: Set by hues.py (lineage: Granite).
PALETTE = {"hue": 198}
VLLM = False
QUIRKS = ["hybrid", "mamba2", "tuple-blocks", "partial-rotary"]

#: Every checkpoint is 9B, so no real weights were run: the shapes and identities below are from the pinned
#: tiny checkpoint (routed, float32, CPU), the sizes from the checkpoints' configs.

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``);
#: every block has the same MLP.
MIXER_NORM = "One norm, two readers: on a Mamba-2 block linear_attn reads its output, on an attention block self_attn."

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "input_layernorm",
            "pre_norm_note": MIXER_NORM,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD scan, a state per head",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": MIXER_NORM,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, half rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm, the MLP's input norm, as Llama's post_attention_layernorm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, no scale and no position embedding: token_embeddings equals layers[0].input. The "
             "tokenizer prepends <|begin_of_text|>.",
    "layers": "Each block returns (hidden_states, attention_weights); layer_output is the first element.",
    "norm": "Native final_layernorm, an RMSNorm whose gain is its weight.",
    "head": "lm_head has its own weight: tie_word_embeddings is false. logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + mixer(input_layernorm(x))     # linear_attn (the Mamba-2 mixer) or self_attn
out = h + mlp(pre_ff_layernorm(h))      # SwiGLU, the same class on every block
return out, attention_weights
```

One block class holds either mixer: `config.attn_layer_indices` names the attention blocks, 9, 18
and 27 of 32 on every checkpoint, and the other 29 hold a Mamba-2 mixer. `config.layer_types`
says the same per block (`linear_attention`, `full_attention`). A block has `linear_attn` or
`self_attn`, never both; `mamba` is `linear_attn`, `feed_forward` is `mlp` and `pre_ff_layernorm`
is `post_attention_layernorm`. Pick blocks outside the trace:

```python
kinds = model.config.layer_types
ssm = [i for i, t in enumerate(kinds) if t == "linear_attention"]
attn = [i for i, t in enumerate(kinds) if t == "full_attention"]     # 9, 18, 27
```

## The contributions are the sublayers' outputs

The block returns a tuple, `(hidden_states, attention_weights)`, and `layer_output` is its first
element. Nothing scales or norms a sublayer's output on its way to the stream, so the identity is
the plain sum, exact in float32 on both kinds of block:

```python
with model.trace(prompt):
    x = model.layers[0].input.save()
    mix = model.layers[0].linear_attn.attention_output.save()
    mlp = model.layers[0].mlp.mlp_output.save()
    out = model.layers[0].layer_output.save()

torch.testing.assert_close(x + mix + mlp, out)
```

On an attention block the middle term is `model.layers[9].self_attn.attention_output`.

## Loading: route the kernels

With `mamba_ssm` installed, transformers runs the Mamba-2 scans in its CUDA kernels: every
`linear_attn` value but `attention_output` is unavailable, and on a CPU no trace runs at all
(`RuntimeError: invalid argument to exchangeDevice`, even for a read of `layer_output`).
`nnterp.route_kernels(model.family, "torch")` before the first trace binds transformers'
pure-torch scans. `states` and `state_after` also need `nnterp.chunk_per_token(model)`: the
scan keeps the state every `mamba_chunk_size` tokens, 256 on every checkpoint.

```python
import nnterp

nnterp.route_kernels(nnterp.families.bamba, "torch")       # before the first trace
model = StandardizedTransformer("ibm-ai-platform/Bamba-9B-v2", attn_implementation="eager")
```

The attention interior on the three attention blocks needs `attn_implementation="eager"`; the
Mamba-2 values are read at the scan's call and need no eager load.

## The Mamba-2 mixer

Every checkpoint has 128 heads of 64 channels in one group, and a state of 128 per head:
`attention_queries` (`C`) and `attention_keys` (`B`) are `[batch, seq, 1, 128]`, one for all 128
heads, `attention_values` (`x`) and `attention_head_outputs` (`y`) are `[batch, seq, 128, 64]`,
`betas` (`dt`) and `decays` (`A * dt`) are `[batch, seq, 128]`, and the state is
`[batch, 128, 128, 64]`, key side first. A width-4 convolution runs over `x`, `B` and `C` before
the scan. After it, `y` is multiplied by `silu(z)` and RMS-normed over all 8192 channels at once
(`BambaRMSNormGated`), then `out_proj` maps it to the stream; `attention_head_outputs` is read
before both.

```python
mix = model.layers[0].linear_attn
with model.trace(prompt):
    C = mix.attention_queries.save()          # [batch, seq, groups, state_dim]
    x = mix.attention_values.save()           # [batch, seq, heads, head_dim]
    dt = mix.betas.save()                     # [batch, seq, heads]
    state = mix.state_output.save()           # [batch, heads, state_dim, head_dim]
```

## A zero step skips a token exactly

`time_step_limit` is `(0, inf)` on every checkpoint, so the scan's clamp on `dt` changes
nothing, and a written `betas` of zero runs as zero. `betas[:, t] = 0` is then the exact
"skip this token": no write and no decay, so the state after token `t` equals the state after
`t - 1` bit for bit. Assign `betas`; an in-place edit does not reach the scan.

```python
nnterp.chunk_per_token(model)
with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0
    mix.betas = betas
    states = mix.states.save()                # [batch, seq, heads, state_dim, head_dim]

assert torch.equal(states[:, 3], states[:, 2])
```

## Attention rotates half of each head

The three attention blocks apply rotary embeddings to the first 64 of each head's 128
dimensions (`partial_rotary_factor` 0.5, which the config class sets for every checkpoint;
`attn_rotary_emb` is 64), with `rope_theta` 10000. The other 64 carry no position:
`attention_queries[..., 64:]` is `q_proj`'s output split into heads, unchanged. There are 32
query heads over 8 key/value heads, so an edit to `attention_values[:, j]` reaches query heads
`4j` to `4j + 3`; the score scale is `128 ** -0.5`.

```python
attn = model.layers[9].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()
    q = attn.attention_queries.save()              # [batch, heads, seq, head_dim]

b, s, _ = q_raw.shape
q_raw = q_raw.view(b, s, -1, model.head_dim).transpose(1, 2)
half = model.head_dim // 2
assert torch.equal(q_raw[..., half:], q[..., half:])
```

## The readout, the embeddings and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits` exactly. `lm_head` and `embed_tokens` are separate weights, and
nothing is added after `embed_tokens`, so `token_embeddings` equals `layers[0].input`. The
tokenizer is Llama 3's (128256 tokens): it prepends `<|begin_of_text|>` (id 128000), so
position 0 is that token, and `<|end_of_text|>` (128001) ends a text. Bamba-9B-v2's config names
ids 1 and 2 as `bos_token_id` and `eos_token_id`; the tokenizer's are 128000 and 128001. The
checkpoints are base models and ship no chat template.

## The checkpoints

Bamba-9B-2T is the end of the first pretraining stage (2T tokens) and Bamba-9B-1.8T a checkpoint
inside it; Bamba-9B-v1 (`Bamba-9B` on the Hub is the same repository) is the end of the second
stage, and Bamba-9B-v2 is trained on 1T tokens more, as their model cards say. All four share one
shape: 32 blocks of width 4096, an MLP of 14336, and the attention blocks at 9, 18 and 27.
Bamba-9B-v2's `max_position_embeddings` is 262144, v1's 4096.
GraniteMoE-Hybrid (Granite 4.0-H) is Bamba's block with Granite's multipliers, and is the
`granitemoehybrid` family.
"""
