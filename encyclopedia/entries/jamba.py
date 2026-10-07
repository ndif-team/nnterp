"""Jamba: AI21's hybrid (JambaForCausalLM), Mamba-1 blocks and attention blocks, a mixture of experts on every other block."""

MODEL_TYPE = "jamba"
TITLE = "Jamba / Jamba 1.5–1.7 / Jamba2"
SUBTITLE = (
    "Llama's pre-norm block whose mixer is a Mamba-1 selective scan with RMS norms on its step, B and C on most "
    "blocks and attention without rotary on one in eight or fourteen, and whose MLP is, on every other block "
    "of the large checkpoints, a mixture of experts."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ai21labs/Jamba-v0.1"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-JambaForCausalLM"
#: Jamba 1.5, 1.6 and 1.7 are gated on the Hub: the build lists them greyed unless the token has access.
CHECKPOINTS = [
    "ai21labs/Jamba-v0.1",
    "ai21labs/AI21-Jamba-Mini-1.5", "ai21labs/AI21-Jamba-Large-1.5",
    "ai21labs/AI21-Jamba-Mini-1.6", "ai21labs/AI21-Jamba-Large-1.6",
    "ai21labs/AI21-Jamba-Mini-1.7", "ai21labs/AI21-Jamba-Large-1.7",
    "ai21labs/AI21-Jamba-Reasoning-3B",
    "ai21labs/AI21-Jamba2-3B", "ai21labs/AI21-Jamba2-Mini",
]

#: Set by hues.py (lineage: Mamba).
PALETTE = {"hue": 8}
VLLM = False
QUIRKS = ["hybrid", "mamba1", "nope-blocks", "mixture-of-experts", "unnormalized-routing"]

#: No real weights were run: shapes and identities are from the pinned tiny checkpoint (routed, float32,
#: CPU), sizes and periods from the checkpoints' configs, routing from transformers' JambaSparseMoeBlock.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One norm, two readers: on a Mamba block linear_attn reads its output, on an attention "
                             "block self_attn.",
            "contribution": "attention_output",
            "interior": [
                "state_input", "attention_values", "betas", "decays", "attention_keys", "attention_queries",
                "states", "attention_head_outputs", "state_output",
            ],
            "detail": "selective scan, normed dt, B, C",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One norm, two readers: on a Mamba block linear_attn reads its output, on an attention "
                             "block self_attn.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, no rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm, the feed-forward's input norm, as Llama's post_attention_layernorm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm, the feed-forward's input norm, as Llama's post_attention_layernorm.",
            "contribution": "mlp_output",
            "interior": ["router_logits", "expert_weights", "expert_indices", "expert_outputs", "routed_output"],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, no scale and no position embedding: token_embeddings equals layers[0].input. The "
             "tokenizer prepends <|startoftext|>.",
    "norm": "Native final_layernorm, an RMSNorm whose gain is its weight.",
    "head": "Its own weight on Jamba-v0.1 and Jamba2-Mini; embed_tokens' weight on the 3B checkpoints "
            "(tie_word_embeddings). logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + mixer(input_layernorm(x))     # linear_attn (the Mamba mixer) or self_attn
out = h + ffn(pre_ff_layernorm(h))      # mlp: a SwiGLU MLP or a mixture of experts
```

Two block classes, `JambaMambaDecoderLayer` and `JambaAttentionDecoderLayer`, differ only in the
mixer, and both return a tensor. Block `i` is an attention block where `i % attn_layer_period ==
attn_layer_offset` and a Mamba block elsewhere; its feed-forward is a mixture where `i %
expert_layer_period == expert_layer_offset` and `num_experts` is above 1, and a dense MLP
elsewhere. From the configs:

- Jamba-v0.1 and Jamba2-Mini: 32 blocks, attention period 8 offset 4 (blocks 4, 12, 20, 28),
  experts period 2 offset 1 (every odd block, 16 experts, top 2). The attention blocks are even,
  so they always have the dense MLP: the shapes are Mamba + MLP, Mamba + MoE and
  Attention + MLP.
- Jamba-Reasoning-3B and Jamba2-3B: 28 blocks, attention period 14 offset 7 (blocks 7 and 21),
  and `num_experts` 1, so no block has a mixture.

`config.layer_types` names the mixer per block (`linear_attention`, `full_attention`);
`config.layers_num_experts` the experts per block. Pick blocks outside the trace:

```python
kinds = model.config.layer_types
mamba = [i for i, t in enumerate(kinds) if t == "linear_attention"]
attn = [i for i, t in enumerate(kinds) if t == "full_attention"]          # 4, 12, 20, 28 on v0.1
moe = [i for i, n in enumerate(model.config.layers_num_experts) if n > 1]
```

## The contributions are the sublayers' outputs

`mamba` is `linear_attn`, `feed_forward` is `mlp` (dense or mixture) and `pre_ff_layernorm` is
`post_attention_layernorm`. Nothing scales or norms a sublayer's output on its way to the
stream, so the identity is the plain sum on every shape, exact in float32:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()   # an attention block
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

On a Mamba block the middle term is `model.layers[i].linear_attn.attention_output`, the
`out_proj` output of the mixer, the same `[batch, seq, hidden]` layout.

## Loading: route the kernels

With `mamba_ssm` installed, transformers runs the selective scan in its CUDA kernels: every
`linear_attn` value but `attention_output` is unavailable, and on a CPU no trace runs at all
(`RuntimeError: Expected u.is_cuda() to be true`), even one that reads only `layer_output`.
`nnterp.route_kernels(model.family, "torch")` before the first trace binds transformers'
pure-torch scan, which is a token loop, so `state`, `states`, `state_after` and
`set_state_after` work without another call:

```python
import nnterp

nnterp.route_kernels(nnterp.families.jamba, "torch")       # before the first trace
model = StandardizedTransformer("ai21labs/Jamba-v0.1", attn_implementation="eager")
```

The attention interior on the attention blocks needs `attn_implementation="eager"`.

## The Mamba mixer norms its step, B and C

The mixer is Mamba-1's, with three RMS norms with learned weights between `x_proj` and the
scan: `dt_layernorm` on the low-rank step (`mamba_dt_rank`, 256 on v0.1 and Jamba2-Mini, 160 on
the 3B checkpoints), `b_layernorm` on `B` and `c_layernorm` on `C`. The values are read at the
scan's call, after the norms: `attention_queries` is `c_layernorm`'s output.

```python
mix = model.layers[0].linear_attn
with model.trace(prompt):
    c = mix.c_layernorm.output.save()          # [batch, seq, state_dim]
with model.trace(prompt):
    C = mix.attention_queries.save()           # [batch, seq, 1, state_dim]

assert torch.equal(c, C[:, :, 0])
```

The mixer is `mamba_expand × hidden_size` channels wide (8192 on v0.1, 5120 on the 3B
checkpoints) with a state of `mamba_d_state` 16 per channel: `attention_values`, `betas` and
`attention_head_outputs` are `[batch, seq, channels]`, `decays` `[batch, seq, channels, 16]`,
the state `[batch, channels, 16]` and `states` `[batch, seq, channels, 16]`. The root's
`intermediate_size` is the MLP's (14336 on v0.1), not the mixer's width.

## The mixture's weights are the softmax's top 2, not renormalized

`router` is a bare `nn.Linear`, and `router_logits` is its output. The mixture takes a
softmax over all experts in float32 and keeps the `num_experts_per_tok` largest; nothing
renormalizes them, so on v0.1 and Jamba2-Mini a token's two `expert_weights` sum to less than
one. There is no shared expert: `routed_output` is `mlp_output`.

```python
moe = model.layers[1].mlp                      # an odd block: a mixture
with model.trace(prompt):
    logits = moe.router_logits.save()          # [batch, seq, num_experts]
    w = moe.expert_weights.save()              # [batch, seq, top_k]
    idx = moe.expert_indices.save()

top = logits.float().softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values.to(w.dtype), w)
assert torch.equal(top.indices, idx)
```

## Attention carries no position

No block has a rotary or position embedding: `attention_queries` is `q_proj`'s output split into
heads, unchanged, and order reaches the attention only through the causal mask and the Mamba
blocks before it. On v0.1 and Jamba2-Mini 32 query heads share 8 key/value heads of 128, so an
edit to `attention_values[:, j]` reaches query heads `4j` to `4j + 3`; the 3B checkpoints have 20
query heads over one key/value head.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()
    q = attn.attention_queries.save()          # [batch, heads, seq, head_dim]

b, s, _ = q_raw.shape
assert torch.equal(q_raw.view(b, s, -1, model.head_dim).transpose(1, 2), q)
```

## The readout, the embeddings and the tokenizer

`logits` is `lm_head.output`, and `project_on_vocab` on the last block's `layer_output` equals
`logits` exactly. Nothing is added after `embed_tokens`, so `token_embeddings` equals
`layers[0].input`. The 3B checkpoints tie `lm_head` to `embed_tokens`; v0.1 and Jamba2-Mini do
not. The tokenizer has 65536 tokens and prepends `<|startoftext|>` (id 1), so position 0 is that
token; on v0.1 `<|endoftext|>` (id 2) ends a text, and the Jamba2 configs name id 519 as
`eos_token_id`. Jamba-v0.1 is a base model and ships no chat template.
"""
