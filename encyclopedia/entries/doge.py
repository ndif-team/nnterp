"""Doge: Llama's tree with a gated residual stream and a dynamic mask on the attention scores."""

MODEL_TYPE = "doge"
TITLE = "Doge"
SUBTITLE = (
    "Llama's pre-norm block whose residual stream is multiplied by a learned per-channel gate before each add, "
    "and whose attention adds a learned bias per key, computed from the values, to the scores."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Doge-260M is the largest dense checkpoint whose weights load in transformers 5.17.
REFERENCE = "SmallDoge/Doge-260M"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-DogeForCausalLM"
#: Dense checkpoints first, then the cross-domain mixtures (is_moe), then the ones whose weights do not load.
CHECKPOINTS = [
    "SmallDoge/Doge-20M", "SmallDoge/Doge-40M", "SmallDoge/Doge-260M",
    "SmallDoge/Doge-40M-MoE", "SmallDoge/Doge-180M-MoE",
    "SmallDoge/Doge-20M-Instruct", "SmallDoge/Doge-60M", "SmallDoge/Doge-60M-Instruct",
    "SmallDoge/Doge-160M", "SmallDoge/Doge-160M-Instruct", "SmallDoge/Doge-320M", "SmallDoge/Doge-320M-Instruct",
    "SmallDoge/Doge-20M-MoE", "SmallDoge/Doge-120M-MoE",
]

#: Checkpoints whose safetensors (read from their headers on 2026-10-07) do not fit transformers 5.17's
#: DogeForCausalLM: they hold the layout of the repositories' own remote code.
OWN_LAYOUT = {
    "SmallDoge/Doge-20M-Instruct", "SmallDoge/Doge-60M", "SmallDoge/Doge-60M-Instruct",
    "SmallDoge/Doge-160M", "SmallDoge/Doge-160M-Instruct", "SmallDoge/Doge-320M", "SmallDoge/Doge-320M-Instruct",
    "SmallDoge/Doge-20M-MoE", "SmallDoge/Doge-120M-MoE",
}


class WeightsDoNotLoad(Exception):
    """A checkpoint the page lists greyed out: its config builds, its weights do not load."""


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device; a checkpoint in ``OWN_LAYOUT`` is refused, so the page lists
    it greyed out with the reason. Its config builds a model, but its weights do not load into it: each block's
    ``A`` and ``dt_proj`` are twice ``num_key_value_heads`` wide where transformers builds ``num_key_value_heads``,
    and the attention has no ``q_norm`` or ``k_norm``, so ``from_pretrained`` raises on the size mismatch."""
    from nnterp import StandardizedTransformer

    if checkpoint in OWN_LAYOUT:
        raise WeightsDoNotLoad(
            "transformers 5.17 does not load its weights: A and dt_proj are twice num_key_value_heads wide, "
            "and the attention has no q_norm or k_norm"
        )
    return StandardizedTransformer(checkpoint, **kwargs)


#: No kin among the entries; the free gap between dbrx and mpt (350) and deepseek_v2 (357).
PALETTE = {"hue": 353}
VLLM = False
QUIRKS = ["scaled-residual-adds", "qk-norm"]

#: Every real value in the notes was measured on Doge-260M in float32 on a GPU; the snippets also ran on the
#: pinned tiny checkpoint (2 blocks, gates and A at their initial values: ones and zeros).

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads over {num_kv_heads} kv, dynamic mask",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
            "pre_norm_note": "The MLP's input norm, as on Llama. It reads the stream after input_residual has "
                             "scaled it and the attention has added to it.",
        },
    ],
    "identity": "post_attention_residual * (input_residual * layers[i].input + self_attn.attention_output) "
                "+ mlp.mlp_output == layer_output",
    "identity_note": "input_residual and post_attention_residual are layers[i]._module's own [hidden_size] parameters, "
                     "ones at initialisation. Exact in float32 on every Doge-260M block.",
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input.",
    "layers": "Each block multiplies the stream by its two gates before adding to it, so a block's terms reach the "
              "last stream multiplied, channel by channel, by every later gate.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings) on every listed checkpoint.",
}

NOTES = """
## The block, in order

```
h   = input_residual * x + self_attn(input_layernorm(x))
out = post_attention_residual * h + mlp(post_attention_layernorm(h))
```

`input_residual` and `post_attention_residual` are `[hidden_size]` parameters of the block. They
multiply the stream the block carries, not what the sublayers add: `attention_output` is `o_proj`'s
output and `mlp_output` the MLP's, unscaled.

## The stream is gated, so the identity carries the gates

The plain sum `layers[i].input + attention_output + mlp_output` does not give `layer_output`; the
identity multiplies the stream by each gate where the block does. It is exact in float32 on every
Doge-260M block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

block = model.layers[1]._module
h = block.input_residual * x + attn
torch.testing.assert_close(block.post_attention_residual * h + mlp, out)
```

On Doge-260M the gates are below one almost everywhere: `input_residual` averages 0.73 to 0.94 per
block (its few negative entries, 0.09% of them, are on block 0), and `post_attention_residual` 0.87 to
0.96. A term added at block `i` therefore reaches the last block's output multiplied by every later
gate, channel by channel, which a direct attribution to the logits has to include. Zeroing
`attention_output` does not leave the block's input in place: the stream still goes on as
`input_residual * x`.

## The last block erases a massive channel

On Doge-260M channel 524 of the stream is a massive activation: at the last block's input it holds
57,662 at the first position and 5,000 to 6,700 at the others, at least 85% of the stream's squared
norm at every position of a 30-token prompt. The last block's
`post_attention_residual` is 0.0008 on that channel and at least 0.72 on every other, so the block
removes it before the final norm. Steering or patching that adds to channel 524 is erased the same
way at the last block.

## The dynamic mask is a bias per key, computed from the values

Each attention projects its values through `dt_proj` to one number per key and key/value head, and
adds `exp(A * softplus(dt))` to every query's score for that key. `A` is a per-head parameter, so a
head with `A > 0` raises the keys `dt_proj` picks out and a head with `A < 0` lowers them.
`attention_scores` carry the bias, and it can be rebuilt from `attention_values`:

```python
import torch.nn.functional as F

attn = model.layers[1].self_attn
with model.trace(prompt):
    values = attn.attention_values.save()          # [batch, kv_heads, seq, head_dim]
    scores = attn.attention_scores.save()          # [batch, heads, query, key]

module = attn._module
v = values.transpose(1, 2).flatten(2)              # [batch, seq, kv_heads * head_dim]
dt = F.softplus(module.dt_proj(v))                 # [batch, seq, kv_heads]
bias = torch.exp(module.A * dt).transpose(1, 2)    # [batch, kv_heads, key]
```

On Doge-260M `A` ranges from −0.29 to 0.29 and the bias from 0 to 8.4. Removing it from the scores
moves 0.02 to 0.47 of each pattern row's mass, averaged per block, on a 30-token prompt; at block 10
the mass on the first token, averaged over heads and queries, is 0.65 with the bias and 0.34 without it. The pinned tiny checkpoint has
`A` at zero, where the bias is 1 on every key and the pattern is unchanged.

## Past `keep_window_size` keys, the bias drops keys

When a row has more than `keep_window_size` keys (1024 on Doge-260M, 2048 on Doge-20M, 4096 on
Doge-40M), the attention keeps the `keep_window_size` visible keys with the largest bias and masks
the rest. The choice reads the bias alone, not the query's score. A dropped key has the dtype's minimum in `attention_scores` and zero in
`attention_probabilities`. Below that length, every causal key is kept.

## Loading

The attention interior needs `attn_implementation="eager"`; the default `sdpa` load adds the same
bias. Doge-20M, Doge-40M and Doge-260M load in transformers 5.17. In the Instruct checkpoints and in
Doge-60M, -160M and -320M, `A` and `dt_proj` are twice `num_key_value_heads` wide and the attention has
no `q_norm` or `k_norm`, so transformers' `DogeForCausalLM` does not load them (they are greyed out in
the selector). Doge-260M's tokenizer prepends `<|begin_of_text|>` (id 128000).

## The cross-domain mixture does not run

On the `is_moe` checkpoints `mlp` is `DogeCDMoE`: a dense SwiGLU MLP of `intermediate_size` plus
`num_experts` one-neuron experts (rows of `down_embed` and `up_embed`) that `router_gate` retrieves
by product keys, `num_experts_per_tok` per token. It returns `(hidden_states, router_logits)`, and
transformers 5.17's block passes that tuple to dropout, so a forward of Doge-40M-MoE or Doge-180M-MoE
raises a `TypeError`. Their pages read the model's structure; every mixture value is unavailable.

## The readout

`logits` is `lm_head.output`, with no cap or scale, and `lm_head` shares its weight with
`embed_tokens` on every listed checkpoint.
"""
