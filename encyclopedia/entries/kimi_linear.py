"""Kimi-Linear: a hybrid whose blocks come in three shapes (a KDA mixer or latent attention, a dense MLP or a mixture)."""

MODEL_TYPE = "kimi_linear"
TITLE = "Kimi-Linear"
SUBTITLE = (
    "Llama's pre-norm block with two kinds of mixer, Kimi Delta Attention on three blocks in four and "
    "latent attention without rotary on the rest, and a mixture of experts after a dense first block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
#: The tiny checkpoint the test suite builds the page from (the suite types its last block in a copy of the config).
PINNED = "yujiepan/kimi-linear-tiny-random"
CHECKPOINTS = ["moonshotai/Kimi-Linear-48B-A3B-Base", "moonshotai/Kimi-Linear-48B-A3B-Instruct"]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 340}
VLLM = False
QUIRKS = ["hybrid", "latent-attention", "nope-blocks", "mixture-of-experts", "dense-first-blocks"]

#: Every number on this page comes from the reference's config or a run on the pinned tiny checkpoint;
#: none has been checked on real weights, which are not cached (48B parameters).


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device. The reference's tokenizer loads through its remote
    code and is not cached; the page reads no tokens, so the Kimi K2.5 test checkpoint's stands in."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    tokenizer = AutoTokenizer.from_pretrained("hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration")
    return StandardizedTransformer(checkpoint, tokenizer=tokenizer, **kwargs)


#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``)
#: and the MLP its ``mlp`` is (the dense one on block 0, the mixture after).
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Linear attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "decays", "betas",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "KDA, {linear_num_heads} heads × {linear_head_dim}",
        },
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
            "detail": "latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}, no rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "head": "lm_head has its own weight (tie_word_embeddings is false).",
}

NOTES = """
## The block, in order

```
h   = x + mixer(input_layernorm(x))          # linear_attn (KDA) or self_attn (latent attention)
out = h + mlp(post_attention_layernorm(h))   # a dense MLP on block 0, a mixture of experts after
```

Llama's pre-norm block, with the mixer chosen per block by `config.layer_types`: on the 48B
checkpoints blocks 3, 7, 11, 15, 19, 23 and 26 are `full_attention` and the other 20 are
`linear_attention`, so the last two attention blocks are three apart. The MLP is chosen by
`config.mlp_layer_types` (`first_k_dense_replace` is 1): block 0 is `dense`, every later block
`sparse`. transformers names both mixers `self_attn`; nnterp calls the KDA one `linear_attn` and
leaves `self_attn` as `None` on its blocks. Pick blocks outside the trace:

```python
kda = [i for i, t in enumerate(model.config.layer_types) if t == "linear_attention"]
mla = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
```

## The contributions are the sublayers' outputs

The block adds the mixer's output and the MLP's output to the stream and returns a tensor, so
the identity is the plain sum, with whichever mixer the block has:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].linear_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

On a `full_attention` block the middle term is `model.layers[i].self_attn.attention_output`.

## Loading

`attn_implementation="eager"` makes the latent attention's interior readable on the
`full_attention` blocks; the KDA values are read at the kernel call and need no eager load. The
per-token `state` and `states` need `nnterp.route_kernels(model.family, "torch")` before the
layer is first traced: a prompt runs `chunk_kimi_delta_attention`, which carries the state
between 64-token chunks, and the route swaps in the token-by-token kernel. `expert_outputs` is
read inside transformers' `grouped_mm` experts forward, the default.

## Kimi Delta Attention decays each key channel

KDA is a gated DeltaNet whose forget gate gives a log decay per key channel: `decays` is
`[batch, seq, heads, key_dim]`, float32 and non-positive, where the gated DeltaNet's is one per
head. `betas` is `[batch, seq, heads]`, a sigmoid. `attention_queries` and `attention_keys` are
served before the kernel's L2 norm and the queries before its `1/sqrt(key_dim)` scale, so a
recurrence written on them normalizes first (`docs/usage/delta-net.md` has it). The state is
`[batch, heads, key_dim, value_dim]`. A width-4 convolution (`linear_conv_kernel_dim`) runs
over the projected queries, keys and values before the rule, so the mixer also carries the last
tokens outside the state.

## The latent attention has no rotary

The `full_attention` blocks are DeepSeek-V3's multi-head latent attention without rotary
embeddings: queries and keys are `qk_head_dim` wide (`qk_nope_head_dim + qk_rope_head_dim`,
192), values `head_dim` (`v_head_dim`, 128). Keys and values are projected for every head, so
`attention_keys` has `num_heads` heads. The last `qk_rope_head_dim` dimensions of each key come
from one projection per token, shared by every head, and carry no position here: only the
causal mask orders the tokens, on these blocks as on the KDA ones. The config's own `head_dim`
(72) is `hidden_size / num_attention_heads`; `model.head_dim` is `v_head_dim`.

## The mixture of experts

256 routed experts, 8 per token, and one shared expert every token runs through.
`router_logits` are the router's logits before the sigmoid. The router picks the top 8 by the
sigmoid plus a selection bias (`e_score_correction_bias`), and `expert_weights` are the chosen
sigmoids without the bias, renormalized and multiplied by `routed_scaling_factor` (2.446), so
they sum to 2.446 at every token. `expert_outputs` summed over the slots is `routed_output`, and
the mixture's output is `routed_output + shared_expert_output`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    weights = moe.expert_weights.save()                     # [batch, seq, top_k]: 8 on 48B
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

torch.testing.assert_close(routed + shared, out)
```

Block 0's MLP is dense: `support()` reports every mixture value missing there.

## The readout and the embeddings

`logits` is `lm_head.output`: no softcap and no scale. `lm_head` has its own weight. The
embedding is not scaled, so `token_embeddings` equals `layers[0].input`.
"""
