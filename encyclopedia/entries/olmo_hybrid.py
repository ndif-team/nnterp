"""OLMo-Hybrid: AI2's gated DeltaNet hybrid (OlmoHybridForCausalLM), pre-norm DeltaNet blocks and OLMo-3's post-norm attention blocks."""

MODEL_TYPE = "olmo_hybrid"
TITLE = "OLMo-Hybrid"
SUBTITLE = (
    "Two block classes that norm in opposite places: three blocks in four are pre-norm, with a gated DeltaNet "
    "mixer; the fourth is OLMo-3's post-norm block, where attention without rotary and the MLP read the raw "
    "stream and their outputs are normed before it receives them."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/Olmo-Hybrid-7B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-OlmoHybridForCausalLM"
CHECKPOINTS = [
    "allenai/Olmo-Hybrid-7B",
    "allenai/Olmo-Hybrid-Instruct-SFT-7B", "allenai/Olmo-Hybrid-Instruct-DPO-7B",
    "allenai/Olmo-Hybrid-Think-SFT-7B",
]

#: OLMo lineage: olmo2 at 58 and olmoe at 70; the hybrid takes 64, between them.
PALETTE = {"hue": 64}
VLLM = False
QUIRKS = ["hybrid", "post-norms", "qk-norm", "nope-blocks"]

#: Every checkpoint is 7B, so no real weights were run: shapes, identities and read places are from the
#: pinned tiny checkpoint with the released config's rope_theta null and linear_allow_neg_eigval written in
#: (tests/families/test_olmo_hybrid.py's copy), routed, float32, CPU; sizes from the checkpoints' configs.

#: The two block classes put the same native norm in two roles and the one MLP in two places: on a DeltaNet
#: block post_attention_layernorm is the MLP's pre-norm, on an attention block the attention's post-norm,
#: and the MLP is pre-normed on one and post-normed on the other. Each sublayer's `block` key names the
#: native block class it is drawn on, so `mlp` is listed once per class with its own norms.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "block": "OlmoHybridLinearAttentionDecoderLayer",
            "kind": "mixer",
            "label": "Linear attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "decays", "betas",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "gated DeltaNet, {linear_num_value_heads} heads",
        },
        {
            "host": "self_attn",
            "block": "OlmoHybridAttentionDecoderLayer",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "post_norm_note": "On an attention block this norm follows the attention, and its output is "
                              "attention_output; on a DeltaNet block the same name is the MLP's pre-norm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "block": "OlmoHybridLinearAttentionDecoderLayer",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "On a DeltaNet block this is the MLP's input norm, as on Llama.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
        },
        {
            "host": "mlp",
            "block": "OlmoHybridAttentionDecoderLayer",
            "kind": "mlp",
            "label": "MLP",
            "post_norm": "post_feedforward_layernorm",
            "post_norm_note": "On an attention block the MLP reads the raw stream, and this norm's output is mlp_output.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, no scale and no position embedding: token_embeddings equals layers[0].input. "
             "The tokenizer prepends nothing.",
    "norm": "An RMSNorm whose gain is its weight; project_on_vocab applies it.",
    "head": "lm_head has its own weight: tie_word_embeddings is false. logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
# a DeltaNet block (OlmoHybridLinearAttentionDecoderLayer): pre-norm
h   = x + linear_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))

# an attention block (OlmoHybridAttentionDecoderLayer): post-norm, no input_layernorm
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

`config.layer_types` is `full_attention` on every fourth block (3, 7, ..., 31) and
`linear_attention` on the others: 24 DeltaNet and 8 attention blocks on every checkpoint. A block
has `linear_attn` or `self_attn`, never both. The name `post_attention_layernorm` is on both
classes in two roles: the MLP's input norm on a DeltaNet block, the attention's output norm on
an attention block. Pick blocks outside the trace:

```python
kinds = model.config.layer_types
delta = [i for i, t in enumerate(kinds) if t == "linear_attention"]
full = [i for i, t in enumerate(kinds) if t == "full_attention"]      # 3, 7, ..., 31
```

## The contributions follow the block's class

On a DeltaNet block `attention_output` and `mlp_output` are the modules' own outputs. On an
attention block they are the post-norms' outputs: `self_attn.attention_output` is
`post_attention_layernorm`'s output and `mlp.mlp_output` is `post_feedforward_layernorm`'s, so
`mlp.output` there is not what the stream receives. One MLP class serves both, and nnterp picks
`mlp_output`'s place by the block that holds it. The identity is the plain sum on both, exact in
float32:

```python
blk = model.layers[full[0]]
with model.trace(prompt):
    x = blk.input.save()
    attn = blk.self_attn.attention_output.save()      # post_attention_layernorm's output
    mlp = blk.mlp.mlp_output.save()                   # post_feedforward_layernorm's output
    out = blk.layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

The attention block has no input norm: `self_attn.input` is the block's input and `mlp.input`
is `x + attention_output`. Scaling `self_attn.output` there does not scale what the block adds,
since the norm follows; edit `attention_output`.

## Loading

The attention interior needs `attn_implementation="eager"`. The DeltaNet values are read at
transformers' pure-torch delta-rule kernels, which run when `flash-linear-attention` is not
installed (it is not here); the per-token `state` and `states` need
`nnterp.route_kernels(model.family, "torch")` before the first trace, which runs a prompt
through the token-by-token kernel.

```python
import nnterp

nnterp.route_kernels(nnterp.families.olmo_hybrid, "torch")      # for state / states
model = StandardizedTransformer("allenai/Olmo-Hybrid-7B", attn_implementation="eager")
```

## The DeltaNet mixer

Each mixer has 30 heads, keys 96 wide and values 192 wide, one value head per key head:
`attention_queries` and `attention_keys` are `[batch, seq, 30, 96]`, `attention_values` and
`attention_head_outputs` `[batch, seq, 30, 192]`, `decays` and `betas` `[batch, seq, 30]`, and
the state `[batch, 30, 96, 192]`, 2.2 MB per block per prompt in float32. The queries and keys
are read after the width-4 convolution and its SiLU, before the kernel's l2 norm. After the
kernel, `y` passes a gated RMSNorm (`o_norm`, gated by `g_proj`) and `o_proj`;
`attention_head_outputs` is `y` before both.

Every checkpoint sets `linear_allow_neg_eigval`: the mixer doubles the sigmoid of `b_proj`
before the kernel, so `betas` lie in `(0, 2)`; write `betas` in that range.

```python
mix = model.layers[delta[0]].linear_attn
with model.trace(prompt):
    betas = mix.betas.save()               # [batch, seq, heads], in (0, 2)
    states = mix.states.save()             # [batch, seq, heads, key_dim, value_dim]; needs route_kernels
    final = mix.state_output.save()

assert torch.equal(states[:, -1], final)
```

## Attention: normed across heads, no rotary

`q_norm` and `k_norm` are RMSNorms over the whole projection (30 heads × 128 = 3840 wide),
applied before the heads are split, so one RMS covers every head and an edit to one head's
slice of `q_proj.output` reaches the others; edit a single head at `attention_queries`. 30
query heads over 30 key/value heads: there is no grouping. The released configs set
`rope_theta` to null, and the model then builds no rotary embedding: `attention_queries` is
`q_norm`'s output split into heads, unchanged, and order reaches the attention only through the
causal mask and the DeltaNet blocks.

```python
attn = model.layers[full[0]].self_attn
with model.trace(prompt):
    qn = attn.q_norm.output.save()
with model.trace(prompt):
    q = attn.attention_queries.save()          # [batch, heads, seq, head_dim]

b, s, _ = qn.shape
assert torch.equal(qn.view(b, s, -1, model.head_dim).transpose(1, 2), q)
```

Olmo-Hybrid-Think-SFT-7B's config sets `rope_parameters` to null rather than a null
`rope_theta`; transformers fills that with a theta of 10000, so loaded through transformers that
checkpoint's attention blocks apply rotary embeddings (its `model.rotary_emb` is an
`OlmoHybridRotaryEmbedding`, where the other three checkpoints' is `None`).

## The readout, the embeddings and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits` exactly. `lm_head` and `embed_tokens` are separate weights, and
nothing is added after `embed_tokens`, so `token_embeddings` equals `layers[0].input`. The
tokenizer is a 100278-token BPE (the embedding has 100352 rows); it prepends nothing, so
position 0 is the text's first token, and `<|endoftext|>` (100257) ends a text.
"""
