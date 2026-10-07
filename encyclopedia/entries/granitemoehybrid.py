"""GraniteMoE-Hybrid: Granite 4.0, Mamba-2 and attention blocks in Granite's scaled block, with a shared MLP and routed experts."""

MODEL_TYPE = "granitemoehybrid"
TITLE = "Granite 4.0"
SUBTITLE = (
    "Granite's scaled block with a Mamba-2 mixer on most blocks and attention without rotary on a few (attention on "
    "every block of the non-H models), and a feed-forward that is a shared MLP plus, on the mixture checkpoints, "
    "routed experts, added as one scaled sum."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-4.0-h-tiny-base"
#: The tiny checkpoint the test suite builds the page from (the suite routes it to the pure-torch kernels).
PINNED = "hf-tiny-v2/tiny-random-GraniteMoeHybridForCausalLM"
CHECKPOINTS = [
    "ibm-granite/granite-4.0-h-350m-base", "ibm-granite/granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-1b-base", "ibm-granite/granite-4.0-h-1b",
    "ibm-granite/granite-4.0-h-micro-base", "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-h-tiny-base", "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-small-base", "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-tiny-base-preview", "ibm-granite/granite-4.0-tiny-preview",
    "ibm-research/granite-4.0-h-3b-ar",
    # attention on every block: their blocks hold no linear_attn, and the page draws the attention block alone
    "ibm-granite/granite-4.0-350m-base", "ibm-granite/granite-4.0-350m",
    "ibm-granite/granite-4.0-1b-base", "ibm-granite/granite-4.0-1b",
    "ibm-granite/granite-4.0-micro-base", "ibm-granite/granite-4.0-micro",
]

#: Granite lineage (205, set by granite); below granite_swa (199), between Cohere 2 (188) and Bamba (196).
PALETTE = {"hue": 192}
VLLM = False
QUIRKS = ["scaled-residual-adds", "hybrid", "mamba2", "nope-blocks", "mixture-of-experts", "embedding-multiplier",
          "scaled-logits"]

#: Numbers in the notes come from granite-4.0-h-1b-base's weights (dense, the H layout) in bfloat16 on an RTX A6000
#: unless they say float32, routed to the pure-torch kernels; the mixture's values and every snippet were checked on
#: the pinned tiny checkpoint, on a copy with Granite's multipliers away from 1.0, and on a copy without routed experts.
NORM_NOTE = ("The block's first norm, before whichever mixer it holds: model.layers[i].linear_attn and "
             "model.layers[i].self_attn both read its output.")

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "input_layernorm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD scan, gated RMSNorm after",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
        },
        {
            # The dense layout (H-350M, H-1B, H-Micro and the non-H models): the shared MLP is the whole feed-forward.
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the mixer's add: it is the feed-forward's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}",
        },
        {
            # The mixture layout (H-Tiny, H-Small, the previews): mlp is the shared MLP and hosts the block's mixture.
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the mixer's add: the router, every routed expert "
                             "and the shared MLP read its output.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, + shared",
        },
    ],
    "identity_note": "The terms are already scaled: each mixer's attention_output is its output times "
                     "residual_multiplier, and mlp_output the feed-forward's sum times it, so the plain sum is exact.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier (12) before block 0, "
             "so layers[0].input is token_embeddings · 12. No position embedding.",
    "layers": "Each block adds its mixer's output and its feed-forward's, each times residual_multiplier (0.22; 0.246 "
              "on H-350M, 0.263 on 350M).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "logits = lm_head.output / logits_scaling (6 on H-Tiny and H-1B, 16 on H-Small). project_on_vocab "
              "divides too.",
}

NOTES = """
## The block, in order

```
x      = embed_tokens(ids) * embedding_multiplier
h      = x + mixer(input_layernorm(x)) * residual_multiplier       # mixer: Mamba-2 or attention
n      = post_attention_layernorm(h)
out    = h + (block_sparse_moe(n) + shared_mlp(n)) * residual_multiplier
logits = lm_head(norm(out)) / logits_scaling
```

Every block has the same class and holds one mixer, `mamba` (`linear_attn`) or `self_attn`, by
`config.layer_types`. The H models (H-350M, H-1B, H-Micro, H-Tiny, H-Small) have 40 blocks with
attention on blocks 5, 15, 25 and 35 (H-350M: 32 blocks, attention on 10, 13, 17 and 27) and a
Mamba-2 mixer on the rest; the non-H 350M, 1B and Micro have attention on every block. The
feed-forward is `shared_mlp`, which is `mlp`, plus `block_sparse_moe`, the routed experts, where
`num_local_experts` is above 0: H-Tiny (64 experts, top 6), H-Small (72, top 10) and the Tiny
previews (62, top 6). On the dense checkpoints `shared_mlp(n)` alone is the feed-forward. Pick
blocks outside the trace:

```python
kinds = model.config.layer_types
ssm = [i for i, t in enumerate(kinds) if t == "linear_attention"]
attn_blocks = [i for i, t in enumerate(kinds) if t == "full_attention"]
```

## The contributions are scaled; the mixture's values are not

Both mixers' `attention_output` are the module's output times `residual_multiplier`, and
`mlp_output` is the block's binding of the feed-forward's sum times it, so the plain identity
holds, bit for bit in bfloat16 on H-1B on both block kinds. `mlp` is the shared MLP and hosts the
mixture's values, read before the multiplier: `routed_output` is `layers[i].block_sparse_moe.output`
and `shared_expert_output` is `mlp.output`. The routed experts run first.

```python
moe = model.layers[1].mlp                    # a mixture checkpoint
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    mlp = moe.mlp_output.save()
    out = model.layers[1].layer_output.save()

m = model.config.residual_multiplier         # 0.22
torch.testing.assert_close((routed + shared) * m, mlp)
torch.testing.assert_close(x + attn + mlp, out)
```

`mlp.output` is the shared MLP's output alone, also on a dense checkpoint, where
`mlp_output == mlp.output * m`. A write to `mlp_output` lands on the sum: `mlp_output[:] = 0`
removes the routed and the shared experts together. `intermediate_size` and
`mlp.intermediate_size` are the shared MLP's width, `shared_intermediate_size`; the routed experts
are `config.intermediate_size` wide (512 on H-Tiny). The router takes the top
`num_experts_per_tok` of its logits and softmaxes those alone, GraniteMoE's routing.

## Route the kernels before the first trace

With `mamba_ssm` installed, transformers runs the Mamba-2 scan in its CUDA kernels, and every
`linear_attn` value but `attention_output` is unavailable until
`nnterp.route_kernels(model.family, "torch")` is called before the first trace. On a CPU the
unrouted model does not run at all. `states` and `state_after` also need
`nnterp.chunk_per_token(model)`: the scan keeps the state every `mamba_chunk_size` tokens, 256 on
every released checkpoint.

```python
import nnterp

nnterp.route_kernels(nnterp.families.granitemoehybrid, "torch")     # before the first trace
model = StandardizedTransformer(
    "ibm-granite/granite-4.0-h-1b-base", attn_implementation="eager", dtype=torch.float32
)
```

The kernels disagree in bfloat16. On H-1B, on a seven-token prompt, the routed model's bfloat16
logits differ from its float32 logits by up to 5.7, the default kernels' by up to 2.4 (with a
different top token at one position), and the two bfloat16 runs differ from each other by up to
5.5. In float32 the routed and default kernels agree to 0.07 with the same top tokens, and eager
and `sdpa` attention to the bit. Compare runs, and measure effects, in float32.

## The Mamba-2 mixer

On H-1B each mixer has 48 heads of 64 channels in one group, with a state of 128:
`attention_queries` (`C`) and `attention_keys` (`B`) are `[batch, seq, 1, 128]`,
`attention_values` (`x`) and `attention_head_outputs` are `[batch, seq, 48, 64]`, and the state is
`[batch, 48, 128, 64]`. A width-4 convolution runs over `x`, `B` and `C` before the scan, and a
gated RMSNorm after it; `attention_head_outputs` is read before the norm. `dt` (`betas`) is
clamped to `time_step_limit`, `(0, inf)`, so a written zero stays zero: `betas[:, t] = 0` skips
token `t`'s write exactly, and the state after `t` equals the state before it.

```python
nnterp.chunk_per_token(model)
mix = model.layers[ssm[0]].linear_attn
with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0
    mix.betas = betas
    states = mix.states.save()                # [batch, seq, heads, 128, 64]

assert torch.equal(states[:, 3], states[:, 2])
```

## Attention without rotary on the H models

The H models set `position_embedding_type` to `nope`: the model builds no rotary embedding, and
`attention_queries` is `q_proj`'s output split into heads, unchanged. Order reaches the attention
only through the causal mask and the Mamba-2 blocks before it. The non-H 350M, 1B and Micro apply
rotary embeddings on every block.

```python
attn = model.layers[attn_blocks[0]].self_attn         # block 5 on H-1B
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()
    q = attn.attention_queries.save()

b, s, _ = q_raw.shape
assert torch.equal(q_raw.view(b, s, -1, model.head_dim).transpose(1, 2), q)
```

The query scale is `config.attention_multiplier`, `1 / head_dim` (1/128 on H-1B and H-Tiny).
H-1B has 12 query heads over 4 key/value heads of 128, so an edit to key/value head `j` reaches
query heads `3j` to `3j + 2`. The six interior values need `attn_implementation="eager"`.

## A write rounds the other positions with 0.22

A write to `attention_output` or `mlp_output` reaches the model as the whole edited copy divided
by `residual_multiplier`, which the block multiplies again. With 0.22, for 502 of the 65280 finite
bfloat16 products `x * 0.22` that round trip does not give the product back, and the Mamba-2
recurrence carries the difference on. On H-1B in bfloat16, adding a unit vector to `mlp_output` at
the last position moved the earlier positions' logits by up to 0.5 from block 1 (a KL of 0.009)
and by up to 4.0 from block 5 (a KL of 0.23), which the edit cannot reach causally. The same edit
on the module's output leaves them bit-identical:

```python
m = model.config.residual_multiplier
with model.trace(prompt):
    model.layers[5].mlp.mlp_output[:, -1] += v    # every position rounds

with model.trace(prompt):
    model.layers[5].mlp.output[:, -1] += v / m    # position -1 only (dense checkpoint)
```

On a mixture checkpoint `mlp.output` is the shared MLP's: edit `block_sparse_moe.output` for the
routed experts' term. With the multipliers of H-350M (0.246) and 350M (0.263) the round trip is
exact for every bfloat16 product.

## Position 0 carries a large norm

The tokenizer prepends nothing, so position 0 is the prompt's own first token. On H-1B the stream
there has a norm of about 270 after block 20 against 21 to 28 elsewhere, whether the prompt starts
with `The`, `def` or `In`, and the four attention blocks put 0.47 to 0.90 of their attention on
it, averaged over heads and queries. Slice `[:, 1:]` before averaging activations for steering
vectors, mean ablation or probes.

```python
with model.trace(prompt):
    resid = model.layers[20].layer_output.save()

resid[0].float().norm(dim=-1)        # H-1B: about 270 at position 0, 21 to 28 elsewhere
```

## Logits and embeddings are Granite's

`model.logits` is `model.lm_head.output / logits_scaling` and `project_on_vocab` divides too, so a
lens at the last block equals `logits`. A softmax over `lm_head.output` is at temperature
1/`logits_scaling`: after "The capital of the United Kingdom is", ` London` has probability 0.56
from `logits` and the top token 1.00 from `lm_head.output` on H-1B. `token_embeddings` is the plain
lookup and `layers[0].input` is it times 12. `lm_head` and `embed_tokens` are one parameter.

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()
    first = model.layers[0].input.save()
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

torch.testing.assert_close(first, emb * model.config.embedding_multiplier)
torch.testing.assert_close(logits, raw / model.config.logits_scaling)
torch.testing.assert_close(model.project_on_vocab(resid), logits)
```

## What this family module covers

Every checkpoint whose config says `granitemoehybrid`: all of Granite 4.0, the H models, the
attention-only 350M, 1B and Micro, the Tiny previews, Granite Guardian 4.0 3B and
granite-4.0-h-3b-ar. Granite 4.1 and 4.2 are `granite`, the Swash models `granite_swa` and
`granitemoe_swa`. `granitemoeshared` has the same feed-forward on an attention-only block.
"""
