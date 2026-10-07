"""Qwen3-Next: a gated DeltaNet hybrid with a mixture of experts on every block, text only."""

MODEL_TYPE = "qwen3_next"
TITLE = "Qwen3-Next / Qwen3-Coder-Next"
SUBTITLE = (
    "Llama's pre-norm block with a gated DeltaNet mixer on three blocks in four, output-gated attention on the "
    "fourth, and a mixture of 512 experts whose shared expert is scaled by a sigmoid gate per token; the mixer "
    "fuses its projections per key head."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-Next-80B-A3B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen3-next-moe-tiny-random"
#: The FP8 and GGUF copies are left out.
CHECKPOINTS = [
    "Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen3-Coder-Next", "Qwen/Qwen3-Coder-Next-Base",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 307}
VLLM = False
QUIRKS = ["hybrid", "gated-query", "mixture-of-experts", "qk-norm", "partial-rotary", "gain-norm"]

#: No checkpoint of this family is under 3B parameters, so no real-weight run backs the notes: shapes, identities and
#: read orders are from the pinned tiny checkpoint, sizes from the configs.

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``); every block of
#: every checkpoint, and of the pinned tiny, has the mixture.
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
            "detail": "gated DeltaNet, {linear_num_value_heads} heads × {linear_value_head_dim}",
            "pre_norm_note": "One norm, two readers: on a DeltaNet block linear_attn reads its output, on an attention block self_attn.",
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
            "detail": "{num_heads} heads, {num_kv_heads} kv, output-gated",
            "pre_norm_note": "One norm, two readers: on a DeltaNet block linear_attn reads its output, on an attention block self_attn.",
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
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}, gated shared",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. No BOS is prepended, so "
             "position 0 holds the text's first token.",
    "norm": "The gain is 1 + weight, as on every RMSNorm of the block but the DeltaNet's own output norm.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is "
            "lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + mixer(input_layernorm(x))          # linear_attn (gated DeltaNet) or self_attn
out = h + mlp(post_attention_layernorm(h))   # routed experts + gate * shared expert
```

`config.layer_types` is `full_attention` on every fourth block (`full_attention_interval` 4,
blocks 3, 7, ..., 47) and `linear_attention` on the others: 36 DeltaNet and 12 attention blocks
of 48 on every checkpoint. A block has `linear_attn` or `self_attn`, never both. Every block's
`mlp` is the mixture (`decoder_sparse_step` 1, `mlp_only_layers` empty); the config's
`intermediate_size` (5120) is the width of a dense MLP no block has. Pick blocks outside the trace:

```python
types = model.config.layer_types
delta = [i for i, t in enumerate(types) if t == "linear_attention"]
full = [i for i, t in enumerate(types) if t == "full_attention"]    # 3, 7, ..., 47
```

## The contributions are the sublayers' outputs

The block adds the mixer's output and the mixture's output to the stream and returns a tensor,
so the identity is the plain sum, exact in float32 on both kinds of block; the mixture's output
is its routed sum plus its gated shared expert:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    mix = model.layers[1].linear_attn.attention_output.save()
    routed = model.layers[1].mlp.routed_output.save()
    shared = model.layers[1].mlp.shared_expert_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.equal(routed + shared, mlp)               # True
torch.testing.assert_close(x + mix + mlp, out)
```

On an attention block the middle term is `model.layers[3].self_attn.attention_output`.

## The mixture: 10 of 512 experts, renormalized, and a gated shared expert

The router, `gate` aliased `router`, gives one logit per expert; the scoring is a softmax, the 10
largest probabilities are kept, and `norm_topk_prob` (true on every checkpoint) divides them by
their sum, so each token's `expert_weights` sum to one. The experts are SwiGLUs 512 wide
(`moe_intermediate_size`); `expert_outputs` is each slot's weighted output,
`[batch, seq, 10, hidden]`, and sums to `routed_output`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    weights = moe.expert_weights.save()

top = logits.softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(weights, top.values / top.values.sum(-1, keepdim=True))
```

Beside them every token runs through one shared expert, a SwiGLU 512 wide
(`shared_expert_intermediate_size`), whose output the mixture multiplies by
`sigmoid(shared_expert_gate(x))`, one scalar per token. `shared_expert_output` is that product,
what the mixture adds; `moe.shared_experts.output` is the expert's output before the gate:

```python
with model.trace(prompt):
    x = moe.input.save()
    ungated = moe.shared_experts.output.save()
with model.trace(prompt):
    shared = moe.shared_expert_output.save()

gate = torch.sigmoid(moe._module.shared_expert_gate(x))   # [batch, seq, 1]
torch.equal(gate * ungated, shared)                       # True
```

Zeroing one routed expert's weight leaves the other slots' weights as they are, so they sum to
less than one:

```python
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = model.logits.save()
```

## The shared expert runs before the router

The mixture computes the shared expert first, then the router and the experts, and multiplies
by the gate last. So in one trace `moe.shared_experts.output` comes before `router_logits`, while
`shared_expert_output` comes after `routed_output`; `router_logits` before
`shared_experts.output` raises `OutOfOrderError`.

## Loading

`attn_implementation="eager"` is needed only for the six attention interior values on the
attention blocks; there is no softcap or window. The DeltaNet values are read at the delta-rule
kernel call and need transformers' pure-torch kernels, which is what runs when
`flash-linear-attention` and `causal-conv1d` are not installed; with either installed, every
`linear_attn` value but `attention_output` is unavailable until
`nnterp.route_kernels(model.family, "torch")`. The same call is what makes `state` and `states`
exist, through the token-by-token kernel; `states` holds one float32 state per token,
32 × 128 × 128, 2 MiB per token per block. `expert_outputs` needs `experts_implementation`
`"grouped_mm"` (the default) or `"batched_mm"`.

## The attention block gates its heads' output

`q_proj` is `2 × num_heads × head_dim` wide (8192) and holds, for each head in turn, its 256
query dimensions followed by its 256 gate dimensions, so the first half of `q_proj.output` is
not the queries. The heads' output is multiplied by the sigmoid of that gate before `o_proj`;
`attention_head_outputs` is read before the gate and `o_proj.input` is the gated tensor:

```python
attn = model.layers[3].self_attn
d = model.head_dim
with model.trace(prompt):
    q_and_gate = attn.q_proj.output.save()            # heads * 2 * head_dim wide
query, gate = q_and_gate.unflatten(-1, (-1, 2 * d)).chunk(2, dim=-1)

with model.trace(prompt):
    heads = attn.attention_head_outputs.save()        # before the gate
    gated = attn.o_proj.input.save()                  # after it

torch.testing.assert_close(gated, (heads * gate.sigmoid()).flatten(2))
```

`q_norm` and `k_norm` are RMSNorms over one head's 256 dimensions with gain `1 + weight`,
applied before the rotary. The rotary (`rotate_half`) turns only the first 64 dimensions of each
query and key head (`partial_rotary_factor` 0.25; `rope_theta` 10⁷ on Qwen3-Next, 5·10⁶ on
Qwen3-Coder-Next); `attention_queries[..., 64:]` equals `q_norm.output[..., 64:]` exactly. There
are 16 query heads over 2 key/value heads, so an edit to key/value head `j` reaches query heads
`8j` to `8j + 7`. The score scale is `256 ** -0.5`.

## The DeltaNet mixer fuses its projections per key head

```
q, k, v, z, b, a = regroup(in_proj_qkvz(x), in_proj_ba(x))   # per key head
q, k, v = split(silu(conv1d(cat(q, k, v))))                  # causal, width 4
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
y, state = delta_rule(l2norm(q) / sqrt(128), l2norm(k), v, g, beta)
out = out_proj(norm(y, gate=silu(z)))                        # gain: weight
```

`in_proj_qkvz` is laid out by key head: for each of the 16 key heads, its query (128), its key
(128), then the values and output gates of the two value heads it serves (2 × 128 each), 768
columns per key head. `in_proj_ba` is laid out the same way, the two value heads' `b` then their
`a` per key head. So no slice of the raw outputs is the queries or the betas; the module's own
`fix_query_key_value_ordering` regroups them:

```python
mix = model.layers[0].linear_attn
with model.trace(prompt):
    qkvz = mix.in_proj_qkvz.output.save()
    ba = mix.in_proj_ba.output.save()
with model.trace(prompt):
    betas = mix.betas.save()

q, k, v, z, b, a = mix._module.fix_query_key_value_ordering(qkvz, ba)
torch.testing.assert_close(b.sigmoid(), betas)
```

`attention_queries`, `attention_keys` and `attention_values` are the kernel's arguments: after
the convolution and the SiLU, before the kernel's l2 norm and the queries' `1/sqrt(128)` scale.
The 16 key heads' queries and keys are repeated up to the 32 value heads (`repeat_interleave`)
before the kernel, so they are served 32 wide, heads `2j` and `2j + 1` being copies of key head
`j`. `decays` is one log decay per head and token (float32, at most 0), `betas` the sigmoid write
strength, and `attention_head_outputs` is `y`, before the gated norm and `out_proj`. The state
after a prompt, `state_output`, is `[batch, 32, 128, 128]`.

## The prompt and the chat template

The tokenizer sets `bos_token` to `None` and prepends nothing; `<|endoftext|>` is 151643,
`<|im_start|>` 151644, `<|im_end|>` 151645. With `add_generation_prompt=True` the Instruct and
Coder templates end the prompt at `<|im_start|>assistant\\n`; the Thinking template adds
`<think>\\n`, so the next token starts a reasoning trace, not the answer. `generate` stops on
`<|im_end|>` or `<|endoftext|>` (`generation_config.json`).

## What loads as this family

Qwen3-Next-80B-A3B (Instruct and Thinking) and Qwen3-Coder-Next (and its Base), each
`Qwen3NextForCausalLM`, text only, with the same sizes. The multi-token-prediction block (`mtp`)
the checkpoints ship is not loaded. Qwen3.5's mixture-of-experts checkpoints, the same hybrid
under a vision-language wrapper with separate DeltaNet projections, are `qwen3_5_moe_text`.
"""
