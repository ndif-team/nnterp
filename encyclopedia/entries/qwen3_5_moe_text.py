"""Qwen3.5 and Qwen3.6, mixture of experts: the text model of the multimodal MoE checkpoints, a gated DeltaNet hybrid."""

MODEL_TYPE = "qwen3_5_moe_text"
TITLE = "Qwen3.5-MoE / Qwen3.6-MoE"
SUBTITLE = (
    "Qwen3.5's hybrid block, a gated DeltaNet mixer on three blocks in four and output-gated attention on the "
    "fourth, with a mixture of experts on every block whose shared expert is scaled by a sigmoid gate per token."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3.5-35B-A3B-Base"
#: The tiny checkpoint the test suite builds the page from: the wrapper itself, as every checkpoint is.
PINNED = "yujiepan/qwen3.5-moe-tiny-random"
#: Every checkpoint is the vision-language wrapper (model_type qwen3_5_moe); the FP8 and GPTQ copies are left out.
CHECKPOINTS = [
    "Qwen/Qwen3.5-35B-A3B-Base", "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.6-35B-A3B",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
#: Shapes and identities were checked on the pinned tiny wrapper; the sizes are the checkpoints' configs.
WRAPPERS = {
    "qwen3_5_moe": {
        "title": "Qwen3.5-MoE",
        "pinned": "yujiepan/qwen3.5-moe-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm on each patch, then an MLP (linear_fc1, GELU, "
                     "linear_fc2) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
        "notes": """
## The merger folds four patches into one image token

`model.projector` is `model.visual.merger`, the vision encoder's last module: `norm` (a LayerNorm over
`vision_hidden`) on each patch, then each 2 × 2 block of consecutive patches concatenated into one
`4 * vision_hidden` vector, `linear_fc1`, GELU and `linear_fc2` to the text model's width. So an image
of `t * h * w` patches (its `image_grid_thw` row) is `t * h * w / 4` image tokens, and
`vision.image_features` is `model.projector.output` as it is, `[image_tokens, hidden]`. The merger maps
4 × 1152 = 4608 to 2048 on 35B-A3B, to 3072 on 122B-A10B and to 4096 on 397B-A17B; a patch is 16
pixels, so an image token covers 32 × 32 pixels.

```python
with model.trace(prompt, images=[red, wide]):
    mask = model.vision.image_token_mask.save()
    fed = model.projector.input.save()          # [patches, vision_hidden]
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

fed.shape[0] // 4 == mask.sum()                 # True: one image token per 2 x 2 block
torch.equal(first[mask], features)              # True
```

## No DeepStack

`vision_config.deepstack_visual_indexes` is empty on every checkpoint: the image reaches the text
model only through `vision.image_features`; the text blocks have no `deepstack_output`. The vision
encoder is the same on every checkpoint: 27 blocks of width 1152, 16 heads, with Qwen3-VL's learned
`pos_embed`. The image token is `<|image_pad|>` (248056), and `vision.image_token_mask` marks it.

## Read the vision values in the forward's order

The merger runs inside the vision encoder, before it returns. So in one trace read the last vision
block's `layer_output`, then `model.projector.input` and `.output`, then `vision.tower_output`, then
`vision.image_features`, then the text blocks. A read out of this order raises `OutOfOrderError`.
""",
    },
}

#: The Qwen lineage sits at 285 (qwen2), 297 (qwen3) and 273 (qwen3_5_text); this family takes 255.
PALETTE = {"hue": 255}
VLLM = False
QUIRKS = ["hybrid", "gated-query", "mixture-of-experts", "qk-norm", "partial-rotary", "gain-norm"]

#: No checkpoint of this family is under 3B parameters, so no real-weight run backs the notes: shapes, identities and
#: read orders are from the pinned tiny checkpoint (as the text model and as the wrapper), sizes from the configs.

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``);
#: every block has the mixture.
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
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input on a text prompt. No BOS is "
             "prepended, so position 0 holds the text's first token.",
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
blocks 3, 7, 11, ...) and `linear_attention` on the others: 30 DeltaNet and 10 attention blocks
on 35B-A3B, 36 and 12 on 122B-A10B, 45 and 15 on 397B-A17B. A block has `linear_attn` or
`self_attn`, never both; the last block is an attention block. Every block's `mlp` is the
mixture (`mlp_only_layers` is empty). Pick blocks outside the trace:

```python
types = model.config.get_text_config().layer_types
delta = [i for i, t in enumerate(types) if t == "linear_attention"]
full = [i for i, t in enumerate(types) if t == "full_attention"]    # 3, 7, ..., 39 on 35B-A3B
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

## The mixture: a softmax router, renormalized, and a gated shared expert

The router, `gate` aliased `router`, gives one logit per expert; the scoring is a softmax, the
top `num_experts_per_tok` probabilities are kept and always divided by their sum (the config has
no `norm_topk_prob` switch), so each token's `expert_weights` sum to one. 35B-A3B and 122B-A10B
route each token to 8 of 256 experts, 397B-A17B to 10 of 512. The experts are one fused module,
`experts`, each a SwiGLU `moe_intermediate_size` wide (512 on 35B-A3B, 1024 on 122B-A10B and
397B-A17B); `expert_outputs` is each slot's weighted output, `[batch, seq, top_k, hidden]`, and
sums to `routed_output`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    weights = moe.expert_weights.save()

top = logits.softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(weights, top.values / top.values.sum(-1, keepdim=True))
```

Beside them every token runs through one shared expert, a SwiGLU as wide as a routed one, whose
output the mixture multiplies by `sigmoid(shared_expert_gate(x))`, one scalar per token.
`shared_expert_output` is that product, what the mixture adds; `moe.shared_experts.output` is
the expert's output before the gate:

```python
with model.trace(prompt):
    x = moe.input.save()
    ungated = moe.shared_experts.output.save()
with model.trace(prompt):
    shared = moe.shared_expert_output.save()

gate = torch.sigmoid(moe._module.shared_expert_gate(x))   # [batch, seq, 1]
torch.equal(gate * ungated, shared)                       # True
```

Zeroing `shared_expert_output` removes the shared expert's whole contribution; zeroing one routed
expert's weight leaves the other slots' weights as they are, so they sum to less than one:

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
`flash-linear-attention` is not installed; with it installed, every `linear_attn` value but
`attention_output` is unavailable until `nnterp.route_kernels(model.family, "torch")`. The same
call is what makes `state` and `states` exist, through the token-by-token kernel. `states` holds
one float32 state per token, `num_v_heads × 128 × 128`: 2 MiB per token per block on 35B-A3B,
4 MiB on 122B-A10B and 397B-A17B. `expert_outputs` needs `experts_implementation` `"grouped_mm"`
(the default) or `"batched_mm"`.

## The attention block gates its heads' output

`q_proj` is `2 × num_heads × head_dim` wide (8192 on 35B-A3B) and holds, for each head in turn,
its 256 query dimensions followed by its 256 gate dimensions, so the first half of
`q_proj.output` is not the queries. The heads' output is multiplied by the sigmoid of that gate
before `o_proj`; `attention_head_outputs` is read before the gate and `o_proj.input` is the
gated tensor:

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
applied before the rotary. The rotary turns only the first 64 dimensions of each query and key
head (`partial_rotary_factor` 0.25, `rope_theta` 10⁷); `attention_queries[..., 64:]` equals
`q_norm.output[..., 64:]` exactly. `mrope_section` `[11, 11, 10]` splits those frequencies between
time, height and width for image tokens; on text the three positions coincide. 35B-A3B has 16
query heads over 2 key/value heads, 122B-A10B and 397B-A17B 32 over 2, so an edit to key/value
head `j` reaches 8 query heads on 35B-A3B and 16 on the others. The score scale is `256 ** -0.5`.

## The DeltaNet mixer

```
q, k, v = split(silu(conv1d(in_proj_qkv(x))))     # causal, width 4
beta = sigmoid(in_proj_b(x))
g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
y, state = delta_rule(l2norm(q) / sqrt(128), l2norm(k), v, g, beta)
out = out_proj(norm(y, gate=silu(in_proj_z(x))))  # gain: weight
```

`attention_queries`, `attention_keys` and `attention_values` are the kernel's arguments: after
the convolution and the SiLU, before the kernel's l2 norm and the queries' `1/sqrt(128)` scale.
Each head's key and value are 128 wide. There are 16 key heads, and 32 value heads on 35B-A3B,
64 on 122B-A10B and 397B-A17B; the queries and keys are repeated up to the value heads
(`repeat_interleave`) before the kernel, so they are served `num_v_heads` wide, heads `2j` and
`2j + 1` on 35B-A3B being copies of key head `j`. `decays` is one log decay per head and token
(float32, at most 0), `betas` the sigmoid write strength, and `attention_head_outputs` is `y`,
before the gated norm and `out_proj`. The state after a prompt, `state_output`, is
`[batch, num_v_heads, 128, 128]`.

## The prompt and the chat template

The tokenizer sets `bos_token` to `None` and prepends nothing; `<|endoftext|>` is 248044,
`<|im_start|>` 248045, `<|im_end|>` 248046. With `add_generation_prompt=True` every
checkpoint's template, the Base one included, ends the prompt at `<think>\\n` unless
`enable_thinking=False` is passed, which closes an empty `<think>\\n\\n</think>\\n\\n`; pass the
flag either way. The post-trained checkpoints' `generation_config.json` stops on `<|im_end|>` or
`<|endoftext|>`; 35B-A3B-Base ships none, so `generate` stops only on `<|endoftext|>`.

## What loads as this family

Every checkpoint is multimodal: `Qwen3_5MoeForConditionalGeneration` (`model_type`
`qwen3_5_moe`) with a `text_config` of `model_type` `qwen3_5_moe_text`, and nnterp picks the
family from the `text_config`. With `task="image-text-to-text"`, as this page builds them, the
wrapper loads with its processor: the text model at `model.language_model`, the vision encoder at
`model.visual` as `model.vision`. With the default `task="text-generation"` it builds
`Qwen3_5MoeForCausalLM` from the `model.language_model` weights, without the vision encoder; the
text model's names and values are the same under both. The multi-token-prediction block (`mtp`)
the checkpoints ship is loaded by neither. The dense Qwen3.5 and Qwen3.6 checkpoints are
`qwen3_5_text`; Qwen3-Next is `qwen3_next`.

## Sparse autoencoders

Qwen-Scope's residual-stream SAEs cover Qwen3.5-35B-A3B-Base:
`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50` (32768 features, `k` 50) and
`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100` (131072 features, `k` 100), TopK, one per block,
blocks 0 to 39. Their hook point is `resid_post`, the decoder block's output, which is
`model.layers[i].layer_output` on DeltaNet and attention blocks alike. Each `layer<i>.sae.pt`
holds `W_enc` `[d_sae, 2048]`, `b_enc`, `W_dec` `[2048, d_sae]` and `b_dec` in float32.
"""
