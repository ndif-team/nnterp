"""Qwen3.5 and Qwen3.6, dense: the text model of the multimodal checkpoints, a gated DeltaNet hybrid."""

MODEL_TYPE = "qwen3_5_text"
TITLE = "Qwen3.5 / Qwen3.6"
SUBTITLE = (
    "Llama's pre-norm block with a gated DeltaNet mixer on three blocks in four and, on the fourth, "
    "attention whose q_proj also yields a sigmoid gate on the heads' output; every RMSNorm but the "
    "mixer's own multiplies by 1 + weight."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3.5-0.8B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen3.5-tiny-random"
CHECKPOINTS = [
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.6-27B",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
WRAPPERS = {
    "qwen3_5": {
        "title": "Qwen3.5",
        "pinned": "yujiepan/qwen3.5-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm on each patch, then an MLP (linear_fc1, GELU, "
                     "linear_fc2) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
        "notes": """
## The merger folds four patches into one image token

`model.projector` is `model.visual.merger`, the vision encoder's last module: `norm` (a LayerNorm over
`vision_hidden`) on each patch, then each 2 × 2 block of consecutive patches concatenated into one
`4 * vision_hidden` vector, `linear_fc1`, GELU and `linear_fc2` to the text model's width. So an image
of `t * h * w` patches (its `image_grid_thw` row) is `t * h * w / 4` image tokens, and
`vision.image_features` is `model.projector.output` as it is, `[image_tokens, hidden]`. On 0.8B the
merger maps 4 × 768 = 3072 to 1024; a patch is 16 pixels, so an image token covers 32 × 32 pixels.

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
model only through `vision.image_features`; the text blocks have no `deepstack_output`. The vision encoder is otherwise Qwen3-VL's, with its learned `pos_embed`: 12 blocks of width 768
on 0.8B, 24 of width 1024 on 2B and 4B, 27 of width 1152 on 9B and 27B.

""",
    },
}

#: The Qwen lineage sits at 285 (qwen2) and 297 (qwen3); this family takes 273.
PALETTE = {"hue": 273}
VLLM = False
QUIRKS = ["hybrid", "gated-query", "qk-norm", "partial-rotary", "gain-norm", "multimodal-rotary"]

#: Real-value numbers below come from runs on Qwen3.5-0.8B (float32) and Qwen3.5-9B (bfloat16), with
#: transformers' pure-torch DeltaNet kernels (flash-linear-attention and causal-conv1d not installed).

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``);
#: every block has the dense MLP.
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
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. No BOS is prepended, "
             "so position 0 holds the text's first token.",
    "norm": "The gain is 1 + weight: the stored weight averages 3.31 on 0.8B and 1.14 on 9B.",
    "head": "lm_head is embed_tokens' weight on 0.8B, 2B and 4B (tie_word_embeddings) and a matrix of its own "
            "on 9B and 27B. logits is lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + mixer(input_layernorm(x))          # linear_attn (gated DeltaNet) or self_attn
out = h + mlp(post_attention_layernorm(h))   # SwiGLU, the same class on every block
```

`config.layer_types` is `full_attention` on every fourth block (`full_attention_interval` 4,
blocks 3, 7, 11, ...) and `linear_attention` on the others: 18 DeltaNet and 6 attention blocks
on 0.8B and 2B, 24 and 8 on 4B and 9B, 48 and 16 on 27B. A block has `linear_attn` or
`self_attn`, never both; the last block is always an attention block. Pick blocks outside the trace:

```python
types = model.config.layer_types
delta = [i for i, t in enumerate(types) if t == "linear_attention"]
full = [i for i, t in enumerate(types) if t == "full_attention"]    # 3, 7, ..., 23 on 0.8B
```

## The contributions are the sublayers' outputs

The block adds the mixer's output and the MLP's output to the stream and returns a tensor, so
the identity is the plain sum, exact in float32 on both kinds of block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    mix = model.layers[1].linear_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + mix + mlp, out)
```

On an attention block the middle term is `model.layers[3].self_attn.attention_output`.

## Loading

`attn_implementation="eager"` is needed only for the six attention interior values on the
attention blocks; there is no softcap or window, and on 0.8B in float32 the default `sdpa` load
gives the same logits as the eager one. The DeltaNet values are read at the delta-rule kernel
call and need no eager load. They need transformers' pure-torch kernels, which is what runs when
`flash-linear-attention` is not installed (transformers then warns that it falls back to them);
with it installed, every `linear_attn` value but `attention_output` is unavailable until
`nnterp.route_kernels(model.family, "torch")`. The same call is what makes `state` and `states`
exist: it routes a prompt through the token-by-token kernel, which on 0.8B on CPU took about twice
as long on a 250-token prompt and moved the logits by 2e-5. `states` holds one float32 state per
token, `num_v_heads × 128 × 128`: 1 MiB per token per block on 0.8B and 2B, 2 MiB on 4B and 9B,
3 MiB on 27B.

## The attention block gates its heads' output

`q_proj` is `2 × num_heads × head_dim` wide (4096 on 0.8B) and holds, for each head in turn,
its 256 query dimensions followed by its 256 gate dimensions, so the first half of
`q_proj.output` is not the queries. The heads' output is multiplied by the sigmoid of that gate
before `o_proj`:

```
q, gate = split per head(q_proj(x))             # [..., heads, 256] each
q = rope(q_norm(q)); k = rope(k_norm(k_proj(x))); v = v_proj(x)
out = o_proj(attention(q, k, v).flatten() * sigmoid(gate.flatten()))
```

`attention_queries` is read at the attention interface, after the split, the norm and the
rotary; `attention_head_outputs` is read there too, before the gate. The gate is mostly
closed: on block 3 its sigmoid averages 0.07 on 0.8B (78% of entries under 0.1) and 0.03 on 9B
(96%), so what reaches `o_proj` is a small fraction of `attention_head_outputs`, and an edit
to a head's output reaches the stream multiplied by its gate. `o_proj.input` is the gated
tensor:

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

## Queries and keys: normed per head, a quarter of each rotated

`q_norm` and `k_norm` are RMSNorms over one head's 256 dimensions with gain `1 + weight`,
applied before the rotary. The rotary turns only the first 64 dimensions of each query and key
head (`partial_rotary_factor` 0.25, `rope_theta` 10⁷); the other 192 carry no position, and
`attention_queries[..., 64:]` equals `q_norm.output[..., 64:]` exactly. The config's
`mrope_section` splits the rotary frequencies between time, height and width for image and video
tokens; on text the three positions coincide, and the rotation is the standard `rotate_half` one.
0.8B and 2B have 8 query heads over 2 key/value heads, 4B and 9B 16 over 4, 27B 24 over 4, so an
edit to key/value head `j` on 0.8B reaches query heads `4j` to `4j + 3`. The score scale is
`256 ** -0.5`.

## The DeltaNet mixer

```
q, k, v = split(silu(conv1d(in_proj_qkv(x))))     # causal, width 4
beta = sigmoid(in_proj_b(x))
g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
y, state = delta_rule(l2norm(q) / sqrt(128), l2norm(k), v, g, beta)
out = out_proj(norm(y, gate=silu(in_proj_z(x))))  # gain: weight
```

`attention_queries`, `attention_keys` and `attention_values` are the kernel's arguments: after
the convolution and the SiLU, before the kernel's l2 norm and the queries' `1/sqrt(128)` scale
(on 0.8B's block 0 the queries' norms run from 0.09 to 4.1). A recurrence written on them without that
normalization leaves a 42% error in block 0's final state on 0.8B; with it, it matches
`attention_head_outputs` to 2e-8. Each head's key and value are 128 wide. 0.8B and 2B have 16
key heads and 16 value heads; 4B and 9B 16 key heads over 32 value heads, 27B over 48. The
queries and keys are repeated up to the value heads (`repeat_interleave`) before the kernel, so
there they are served `num_v_heads` wide, with heads `2j` and `2j + 1` (on 4B and 9B) copies of
key head `j`, and an edit to one copy reaches that one value head.

`decays` is one log decay per head and token, `betas` the sigmoid write strength, so each
token keeps `decays.exp()` of the state. On 0.8B's block 0 the heads' mean kept fraction runs
from 0.05 (a head that forgets within a few tokens) to above 0.999. `attention_head_outputs` is
`y`, before the gated norm and `out_proj`. The state after a prompt, `state_output`, is
`[batch, num_v_heads, 128, 128]`.

## Position 0 is the text's first token, with no massive activation

The tokenizer sets `bos_token` to `None` and prepends nothing; `<|endoftext|>` is 248044,
`<|im_start|>` 248045, `<|im_end|>` 248046. Position 0 does not grow into an outlier in the
stream: across the prompts tried, its `layer_output` norm is at most 6 times the median of the
other positions on 0.8B (18.6 at most, on block 14) and at most 3.5 times on 9B. It is still a
place attention goes: from query 8 on, the attention blocks put a mean 0.10 to 0.42 of their
probability on key 0 on 0.8B and 0.13 to 0.61 on 9B, up to 0.80 for single heads. Block 6, a
DeltaNet block, writes a vector at position 0 several times larger than at any other position
(10.6 against at most 1.0 on 0.8B, 18 to 30 against at most 6.4 on 9B). Prepending
`<|endoftext|>` is not neutral: on 0.8B it takes `' Paris'` after `"The Eiffel Tower is in the
city of"` from 0.592 to 0.019, and `' cold'` after `"The opposite of hot is"` from 0.254 to 0.755.

## The chat template, and thinking by size

The template writes no system turn of its own. With `add_generation_prompt=True`, the 0.8B and
2B templates (Base included) end the prompt at an empty `<think>\\n\\n</think>\\n\\n` unless
`enable_thinking=True` is passed; the 4B, 9B (Base included), 27B and Qwen3.6-27B templates end it at
`<think>\\n` unless `enable_thinking=False` is passed. After `<think>\\n` the next token is
`'Thinking'` (1.000 on 0.8B and 9B), the start of a reasoning trace, not the answer. Pass the
flag either way rather than relying on the default:

```python
text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
with model.trace(text):
    probs = model.next_token_probs.save()
```

After the closed think block the answer's first token differs by size: on 0.8B `' Paris'` 0.784
and `'Paris'` 0.174, on 9B `'Paris'` 1.000, so a target should count both forms. 0.8B to 9B
ship no `generation_config.json`, so `generate` stops only on the config's `eos_token_id`,
`<|endoftext|>`: a templated reply on 0.8B runs on as `' Paris<|im_end|>\\n<|endoftext|>'`.
27B and Qwen3.6-27B stop on `<|im_end|>` or `<|endoftext|>`.

## The readout is the plain projection

`model.logits` equals `model.lm_head.output`, and `project_on_vocab` on the last block's
`layer_output` equals `logits` exactly. The final norm's gain is `1 + model.norm._module.weight`
(about 4.3 on average on 0.8B), as for `input_layernorm`, `post_attention_layernorm`, `q_norm`
and `k_norm`; the DeltaNet's own output norm multiplies by its `weight`. On 0.8B, 2B and 4B the
embedding and the unembedding are one matrix, so an edit to `embed_tokens.weight` edits
`lm_head`; on 9B and 27B they are separate.

## What loads as this family

Every checkpoint here is multimodal: its config is `Qwen3_5ForConditionalGeneration`
(`model_type` `qwen3_5`) with a `text_config` of `model_type` `qwen3_5_text`, and nnterp picks the
family from the `text_config`. With `task="image-text-to-text"`, as this page builds them, the
wrapper loads with its processor: the text model at `model.language_model`, the vision encoder at
`model.visual` as `model.vision`. With the default `task="text-generation"` it builds
`Qwen3_5ForCausalLM` from the `model.language_model` weights, without the vision encoder, and takes text
only; the text model's names and values are the same under both. The multi-token-prediction block
(`mtp`) the checkpoints ship is loaded by neither. The mixture-of-experts releases,
Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B and Qwen3.6-35B-A3B, are `qwen3_5_moe_text`.

## Sparse autoencoders

Qwen-Scope's residual-stream SAEs cover Qwen3.5-2B-Base (`Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`
and `-L0_100`, 32768 features), Qwen3.5-9B-Base (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` and
`-L0_100`, 65536) and the post-trained Qwen3.5-27B (`Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` and
`-L0_100`, 81920), one per block, TopK with `k` 50 or 100. Their hook point is `resid_post`, a
forward hook on the decoder block's output, which is `model.layers[i].layer_output`, on DeltaNet
and attention blocks alike. Each `layer<i>.sae.pt` holds `W_enc` `[d_sae, hidden]`, `b_enc`,
`W_dec` `[hidden, d_sae]` and `b_dec` in float32.
"""
