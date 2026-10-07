"""GLM-4.5, GLM-4.5-Air, GLM-4.6 and GLM-4.7: ``Glm4MoeForCausalLM``, dense first blocks and a sigmoid-routed mixture after."""

MODEL_TYPE = "glm4_moe"
TITLE = "GLM-4.5 / GLM-4.6 / GLM-4.7"
SUBTITLE = (
    "Llama's pre-norm block with rotary on half of each head and, after the first dense blocks, a mixture "
    "whose sigmoid router picks 8 experts with a selection bias, beside one shared expert."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-4.5-Air"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Glm4MoeForCausalLM"
CHECKPOINTS = [
    "zai-org/GLM-4.5-Air-Base", "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.5-Base", "zai-org/GLM-4.5",
    "zai-org/GLM-4.6", "zai-org/GLM-4.7",
]

#: GLM lineage: glm sets 84; glm4_moe sits 5 degrees from it on the other side from glm4 (79).
PALETTE = {"hue": 89}
VLLM = False
QUIRKS = ["mixture-of-experts", "dense-first-blocks", "qkv-bias", "qk-norm", "partial-rotary"]

#: Every claim in the notes was checked on the pinned tiny checkpoint (4 experts, top 2, block 0 dense),
#: on a copy of its config with head_dim 8 for the rotary, or read off the listed checkpoints' configs and
#: tokenizers; no real weights were run.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
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
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. The tokenizer prepends "
             "nothing; the chat template writes [gMASK]<sop>.",
    "layers": "The checkpoints also store one multi-token-prediction block after the last; transformers "
              "does not load it, so layers holds num_hidden_layers blocks.",
    "head": "lm_head has its own weight (tie_word_embeddings is false). logits is lm_head.output, with no "
            "softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))        # q_norm, k_norm inside on 4.5, 4.6, 4.7
out = h + mlp(post_attention_layernorm(h))     # dense first, then routed + shared
```

Llama's pre-norm block and names. `first_k_dense_replace` blocks come first with a dense
SwiGLU `mlp` (3 on GLM-4.5, 4.6 and 4.7, 1 on Air); every later block's `mlp` is a mixture.
Nothing norms a sublayer's output, so the identity is the plain sum on both kinds of block,
exact:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal(x + attn + mlp, out)
```

On the dense blocks `support()` reports every mixture value missing (`"no router_logits value on
this block's mlp"`).

## The router: sigmoid scores, a selection bias, renormalized weights

`router_logits` are the router's logits, computed in float32. The router takes their sigmoid,
adds `router.e_score_correction_bias` (a per-expert buffer) and keeps the 8 highest
(`n_group` and `topk_group` are 1, so every expert competes). `expert_weights` are the chosen
experts' sigmoids *without* the bias, renormalized to sum to one and multiplied by
`routed_scaling_factor`, so a token's weights sum to 2.5 on GLM-4.5, 4.6 and 4.7 and to 1.0 on
Air. They stay float32 under a bfloat16 load. The bias decides which experts are chosen but not
how much each counts; the slots come in no promised order.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts], before the sigmoid
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

s = logits.sigmoid()
chosen = (s + moe.router._module.e_score_correction_bias).topk(moe.top_k, dim=-1).indices
assert torch.equal(chosen.sort(-1).values, idx.sort(-1).values)
expected = s.gather(-1, idx)
expected = expected / expected.sum(-1, keepdim=True) * model.config.routed_scaling_factor
torch.testing.assert_close(expected, w)
```

## The mixture's output is routed plus shared

GLM-4.5, 4.6 and 4.7 route over 160 experts, Air over 128, 8 per token on both, and each mixture
has one shared expert (`mlp.shared_experts`, a dense SwiGLU `moe_intermediate_size` wide) that
every token runs through. The shared expert runs after the routed experts, so read
`routed_output` before `shared_expert_output`; their sum is `mlp_output` exactly:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

assert torch.equal(routed + shared, out)
```

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term from
`routed_output` and leaves the token's other weights as they were (they are not renormalized
again). `mlp_output` changes on exactly the tokens that chose `e`, by that slot's
`expert_outputs`; the shared expert is untouched.

```python
moe, e = model.layers[1].mlp, 2
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

changed = (ablated != clean).any(-1)                 # [batch, seq]
assert torch.equal(changed, (idx == e).any(-1))
```

## Rotary turns half of each head, rotate_half style

The rotary covers the first `partial_rotary_factor × head_dim` dimensions of each query and key
head, 64 of 128 (`partial_rotary_factor` 0.5), pairing `i` with `i + 32` as Llama's
`rotate_half` does; dimensions 64 to 127 carry no position. The base (`rope_theta`) is 10⁶.
`attention_queries` and `attention_keys` are read after the rotary; recomputing them from
`q_proj.output` and `k_proj.output` with `rotate_half` pairing matches exactly, and the unturned
half equals the projection (after `q_norm` where there is one). The GLM-4-9B families (`glm`, `glm4`) pair
adjacent dimensions instead.

## Query and key norms on GLM-4.5, 4.6 and 4.7, not on Air

Where `use_qk_norm` is true (GLM-4.5, 4.5-Base, 4.6, 4.7) the attention has `q_norm` and
`k_norm`, RMSNorms over one head's 128 dimensions with one gain shared by every head, applied
to `[batch, seq, heads, head_dim]` before the transpose and the rotary. An edit to one head's
slice of `q_proj.output` is renormalized and loses its size; edit `attention_queries` or
`attention_keys`, which come after the norm. GLM-4.5-Air and Air-Base have no `q_norm` or
`k_norm`.

```python
with model.trace(prompt):
    normed = model.layers[1].self_attn.q_norm.output.save()   # [batch, seq, heads, head_dim]
```

## Attention

Every checkpoint has 96 query heads over 8 key/value heads (an edit to key/value head `j`
reaches query heads `12j` to `12j + 11`), 128 wide, with biases on `q_proj`, `k_proj` and
`v_proj` (`attention_bias`) and none on `o_proj`. `head_dim` is the config's, so
`num_heads × head_dim` is 12288, wider than `hidden_size` (4096 on Air, 5120 on GLM-4.5):
`q_proj` widens the stream and `o_proj` narrows it back, and `attention_head_outputs`
flattened is wider than `layer_output`. The score scale is `128 ** -0.5`; there is no window,
sink or softcap, the default `sdpa` load computes the same function as an eager one, and the
six attention interior values need `attn_implementation="eager"`.

## The tokenizer and the chat template

The tokenizer prepends nothing: a raw prompt is the text's tokens from position 0. The chat
template starts with `[gMASK]<sop><|user|>`; on GLM-4.7 `add_generation_prompt=True` ends the
prompt at `<|assistant|><think>`, so the next token begins a reasoning trace.

## What loads as this family

The configs that say `Glm4MoeForCausalLM` are GLM-4.5, GLM-4.5-Air, their Base models, GLM-4.6
and GLM-4.7 (and their FP8 copies). Each stores one multi-token-prediction block after the last
(`num_nextn_predict_layers` 1); transformers skips its weights at load, so `model.layers` has
`num_hidden_layers` blocks (92, or 46 on Air). GLM-4.7-Flash is `glm4_moe_lite` (latent
attention), and GLM-5 is `glm_moe_dsa`.
"""
