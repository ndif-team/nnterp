"""GLM-4-0414 and GLM-Z1: ``Glm4ForCausalLM``, GLM's block with a norm after each sublayer."""

MODEL_TYPE = "glm4"
TITLE = "GLM-4-0414 / GLM-Z1"
SUBTITLE = (
    "GLM's block with a second norm after each sublayer, so the stream receives the post-norms' outputs; "
    "post_attention_layernorm keeps Llama's meaning, the MLP's input norm."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-4-9B-0414"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Glm4ForCausalLM"
CHECKPOINTS = [
    "zai-org/GLM-4-9B-0414", "zai-org/GLM-Z1-9B-0414",
    "zai-org/GLM-4-32B-Base-0414", "zai-org/GLM-4-32B-0414",
    "zai-org/GLM-Z1-32B-0414", "zai-org/GLM-Z1-Rumination-32B-0414",
]

#: GLM lineage: glm sets 84; glm4 sits 5 degrees from it.
PALETTE = {"hue": 79}
VLLM = False
QUIRKS = ["sandwich-norms", "qkv-bias", "partial-rotary", "interleaved-rotary"]

#: Every claim in the notes was checked on the pinned tiny checkpoint (shapes, identities, the rotary
#: recomputed exactly) or read off the listed checkpoints' configs and tokenizers; no real weights were run.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_self_attn_layernorm",
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
            "post_norm": "post_mlp_layernorm",
            "contribution": "mlp_output",
            "detail": "fused gate_up_proj: {hidden_size} → 2 × {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "Despite its name, the MLP's input norm, as on Llama. On Gemma-2 the same name is the norm after the attention.",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. The tokenizer prepends "
             "nothing; the chat template writes [gMASK]<sop>.",
    "norm": "An RMSNorm whose gain is norm.weight; project_on_vocab applies it.",
    "head": "lm_head has its own weight (tie_word_embeddings is false). logits is lm_head.output, with no "
            "softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + post_self_attn_layernorm(self_attn(input_layernorm(x)))
out = h + post_mlp_layernorm(mlp(post_attention_layernorm(h)))
```

Four RMSNorms per block, two per sublayer. The names mix two conventions:
`post_attention_layernorm` is the MLP's *input* norm (Llama's meaning), while the norms after
the sublayers are `post_self_attn_layernorm` and `post_mlp_layernorm`. So `self_attn.input` is
`input_layernorm`'s output and `mlp.input` is `post_attention_layernorm`'s.

## The contributions are the post-norms' outputs

`attention_output` is `post_self_attn_layernorm`'s output and `mlp_output` is
`post_mlp_layernorm`'s, the tensors the block adds. `self_attn.output[0]` is `o_proj`'s output,
the tensor entering the post-norm, and `mlp.output` is `down_proj`'s. The identity is the plain
sum of the post-norms' outputs, exact:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal(x + attn + mlp, out)
```

An RMSNorm is scale-invariant up to its `eps` (`1e-5` on every checkpoint here), so scaling a
module's output does not scale what the block adds: `post_mlp_layernorm(0.5 * y)` is
`post_mlp_layernorm(y)`. Scale, steer or attribute through `attention_output` and `mlp_output`:

```python
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:] *= 0.5     # halves what the block adds

with model.trace(prompt):
    model.layers[1].mlp.output[:] *= 0.5         # the post-norm rescales it back
```

Zeroing `mlp.output` does zero the contribution, since the norm of zero is zero.

## Rotary turns adjacent pairs, on half of each head

The attention is GLM's: the rotary embedding turns dimension `2i` with `2i + 1` (each frequency
twice in a row in the cos/sin table), where Llama turns `i` with `i + rot / 2`, and only the first
`partial_rotary_factor × head_dim` dimensions of each query and key head, 64 of 128 on every
checkpoint here (`partial_rotary_factor` 0.5); dimensions 64 to 127 carry no position. The base
(`rope_theta`) is 10,000. `attention_queries` and `attention_keys` are read after the rotary;
recomputing them from `q_proj.output` and `k_proj.output` with adjacent pairs matches exactly.
To put the turned part in `rotate_half`'s order:

```python
hd = model.head_dim
rot = int(hd * model.config.rope_parameters["partial_rotary_factor"])
perm = torch.cat([torch.arange(0, rot, 2), torch.arange(1, rot, 2), torch.arange(rot, hd)])

with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.save()

q_half = q[..., perm]
```

## Biases on the 9B models only

`attention_bias` is true on GLM-4-9B-0414 and GLM-Z1-9B-0414: `q_proj`, `k_proj` and `v_proj`
add a bias (`o_proj` has none), so a zero input to `self_attn` still gives each value head its
slice of `v_proj.bias`. The 32B models set it false. The MLP is GLM's fused one:
`mlp.gate_up_proj.output` is the gate (first `intermediate_size` columns) then the up
projection, and `mlp.output` is `down_proj(act(gate) * up)`.

## Attention

The 9B models have 32 query heads over 2 key/value heads (an edit to key/value head `j`
reaches query heads `16j` to `16j + 15`); GLM-4-32B and GLM-Z1-32B 48 over 2; GLM-Z1-Rumination-32B
48 over 8. `head_dim` is 128 everywhere, the score scale `128 ** -0.5`, and `num_heads × head_dim`
is `hidden_size` on both sizes. There is no window, sink or softcap; the default `sdpa` load
computes the same function as an eager one, and the six attention interior values need
`attn_implementation="eager"`.

## The tokenizer prepends nothing; the chat template writes `[gMASK]<sop>`

A raw prompt is the text's tokens alone, from position 0. The chat template starts with
`[gMASK]<sop><|user|>`, and on GLM-Z1-9B-0414 `add_generation_prompt=True` ends the prompt
at `<|assistant|>\\n<think>`, so the next token begins a reasoning trace rather than the answer.
The 9B-0414 tokenizer's `eos_token` is `<|user|>`.

## What loads as this family

The configs that say `Glm4ForCausalLM` are the 0414 releases: GLM-4-9B, GLM-4-32B and its Base,
and the GLM-Z1 reasoning models (9B, 32B, Rumination-32B). `max_position_embeddings` is 32768,
131072 on Rumination. GLM-4-9B's first release is `glm` (no post-norms), and GLM-4.5, 4.6 and
4.7 are `glm4_moe`.
"""
