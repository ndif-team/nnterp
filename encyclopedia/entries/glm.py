"""GLM-4-9B in transformers' own format (the ``-hf`` repositories) and GLM-Edge: ``GlmForCausalLM``."""

MODEL_TYPE = "glm"
TITLE = "GLM-4 / GLM-Edge"
SUBTITLE = (
    "Llama's block with a fused gate and up projection, biased queries, keys and values, and a rotary "
    "embedding that turns adjacent pairs of dimensions, on the first half of each head only on GLM-4-9B."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/glm-4-9b-hf"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GlmForCausalLM"
CHECKPOINTS = [
    "zai-org/glm-4-9b-hf", "zai-org/glm-4-9b-chat-hf", "zai-org/glm-4-9b-chat-1m-hf",
    "zai-org/glm-edge-1.5b-chat", "zai-org/glm-edge-4b-chat",
]

#: GLM lineage, set here (glm4 79, glm4_moe 89). No hue is 15 degrees from every other entry's;
#: 84 sits 14 from OLMoE's 70 and 15 from GPT-NeoX's 99.
PALETTE = {"hue": 84}
VLLM = False
QUIRKS = ["qkv-bias", "partial-rotary", "interleaved-rotary"]

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
            "detail": "fused gate_up_proj: {hidden_size} → 2 × {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "The MLP's input norm, as on Llama.",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. GLM-4-9B's tokenizer "
             "prepends [gMASK]<sop> to every string; GLM-Edge's prepends nothing.",
    "norm": "An RMSNorm whose gain is norm.weight; project_on_vocab applies it.",
    "head": "lm_head has its own weight on GLM-4-9B and GLM-Edge-4B, and is embed_tokens' weight on "
            "GLM-Edge-1.5B (tie_word_embeddings). logits is lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))      # biased q, k, v; paired rotary
out = h + mlp(post_attention_layernorm(h))   # one gate_up_proj
```

Llama's pre-norm block and Llama's names: `post_attention_layernorm` is the MLP's input norm.
Nothing norms a sublayer's output, so `attention_output` is `o_proj`'s output, `mlp_output` is
`down_proj`'s, and the identity is the plain sum, exact:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal(x + attn + mlp, out)
```

## Rotary turns adjacent pairs, on half of each head on GLM-4-9B

The rotary embedding turns dimension `2i` with `2i + 1`, where Llama turns `i` with
`i + rot / 2`: each frequency appears twice in a row in the cos/sin table, and `rotate_half`
takes the even and odd dimensions. It covers the first `partial_rotary_factor × head_dim`
dimensions of each query and key head and passes the rest through unchanged. On GLM-4-9B the
factor is 0.5 (the config does not say it; `GlmConfig` sets it), so dimensions 0 to 63 of each
128-wide head are turned and 64 to 127 carry no position. GLM-Edge's configs set 1.0: the whole
head turns. The base (`rope_theta`) is 10,000 on every checkpoint here.

`attention_queries` and `attention_keys` are read after the rotary. Recomputing them from
`q_proj.output` and `k_proj.output` with the pairs adjacent matches exactly; with Llama's
pairing it does not. To compare with a `rotate_half` family or reuse rotary code written for
one, reorder the turned part, evens then odds:

```python
hd = model.head_dim
rot = int(hd * model.config.rope_parameters["partial_rotary_factor"])
perm = torch.cat([torch.arange(0, rot, 2), torch.arange(1, rot, 2), torch.arange(rot, hd)])

with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.save()

q_half = q[..., perm]          # rotate_half's order; q[..., rot:] carries no position
```

## Queries, keys and values carry a bias

`attention_bias` is true on GLM-4-9B: `q_proj`, `k_proj` and `v_proj` add a bias, and `o_proj`
has none. GLM-Edge sets it false. Where the bias is on, a zero input still gives keys, values
and queries: zeroing `self_attn.input` makes every value head equal to its slice of
`v_proj.bias`, so every head outputs that bias whatever its pattern, and `attention_output`
is `o_proj` of the biases, not zero. A zero ablation of the attention is `attention_output`
set to zero, not its input.

```python
with model.trace(prompt):
    model.layers[1].self_attn.input[:] = 0
    v = model.layers[1].self_attn.attention_values.save()

# every value head is its slice of v_proj.bias
```

## The MLP fuses its gate and up projections

`mlp.gate_up_proj` is one `Linear` from `hidden_size` to `2 × intermediate_size`; its output's
first half is the gate and the second half the up projection, and `mlp.output` is
`down_proj(act(gate) * up)`. There is no `gate_proj` or `up_proj` module to hook: a neuron's
pre-activation is a column of the first half.

```python
I = model.config.intermediate_size
with model.trace(prompt):
    gate_up = model.layers[1].mlp.gate_up_proj.output.save()

gate, up = gate_up[..., :I], gate_up[..., I:]
```

## Attention

GLM-4-9B and its 128k chat model have 32 query heads over 2 key/value heads, so an edit to
key/value head `j` reaches query heads `16j` to `16j + 15`; the 1M-context chat model has 4
key/value heads, GLM-Edge-1.5B 16 over 4 and GLM-Edge-4B 24 over 6. `head_dim` is 128
throughout and the scores are scaled by `128 ** -0.5`. There is no window, sink or softcap,
and the default `sdpa` load computes the same function as an eager one; the six attention
interior values need `attn_implementation="eager"`.

## The tokenizer prepends `[gMASK]<sop>` on GLM-4-9B

GLM-4-9B's tokenizers (base and chat) put `[gMASK]` and `<sop>` before every encoded string, so
`model.input_ids` starts with them and the prompt's first token is at position 2. The chat
template writes the same two tokens itself: tokenize a templated prompt with
`add_special_tokens=False`, or the prefix appears twice. GLM-Edge's tokenizer prepends nothing,
and its chat template starts at `<|user|>`.

## The readout is the plain projection

`model.logits` equals `lm_head.output`; there is no softcap or scale. `rms_norm_eps` is
`1.5625e-07` on GLM-4-9B and `1e-5` on GLM-Edge.

## What loads as this family

The configs that say `GlmForCausalLM` are GLM-4-9B's transformers-format releases
(`glm-4-9b-hf`, `glm-4-9b-chat-hf`, `glm-4-9b-chat-1m-hf`) and the GLM-Edge chat models. The
repositories without `-hf` (`zai-org/glm-4-9b`, `glm-4-9b-chat`) are `chatglm`, loaded through
their own remote code, which no family covers. GLM-4-0414 and GLM-Z1 are `glm4`, and GLM-4.5,
4.6 and 4.7 are `glm4_moe`.
"""
