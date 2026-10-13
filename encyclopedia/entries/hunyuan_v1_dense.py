"""Hunyuan dense V1: Llama's block whose attention RMS-norms each query and key head after the rotary embedding."""

MODEL_TYPE = "hunyuan_v1_dense"
TITLE = "Hunyuan"
SUBTITLE = (
    "Llama's block with each query and key head RMS-normed after the rotary embedding "
    "(query_layernorm, key_layernorm), so a head's query and key lengths are set by the norms' gains."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tencent/Hunyuan-0.5B-Pretrain"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HunYuanDenseV1ForCausalLM"
CHECKPOINTS = [
    "tencent/Hunyuan-0.5B-Pretrain", "tencent/Hunyuan-0.5B-Instruct",
    "tencent/Hunyuan-1.8B-Pretrain", "tencent/Hunyuan-1.8B-Instruct",
    "tencent/Hunyuan-4B-Pretrain", "tencent/Hunyuan-4B-Instruct",
    "tencent/Hunyuan-7B-Pretrain", "tencent/Hunyuan-7B-Instruct", "tencent/Hunyuan-7B-Instruct-0124",
    "tencent/Hunyuan-MT-7B", "tencent/Hunyuan-MT-Chimera-7B",
]

#: Set by hues.py (lineage: HunYuan).
PALETTE = {"hue": 26}
VLLM = False
QUIRKS = ["qk-norm"]

#: Every real value in the notes was measured on Hunyuan-0.5B-Pretrain in float32 on a GPU unless the note
#: says bfloat16; shapes, identities and read orders also ran on the pinned tiny checkpoint.

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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings) on every checkpoint; "
            "logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
q, k = rope(q_proj(n), k_proj(n))                 # n = input_layernorm(x)
q, k = query_layernorm(q), key_layernorm(k)       # per head, after the rotary
h    = x + o_proj(attention(q, k, v_proj(n)))
out  = h + mlp(post_attention_layernorm(h))
```

Llama's pre-norm block and Llama's names, `post_attention_layernorm` included (the MLP's input
norm). Nothing norms a sublayer's output, so `attention_output` is `o_proj`'s output and `mlp_output`
the MLP's, and the identity is the plain sum, exact in bfloat16:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Each query and key head is normed after the rotary embedding

`query_layernorm` and `key_layernorm` are RMSNorms `head_dim` wide (128 on every checkpoint), one gain
vector shared by every head, applied to each head's rotated query and key. `attention_queries` and
`attention_keys` are read after them: `attention_queries` equals `query_layernorm.output` exactly. The
norm removes the length a head's query or key had before it, so the dot products' scale comes from
the two gains (their means are 1.15 to 1.41 across the 24 blocks of 0.5B). Tripling head 0's slice
of `q_proj.output` at block 8 moves the logits by at most 7e-4; tripling head 0 at
`attention_queries` sharpens its pattern, from 0.71 nats of entropy to 0.01. Scale or steer a
head's query or key where no norm follows:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    attn.attention_queries[:, 0] *= 3        # head 0 attends more sharply
    logits = model.logits.save()
```

Each head is normed on its own, so an edit to one head's slice of `q_proj.output` stays in that
head.

## Grouped-query attention, scaled by head_dim ** -0.5

0.5B has 16 query heads over 8 key/value heads (1.8B 16 over 4, 4B and 7B 32 over 8).
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, 8, seq, 128]` on 0.5B,
so an edit to key/value head `j` reaches query heads `2j` and `2j + 1`. The query scale is
`head_dim ** -0.5`, and `attention_scores` are the scaled products with the causal mask added:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    scores = attn.attention_scores.save()

k = k.repeat_interleave(model.num_heads // model.num_kv_heads, dim=1)
qk = q @ k.transpose(-1, -2) * model.head_dim ** -0.5
causal = torch.ones_like(qk[0, 0], dtype=torch.bool).tril()
torch.testing.assert_close(qk[..., causal], scores[..., causal])
```

The interior needs `attn_implementation="eager"`. A default load runs `sdpa`, which computes the
same function: its logits differ from eager's by at most 3e-5.

## The rotary base is raised by alpha at load

`rope_scaling` is `{"type": "dynamic", "alpha": 1000.0}` with `rope_theta` 10000 on every checkpoint
but the two translation models, whose `alpha` is 100000. The rotary embedding builds its frequencies
once, from a base of `rope_theta * alpha ** (head_dim / (head_dim - 2))`: 1.116e7 at `alpha` 1000,
1.2e9 at 100000. Code that recomputes the rotation takes that base, not `rope_theta`:

```python
rotary = model._module.model.rotary_emb
base = 1 / rotary.inv_freq[1] ** (model.head_dim / 2)    # 1.116e7 on 0.5B
```

## Position 0 is a sink, and no BOS token is added

The tokenizer prepends nothing to a plain prompt, so the first real token takes the role of a
sink. On 0.5B its residual norm is 78 after blocks 8 and 16, against 0.6 to 1.5 at the other
positions, and at block 16 heads put 0.83 to 0.90 of their attention on it (two test prompts). The
chat templates start with the BOS token (`<｜hy_begin▁of▁sentence｜>` on 0.5B to 4B,
`<|startoftext|>` on 7B), which then is position 0. Slice `[:, 1:]` before averaging activations for
steering vectors, mean ablation or probes.

## The readout is plain, and the head is the embedding

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `lm_head` and `embed_tokens` are one parameter on every checkpoint
(`tie_word_embeddings`), so an edit to `embed_tokens.weight` is an edit to the unembedding.
`token_embeddings` equals `layers[0].input`.

## What this family module covers

The module serves every checkpoint whose config says `model_type` `hunyuan_v1_dense`: Hunyuan
0.5B, 1.8B, 4B and 7B, base and instruct, the earlier 7B-Instruct-0124, and the translation models
Hunyuan-MT-7B and MT-Chimera-7B. The 0.5B to 4B checkpoints share a 120818-token vocabulary, the
7B ones about 128K. Hunyuan-A13B is `hunyuan_v1_moe`; `Hunyuan-7B-Pretrain-0124` says `model_type`
`hunyuan` and loads only with its own remote code.
"""
