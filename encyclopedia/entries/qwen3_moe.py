"""Qwen3-MoE: Qwen3's attention with a mixture of experts on every block."""

MODEL_TYPE = "qwen3_moe"
TITLE = "Qwen3-MoE"
SUBTITLE = (
    "Qwen3's attention, an RMSNorm on each query and key head, with a mixture of experts on every block "
    "in place of the MLP: 8 experts per token, their softmax weights renormalized to sum to one, and no "
    "shared expert."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-30B-A3B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
CHECKPOINTS = [
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-480B-A35B-Instruct",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 298}
VLLM = True
QUIRKS = ["qk-norm", "mixture-of-experts"]

#: Every number in the notes comes from a checkpoint's config or tokenizer; the shapes, identities and
#: snippets were run on the pinned tiny checkpoint (4 experts, top 2, 2 blocks), in float32 and bfloat16.
#: No real weights were run: the smallest checkpoint has 30B parameters.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k RMS-normed",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. No BOS is prepended, "
             "so position 0 holds the text's first token.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is "
            "lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))          # q_norm, k_norm per head, then rope
out = h + mlp(post_attention_layernorm(h))       # 8 of 128 experts per token
```

Qwen3's pre-norm block with Llama's names, and a mixture where Qwen3 has its MLP. Nothing norms
a sublayer's output: `attention_output` is `o_proj`'s output and `mlp_output` the mixture's,
which has no shared expert, so `mlp_output` is `routed_output`. The identity is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    routed = model.layers[1].mlp.routed_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(routed, mlp)
torch.testing.assert_close(x + attn + mlp, out)
```

## Every block is a mixture

A block's `mlp` is a mixture where its index is not in `mlp_only_layers` and
`(i + 1) % decoder_sparse_step == 0`. Every checkpoint here sets `mlp_only_layers` to `[]` and
`decoder_sparse_step` to 1, so all 48 blocks of 30B-A3B (94 of 235B-A22B, 62 of
Coder-480B-A35B) are mixtures. The config's `intermediate_size` (6144 on 30B-A3B, 12288 on
235B-A22B) is the width of a dense MLP no block has; the experts are `moe_intermediate_size`
wide (768, 1536, 2560 on Coder-480B-A35B), and `model.layers[i].mlp.intermediate_size` reads that.

## The router renormalizes its top 8

The router (`gate`, aliased `router`) gives one logit per expert. It takes a softmax over all of
them in float32, keeps the 8 largest, and, with `norm_topk_prob` true on every checkpoint, divides
them by their sum before casting back to the model's dtype, so a token's `expert_weights` sum to
one (to bfloat16's rounding in a bfloat16 load). `expert_indices` are the softmax's top 8, slot 0
the largest:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts], before the softmax
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

top = logits.float().softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close((top.values / top.values.sum(-1, keepdim=True)).to(w.dtype), w)
assert torch.equal(top.indices, idx)
```

`moe.SCORING` is `"softmax"`. 30B-A3B and 235B-A22B have 128 experts and Coder-480B-A35B has
160, all with 8 per token.

## Ablating an expert

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term; the token's
other weights are not renormalized again, so they sum to less than one. `mlp_output` changes on
exactly the tokens that chose `e`, and an expert no token chose has no effect:

```python
moe, e = model.layers[1].mlp, 3
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

changed = (ablated != clean).any(-1)                       # [batch, seq]
assert torch.equal(changed, (idx == e).any(-1))            # only the tokens that chose e
```

This holds on the pinned tiny checkpoint for every expert, in float32 and in bfloat16.

## expert_outputs needs the default experts implementation

The experts are one module, `experts`, with every expert's weights stacked. `expert_outputs`, each
slot's weighted output, is read inside transformers' `grouped_mm` (the default) and `batched_mm`
forwards; under `experts_implementation="eager"` `support()` reports it missing and names the
kwarg. The other mixture values read the same under every implementation. `expert_outputs` sums
over its slot axis to `routed_output`.

## Queries and keys are normed per head

`q_norm` and `k_norm` are RMSNorms over one head's 128 dimensions, one gain for every head,
applied before the rotary embedding; `attention_queries` and `attention_keys` are read after both.
An edit to a head's slice of `q_proj.output` or `k_proj.output` passes through that norm, which
undoes its scale; edit `attention_queries` or `attention_keys` to change a head's scores. No
projection has a bias (`attention_bias` false).

`head_dim` is 128 on every checkpoint, so the attention is wider than the stream:
30B-A3B has 32 query heads over 4 key/value heads (4096 wide against `hidden_size` 2048),
235B-A22B 64 over 4 (8192 against 4096), Coder-480B-A35B 96 over 8 (12288 against 6144). An
edit to key/value head `j` on 30B-A3B reaches query heads `8j` to `8j + 7`.

## Loading

The attention interior needs `attn_implementation="eager"`. Every checkpoint sets
`use_sliding_window` false, so `sliding_window` is `None` and no block has a window; there is no
softcap or sink. `rope_theta` is 10⁶ on 30B-A3B, 30B-A3B-Base and 235B-A22B, 5·10⁶ on
235B-A22B-Instruct-2507 and -Thinking-2507, and 10⁷ on the 30B-A3B 2507 pair and both Coder models.

## The chat templates

The tokenizer prepends nothing (`bos_token` is `None`). On 30B-A3B and 235B-A22B (and the Base
model, which ships the same template) `add_generation_prompt=True` ends the prompt at
`<|im_start|>assistant\\n`, where the model opens a `<think>` block; `enable_thinking=False`
appends an empty `<think>\\n\\n</think>\\n\\n`. The Thinking-2507 templates end the prompt at
`<think>\\n` and ignore `enable_thinking`. The Instruct-2507 and Coder templates have no thinking
block. No template writes a system turn of its own.

```python
text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
with model.trace(text):
    probs = model.next_token_probs.save()
```

## The readout

`model.logits` equals `model.lm_head.output`, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `lm_head` and `embed_tokens` are separate weights on every
checkpoint here.
"""
