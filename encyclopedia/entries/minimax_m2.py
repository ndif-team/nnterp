"""MiniMax-M2: Llama's pre-norm block with query and key norms over the whole projection, and a mixture of experts
on every block chosen by sigmoid scores plus a selection bias."""

MODEL_TYPE = "minimax_m2"
TITLE = "MiniMax-M2 / MiniMax-M2.1 / MiniMax-M2.5 / MiniMax-M2.7"
SUBTITLE = (
    "Llama's pre-norm block with queries and keys RMS-normed across all heads, and a mixture of 256 experts on "
    "every block whose 8 per token are chosen by sigmoid scores plus a bias and weighted by the sigmoids alone."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "MiniMaxAI/MiniMax-M2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-MiniMaxM2ForCausalLM"
CHECKPOINTS = ["MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2.1", "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7"]

#: Set by hues.py (lineage: MiniMax / MiMo).
PALETTE = {"hue": 346}
VLLM = False
QUIRKS = ["qk-norm", "mixture-of-experts"]

#: The checkpoints have 230B parameters: nothing here ran on real weights. Every identity, shape and snippet ran on
#: the pinned tiny checkpoint (2 blocks, 8 experts, top 2; float32, eager); the sizes are the Hub configs', read on
#: meta builds, and the routing arithmetic is transformers' MiniMaxM2TopKRouter.

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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv, q/k normed",
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
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # q_norm, k_norm over the whole projection
out = h + mlp(post_attention_layernorm(h))    # 8 of 256 experts per token, on every block
```

Llama's pre-norm block and Llama's names. Every block's MLP is a mixture: there is no dense MLP and
no shared expert, so `intermediate_size` (1536) is one expert's width. Nothing norms a sublayer's
output, so `attention_output` is `o_proj`'s output and `mlp_output` the mixture's, and the identity
is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Queries and keys are normed across all heads

`q_norm` is one RMSNorm over `q_proj`'s whole output (48 heads × 128 = 6144 wide) and `k_norm` one over
`k_proj`'s (8 × 128 = 1024), applied before the heads are split and before the rotary embedding;
`attention_queries` and `attention_keys` are read after both. One RMS covers every head, so an edit to
one head's slice of `q_proj.output` changes the other heads' queries too. Edit a single head at
`attention_queries` or `attention_keys`, which no norm follows:

```python
with model.trace(prompt):
    model.layers[1].self_attn.attention_queries[:, 0] = 0   # head 0 only
```

48 query heads share 8 key/value heads, so `attention_keys` and `attention_values` are
`[batch, 8, seq, 128]` and query head `h` reads key/value head `h // 6`: an edit to one key/value head
reaches 6 query heads. `head_dim` is the config's 128, so `num_heads * head_dim` (6144) is wider
than the 3072-wide stream.

## Load with eager for the attention interior

The attention interior needs `attn_implementation="eager"`; the default load runs `sdpa`. There is no
softcap, sink or window, and on the pinned tiny checkpoint the two loads' logits differ by 2e-7 in
float32. The checkpoints' configs carry an FP8 `quantization_config` (128 × 128 weight blocks) that
leaves the router, `e_score_correction_bias` and `lm_head` unquantized.

## The router: sigmoid scores, a selection bias, renormalized weights

`router_logits` is the router's projection, `[batch, seq, 256]`, in the model's dtype. The router takes
their sigmoid in float32 and adds the mixture's selection bias, `mlp.e_score_correction_bias` (a
buffer on the mixture module, not on the router), to choose 8 experts. `expert_weights` are the
chosen experts' sigmoids without the bias, renormalized to sum to one, with no scaling factor; they
stay float32 under a bfloat16 load. There is no shared expert, so `routed_output` is `mlp_output`.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()           # [batch, seq, num_experts]
    w = moe.expert_weights.save()               # [batch, seq, top_k]
    idx = moe.expert_indices.save()
    each = moe.expert_outputs.save()            # [batch, seq, top_k, hidden]
    out = moe.mlp_output.save()

scores = logits.float().sigmoid()
choice = (scores + moe._module.e_score_correction_bias).topk(moe.top_k, dim=-1).indices
assert torch.equal(choice.sort(-1).values, idx.sort(-1).values)     # chosen with the bias
chosen = scores.gather(-1, idx)
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True))  # weighted without it
torch.testing.assert_close(each.sum(2), out)                         # no shared expert
```

The router asks `topk` for unsorted slots (`sorted=False`), so slot 0 need not hold the largest
weight: compare sets of experts, not slots. A written `router_logits` moves both the choice and the
weights; a written bias moves only the choice.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other weights as they were, so they then sum to less than one. `mlp_output` changes on exactly
the tokens that chose `e`:

```python
moe, e = model.layers[1].mlp, 4
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

changed = (ablated != clean).any(-1)                       # [batch, seq]
assert torch.equal(changed, (idx == e).any(-1))            # only the tokens that chose e
```

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. The final norm is a plain RMSNorm whose gain is `norm.weight`.
`lm_head` and `embed_tokens` are separate weights. The tokenizer prepends nothing, so
`model.input_ids` is the prompt's tokens alone; its begin and end tokens are `]~!b[` (id 200034) and
`[e~[` (id 200020).
"""
