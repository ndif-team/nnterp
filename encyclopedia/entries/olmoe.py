"""OLMoE: Llama's pre-norm block with OLMo's query and key norms, and a mixture of 64 experts on every block."""

MODEL_TYPE = "olmoe"
TITLE = "OLMoE"
SUBTITLE = (
    "Llama's pre-norm block with queries and keys RMS-normed across all heads, and a mixture of 64 experts "
    "on every block whose 8 weights per token are the softmax's own entries, not renormalized, so they sum "
    "to less than one."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/OLMoE-1B-7B-0924"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
CHECKPOINTS = [
    "allenai/OLMoE-1B-7B-0924", "allenai/OLMoE-1B-7B-0924-SFT", "allenai/OLMoE-1B-7B-0924-Instruct",
    "allenai/OLMoE-1B-7B-0125", "allenai/OLMoE-1B-7B-0125-SFT", "allenai/OLMoE-1B-7B-0125-DPO",
    "allenai/OLMoE-1B-7B-0125-Instruct",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 104}
VLLM = True
QUIRKS = ["qk-norm", "mixture-of-experts", "unnormalized-routing"]

#: Every real value in the notes was measured on OLMoE-1B-7B-0924 (bf16 as stored unless the note says
#: float32) on a GPU; the snippets also ran on the pinned tiny checkpoint (4 experts, top 2).

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
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
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
            "pre_norm_note": "The MLP's input norm, as on Llama. On OLMo-2 the same name is the norm after the attention.",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "norm": "A plain RMSNorm: the gain is norm.weight (0.13 to 2.61 on 0924, mean 2.18), eps 1e-5. "
            "project_on_vocab applies it.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # q_norm, k_norm inside
out = h + mlp(post_attention_layernorm(h))    # 8 of 64 experts per token
```

Llama's pre-norm block and Llama's names: `post_attention_layernorm` is the MLP's input norm here,
where on OLMo-2 the same name is a norm after the attention. Nothing norms a sublayer's output, so
`attention_output` is `o_proj`'s output and `mlp_output` the mixture's, and scaling either module's
output scales what the block adds. The identity is the plain sum, exact in bfloat16:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Queries and keys are normed across all heads

`q_norm` and `k_norm` are RMSNorms over the whole projection, 2048 wide on every checkpoint
(16 heads of 128), applied before the heads are split and before the rotary embedding;
`attention_queries` and `attention_keys` are read after both. One RMS covers every head, so an edit
to one head's slice of `q_proj.output` reaches the others: tripling head 0's slice at block 8 changes
head 1's queries by 17% on 0924. Edit a single head at `attention_queries` or `attention_keys`,
which no norm follows:

```python
with model.trace(prompt):
    model.layers[1].self_attn.attention_queries[:, 0] = 0   # head 0 only
```

Every checkpoint has 16 key/value heads for 16 query heads, so there is no grouping, and `clip_qkv`,
the optional clamp on queries, keys and values, is off (`null`) on all of them.

## Load with eager for the attention interior only

The attention interior needs `attn_implementation="eager"`. The default `sdpa` load computes the same
function: in float32 the two loads' logits differ by at most 2e-5 on 0924. There is no softcap,
sink or window. 0924's weights are stored in bfloat16 (13.8 GB); 0125's base checkpoint is stored
in float32 (`torch_dtype` in its config), so a default load of it takes about 28 GB, and 0924's
`fp32` branch holds 0924 in float32.

## The router's weights are the softmax's top 8, not renormalized

The router takes a softmax over all 64 logits in float32, keeps the 8 largest, and casts them back to
the model's dtype; `norm_topk_prob` is false on every checkpoint, so nothing rescales the 8 to sum to
one. `expert_weights` is exactly those 8 probabilities, slot 0 the largest, and there is no shared
expert: `routed_output` is `mlp_output`.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts], before the softmax
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

top = logits.float().softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values.to(w.dtype), w)    # the softmax's own entries
assert torch.equal(top.indices, idx)                     # slots in descending weight
```

On 3,072 tokens of English documentation and Python, a token's 8 weights sum to between 0.19 and
0.97, with a median of 0.30 to 0.37 on blocks 0 to 8 rising to 0.55 on block 15, and the median
routing entropy is 3.45 to 3.99 nats against 4.16 for a uniform router: on 14 of the 16 blocks
the median token's 8 chosen experts hold less than half its softmax mass. Each expert's output is
scaled by its own probability, so `mlp_output`'s size moves with how much mass the router puts on
the 8 it chose. A logit written at `router_logits` changes both the choice and the weights: forcing 8 experts to
+10 and the rest to −10 gives each about 1/8.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other 7 weights as they were. At the ablated block `mlp_output` changes on exactly the
tokens that chose `e`, and an expert no token chose has an effect of exactly zero:

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

This held at every expert of blocks 4, 8 and 12 on 0924 in float32, and at block 8 in bfloat16.
On `"The Eiffel Tower is in the city of"` (p(` Paris`) = 0.87) the largest single-expert effect on
log p(` Paris`) is 0.04, 0.06 and 0.07 nats at blocks 4, 8 and 12; zeroing block 8's whole
`mlp_output` moves it by 0.02. Rerouting a slot by writing `expert_indices` keeps the old slot's
weight and changes the expert's group sizes, so other tokens move by rounding (6e-5 in bfloat16).

## Expert usage is uneven and routing changes token to token

On the same 3,072 tokens, every block's busiest expert takes 4.5 to 6.6 times its even share
(1/64 of the slots), and its least-used one under a fifth of it; blocks 0 to 4 used all 64
experts, and deeper blocks left up to 4 unused on this text. Adjacent tokens share 2.4 of their
8 experts on average at block 0, and 3.8 to 4.3 at blocks 4, 8 and 15.

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `lm_head` and `embed_tokens` are separate weights. The tokenizer is
GPT-NeoX's: it prepends nothing, so `model.input_ids` is the prompt's tokens alone, and
`<|endoftext|>` (id 50279) is the end of text. The SFT, DPO and Instruct chat templates start
with `bos_token`, id 50279. On 0125-SFT, -DPO and -Instruct the tokenizer swaps two names: id
50279 decodes as `|||IP_ADDRESS|||` and the literal string `<|endoftext|>` encodes to id 0, so
write the id, not the string, when you build a prompt by hand.

## Intermediate checkpoints

`allenai/OLMoE-1B-7B-0924` keeps 244 pretraining checkpoints as Hub branches, from
`step5000-tokens20B` to `step1220000-tokens5117B`; `main` is annealed from `step1200000-tokens5033B`.
`allenai/OLMoE-1B-7B-0125` keeps three stage-2 runs, five checkpoints each, as
`stage2-ingredient{1,2,3}-step…-tokens…B`. Each loads with `revision=`:

```python
model = StandardizedTransformer("allenai/OLMoE-1B-7B-0924", revision="step10000-tokens41B")
```

The OLMoE paper (arXiv 2409.02060) analyses these checkpoints' routing: router saturation over
training, expert co-activation, and domain and vocabulary specialization.
"""
