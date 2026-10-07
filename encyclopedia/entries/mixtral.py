"""Mixtral: Mistral's block with a mixture of 8 experts, 2 per token, in place of the MLP."""

MODEL_TYPE = "mixtral"
TITLE = "Mixtral"
SUBTITLE = (
    "Mistral's pre-norm block with a mixture of 8 SwiGLU experts in place of the MLP: each token goes to "
    "2 of them, whose softmax weights are renormalized to sum to one, and no shared expert."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "mistralai/Mixtral-8x7B-v0.1"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MixtralForCausalLM"
CHECKPOINTS = [
    "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 263}
VLLM = True
QUIRKS = ["mixture-of-experts"]

#: Every number in the notes comes from a checkpoint's config or tokenizer; the shapes, identities and
#: snippets were run on the pinned tiny checkpoint (4 experts, top 2, 2 blocks), in float32 and bfloat16.
#: No real weights were run: the smallest checkpoint has 47B parameters.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv",
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
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. The tokenizer prepends "
             "<s>, so position 0 is the BOS token.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is "
            "lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))          # grouped-query attention, no window
out = h + mlp(post_attention_layernorm(h))       # 2 of 8 experts per token
```

Mistral's pre-norm block with Llama's names, and a mixture where Mistral has its MLP. Nothing
norms a sublayer's output, and the mixture has no shared expert, so `mlp_output` is
`routed_output` and the identity is the plain sum:

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

Every block of every checkpoint is a mixture. The experts are `intermediate_size` wide (14336 on
8x7B, 16384 on 8x22B), and there is no dense width beside it.

## The router renormalizes its top 2, in float32

The router (`gate`, aliased `router`) gives one logit per expert. It takes a softmax over the 8
in float32, keeps the 2 largest and divides them by their sum, so a token's two `expert_weights`
sum to one; there is no config flag for it. The weights stay float32 whatever the load dtype: in a
bfloat16 load `router_logits` and `routed_output` are bfloat16 and `expert_weights` float32.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, 8], before the softmax
    w = moe.expert_weights.save()           # [batch, seq, 2], float32
    idx = moe.expert_indices.save()

top = logits.float().softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values / top.values.sum(-1, keepdim=True), w)
assert torch.equal(top.indices, idx)        # slot 0 the larger weight
```

`moe.SCORING` is `"softmax"`. With two slots that sum to one, a token's weights are `p` and
`1 - p`: the mixture's output is a convex combination of two experts' outputs. `router_jitter_noise`
is 0 on every checkpoint, and the noise it would add runs only in training.

## Ablating an expert

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other weight as it was, so the token keeps one expert scaled by its own weight, below one.
`mlp_output` changes on exactly the tokens that chose `e`:

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

This holds on the pinned tiny checkpoint for every expert, in float32 and in bfloat16. To give
the token's remaining expert the whole weight, write 1 into the other slot as well.

## expert_outputs needs the default experts implementation

The 8 experts are one module, `experts`, with their weights stacked (`gate_up_proj`,
`down_proj`). `expert_outputs`, each slot's weighted output, is read inside transformers'
`grouped_mm` (the default) and `batched_mm` forwards; under `experts_implementation="eager"`
`support()` reports it missing and names the kwarg. The other mixture values read the same under
every implementation, and `expert_outputs` sums over its slot axis to `routed_output`.

## Attention has no window

Every checkpoint here sets `sliding_window` to `null`, so every block attends causally over the
whole prompt with transformers' plain causal mask. 8x7B has 32 query heads over 8 key/value
heads, 8x22B 48 over 8, so an edit to key/value head `j` reaches query heads `4j` to `4j + 3` on
8x7B and `6j` to `6j + 5` on 8x22B. `head_dim` is 128 on both, and no projection has a bias. The
attention interior needs `attn_implementation="eager"`. `rope_theta` is 10⁶ on every checkpoint.

## The tokenizer prepends a BOS, and the chat templates write one

The tokenizers prepend `<s>` (id 1), so position 0 is the BOS. The Instruct templates begin
the text with the string `<s>` (`<s> [INST] ... [/INST]` on 8x7B, `<s>[INST] ... [/INST]` on
8x22B), and a string passed to `model.trace` is tokenized with special tokens, so a templated
prompt starts with two BOS tokens. Tokenize it without them:

```python
text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
batch = model.tokenizer(text, add_special_tokens=False, return_tensors="pt")
with model.trace(dict(batch)):
    ids = model.input_ids.save()            # one <s>
```

8x22B-Instruct-v0.1 has 32768 tokens, the others 32000; its `[INST]` and `[/INST]` are single
control tokens.

## The readout

`model.logits` equals `model.lm_head.output`, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `lm_head` and `embed_tokens` are separate weights on every
checkpoint here.
"""
