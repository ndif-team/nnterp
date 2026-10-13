"""Ministral: Llama's block with full-attention and sliding-window blocks interleaved by config.layer_types."""

MODEL_TYPE = "ministral"
TITLE = "Ministral"
SUBTITLE = (
    "Llama's pre-norm block with grouped-query attention, each block attending either over every earlier "
    "position or over a sliding window, as config.layer_types says."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Its config.json says `mistral`; AutoConfig reads a `mistral` config with `layer_types` as `ministral`.
REFERENCE = "mistralai/Ministral-8B-Instruct-2410"
#: The tiny checkpoint the test suite builds the page from (its test rewrites it into a local copy with a tokenizer).
PINNED = "hf-tiny-v2/tiny-random-MinistralForCausalLM"
CHECKPOINTS = ["mistralai/Ministral-8B-Instruct-2410"]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 266}
VLLM = False
QUIRKS = ["sliding-window"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
            "variants": {
                "full_attention": "full causal attention",
                "sliding_attention": "sliding window of {sliding_window} tokens",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm, "
                             "and its input is the stream between the two adds.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding is added, so token_embeddings equals "
             "layers[0].input. Positions enter as a rotation of queries and keys inside each attention.",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "Untied from embed_tokens. Nothing follows it: logits equals lm_head.output, and project_on_vocab "
            "is lm_head(norm(hidden)).",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's block: two RMSNorms per block, each before its sublayer, none after, and a SwiGLU MLP.
`attention_output` is the attention module's output and `mlp_output` the MLP's, added to the stream
with nothing in between, so the identity holds exactly (float32, pinned checkpoint):

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Full and sliding blocks, and a window as long as the context

`config.layer_types` gives each block `full_attention` or `sliding_attention`. On
Ministral-8B-Instruct-2410 block 0 and every fourth after it (0, 4, ..., 32) are full, the other 27
sliding; the window is `config.sliding_window`, 32768 tokens, which is also the config's
`max_position_embeddings`. A sliding block's query attends to its own position and the 32767 before
it, so up to that length the sliding mask is the causal mask and the two kinds of block compute the
same function. The window is in the mask the attention receives, so under
`attn_implementation="eager"` `attention_probabilities` is exactly zero beyond it (checked on the
pinned checkpoint loaded with `sliding_window=3`: every entry three or more positions back is
`0.0`). Each block's own window is on its module:

```python
[layer.self_attn._module.sliding_window for layer in model.layers][:5]
# [None, 32768, 32768, 32768, None] on 8B-Instruct-2410
```

## Grouped-query attention

32 query heads over 8 key/value heads of width 128, 4096 wide together, the stream's width.
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, 8, seq, 128]`, and
query head `h` reads key/value head `h // 4`: an edit to key/value head `j` reaches query heads
`4j` to `4j + 3`.

## Target tokens

The tokenizer is the 131072-token Tekken vocabulary: `" Paris"` is `ĠParis` (`6993`) and `"Paris"`
is `Paris` (`42572`), two different tokens. Pick the one the prompt's spacing produces:

```python
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1
```

## What this family module covers

`mistralai/Ministral-8B-Instruct-2410`'s `config.json` says `model_type` `mistral` and
`MistralForCausalLM`, with `layer_types`; transformers' `AutoConfig` reads a `mistral` config that
has `layer_types` as `ministral`, so it loads as `MinistralForCausalLM` and this family serves it.
The tiny `ministral` checkpoints on the Hub ship only `tekken.json`, so the suite's pinned checkpoint
is re-initialised with the tiny Mistral checkpoint's tokenizer and vocabulary.
"""
