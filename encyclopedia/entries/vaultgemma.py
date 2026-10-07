"""VaultGemma: Gemma 2's tree with the pre-norms only, so the contributions are the modules' outputs."""

MODEL_TYPE = "vaultgemma"
TITLE = "VaultGemma"
SUBTITLE = (
    "Gemma 2's names on a pre-norm block: each sublayer is normed on the way in only, so what reaches the "
    "residual stream is the module's own output, and the MLP's input norm is pre_feedforward_layernorm."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/vaultgemma-1b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-VaultGemmaForCausalLM"
CHECKPOINTS = ["google/vaultgemma-1b"]

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 168}
VLLM = False
QUIRKS = ["scaled-embeddings", "gain-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads × {head_dim}",
            "variants": {
                "full_attention": "full causal attention",
                "sliding_attention": "sliding window of {sliding_window} tokens",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_feedforward_layernorm",
            "pre_norm_note": "The MLP's input norm. nnterp also answers to post_attention_layernorm here, the name Llama "
                             "gives its pre-MLP norm; the block has no norm after the attention.",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_activation}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The embedding module multiplies its lookup by √hidden_size, cast to the weight's dtype (33.94 on 1B, "
             "34.0 in bfloat16), so token_embeddings is the scaled tensor and equals layers[0].input.",
    "norm": "VaultGemmaRMSNorm: the gain is 1 + norm.weight, not the stored weight.",
    "head": "lm_head shares its weight with embed_tokens on 1B (tie_word_embeddings). final_logit_softcapping is null "
            "on 1B, so logits equals lm_head.output there.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(pre_feedforward_layernorm(h))
```

Two RMSNorms per block, each before its sublayer, none after. The pre-MLP norm keeps Gemma 2's
name, `pre_feedforward_layernorm`, and nnterp aliases it `post_attention_layernorm`, the name Llama
gives the same place: both reach one module. There is no norm after the attention, so a hook
ported from Gemma 2 that reads `post_attention_layernorm` as the attention's post-norm lands on the
MLP's input norm here.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream
with nothing in between. The stream between the two adds is the pre-MLP norm's input. On the
pinned checkpoint, in float32:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mid = model.layers[1].pre_feedforward_layernorm.input.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn, mid)
torch.testing.assert_close(x + attn + mlp, out)
```

Scaling a module's output scales its contribution: `mlp.output[:] *= 0.5` and
`mlp.mlp_output[:] *= 0.5` are the same edit.

## The released checkpoint sets no softcap and no window

The modeling code carries Gemma 2's machinery: a score softcap inside the eager attention
(`attn_logit_softcapping`), a logit softcap after `lm_head` (`final_logit_softcapping`) and
`config.layer_types` with sliding blocks. On `google/vaultgemma-1b` both caps are `null` and all 26
blocks are `full_attention` (the config's `sliding_window` of 512 is read by no block), so
`model.logits` equals `model.lm_head.output`, `project_on_vocab` is `lm_head(norm(hidden))`, and
`attention_scores` are the plain scaled scores. The pinned tiny checkpoint sets both caps (50 and
30) and alternates sliding and full blocks: there `logits` is `30 · tanh(lm_head.output / 30)`, and
`project_on_vocab` applies the same cap.

## Attention

1B has 4 query heads and 4 key/value heads, each 256 wide, so the heads are 1024 wide against a
`hidden_size` of 1152. The query scale is `query_pre_attn_scalar ** -0.5`, which is
`256 ** -0.5`, equal to `head_dim ** -0.5` on 1B. With no score cap the default `sdpa` load and
`attn_implementation="eager"` run the same attention on 1B; eager serves the interior values. On a
config that sets `attn_logit_softcapping`, only eager applies the cap: transformers' `sdpa` path
takes no softcap.

## Embeddings and the norm gain

`embed_tokens` multiplies its lookup by √hidden_size, cast to the weight's dtype (33.94 on 1B, 34.0
in bfloat16), so `token_embeddings` is the scaled tensor and equals `layers[0].input`. Every
`VaultGemmaRMSNorm` multiplies by `1 + weight`: folding the final norm into the unembedding uses
`1 + model.norm._module.weight`. On 1B `lm_head` shares the weight of `embed_tokens`
(`tie_word_embeddings`), so an edit to the embedding matrix edits the readout.
"""
