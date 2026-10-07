"""OLMo 2: a post-norm block, and queries and keys RMS-normed across all heads."""

MODEL_TYPE = "olmo2"
TITLE = "OLMo 2"
SUBTITLE = (
    "Llama's tree with the norms moved after the sublayers: attention and MLP read the raw residual stream, "
    "what reaches the stream is a post-norm's output, and the queries and keys are RMS-normed across all heads "
    "before the rotary embedding."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/OLMo-2-0425-1B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Olmo2ForCausalLM"
CHECKPOINTS = [
    "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT", "allenai/OLMo-2-0425-1B-DPO",
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B", "allenai/OLMo-2-1124-13B-SFT", "allenai/OLMo-2-1124-13B-DPO",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B", "allenai/OLMo-2-0325-32B-SFT", "allenai/OLMo-2-0325-32B-DPO",
    "allenai/OLMo-2-0325-32B-Instruct",
]

#: Checkpoints that ship their tokenizer as `vocab.json` and `merges.txt` only, with no `tokenizer.json`.
GPT2_FILES_ONLY = {"allenai/OLMo-2-0325-32B-SFT"}


def load(checkpoint, **kwargs):
    """The page's model, built on the meta device. ``allenai/OLMo-2-0325-32B-SFT`` has no ``tokenizer.json``,
    and transformers' ``AutoTokenizer`` maps ``olmo2`` to the generic ``TokenizersBackend``, which cannot
    build from ``vocab.json`` and ``merges.txt``; the GPT-2 tokenizer its ``tokenizer_config.json`` names
    can, so it is loaded by type and handed over."""
    from transformers import AutoTokenizer

    from nnterp import StandardizedTransformer

    if checkpoint in GPT2_FILES_ONLY:
        kwargs["tokenizer"] = AutoTokenizer.from_pretrained(checkpoint, tokenizer_type="gpt2")
    return StandardizedTransformer(checkpoint, **kwargs)


VLLM = False
QUIRKS = ["post-norms", "qk-norm"]

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
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, q_norm and k_norm",
            "post_norm_note": "On OLMo-2 this norm follows the attention. On Llama the same name is the norm "
                              "before the MLP; OLMo-2 has no norm before either sublayer.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "norm": "A plain RMSNorm: the gain is norm.weight (not 1 + weight), eps 1e-6. project_on_vocab applies it.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every size.",
}

NOTES = """
## The block, in order

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

Two RMSNorms on the block, both after a sublayer, none before. `post_attention_layernorm` has
Llama's name and the opposite place: on Llama it is the norm before the MLP, here it follows the
attention, and nothing norms the MLP's input. There is no `input_layernorm`, so code that hooks
`input_layernorm.output` to read what the attention sees has nothing to attach to.

## The sublayers read the raw stream

`self_attn.input` is `layers[i].input` itself and `mlp.input` is
`layers[i].input + attention_output`, unnormalized. A probe or dictionary trained on a
sublayer's input sees the stream at its own scale, which grows with depth.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn_in = model.layers[1].self_attn.input.save()           # equals x
    attn = model.layers[1].self_attn.attention_output.save()
    mlp_in = model.layers[1].mlp.input.save()                  # equals x + attn
```

The attention's contribution depends only on the direction of its input. `q_norm` and `k_norm`
remove the scale from the queries and keys, the value and output projections are linear with no
bias, and `post_attention_layernorm` removes the scale again: doubling what the attention reads
changes `attention_output` by about 1e-5 relative on 1B. A steering vector added to the stream
reaches the attention only by turning the stream's direction. The MLP has no such invariance:
doubling what it reads changes `mlp_output` by 19 to 49% on 1B's blocks 2, 8 and 14.

## The contributions are the post-norms' outputs

`attention_output` and `mlp_output` are what the block adds, the outputs of
`post_attention_layernorm` and `post_feedforward_layernorm`, not of the modules. The identity
holds exactly, in float32 and bfloat16 alike:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

**RMSNorm is scale-invariant, so scale the contribution, not the module.** On 1B, halving
`mlp.output` at block 8 moves `layer_output` by at most 4e-4, while halving `mlp_output` moves it
by 1.2; halving `self_attn.output[0]` moves it by 1e-5. Zeroing the module output does work,
because the norm of zero is zero. Partial ablation and steering of a sublayer's effect go through
`mlp_output` / `attention_output`:

```python
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:] *= 0.5     # halves what the block adds

with model.trace(prompt):
    model.layers[1].mlp.output[:] *= 0.5         # close to a no-op on 1B
```

Direct logit attribution and any per-sublayer decomposition use the post-norm outputs as the
terms. The post-norms' gains are small and set the size of each term: their means run from about
0.08 to 0.37 on the attention side and from about 0.15 to 0.62 on the MLP side across 1B's 16 blocks,
largest in the last blocks.

## Queries and keys are normed across all heads

`q_norm` and `k_norm` are RMSNorms over the whole projection, `num_heads * head_dim` (2048 on 1B)
and `num_kv_heads * head_dim`, applied before the heads are split and before the rotary embedding.
`attention_queries` and `attention_keys` are read after both. Because one RMS is shared by every
head, an edit to one head's slice of `q_proj.output` reaches the other heads: tripling head 0's
slice at block 8 of 1B changes head 1's queries by 15%. Edit a single head at
`attention_queries` or `attention_keys`, which no norm follows.

```python
with model.trace(prompt):
    model.layers[1].self_attn.attention_queries[:, 0] = 0   # head 0 only
```

Every size but 32B has as many key/value heads as query heads; 32B has 40 query heads over 8
key/value heads, so there an edit to key/value head `j` reaches query heads `5j` to `5j + 4`.

## Load with eager for the attention interior only

The attention interior needs `attn_implementation="eager"`. The default `sdpa` load computes the
same function: on 1B in float32 the two loads' logits differ by at most 5e-5. There is no softcap,
sink or window.

## The readout is lm_head after a plain RMSNorm

`logits` is `lm_head.output`, with no cap or scale after it, and `project_on_vocab` is
`lm_head(norm(h))`, equal to `logits` on the last block's `layer_output`. The final norm's gain is
`model.norm._module.weight` itself, between 0.92 and 5.11 on 1B (mean 2.26), and `eps` is 1e-6.
`lm_head` and `embed_tokens` are separate weights on every size.

## The tokenizer adds no BOS

The base tokenizer prepends nothing: `model.input_ids` is the prompt's tokens alone.
`<|endoftext|>` (id 100257) is both `bos_token` and `eos_token`. The chat templates of the 1B, 7B
and 13B instruct checkpoints start with it; 32B-Instruct's does not. `token_embeddings` is the
unscaled lookup and equals `layers[0].input`.

## Intermediate checkpoints and kin

Each base repository keeps its pretraining checkpoints as Hub branches, `stage1-step…-tokens…B`
and `stage2-ingredient…-step…-tokens…B`, which load with `revision=`:

```python
model = StandardizedTransformer(
    "allenai/OLMo-2-0425-1B", revision="stage1-step10000-tokens21B"
)
```

The older `olmo` family is a pre-norm Llama block with a weightless LayerNorm, `olmo3` keeps this
block and mixes sliding-window and full attention, and `olmo_hybrid` puts gated DeltaNet blocks
between post-norm attention blocks; each is its own nnterp family.
"""
