"""Gemma 2: the first entry, and the reference for writing the others; PaliGemma 2 wraps it."""

import os
import tempfile

MODEL_TYPE = "gemma2"
TITLE = "Gemma 2 / PaliGemma 2"
SUBTITLE = (
    "Llama's module tree with a sandwich block: every sublayer is normed on the way in and on the way out, "
    "so what reaches the residual stream is a post-norm's output, and the logits are softcapped."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-2-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Gemma2ForCausalLM"
CHECKPOINTS = [
    "google/gemma-2-2b", "google/gemma-2-2b-it",
    "google/gemma-2-9b", "google/gemma-2-9b-it",
    "google/gemma-2-27b", "google/gemma-2-27b-it",
    # Vision-language wrappers around a Gemma 2 text model (text_config.model_type gemma2): model_type paligemma.
    "google/paligemma2-3b-pt-224", "google/paligemma2-3b-pt-448", "google/paligemma2-3b-pt-896",
    "google/paligemma2-3b-mix-224", "google/paligemma2-3b-mix-448", "google/paligemma2-3b-ft-docci-448",
    "google/paligemma2-10b-pt-224", "google/paligemma2-10b-pt-448", "google/paligemma2-10b-pt-896",
    "google/paligemma2-10b-mix-224", "google/paligemma2-10b-mix-448", "google/paligemma2-10b-ft-docci-448",
    "google/paligemma2-28b-pt-224", "google/paligemma2-28b-pt-448", "google/paligemma2-28b-pt-896",
    "google/paligemma2-28b-mix-224", "google/paligemma2-28b-mix-448",
]


def _tiny_wrapper() -> str:
    """The tiny PaliGemma 2 ``tests/families/test_gemma2.py`` builds (``_paligemma2_checkpoint``): no tiny PaliGemma 2
    is published, so the suite writes one into the temp dir, named after the snapshot of the tiny text checkpoint it
    wraps. The same path, so the wrapper's ``pinned`` is the suite's ``REPO``."""
    ref = os.path.expanduser(f"~/.cache/huggingface/hub/models--{PINNED.replace('/', '--')}/refs/main")
    snapshot = open(ref).read().strip() if os.path.exists(ref) else "missing"
    return os.path.join(tempfile.gettempdir(), f"nnterp-paligemma2-{snapshot}")


#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "paligemma": {
        "title": "PaliGemma 2",
        "pinned": _tiny_wrapper(),
        "projector": "multi_modal_projector: one linear with a bias, from the vision encoder's width to the text "
                     "model's, over vision.tower_output",
        "projector_input": "`vision.tower_output`",
        "notes": """
## PaliGemma 1's wrapper around a Gemma 2

PaliGemma 2 is PaliGemma 1's wrapper class (`PaliGemmaForConditionalGeneration`, `model_type`
`paligemma`) with the same SigLIP vision encoder (27 blocks, 1152 wide) and a Gemma 2 text model:
`gemma-2-2b` in 3B (26 blocks, 2304 wide), `gemma-2-9b` in 10B (42 blocks, 3584 wide) and
`gemma-2-27b` in 28B (46 blocks, 4608 wide). Every size comes at 224, 448 and 896 pixels (`pt`),
`mix` at 224 and 448, and `ft-docci` at 448 on 3B and 10B. The text model keeps Gemma 2's
sandwich block, its alternating sliding and full blocks and its attention softcap, so
`attn_implementation="eager"` is the faithful forward here too.

## The projector is one linear over `tower_output`

`model.multi_modal_projector` (the `projector`) is a single `linear` with a bias, from 1152 to the
text width (2304, 3584 or 4608). It reads the vision encoder's `last_hidden_state`, the patches
after `post_layernorm`, so `model.projector.input == model.vision.tower_output`, and a write to the
last block or to `tower_output` reaches the text model. `vision.image_features` is
`model.projector.output` flattened over the images. On the pinned tiny checkpoint:

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    out = model.vision.tower_output.save()
    fed = model.projector.input.save()
    projected = model.projector.output.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, out)                             # True: the projector reads tower_output
torch.equal(projected.flatten(0, 1), features)    # True
torch.equal(first[mask], features)                # True: the scatter
```

## One image token per patch

The projector maps each patch to one token, so an image is `(image_size // patch_size) ** 2` image
tokens: 256 on the 224 checkpoints (16 × 16 patches of 14 pixels), 1024 on the 448 checkpoints and
4096 on the 896 checkpoints (the processor's `image_seq_length`). The processor resizes every image
to the checkpoint's resolution, so every image of a checkpoint gives the same count.

## The image comes first, unscaled

The processor puts the image tokens before the beginning-of-sequence token, then the prompt and a
newline. The wrapper scatters `model.projector.output` into the output of `embed_tokens` as it is,
so the text rows of `layers[0].input` carry the √hidden_size scale and the image rows carry none:
`layers[0].input[~mask] == token_embeddings[~mask]`. The wrapper numbers positions from 1
(`position_ids` is `arange(seq) + 1`), so a text token sits one rotary step later than on a
`Gemma2ForCausalLM`.

## The prefix attends in both directions

The processor marks the image and the prompt as the prefix (`token_type_ids` 0) and any suffix
passed to it as 1. The wrapper's mask lets every prefix position attend to every other prefix
position, later ones included, on the sliding blocks as on the full ones: the window does not cut
the prefix. Only suffix tokens are causal, and on a sliding block a suffix token sees the last
`sliding_window` (4096) positions only, so on the 896 checkpoints, whose prefix is longer than
4096, the first image tokens fall outside its window on the even blocks. A trace with an image and
a prompt is bidirectional over the whole sequence: on the pinned tiny checkpoint block 0's and
block 1's `attention_probabilities` are nonzero everywhere. A text-only trace with the tokenizer's
encoding passes no `token_type_ids` and is causal.

## The logits are not softcapped

The text config sets `final_logit_softcapping` to 30, but `PaliGemmaForConditionalGeneration`
never applies it: `model.logits` is `model.lm_head.output`, and `project_on_vocab` follows the
class, so a logit lens at the last block equals `logits` here too. The attention softcap still
applies inside the eager attention.

## A text-only trace takes the tokenizer's encoding

PaliGemma's processor refuses a prompt without an image. A text-only trace of the wrapper takes
the tokenizer's encoding instead; the mask is then all false and `vision.image_features` is never
reached.

```python
with model.trace(dict(model.tokenizer(text, return_tensors="pt"))):
    logits = model.logits.save()
```
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 156, "paper": "#EFE8CE"}
QUIRKS = ["sandwich-norms", "softcapped-logits", "sliding-window", "scaled-embeddings", "gain-norm"]

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
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads, head_dim {head_dim}; "
                      "scores softcapped at {attn_logit_softcapping} under eager",
            "variants": {
                "sliding_attention": "sliding window of {sliding_window} tokens",
                "full_attention": "full causal attention",
            },
            "post_norm_note": "On Gemma 2 this norm follows the attention. On Llama the same name is the norm before the MLP. "
                              "layer_mid is the stream after this norm's output is added, which the MLP's pre-norm reads.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_feedforward_layernorm",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, gelu_pytorch_tanh",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The embedding module multiplies by √hidden_size (48.0 on 2B) itself, so token_embeddings is the "
             "scaled tensor and equals layers[0].input.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "A tanh softcap at 30: logits = 30 · tanh(lm_head.output / 30). project_on_vocab applies it too. "
              "PaliGemma 2's wrapper applies none: there logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
```

Four RMSNorms per block, two per sublayer. The names are Llama's where Llama has them, and
that is the trap: on Llama `post_attention_layernorm` is the norm *before* the MLP; on Gemma 2 it
is the norm *after* the attention, and the pre-MLP norm is `pre_feedforward_layernorm`. nnterp
keeps the native names, so write `pre_feedforward_layernorm` when you mean the MLP's input norm.

## The contributions are the post-norms' outputs

`attention_output` and `mlp_output` are what the block adds to the stream, and on this family
that is a norm's output, not the module's. `self_attn.output[0]` is the tensor *entering*
`post_attention_layernorm`; `attention_output` is the one leaving it. The identity still holds
exactly, in float32 and bfloat16 alike:

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

**RMSNorm is scale-invariant, so scale the contribution, not the module.** `post_norm(0.5 * y)`
equals `post_norm(y)` up to `eps`: halving `mlp.output` does nothing to the stream, and zeroing
it happens to work only because the norm of zero is zero. Partial ablations and steering of a
sublayer's effect go through `mlp_output` / `attention_output`, which are the tensors the block
actually adds:

```python
with model.trace(prompt):
    model.layers[5].mlp.mlp_output[:] *= 0.5          # halves what the block adds

with model.trace(prompt):
    model.layers[5].mlp.output[:] *= 0.5              # a no-op: the post-norm rescales it back
```

The same holds for direct logit attribution and for any decomposition that writes a block as a
sum of sublayer terms: the terms are the post-norm outputs. The input side is ordinary:
`self_attn.input` is `input_layernorm`'s output, `mlp.input` is `pre_feedforward_layernorm`'s.

## Load with eager, and not only for the pattern

Gemma 2 softcaps its attention scores, `50 · tanh(qkᵀ / 50)`, inside the eager and flex
attention paths. transformers' sdpa path ignores the `softcap` argument, so a default load
(`sdpa`) runs a different attention than the model was trained with. `attn_implementation="eager"`
is the faithful forward here, and it is also what makes the attention interior readable:

```python
model = StandardizedTransformer("google/gemma-2-2b", dispatch=True, attn_implementation="eager")
```

`attention_scores` are read after the softcap and the mask, at the softmax's input, so
`softmax(attention_scores)` reproduces `attention_probabilities` up to the dtype cast. The
query scale is `query_pre_attn_scalar ** -0.5`, not `head_dim ** -0.5`. On 2B and 9B both are
`256 ** -0.5`; on 27B the scale is `144 ** -0.5` with a `head_dim` of 128.

## Grouped-query attention

8 query heads share 4 key/value heads on 2B. `attention_keys` and `attention_values` are served
before `repeat_kv`, `[batch, 4, seq, 256]`, so an edit to key/value head `j` reaches query heads
`2j` and `2j + 1`. `attention_queries`, `attention_scores` and `attention_probabilities` are
`num_heads` wide.

## Sliding and full attention alternate

`config.layer_types` is `sliding_attention` on even blocks and `full_attention` on odd ones; the
window is 4096 tokens. On a prompt shorter than the window the two masks coincide and the two
kinds of block are not distinguishable from their patterns. The block's own kind is
`model.layers[i].self_attn._module.sliding_window` (`4096` or `None`).

## Softcapped logits

`model.logits` is `30 · tanh(lm_head.output / 30)`; `model.lm_head.output` is the raw projection.
`project_on_vocab` applies the cap, so a logit lens at the last block equals `logits` exactly,
and lens probabilities at earlier blocks are comparable with the model's own. Take KL
divergences on `logits`, not on `lm_head.output`: the cap changes the distribution.

```python
with model.trace(prompt):
    resid = model.layers[12].layer_output[:, -1].save()
    logits = model.logits[:, -1].save()

lens = model.project_on_vocab(resid)                 # softcap applied, like the model's own readout
```

## The norm gain is 1 + weight

`Gemma2RMSNorm` multiplies the normalized input by `1 + weight`. Folding the final norm into
the unembedding, or reading a norm's gain for any reason, uses `1 + model.norm._module.weight`,
not the stored weight, whose values sit near zero.

## Embeddings are scaled before block 0

The embedding module multiplies its lookup by √hidden_size (48.0 on 2B), so
`token_embeddings` is the scaled tensor and equals `layers[0].input`. The stored embedding
weight is shared with `lm_head` (`tie_word_embeddings`), so an edit to `embed_tokens.weight`
is an edit to the unembedding.

## Sparse autoencoders

Gemma Scope's SAEs were trained on Gemma 2 (2B, 9B, 27B). Their residual-stream SAEs read the
stream after a block, which is `model.layers[i].layer_output`. For a sublayer SAE check whether
it was trained before or after the post-norm: `self_attn.output[0]` and `attention_output` are
different tensors here, and only the second is what the stream receives.

## What this family module covers

Every checkpoint whose config says `model_type` `gemma2`: Gemma 2 2B, 9B and 27B with their
instruction-tuned `-it` releases, and the text model inside PaliGemma 2, whose
`text_config.model_type` is `gemma2`. PaliGemma 1 wraps a Gemma 1 text model and is the `gemma`
family's.
"""
