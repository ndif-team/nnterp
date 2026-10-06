---
title: Vision-language models
one_liner: "Load an image-text-to-text checkpoint with `task=\"image-text-to-text\"`, pass an image, and read the tower (`model.vision`, `vision.layers[i]`), the `projector`, and the tower's `vision.image_token_mask` and `vision.image_features`, where `layers[0].input[vision.image_token_mask] == vision.image_features`."
tags: [usage, vision, multimodal, image-text-to-text, vision tower, projector, image_features, image_token_mask, Patches, siglip, clip, llava, gemma3, llama4, gemma4, encoder-free]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/layouts.md, docs/developing/vision-design.md, docs/reference/families.md]
sources: [nnterp/components/vision.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, nnterp/families/llama4_text.py, nnterp/families/gemma4_text.py, nnterp/families/gemma4_unified_text.py, tests/families/vision_suite.py, tests/families/test_gemma3_text.py, tests/families/test_llama.py, tests/families/test_llama4_text.py, tests/families/test_gemma4_text.py, tests/families/test_gemma4_unified_text.py]
tags: [usage, vision, multimodal, image-text-to-text, vision tower, projector, image_features, image_token_mask, Patches, siglip, clip, llava, gemma3, qwen-vl, packed tower, deepstack, m-rope]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/layouts.md, docs/developing/vision-design.md, docs/reference/families.md]
sources: [nnterp/components/vision.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, nnterp/families/qwen2_vl_text.py, nnterp/families/qwen2_5_vl_text.py, nnterp/families/qwen3_vl_text.py, nnterp/families/qwen3_vl_moe_text.py, nnterp/families/qwen3_5_text.py, nnterp/families/qwen3_5_moe_text.py, tests/families/vision_suite.py, tests/families/qwen_vision_suite.py, tests/families/test_gemma3_text.py, tests/families/test_llama.py, tests/families/test_qwen3_vl_text.py]
tags: [usage, vision, multimodal, image-text-to-text, vision tower, projector, image_features, image_token_mask, Patches, siglip, clip, pixtral, llava, llava-next, llava-onevision, paligemma, idefics3, smolvlm, aya-vision, mistral3, gemma3]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/layouts.md, docs/developing/vision-design.md, docs/reference/families.md]
sources: [nnterp/components/vision.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, nnterp/families/gemma.py, nnterp/families/qwen2.py, nnterp/families/cohere2.py, nnterp/families/mistral.py, nnterp/families/ministral3.py, tests/families/vision_suite.py, tests/families/test_gemma3_text.py, tests/families/test_llama.py, tests/families/test_mistral.py]
---

# Vision-language models

## What this is for

An image-text-to-text checkpoint (Gemma 3, Llava, Qwen3.5, Mistral 3, ...) is a text model
plus a vision tower, a projector, and a step that scatters the projected image features
into the text stream at the image tokens. nnterp keeps the text model's standard names
(the family is the text model's: `gemma3_text`, `llama`, ...) and adds names for the
vision side: the tower, its blocks, the projector, and two values of the tower for where
the image enters the text model.

## Canonical pattern

Load the wrapper with its processor by passing `task="image-text-to-text"`; without it
nnterp loads for `text-generation`, which is the text-only load.

```python
import torch
from PIL import Image
from nnterp import StandardizedTransformer

model = StandardizedTransformer("llava-hf/llava-1.5-7b-hf", task="image-text-to-text", dispatch=True, attn_implementation="eager")
image = Image.new("RGB", (64, 64), "red")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()                  # [batch, seq] bool
    pattern = model.vision.layers[0].self_attn.attention_probabilities.save()   # [images, heads, patches, patches]
    patches = model.vision.layers[0].layer_output.save()          # [images, patches, vision_hidden]
    features = model.vision.image_features.save()                # [image_tokens, hidden]
    first = model.layers[0].input.save()
    logits = model.logits.save()

torch.equal(first[mask], features)                    # True: the features are what enters the text model
model.vision.num_layers, model.vision.hidden_size     # the tower's sizes; model.num_layers is the text model's

with model.trace(prompt, images=[image]):
    model.vision.image_features[:] = 0                # ablate every image token at once
    ablated = model.logits.save()
```

Run on `trl-internal-testing/tiny-LlavaForConditionalGeneration`: `mask` is `(1, 592)`
with 576 image tokens, `pattern` `(1, 4, 577, 577)`, `patches` `(1, 577, 16)` (CLIP's CLS
token first), `features` `(576, 16)`. The same body runs on Gemma 3
(`google/gemma-3-4b-pt`) and Gemma 4 (`google/gemma-4-E2B`, whose base checkpoint has no
chat template: write the prompt as `f"{model.processor.image_token} ..."`). On E2B with a
red 224x224 image, under `torch.no_grad()` in bfloat16: 256 image tokens, `pattern`
`(1, 12, 2520, 2520)` (2304 patches and 216 padded rows), `features` `(256, 1536)`, the
scatter equality exact, the next token `" red"`, and `" of"` once `image_features` is zeroed;
it peaked at 11 GB.

## The names

| standard name | what it is | Gemma 3 (SigLIP) | Llava 1.5 (CLIP) | Llama 4 (ViT) | Gemma 4 (ViT) | Gemma 4 unified (encoder-free) |
| --- | --- | --- | --- | --- | --- | --- |
| `model.vision` | the tower's root, a `Vision` | `model.model.vision_tower` | `model.model.vision_tower` | `model.vision_model` | `model.model.vision_tower` | `model.model.embed_vision`, the image embedder |
| `model.vision.patch_embed` | the patch embedding | `vision_tower.embeddings.patch_embedding` | same | `vision_model.patch_embedding` (unfold + linear) | `vision_tower.patch_embedder` (a linear plus 2D position embeddings) | `embed_vision.patch_dense` |
| `model.vision.layers[i]` | the tower's blocks, `VisionLayer` | `vision_tower.encoder.layers[i]` | same | `vision_model.model.layers[i]` | `vision_tower.encoder.layers[i]` | none: no blocks |
| `vision.layers[i].self_attn`, `.mlp` | `VisionAttention`, `VisionMlp` | native | native | native | native; the contributions are the post-norms' outputs (a sandwich block) | none |
| `vision.layers[i].input_layernorm`, `.post_attention_layernorm` | the block's norms | `layer_norm1`, `layer_norm2` | same | native | native: `post_attention_layernorm` *follows* the attention, as on the text block | none |
| `model.vision.norm` | the final norm over the patches | `post_layernorm` | none: CLIP's `post_layernorm` norms only the pooled CLS token | `layernorm_post` | none | none |
| `model.projector` | the module whose output is scattered into the text stream | `model.multi_modal_projector` (pools 4096 patches to 256 tokens) | `model.multi_modal_projector` | `model.multi_modal_projector` (a linear) | `model.model.embed_vision` (an RMS norm and a linear) | `embed_vision.multimodal_embedder` (an RMS norm and a linear) |
| standard name | what it is | Gemma 3 (SigLIP) | Llava 1.5 (CLIP) | Qwen-VL, Qwen3.5 (Qwen ViT) |
| --- | --- | --- | --- | --- |
| `model.vision` | the tower's root, a `Vision` | `model.model.vision_tower` | `model.model.vision_tower` | `model.model.visual` (a `QwenVision`) |
| `model.vision.patch_embed` | the patch embedding | `vision_tower.embeddings.patch_embedding` | same | `visual.patch_embed` |
| `model.vision.layers[i]` | the tower's blocks, `VisionLayer` | `vision_tower.encoder.layers[i]` | same | `visual.blocks[i]` (`PackedVisionLayer`) |
| `vision.layers[i].self_attn`, `.mlp` | `VisionAttention`, `VisionMlp` | native | native | `attn`, `mlp` (`PackedVisionAttention`, `PackedVisionMlp`) |
| `vision.layers[i].input_layernorm`, `.post_attention_layernorm` | the pre-norms | `layer_norm1`, `layer_norm2` | same | `norm1`, `norm2` |
| `model.vision.norm` | the final norm over the patches | `post_layernorm` | none: CLIP's `post_layernorm` norms only the pooled CLS token | none: the merger norms its own input |
| `model.projector` | the module whose output is scattered into the text stream | `model.multi_modal_projector` (pools 4096 patches to 256 tokens) | `model.multi_modal_projector` | `visual.merger`, inside the tower (folds each 2x2 block of patches into one token) |
token first), `features` `(576, 16)`. The same body runs on every wrapper in
[the coverage list](#which-wrappers), from Gemma 3 (`google/gemma-3-4b-pt`) to Mistral 3.

## The names

| standard name | what it is | SigLIP (Gemma 3, PaliGemma, llava-interleave, LLaVA-OneVision, Aya Vision, Cohere2-Vision) | CLIP (Llava 1.5, VipLlava, LLaVA-NeXT, BakLLaVA) | Pixtral (Mistral 3, Pixtral-12B) |
| --- | --- | --- | --- | --- |
| `model.vision` | the tower's root, a `Vision` | `model.model.vision_tower` (DeepSeek-VL, Idefics 3, SmolVLM: `model.model.vision_model`) | `model.model.vision_tower` | `model.model.vision_tower`, a `PixtralVision` |
| `model.vision.patch_embed` | the patch embedding | `embeddings.patch_embedding` | same | `patch_conv` |
| `model.vision.layers[i]` | the tower's blocks, `VisionLayer` | `encoder.layers[i]` | same | `transformer.layers[i]` |
| `vision.layers[i].self_attn`, `.mlp` | `VisionAttention`, `VisionMlp` | native | native | `attention`, `feed_forward` |
| `vision.layers[i].input_layernorm`, `.post_attention_layernorm` | the pre-norms | `layer_norm1`, `layer_norm2` | same | `attention_norm`, `ffn_norm` |
| `model.vision.norm` | the final norm over the patches | `post_layernorm` | none: CLIP's `post_layernorm` norms only the pooled CLS token | none |
| `model.projector` | the last module before the scatter | `model.multi_modal_projector` (Gemma 3's pools 4096 patches to 256 tokens; Aya Vision's and Cohere2-Vision's pixel-shuffle); DeepSeek-VL: `model.aligner`; Idefics 3, SmolVLM: `model.connector` (pixel shuffle) | `model.multi_modal_projector` | `model.multi_modal_projector` (merges each 2x2 block of patches on Mistral 3) |

Idefics 3's and SmolVLM's ViT is SigLIP-shaped and named as SigLIP. A family that hosts the
tower at two paths keys both, as it keys both spellings of the text stack.

`embed_tokens`, `layers`, `norm` and `lm_head` stay the language model's
(`model.language_model.*` on the wrapper; `language_model.model.*` on Llama 4's), and
native names keep working. Gemma 4's audio tower and its embedder keep their native names
(`model.model.audio_tower`, `model.model.embed_audio`), and an audio prompt
(`model.trace(prompt, audio=[waveform])`) runs as before.

## The values

| value | host | layout | meaning |
| --- | --- | --- | --- |
| `layer_output` | `vision.layers[i]` | `Patches` | the tower's stream leaving the block |
| `attention_output`, `mlp_output` | `vision.layers[i].self_attn`, `.mlp` | `Patches` | what each sublayer adds: `input + attention_output + mlp_output == layer_output` |
| `attention_probabilities`, `attention_queries`, ... | `vision.layers[i].self_attn` | `Pattern`, `Queries`, ... | as on a text block, the `batch` axis being the tower's images; unmasked (the tower attends both ways); needs eager |
| `patch_embeddings` | `vision` | `Patches` | the patch embedding's output, one row per patch, before position embeddings (and CLIP's CLS token and pre-norm) |
| `tower_output` | `vision` | `Patches` | what the tower returns over the patches (`last_hidden_state`) |
| `image_token_mask` | `vision` | `ImageTokenMask` `[batch seq]` bool | `input_ids == config.image_token_id`, read off the model's inputs; read-only |
| `image_features` | `vision` | `ImageFeatures` `[image_tokens hidden]` | the projector's output, flat over every image token in row-major order; assignable, in-place edits land |

`Patches` is `[images, patches, vision_hidden]`: one row per image, the image's patches in
raster order. Per tower:

| tower | rows | the patch axis |
| --- | --- | --- |
| SigLIP (Gemma 3) | one per image | the patches |
| CLIP (Llava 1.5) | one per image | the CLS token *first*, then the patches |
| Llama 4's ViT | one per image tile | the patches, then the CLS token *last* (`patches + 1` rows); the tower drops it after `vision.norm` |
| Gemma 4's ViT | one per image | the patches *padded* to `max_soft_tokens * pooling_kernel_size**2` rows (2520 by default). The padded rows (zero pixels at position `(-1, -1)`, `image_position_ids` in the processor's encoding) are masked as keys but run through every block, so they are rows of `layer_output`, with values; the pooler zeroes and strips them |
| Gemma 4 unified (encoder-free) | one per image | `patch_embeddings` and `tower_output` only, padded to `max_soft_tokens` rows (280) the same way; one row per image token once stripped |

The tower's sizes are on `model.vision`: `num_layers`, `hidden_size`, `num_heads`,
`head_dim`, `intermediate_size`, `patch_size`, `image_size`. Gemma 4's tower takes
variable-resolution images, so its `image_size` raises `Unavailable`, and its `head_dim`
is the config's (64 on E2B, with 12 heads over a 768-wide stream). The encoder-free
embedder has `num_layers == 0`, `hidden_size` its `mm_embed_dim` and `patch_size` the
48-pixel merged patch it embeds; its `num_heads`, `head_dim`, `intermediate_size` and
`image_size` raise `Unavailable`.
| `image_features` | `vision` | `ImageFeatures` `[image_tokens hidden]` | what the wrapper scatters into the token embeddings, flat over every image token in row-major order; assignable, in-place edits land |

`Patches` is `[images, patches, vision_hidden]`: one row per image (per crop on LLaVA-NeXT
and LLaVA-OneVision, per tile on Idefics 3 and SmolVLM), the image's patches in raster
order (CLS first on CLIP). Pixtral is *packed*: one row holding every image's patches, image
after image, `[1, all patches, vision_hidden]`; each image has `(height // patch_size) *
(width // patch_size)` of them, its `image_sizes` entry from the processor. Its blocks
attend under a block-diagonal mask, so `attention_probabilities` is `[1, heads, all patches,
all patches]`, zero between images, and its `patch_embeddings` is the packed row as it
enters `ln_pre` (the convolution's output, `patch_embed.output`, is the padded grid). The tower's sizes are on `model.vision`: `num_layers`,
`hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size`.

`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly:
`image_features` is read at the scatter, the tensor the wrapper's forward writes into the
token embeddings at the image tokens, and nothing touches it before block 0. So
`vision.image_features` is the place to ablate, patch or steer the image as the text model
sees it. On most wrappers it is the projector's output, reshaped; on LLaVA-NeXT and
LLaVA-OneVision it is not, since they unpad the projector's output and add a newline token
per row (`model.model.image_newline`), so `projector.output` has another row count there.
Both values are the tower's although neither is read inside it: the mask comes off the
model's inputs and the features off the wrapper's forward.

What feeds the projector differs per host: Gemma 3 pools `tower_output`; Llava takes
`vision.layers[-2].layer_output` without its CLS token (`vision_feature_layer=-2`), so a
write to `tower_output` or the last block does not reach Llava's text model.
`model.projector.input` is what the projector actually receives. `tower_output` is always
the last block's stream after `vision.norm` where there is one, before any pooling: on
Llama 4 that is `layernorm_post`'s output, CLS included, and the tower then runs a
pixel-shuffle adapter (`vision.vision_adapter`, a quarter as many rows) whose output,
flattened over the tiles, is `projector.input`; on Gemma 4 it is the encoder's output over
the padded patches, and the tower's `pooler` average-pools 3x3 patches into each soft
token and strips the padding, which is `projector.input`; on the encoder-free embedder it
is the states before the projection, `projector.input` itself.

On Gemma 4 unified the projector runs on the padded rows too, and the wrapper strips them
before scattering, so `vision.image_features` is read at the scatter
(`inputs_embeds.masked_scatter` in the wrapper model's forward), not at the projector:
`image_features == projector.output[valid]`, where `valid` is
`(image_position_ids != -1).all(-1)`.
write to `tower_output` or the last block does not reach Llava's text model; VipLlava
concatenates several blocks' streams. `model.projector.input` is what the projector
actually receives.

## Inputs

- `model.trace(prompt, images=[image])`, the image placeholder in the prompt (the
  processor's chat template puts it there).
- `model.trace(encoding)` with an encoding built by `model.processor`.
- `model.trace("text")`: a text-only trace of the wrapper works (the mask is all false,
  `vision.image_features` is never reached), except on PaliGemma, whose processor demands an image;
  pass `dict(model.tokenizer(text, return_tensors="pt"))` there.
- One invoke per trace while it carries an image; several images go in one invoke, as lists.
- Llama 4's image processor returns bfloat16 pixels, which a float32 tower refuses (as in
  plain transformers): load Llama 4 in bfloat16, or cast `pixel_values` in an encoding.

Gemma 3's tower attends over 4096 patches, so under eager every block's pattern is
16 x 4096 x 4096. A trace keeps them all for autograd unless it runs under
`torch.no_grad()`: on `google/gemma-3-4b-pt` an eager trace without it ran out of a 48 GB
card, and with it peaked at 11 GB.

Read order is the forward's: `vision.image_token_mask` first (it comes off the inputs,
like `input_ids`), then the tower's values, a block's attention interior before its
`layer_output`, then `vision.image_features`, then the text model's.

## A packed tower: the Qwen ViT

Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5 and Qwen3.5-MoE share one tower at
`model.visual`, and it is *packed*: it runs on `[patches, vision_hidden]`, every image of
the invoke concatenated, and its attention calls the interface once per image (once per
window on Qwen2.5-VL's windowed blocks). So:

- Its `Patches` values (`patch_embeddings`, `tower_output`, `layer_output`,
  `attention_output`, `mlp_output`) are `[1, patches, vision_hidden]`: the packed tensor with
  a leading images axis of 1, a view, so in-place edits land; assign the same shape.
  `vision.layers[i].output` stays the native `[patches, vision_hidden]`.
- The processor's `image_grid_thw` (`[t, h, w]` per image, in patches) splits the row: image
  `j` has `t * h * w` patches. An image's patches are in merge-block order (each 2x2 block
  the merger folds is consecutive), not raster order. On Qwen2.5-VL the tower permutes them
  into attention windows at entry, so its block values and `tower_output` are in window
  order, served as they are.
- The attention interior (`attention_queries` ... `attention_head_outputs`,
  `attention_probabilities`) is `Unavailable` on every tower block, eager or not: "a packed
  tower: the attention makes one interface call per image ..., so the block's pattern is
  not one tensor". `attention_output` and the block values are whole.
- `vision.image_features` is the tower's `pooler_output`: the merger's output in the order
  the wrapper scatters it. On Qwen2.5-VL that is not `projector.output`, which is still in
  window order once an image spans more than one 112-pixel window.
- The sizes are the vision config's: `hidden_size` is the tower's width (`embed_dim` on
  Qwen2-VL), `num_heads`, `intermediate_size`, `patch_size`, `spatial_merge_size`, and
  `window_size` (Qwen2.5-VL; `None` elsewhere). `image_size` is `None`: any resolution goes.

```python
import torch
from PIL import Image
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3-VL-4B-Instruct", task="image-text-to-text", dispatch=True, dtype=torch.bfloat16)
red, blue = Image.new("RGB", (448, 448), "red"), Image.new("RGB", (320, 256), "blue")
content = [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Describe both images."}]
prompt = model.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)

with torch.no_grad(), model.trace(prompt, images=[red, blue]):
    mask = model.vision.image_token_mask.save()        # 276 image tokens, both images' (read first: off the inputs)
    patches = model.vision.patch_embeddings.save()     # [1, 1104, 1024]: both images' patches in one row
    features = model.vision.image_features.save()      # [276, 2560]
    first = model.layers[0].input.save()

torch.equal(first[mask], features)                     # True
```

**Qwen3-VL's DeepStack.** On `qwen3_vl_text` and `qwen3_vl_moe_text` the tower also taps
three of its blocks, and the text model adds each tap's merged features at the image
positions after text blocks 0, 1 and 2, outside the blocks. `layers[k].deepstack_output`
(`[image_tokens, hidden]`, assignable) is what is added after block `k`:
`layers[k+1].input[mask] == layers[k].layer_output[mask] + layers[k].deepstack_output`;
on the other blocks it is `Unavailable` and `layers[k+1].input == layers[k].layer_output`.
So `image_features` is not the only way the image reaches the text model there: on
`Qwen/Qwen3-VL-4B-Instruct`, asked the color of a red square on white, zeroing
`image_features` alone still answers "Red"; zeroing `deepstack_output` on blocks 0-2 too
answers "White".

```python
with torch.no_grad(), model.trace(prompt, images=[red, blue]):
    model.vision.image_features[:] = 0
    for k in range(3):
        model.layers[k].deepstack_output[:] = 0         # read in forward order: after block k
    ablated = model.logits.save()
```

The text blocks of these families use multimodal rotary embeddings (M-RoPE: temporal,
height and width position streams), which the model folds into one `cos`/`sin` before the
blocks; the attention applies it before the interface, so `attention_queries` and
`attention_keys` are the rotated queries and keys, as on any rotary family.

## Availability

`model.vision.support()` lists the tower's values (`image_token_mask`, `patch_embeddings`,
`tower_output`, `image_features`) and its block values the way `model.support()` lists the
text blocks', and `model.support()` carries the same rows under the `vision` host
(`"vision.image_features"`, `"vision.self_attn.attention_probabilities"`), as it carries
`self_attn` rows for the text blocks. Only where an image can reach the model, a wrapper
loaded with its processor: a text-only checkpoint has no `model.vision` at all, and a
`task="text-generation"` load of a wrapper (on Gemma 3 that builds the wrapper without a
processor) has a tower that never runs, so `model.vision.support()` is empty,
`model.support()` has no `vision` rows, and reading an image value raises `Unavailable`
("a text-only load").

`vision.image_features` is served only on the wrappers a family lists in `IMAGE_WRAPPERS`
(`gemma3` for `gemma3_text`, `llava` for `llama`, `llama4` for `llama4_text`, `gemma4` for
`gemma4_text`, `gemma4_unified` for `gemma4_unified_text`), where the suite checks that it
is what is scattered. A wrapper that binds the same names but rearranges the projector's output
(`gemma3` for `gemma3_text`, `llava` for `llama`, each Qwen wrapper for its text family),
where the value read is what is scattered. A wrapper that binds the same names but rearranges the projector's output
first (LLaVA-NeXT's unpadding and newline tokens) is not listed, and the value says so.

## What is not covered yet

- Towers other than SigLIP (Gemma 3), CLIP (Llava 1.5), Llama 4's and Gemma 4's ViTs and
  Gemma 4 unified's embedder. The text names bind on the
  wrappers of Qwen3.5, Qwen3.5-MoE, Mistral 3, PaliGemma, LLaVA-OneVision, llava-interleave,
  Idefics 3, Aya Vision, EXAONE 4.5 and LightOnOCR, but their towers and projectors are
`vision.image_features` needs the family to know where the wrapper writes the features in
(an `ImageScatter` keyed on the wrapper's model); on a wrapper whose family does not, the
tower's names bind but `image_features` is `Unavailable` and says so.

## Which wrappers

| family | wrapper (`model_type`) | tower |
| --- | --- | --- |
| `gemma3_text` | Gemma 3 (`gemma3`) | SigLIP |
| `gemma` | PaliGemma (`paligemma`) | SigLIP |
| `qwen2` | llava-interleave (`llava`), LLaVA-OneVision (`llava_onevision`) | SigLIP |
| `cohere2` | Aya Vision (`aya_vision`), Cohere2-Vision (`cohere2_vision`) | SigLIP |
| `llama` | Llava 1.5 (`llava`), VipLlava (`vipllava`), LLaVA-NeXT (`llava_next`) | CLIP |
| `llama` | DeepSeek-VL (`deepseek_vl`) | SigLIP |
| `llama` | Idefics 3 (`idefics3`), SmolVLM (`smolvlm`) | their SigLIP-shaped ViT, one row per tile |
| `mistral` | LLaVA-NeXT (`llava_next`, `llava-v1.6-mistral`), BakLLaVA (`llava`; no tiny checkpoint, so untested: the keys are LLaVA-NeXT's tower and Llava's scatter) | CLIP |
| `mistral` | Mistral 3 (`mistral3`, Mistral Small 3.1 / 3.2), Pixtral-12B (`llava`) | Pixtral |
| `ministral3` | Mistral 3 (`mistral3`, Ministral 3) | Pixtral |

## What is not covered yet

- Towers other than SigLIP, CLIP and Pixtral. The text names bind on the wrappers of
  Qwen3.5, Qwen3.5-MoE, EXAONE 4.5 and LightOnOCR, but their towers and projectors are
  native-only, and there is no `model.vision` there.
- Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Mllama: their text models are types nnterp has no
  family for yet.
- Towers other than SigLIP (Gemma 3), CLIP (Llava 1.5) and the Qwen ViT (Qwen2-VL,
  Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE). The text names bind on the
  wrappers of Mistral 3, PaliGemma, LLaVA-OneVision, llava-interleave, Idefics 3, Aya
  Vision, EXAONE 4.5 and LightOnOCR, but their towers and projectors are native-only, and
  there is no `model.vision` there.
- Mllama: its text model is a type nnterp has no family for yet.
- Video and audio values; batching several image-carrying invokes in one trace.
- The design and the phases: [docs/developing/vision-design.md](../developing/vision-design.md).
