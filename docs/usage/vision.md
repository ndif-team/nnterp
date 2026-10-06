---
title: Vision-language models
one_liner: "Load an image-text-to-text checkpoint with `task=\"image-text-to-text\"`, pass an image, and read the tower (`model.vision`, `vision.layers[i]`), the `projector`, and the tower's `vision.image_token_mask` and `vision.image_features`, where `layers[0].input[vision.image_token_mask] == vision.image_features`."
tags: [usage, vision, multimodal, image-text-to-text, vision tower, projector, image_features, image_token_mask, Patches, siglip, clip, pixtral, qwen-vl, llama4, gemma3, gemma4, encoder-free, packed tower, deepstack, m-rope, llava, llava-next, llava-onevision, paligemma, idefics3, smolvlm, aya-vision, mistral3]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/layouts.md, docs/developing/vision-design.md, docs/reference/families.md]
sources: [nnterp/components/vision.py, nnterp/standardized.py, nnterp/families/gemma3_text.py, nnterp/families/gemma.py, nnterp/families/llama.py, nnterp/families/qwen2.py, nnterp/families/cohere2.py, nnterp/families/mistral.py, nnterp/families/ministral3.py, nnterp/families/llama4_text.py, nnterp/families/gemma4_text.py, nnterp/families/gemma4_unified_text.py, nnterp/families/qwen2_vl_text.py, nnterp/families/qwen2_5_vl_text.py, nnterp/families/qwen3_vl_text.py, nnterp/families/qwen3_vl_moe_text.py, nnterp/families/qwen3_5_text.py, nnterp/families/qwen3_5_moe_text.py, tests/families/vision_suite.py, tests/families/qwen_vision_suite.py]
---

# Vision-language models

## What this is for

An image-text-to-text checkpoint (Gemma 3, Llava, Qwen3-VL, Mistral 3, Llama 4, ...) is a
text model plus a vision tower, a projector, and a step that scatters the projected image
features into the text stream at the image tokens. nnterp keeps the text model's standard
names (the family is the text model's: `gemma3_text`, `llama`, ...) and adds names for the
vision side: the tower, its blocks, the projector, and two values of the tower for where the
image enters the text model.

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
token first), `features` `(576, 16)`. The same body runs on every wrapper in
[the coverage list](#which-wrappers), less the lines a tower does not serve: the Qwen ViT has
no `pattern`, and Gemma 4 unified's embedder has no blocks. On Gemma 4 (`google/gemma-4-E2B`, whose base checkpoint has no chat
template: write the prompt as `f"{model.processor.image_token} ..."`) with a red 224x224
image, under `torch.no_grad()` in bfloat16: 256 image tokens, `pattern`
`(1, 12, 2520, 2520)` (2304 patches and 216 padded rows), `features` `(256, 1536)`, the
scatter equality exact, the next token `" red"`, and `" of"` once `image_features` is
zeroed; it peaked at 11 GB.

## The names

`model.vision` is the tower's root (a `Vision`), `vision.layers[i]` its blocks
(`VisionLayer`, with `self_attn` a `VisionAttention` and `mlp` a `VisionMlp`),
`vision.patch_embed` the patch embedding, `vision.norm` the final norm over the patches where
the tower has one, and `model.projector` the last module before the scatter. Per tower, the
native names they alias:

| tower | `vision` | `patch_embed` | `layers[i]` | `self_attn`, `mlp` | `input_layernorm`, `post_attention_layernorm` | `norm` | `projector` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SigLIP (Gemma 3, PaliGemma, llava-interleave, LLaVA-OneVision, Aya Vision, Cohere2-Vision, DeepSeek-VL, Idefics 3, SmolVLM) | `model.model.vision_tower` (DeepSeek-VL, Idefics 3, SmolVLM: `model.model.vision_model`) | `embeddings.patch_embedding` | `encoder.layers[i]` | native | `layer_norm1`, `layer_norm2` | `post_layernorm` | `model.model.multi_modal_projector` (Gemma 3's pools 4096 patches to 256 tokens; Aya Vision's and Cohere2-Vision's pixel-shuffle); DeepSeek-VL: `model.model.aligner`; Idefics 3, SmolVLM: `model.model.connector` (pixel shuffle) |
| CLIP (Llava 1.5, VipLlava, LLaVA-NeXT, BakLLaVA) | `model.model.vision_tower` | `embeddings.patch_embedding` | `encoder.layers[i]` | native | `layer_norm1`, `layer_norm2` | none: CLIP's `post_layernorm` norms only the pooled CLS token | `model.model.multi_modal_projector` |
| Pixtral (Mistral 3, Pixtral-12B), a `PixtralVision` | `model.model.vision_tower` | `patch_conv` | `transformer.layers[i]` | `attention`, `feed_forward` | `attention_norm`, `ffn_norm` | none | `model.model.multi_modal_projector` (merges each 2x2 block of patches on Mistral 3) |
| the Qwen ViT (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE), a `QwenVision` | `model.model.visual` | `patch_embed` | `blocks[i]` | `attn` (a `QwenVisionAttention`), `mlp` | `norm1`, `norm2` | none: the merger norms its own input | `visual.merger`, inside the tower (folds each 2x2 block of patches into one token) |
| Llama 4's ViT | `model.vision_model` | `patch_embedding` (unfold + linear) | `model.layers[i]` | native | native | `layernorm_post` | `model.multi_modal_projector` (a linear) |
| Gemma 4's ViT | `model.model.vision_tower` | `patch_embedder.input_proj` (a linear; the 2D position embedding comes after) | `encoder.layers[i]` | native; the contributions are the post-norms' outputs (a sandwich block) | native: `post_attention_layernorm` *follows* the attention, as on the text block | none | `model.model.embed_vision` (an RMS norm and a linear) |
| Gemma 4 unified's encoder-free embedder | `model.model.embed_vision` | `patch_dense` | none: no blocks | none | none | none | `embed_vision.multimodal_embedder` (an RMS norm and a linear) |

`embed_tokens`, `layers`, `norm` and `lm_head` stay the language model's
(`model.language_model.*` on the wrapper; `language_model.model.*` on Llama 4's), and
native names keep working. A family that hosts the tower at two paths keys both, as it keys
both spellings of the text stack. Gemma 4's audio tower and its embedder keep their native
names (`model.model.audio_tower`, `model.model.embed_audio`), and an audio prompt
(`model.trace(prompt, audio=[waveform])`) runs as before.

## The values

| value | host | layout | meaning |
| --- | --- | --- | --- |
| `layer_output` | `vision.layers[i]` | `Patches` | the tower's stream leaving the block |
| `attention_output`, `mlp_output` | `vision.layers[i].self_attn`, `.mlp` | `Patches` | what each sublayer adds: `input + attention_output + mlp_output == layer_output` |
| `attention_probabilities`, `attention_queries`, ... | `vision.layers[i].self_attn` | `Pattern`, `Queries`, ... | as on a text block, the `batch` axis being the tower's rows; no causal mask (Pixtral masks between packed images, Gemma 4 masks its padded keys); needs eager (the Qwen ViT's: [below](#the-qwen-vit)) |
| `patch_embeddings` | `vision` | `Patches` | the patch embedding's output, one row per patch, before any position embedding, CLS token or pre-norm |
| `tower_output` | `vision` | `Patches` | the last block's stream after `vision.norm` where there is one, before any pooling, CLS dropping or adapter |
| `image_token_mask` | `vision` | `ImageTokenMask` `[batch seq]` bool | `input_ids == config.image_token_id`, read off the model's inputs; read-only |
| `image_features` | `vision` | `ImageFeatures` `[image_tokens hidden]` | what the wrapper scatters into the token embeddings, flat over every image token in row-major order; assignable, in-place edits land |

`Patches` is `[images, patches, vision_hidden]`. What a row and the patch axis are, per tower:

| tower | rows | the patch axis |
| --- | --- | --- |
| SigLIP | one per image (per crop on LLaVA-OneVision, per tile on Idefics 3 and SmolVLM) | the patches, in raster order |
| CLIP | one per image (per crop on LLaVA-NeXT) | the CLS token *first*, then the patches |
| Pixtral | one, *packed*: every image's patches, image after image | each image has `(height // patch_size) * (width // patch_size)` patches, its `image_sizes` entry from the processor; `patch_embeddings` is the packed row as it enters `ln_pre` (the convolution's own output, `patch_embed.output`, is the padded grid) |
| the Qwen ViT | one, *packed* (the tower runs on `[patches, vision_hidden]`, served with a leading 1) | every image's patches in merge-block order; on Qwen2.5-VL in window order inside the tower ([below](#the-qwen-vit)) |
| Llama 4's ViT | one per image tile | the patches, then the CLS token *last* (`patches + 1` rows); the tower drops it after `vision.norm` |
| Gemma 4's ViT | one per image | the patches *padded* to `max_soft_tokens * pooling_kernel_size**2` rows (2520 by default). The padded rows (zero pixels at position `(-1, -1)`, `image_position_ids` in the processor's encoding) are masked as keys but run through every block, so they are rows of `layer_output`, with values; the pooler zeroes and strips them |
| Gemma 4 unified's embedder | one per image | `patch_embeddings` and `tower_output` only, padded to `max_soft_tokens` rows (280) the same way |

The tower's sizes are on `model.vision`: `num_layers`, `hidden_size`, `num_heads`,
`head_dim`, `intermediate_size`, `patch_size`, `image_size`, read off the tower's own config
(a tower that spells one its own way reports it under the plain name). The Qwen ViT, Pixtral,
Gemma 4 and Gemma 4 unified take images of any resolution, so their `image_size` raises
`Unavailable` saying where each image's grid is (`image_grid_thw`, `image_sizes`,
`image_position_ids`). Gemma 4's `head_dim` is the config's (64 on E2B, with 12 heads over a
768-wide stream). The encoder-free embedder has `num_layers == 0`, `hidden_size` its
`mm_embed_dim` and `patch_size` the 48-pixel merged patch it embeds; its `num_heads`,
`head_dim` and `intermediate_size` raise `Unavailable`.

## Where the image meets the text model

`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly:
`image_features` is read at the scatter, the tensor the wrapper's forward writes into the
token embeddings at the image tokens, and nothing touches it before block 0. So
`vision.image_features` is the place to ablate, patch or steer the image as the text model
sees it. On most wrappers it is the projector's output, reshaped. Where the wrapper changes
that output before scattering it, it is not:

- LLaVA-NeXT and LLaVA-OneVision unpad the projector's output and add a newline token per
  row (`model.model.image_newline`), so `projector.output` has another row count.
- Gemma 4 unified runs the projector on the padded rows too and strips them:
  `image_features == projector.output[valid]`, where `valid` is
  `(image_position_ids != -1).all(-1)`.
- Qwen2.5-VL restores the merge-block order after the merger, so `projector.output` is in
  window order once an image spans more than one 112-pixel window.

Both values are the tower's although neither is read inside it: the mask comes off the
model's inputs and the features off the wrapper's forward.

What feeds the projector differs per host: Gemma 3 pools `tower_output`; Llava takes
`vision.layers[-2].layer_output` without its CLS token (`vision_feature_layer=-2`), so a
write to `tower_output` or the last block does not reach Llava's text model; VipLlava
concatenates several blocks' streams. `model.projector.input` is what the projector
actually receives. On Llama 4 the tower runs a pixel-shuffle adapter after `tower_output`
(`vision.vision_adapter`, a quarter as many rows) whose output, flattened over the tiles, is
`projector.input`; on Gemma 4 the tower's `pooler` average-pools 3x3 patches of
`tower_output` into each soft token and strips the padding, which is `projector.input`; the
encoder-free embedder has no blocks to end a stream, so its `tower_output` is the states
before the projection, `projector.input` itself.

## Inputs

- `model.trace(prompt, images=[image])`, the image placeholder in the prompt (the
  processor's chat template puts it there).
- `model.trace(encoding)` with an encoding built by `model.processor`.
- `model.trace("text")`: a text-only trace of the wrapper works (the mask is all false,
  `vision.image_features` is never reached), except on PaliGemma, whose processor demands an
  image; pass `dict(model.tokenizer(text, return_tensors="pt"))` there.
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

## The Qwen ViT

Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5 and Qwen3.5-MoE share one tower at
`model.visual`. It runs on `[patches, vision_hidden]`, every image of the invoke
concatenated, and its attention calls the interface once per image (once per window on
Qwen2.5-VL's windowed blocks). So:

- Its `Patches` values (`patch_embeddings`, `tower_output`, `layer_output`,
  `attention_output`, `mlp_output`) are `[1, patches, vision_hidden]`: the packed tensor with
  a leading images axis of 1, a view, so in-place edits land; assign the same shape.
  `vision.layers[i].output` stays the native `[patches, vision_hidden]`.
- The processor's `image_grid_thw` (`[t, h, w]` per image, in patches) splits the row: image
  `j` has `t * h * w` patches. An image's patches are in merge-block order (each 2x2 block
  the merger folds is consecutive), not raster order. On Qwen2.5-VL the tower permutes them
  into attention windows at entry, so its block values and `tower_output` are in window
  order, served as they are.
- The attention interior is read around the per-image calls: `attention_queries`,
  `attention_keys` and `attention_values` are the whole `[1, heads, patches, head_dim]`
  tensors before the module splits them (the queries and keys after the 2D rotary
  embedding), and `attention_head_outputs` is the calls' outputs concatenated back,
  `[1, patches, heads, head_dim]` (under any `attn_implementation` but flash).
  `attention_scores` and `attention_probabilities` are `Unavailable` under every
  implementation: no one tensor is the block's pattern. Split the queries and keys at the
  attention's `cu_seqlens` argument (`self_attn.inputs[1]["cu_seqlens"]`) to compute a
  per-image pattern.
- The sizes are the vision config's: `hidden_size` is the tower's width (`embed_dim` on
  Qwen2-VL), `num_heads`, `intermediate_size`, `patch_size`, `spatial_merge_size`, and
  `window_size` (Qwen2.5-VL's, in pixels; `None` elsewhere).

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
`task="text-generation"` load that builds the wrapper has no processor, so its tower never
runs. Gemma 3, Gemma 4 and Gemma 4 unified build the wrapper for text generation; so do the Llava-class and
Qwen-VL wrappers when dispatched, transformers having no text-only class for their configs
(Llama 4 and Qwen3.5 build their text-only class). There `model.vision.support()` is empty,
`model.support()` has no `vision` rows, and reading any tower or block value raises
`Unavailable` ("a text-only load: no processor, so no image reaches the model; load with
task='image-text-to-text'"), inside a trace too.

`vision.image_features` needs the family to know where the wrapper writes the features in:
an `ImageScatter` keyed on the wrapper's model, or, on Llama 4, whose top-level forward
scatters, the family's `ROOT_SCATTER`. On a wrapper whose family keys neither, the tower's
names bind but `image_features` is `Unavailable` and says so.

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
| `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text`, `qwen3_vl_moe_text` | Qwen2-VL (`qwen2_vl`), Qwen2.5-VL (`qwen2_5_vl`), Qwen3-VL (`qwen3_vl`), Qwen3-VL-MoE (`qwen3_vl_moe`) | the Qwen ViT |
| `qwen3_5_text`, `qwen3_5_moe_text` | Qwen3.5 (`qwen3_5`), Qwen3.5-MoE (`qwen3_5_moe`) | the Qwen ViT |
| `llama4_text` | Llama 4 (`llama4`) | Llama 4's ViT |
| `gemma4_text` | Gemma 4 (`gemma4`) | Gemma 4's ViT |
| `gemma4_unified_text` | Gemma 4 unified (`gemma4_unified`) | the encoder-free embedder |

## What is not covered yet

- The towers of EXAONE 4.5 and LightOnOCR: the text names bind on their wrappers, but the
  towers and projectors are native-only, and there is no `model.vision` there.
- Mllama: its text model is a type nnterp has no family for yet.
- Video and audio values; batching several image-carrying invokes in one trace.
- The design: [docs/developing/vision-design.md](../developing/vision-design.md).
