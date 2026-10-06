---
title: Vision-language models
one_liner: "Load an image-text-to-text checkpoint with `task=\"image-text-to-text\"`, pass an image, and read the tower (`model.vision`, `vision.layers[i]`), the `projector`, and the tower's `vision.image_token_mask` and `vision.image_features`, where `layers[0].input[vision.image_token_mask] == vision.image_features`."
tags: [usage, vision, multimodal, image-text-to-text, vision tower, projector, image_features, image_token_mask, Patches, siglip, clip, llava, gemma3, qwen-vl, packed tower, deepstack, m-rope]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/layouts.md, docs/developing/vision-design.md, docs/reference/families.md]
sources: [nnterp/components/vision.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, nnterp/families/qwen2_vl_text.py, nnterp/families/qwen2_5_vl_text.py, nnterp/families/qwen3_vl_text.py, nnterp/families/qwen3_vl_moe_text.py, nnterp/families/qwen3_5_text.py, nnterp/families/qwen3_5_moe_text.py, tests/families/vision_suite.py, tests/families/qwen_vision_suite.py, tests/families/test_gemma3_text.py, tests/families/test_llama.py, tests/families/test_qwen3_vl_text.py]
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
(`google/gemma-3-4b-pt`).

## The names

| standard name | what it is | Gemma 3 (SigLIP) | Llava 1.5 (CLIP) | Qwen-VL, Qwen3.5 (Qwen ViT) |
| --- | --- | --- | --- | --- |
| `model.vision` | the tower's root, a `Vision` | `model.model.vision_tower` | `model.model.vision_tower` | `model.model.visual` (a `QwenVision`) |
| `model.vision.patch_embed` | the patch embedding | `vision_tower.embeddings.patch_embedding` | same | `visual.patch_embed` |
| `model.vision.layers[i]` | the tower's blocks, `VisionLayer` | `vision_tower.encoder.layers[i]` | same | `visual.blocks[i]` (`PackedVisionLayer`) |
| `vision.layers[i].self_attn`, `.mlp` | `VisionAttention`, `VisionMlp` | native | native | `attn`, `mlp` (`PackedVisionAttention`, `PackedVisionMlp`) |
| `vision.layers[i].input_layernorm`, `.post_attention_layernorm` | the pre-norms | `layer_norm1`, `layer_norm2` | same | `norm1`, `norm2` |
| `model.vision.norm` | the final norm over the patches | `post_layernorm` | none: CLIP's `post_layernorm` norms only the pooled CLS token | none: the merger norms its own input |
| `model.projector` | the module whose output is scattered into the text stream | `model.multi_modal_projector` (pools 4096 patches to 256 tokens) | `model.multi_modal_projector` | `visual.merger`, inside the tower (folds each 2x2 block of patches into one token) |

`embed_tokens`, `layers`, `norm` and `lm_head` stay the language model's
(`model.language_model.*` on the wrapper), and native names keep working.

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
raster order (CLS first on CLIP). The tower's sizes are on `model.vision`: `num_layers`,
`hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size`.

`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly: the
wrapper writes the projector's output into the token embeddings at the image tokens and
nothing touches it before block 0. So `vision.image_features` is the place to ablate, patch
or steer the image as the text model sees it. Both values are the tower's although neither
is read inside it: the mask comes off the model's inputs and the features off the
projector, the tower's sibling.

What feeds the projector differs per host: Gemma 3 pools `tower_output`; Llava takes
`vision.layers[-2].layer_output` without its CLS token (`vision_feature_layer=-2`), so a
write to `tower_output` or the last block does not reach Llava's text model.
`model.projector.input` is what the projector actually receives.

## Inputs

- `model.trace(prompt, images=[image])`, the image placeholder in the prompt (the
  processor's chat template puts it there).
- `model.trace(encoding)` with an encoding built by `model.processor`.
- `model.trace("text")`: a text-only trace of the wrapper works (the mask is all false,
  `vision.image_features` is never reached), except on PaliGemma, whose processor demands an image;
  pass `dict(model.tokenizer(text, return_tensors="pt"))` there.
- One invoke per trace while it carries an image; several images go in one invoke, as lists.

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
(`gemma3` for `gemma3_text`, `llava` for `llama`, each Qwen wrapper for its text family),
where the value read is what is scattered. A wrapper that binds the same names but rearranges the projector's output
first (LLaVA-NeXT's unpadding and newline tokens) is not listed, and the value says so.

## What is not covered yet

- Towers other than SigLIP (Gemma 3), CLIP (Llava 1.5) and the Qwen ViT (Qwen2-VL,
  Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE). The text names bind on the
  wrappers of Mistral 3, PaliGemma, LLaVA-OneVision, llava-interleave, Idefics 3, Aya
  Vision, EXAONE 4.5 and LightOnOCR, but their towers and projectors are native-only, and
  there is no `model.vision` there.
- Mllama: its text model is a type nnterp has no family for yet.
- Video and audio values; batching several image-carrying invokes in one trace.
- The design and the phases: [docs/developing/vision-design.md](../developing/vision-design.md).
