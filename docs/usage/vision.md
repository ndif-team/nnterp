---
title: Vision-language models
one_liner: "Load an image-text-to-text checkpoint with `task=\"image-text-to-text\"`, pass an image, and read the tower (`model.vision`, `vision.layers[i]`), the `projector`, and the tower's `vision.image_token_mask` and `vision.image_features`, where `layers[0].input[vision.image_token_mask] == vision.image_features`."
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

Gemma 3's tower attends over 4096 patches, so under eager every block's pattern is
16 x 4096 x 4096. A trace keeps them all for autograd unless it runs under
`torch.no_grad()`: on `google/gemma-3-4b-pt` an eager trace without it ran out of a 48 GB
card, and with it peaked at 11 GB.

Read order is the forward's: `vision.image_token_mask` first (it comes off the inputs,
like `input_ids`), then the tower's values, a block's attention interior before its
`layer_output`, then `vision.image_features`, then the text model's.

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
- Video and audio values; batching several image-carrying invokes in one trace.
- The design and the phases: [docs/developing/vision-design.md](../developing/vision-design.md).
