---
title: Vision design
one_liner: How nnterp standardizes the vision side of image-text-to-text checkpoints — the tower under model.vision with its blocks and the image values, the projector, where the code lives, how loading and the suite work, what does not fit, and the phases.
tags: [developing, design, vision, multimodal, families]
related: [docs/usage/vision.md, docs/developing/architecture.md, docs/developing/eproperty-internals.md, docs/developing/testing.md, docs/extending/adding-a-family.md]
sources: [nnterp/components/vision.py, nnterp/components/standard.py, nnterp/standardized.py, nnterp/components/eproperty.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, tests/families/vision_suite.py, tests/families/suite.py]
---

# Vision design

## What this is for

An image-text-to-text checkpoint is a text model plus a vision tower, a projector, and a
step that puts the projected features into the text stream. nnterp standardizes the text
model already; this page is how the vision side is standardized so the same names and
values hold on every wrapper: what the tower and projector are called, which values the
tower serves, where the code lives, and how loading and the suite work. Facts quoted here
were run on transformers 5.17 and nnsight `dev` on the tiny checkpoints named in
[the suite section](#the-suite).

The rules:

- **One family per text `model_type`, wrapper included.** The family is chosen from
  `text_config.model_type`. Its `RENAME` carries the wrapper's spellings of the text stack,
  the tower and the projector, and the tower's own names, keyed so that on a text-only
  checkpoint none of them resolve.
- **The root stays the text model's.** `model.num_layers`, `model.hidden_size`,
  `model.layers` are the language model's; the tower's sizes are on `model.vision`, and so
  are the values where the image meets the text model. Vision is a component, like `Moe`:
  `StandardizedTransformer` has no vision code.
- **Shared tower classes live in `nnterp/components/vision.py`; per-family paths live in the
  family file.** A tower's names are a handful of `RENAME` lines and four `ENVOYS` entries,
  inline where the family needs them.

## The vocabulary

### Names

| standard name | what it is | Gemma 3 | Llava | Qwen3.5 / Qwen-VL | Llama 4 | Mistral 3 | Kimi K2.5 | Gemma 4 |
|---|---|---|---|---|---|---|---|---|
| `vision` | the tower's root module | `model.vision_tower` | `model.vision_tower` | `model.visual` | `vision_model` | `model.vision_tower` | `model.vision_tower` | `model.vision_tower` |
| `vision.patch_embed` | the patch embedding | `embeddings.patch_embedding` | `embeddings.patch_embedding` | `patch_embed` (native) | `patch_embedding` | `patch_conv` | `patch_embed` (native) | `patch_embedder` |
| `vision.layers` | the tower's blocks | `encoder.layers` | `encoder.layers` | `blocks` | `model.layers` | `transformer.layers` | `layers` (native) | `encoder.layers` |
| `vision.layers[i].self_attn`, `.mlp`, `.input_layernorm`, `.post_attention_layernorm` | the block's sublayers and norms | `self_attn`, `mlp`, `layer_norm1`, `layer_norm2` | same as Gemma 3 | `attn`, `mlp`, `norm1`, `norm2` | native | `attention`, `feed_forward`, `attention_norm`, `ffn_norm` | `attn`, `mlp`, `norm1`, `norm2` | native (sandwich) |
| `vision.norm` | the tower's final norm over the patches, where it has one | `post_layernorm` | none (CLIP's `post_layernorm` norms the pooled CLS only) | none (the merger norms) | `layernorm_post` | none | `final_layernorm` | none |
| `projector` | the module whose output is scattered into the text stream | `model.multi_modal_projector` | `model.multi_modal_projector` | `model.visual.merger` | `multi_modal_projector` | `model.multi_modal_projector` | `model.mm_projector` | `model.embed_vision` |

Gemma 3 and Llava are implemented; the other columns are what the later phases bind. The
text names (`embed_tokens`, `layers`, `norm`, `lm_head`) keep their meaning; on a wrapper
they alias `model.language_model.*` (Llava's family included) or `model.text_model.*`
(Idefics 3, SmolVLM). Native names keep working everywhere.

`projector` names the last module before the scatter. A pooling or token-merging step
between the tower and the projector keeps its native name (Gemma 3's pooling is inside the
projector; Llama 4's pixel-shuffle is `vision.vision_adapter`; Kimi's temporal merge is a
method of the tower; Idefics 3's pixel shuffle is inside `model.connector`).
`projector.input` is therefore what the host feeds its projector, which is not always the
tower's output: Llava feeds `vision.layers[-2].layer_output` without the CLS token.

### How the tower keys bind

nnsight resolves a `rename` key from every envoy and binds the alias on the envoy it
resolves from. So the tower root and the projector are keyed from the model root
(`"model.vision_tower": "vision"`), and the tower's inner names are keyed relative to the
tower: multi-component keys no text model has (`"encoder.layers": "layers"`,
`"embeddings.patch_embedding": "patch_embed"`) and single names no text block has
(`"post_layernorm": "norm"`, `"layer_norm1": "input_layernorm"`). A bare name that a text
block also has is never a tower key. Where a family hosts the same tower spelled the same
way under another wrapper (DeepSeek-VL's `model.vision_model`), the inner keys bind there
too, on that native envoy; only the root keys decide what `model.vision` is.

### Values

| value | host | layout | meaning |
|---|---|---|---|
| `layer_output` | `vision.layers[i]` | `Patches` | the tower's stream leaving the block |
| `attention_output`, `mlp_output` | `vision.layers[i].self_attn`, `.mlp` | `Patches` | what each sublayer adds to the tower's stream: `input + attention_output + mlp_output == layer_output` |
| `attention_probabilities`, `attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_head_outputs` | `vision.layers[i].self_attn` | `Pattern`, `Queries`, ... | as on the text blocks, read inside the shared eager interface; the `batch` axis is the tower's images; need `attn_implementation="eager"` |
| `patch_embeddings` | `vision` | `Patches` | the patch embedding's output, one row per patch (a view of the convolution's `[images, hidden, rows, columns]`) |
| `tower_output` | `vision` | `Patches` | what the tower returns over the patches (`last_hidden_state`) |
| `image_token_mask` | `vision` | `ImageTokenMask` `[batch seq]` | `input_ids == config.image_token_id`, off the model's inputs; read-only |
| `image_features` | `vision` | `ImageFeatures` `[image_tokens hidden]` | the projector's output flat over every image token of the batch, in scatter order: `layers[0].input[vision.image_token_mask] == vision.image_features` |

```python
from jaxtyping import Bool, Float
from torch import Tensor

#: A tower's stream: the tower's own batch (images, tiles, frames; 1 on a packed tower) by its tokens.
Patches = Float[Tensor, "images patches vision_hidden"]                 # nnterp/components/vision.py
#: What the text model receives at the image positions, flat in scatter (row-major) order.
ImageFeatures = Float[Tensor, "image_tokens hidden"]                     # nnterp/components/vision.py
#: Which positions of the text batch hold image tokens.
ImageTokenMask = Bool[Tensor, "batch seq"]                               # nnterp/components/vision.py
```

The sizes are the tower's, read off its own config on `model.vision`: `num_layers`,
`hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size`. A
tower whose config spells one its own way overrides the property on a `Vision` subclass.

### What `Patches` is on each tower

| tower | `images` axis | `patches` axis |
|---|---|---|
| SigLIP (Gemma 3, PaliGemma, LLaVA-OneVision, DeepSeek-VL, Aya Vision, Cohere2-Vision) | one row per image | the image's patches in raster order |
| CLIP (Llava 1.5, LLaVA-NeXT, VipLlava, Video-Llava) | one row per image or crop | CLS first, then patches |
| Idefics 3 / SmolVLM ViT | one row per image tile | patches |
| Llama 4 ViT | one row per image tile | patches, then CLS last |
| Gemma 4 ViT | one row per image | patches padded to the batch's longest; the padding is masked and stripped by the pooler |
| Pixtral (Mistral 3, LightOnOCR, Pixtral-12B's Llava wrapper) | 1 | every image's patches concatenated (block-diagonal mask) |
| Qwen2-VL, Qwen3-VL, Qwen3.5 ViT; MoonViT (Kimi K2.5) | 1 | every image's patches concatenated (`cu_seqlens`); natively `[patches, hidden]`, served with a leading 1 |
| Qwen2.5-VL ViT (also EXAONE 4.5) | 1 | as Qwen2-VL, but in *window order*: the tower permutes the patches into attention windows at entry and restores raster order after the merger |

A video enters as frames on the per-image towers and as temporal patches on the packed
ones. A user splits a packed row per image with the processor's `image_grid_thw` (Qwen,
Kimi) or `image_sizes` (Pixtral).

### What does not fit, and how it is handled

- **Attention interior on a packed tower.** Under eager attention, Qwen's ViT, MoonViT and
  Pixtral call the attention interface once per image, inside a list comprehension (op
  `attention_interface_2`; `_1` is the flash branch). A value at that call is one image's: a
  trace with two images served the first image's 256x256 pattern while the block held 536
  patches. The interior values are `Unavailable` on those towers, with the op named in the
  reason; `attention_output` and the block values are whole and available.
- **Qwen2.5-VL's window order.** The block values are served in the tower's own (window)
  order and documented as such; a raster view needs the window index, which the tower
  computes inside its forward, so nnterp does not reorder.
- **Qwen3-VL's DeepStack.** Three tower blocks (`deepstack_visual_indexes`) each feed a
  `deepstack_merger_list[k]` whose output the text model adds at the image positions after
  text block `k`, outside the block. So on `qwen3_vl_text`,
  `layers[k+1].input != layers[k].layer_output` at image positions for `k < 3`. The family
  serves it as a value of its own `Layer`: `layers[k].deepstack_output`, `[image_tokens
  hidden]`, read at the text model's `_deepstack_process` call (third argument), with
  `layers[k+1].input[mask] == layers[k].layer_output[mask] + deepstack_output`; unavailable
  on the other blocks. The tower side needs no new name: the taps are
  `vision.layers[i].layer_output` and the mergers keep their native path.
- **Mllama (Llama 3.2 Vision).** Its text model (`mllama_text_model`) interleaves
  cross-attention blocks that attend to `cross_attention_states` (the projector's output) and
  are skipped entirely on a text-only input; nothing is scattered. It needs its own family
  with a `CrossAttention` component; `image_features` is unavailable there with that reason.
  Its tower has two encoders (`transformer`, `global_transformer` with gated blocks) over
  tiles and concatenates intermediate layers' outputs.
- **An encoder-free wrapper** (Gemma 4 unified, `Gemma4UnifiedForConditionalGeneration`;
  Fuyu) has no tower: raw patches go through one embedder (`model.embed_vision`) into the
  text stream. `projector` names that embedder and so does `vision`, a `Vision` with no
  blocks that serves only the two image values, and `vision.image_features` is read at the
  scatter, since the embedder's output still holds the
  padding patches the forward strips.
- **Gemma 4's audio tower** is a Conformer; its blocks are not `Layer`/`Attention`/`Mlp`.
  The audio tower's component gets `audio_token_mask` and `audio_features`, as `Vision`
  carries the image values (read at the audio scatter, since the embedder's output still
  holds padding); the tower's blocks stay native.
- **InternVL's ViT** scales each sublayer by a learned `lambda_1`/`lambda_2` before the
  residual add, so its `attention_output`/`mlp_output` point at those products, the way
  Gemma's point at the post-norms.

## Where the code lives

```
nnterp/
  components/
    vision.py        Patches, ImageTokenMask, ImageFeatures; Vision (the tower root: sizes, image_token_mask,
                     patch_embeddings, tower_output, image_features, no_images, support);
                     VisionLayer, VisionAttention, VisionMlp (Layer/Attention/Mlp with Patches stream values)
    standard.py      blocks_support: the support walk over a block list, shared by model.support()
                     and model.vision.support()
    eproperty.py     root-anchored keys ("/projector.output"), walked from envoy.root
  standardized.py    support() lists a Standard child of the root (the tower) under its name
  families/
    gemma3_text.py   SigLIP's paths in RENAME, its module types in ENVOYS, IMAGE_WRAPPERS = ("gemma3",)
    llama.py         CLIP's paths in RENAME, its module types in ENVOYS, IMAGE_WRAPPERS = ("llava",)
```

A tower whose blocks are plain pre-norm attention + MLP on the shared attention interface
(SigLIP, CLIP) needs no class of its own: the family keys `VisionLayer`, `VisionAttention`,
`VisionMlp` and `Vision` on the tower's module types. A tower that differs (a sandwich
block, a packed attention, a scaled residual) gets a small subclass in
`components/vision.py`, beside the base, and the family keys that one. A tower hosted by
several families (SigLIP under Gemma 3, PaliGemma, LLaVA-OneVision, Aya Vision) repeats its
five or so `RENAME` lines and four `ENVOYS` entries in each family file.

What `gemma3_text.py` carries for the tower:

```python
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import Vision, VisionAttention, VisionLayer, VisionMlp

#: The wrappers (config ``model_type``) whose projector's output is what they scatter into the text stream.
IMAGE_WRAPPERS = ("gemma3",)

RENAME = {
    ...,                                             # the text stack, both spellings
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "post_layernorm": "norm",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
}

ENVOYS = {
    ...,                                             # the text blocks
    SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
}
```

## The image values

`image_token_mask` and `image_features` are `EProperty`s on `Vision`, so they read as
`model.vision.image_token_mask` and `model.vision.image_features`. Neither is read inside
the tower, so each key is anchored at the model's root with a leading `/` and walked from
`vision.root` by standard names, aliases included.

- `image_token_mask` is keyed `"/inputs"`: the root's inputs, the location
  `model.input_ids` reads, so it is read before anything else in the invoke. It returns
  `input_ids == image_token_id(config)` (`image_token_id`, else `image_token_index`, off the
  wrapper's config). Assigning raises.
- `image_features` is keyed `"/projector.output"`, `model.model.multi_modal_projector.output`
  on both wrappers: the projector, the tower's sibling under the root. It returns the output
  flattened to `[image_tokens, hidden]`, a view, so in-place edits land; an assignment is
  reshaped back to the projector's output.

What the tower reads outside itself (the wrapper's config, its processor, the family's
`IMAGE_WRAPPERS`, the root's `projector` alias) it reads through `self.root`, nnsight's
walk up the envoys' parent links to the `StandardizedTransformer`.

**Where `image_features` is read.** At the projector's output, on the wrappers where that
output is what the wrapper scatters into the token embeddings. The family lists those
wrappers by config `model_type` in `IMAGE_WRAPPERS`, each one checked by the suite
(`layers[0].input[image_token_mask] == image_features` exactly); `image_features`'
`unavailable=` predicate reads the list through `model.family`. The projector is a module
boundary: no instrumentation, no read-order trap. A wrapper that rearranges the projector's
output before the scatter (LLaVA-NeXT's unpadding and newline tokens; an encoder-free
embedder's padding) reads `image_features` at the scatter instead: the family names the
wrapper's `inputs_embeds.masked_scatter` operation and keys a `Standard` envoy with
`sourced = True` on the wrapper model's type (the scatter runs after the tower, inside the
same forward). Until a family does that for a wrapper, the wrapper is not in
`IMAGE_WRAPPERS` and `image_features` is `Unavailable` there with that reason, even where
the names bind.

**Availability.** `Vision.no_images()` is the one place that decides whether an image can
reach the model: not where the family names no `projector`, nor on a load with no processor
(`task="text-generation"`, which on Gemma 3 builds the wrapper with a tokenizer only).
There both image values are `Unavailable` with that reason, `model.vision.support()` is
empty (the tower never runs), and `model.support()` has no `vision` rows; a text-only
checkpoint has no `model.vision` at all. Otherwise `model.vision.support()` reports the
tower's values and its block values over `vision.layers` the way `model.support()` reports
the text blocks', and `model.support()` carries the same rows under the `vision` host
(`"vision.image_features"`, `"vision.self_attn.attention_probabilities"`): the root lists
every `Standard` child it has an alias for, as it lists the `Standard` children of a block.

## Loading

`StandardizedTransformer` loads for `task="text-generation"` unless the caller passes a
task; `task="image-text-to-text"` loads the wrapper with its processor. What each task
builds, run on the tiny checkpoints:

| config | `text-generation` | `image-text-to-text` |
|---|---|---|
| Gemma 3, Gemma 4, Gemma 4 unified | the wrapper (transformers maps the composite config to it under causal LM too), tokenizer only | the wrapper, with its processor |
| Llama 4, Qwen3.5, Qwen3.5-MoE, Mllama | the text-only class (`Llama4ForCausalLM`, `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, `MllamaForCausalLM`), weights read out of the wrapper checkpoint | the wrapper, with its processor |
| Llava, Mistral 3, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Kimi K2.5, PaliGemma, Idefics 3, DeepSeek-VL | refused on the lazy (meta) load: no causal-LM class for the config; `dispatch=True` falls back through the pipeline and loads the wrapper with a warning | the wrapper, with its processor |

So the text-only load is `task="text-generation"`, and nnterp has no option for it. With a
processor loaded, `model.tokenizer` is the processor's tokenizer, so every text helper keeps
working.

What a trace takes on a wrapper:

- `model.trace(prompt, images=[image])`, with the image placeholder in the prompt (the
  processor's chat template puts it there).
- `model.trace(encoding)` with an encoding built with `model.processor`.
- `model.trace("text")`, text only: works on every wrapper tested except PaliGemma, whose
  processor demands an image (pass the tokenizer's encoding there).

An invoke carrying `pixel_values` cannot share a trace with another invoke (nnsight refuses
it); several images or prompts go in one invoke, as lists. Two text-only invokes work.

## The suite

`tests/families/vision_suite.py` holds two bases.

`VisionSuite`, subclassed per wrapper in the family's test file (`TestGemma3Vision`,
`TestLlavaVision`), loads the pinned tiny wrapper with `task="image-text-to-text"`, eager, in
float32, and traces a fixed random image through the processor's chat template:

- the tower names alias the native modules; the tower's envoys are `Vision`, `VisionLayer`,
  `VisionAttention`, `VisionMlp`; no tower alias binds on a text block;
- `vision.num_layers`, `hidden_size`, `num_heads`, `intermediate_size`, `patch_size` are the
  vision config's; the root's sizes are the text config's;
- on every tower block, `input + attention_output + mlp_output == layer_output`, and under
  eager `attention_probabilities` rows sum to one;
- `vision.support()` lists the tower's four values and the block values, all available, and
  `model.support()` carries them as `vision.*` rows; the root has no image values;
- `image_token_mask == (input_ids == image_token_id)`, and its count is
  `image_features.shape[0]`;
- `layers[0].input[vision.image_token_mask] == vision.image_features` exactly;
- writes are causal: zeroing `image_features` lands at the image positions only and moves
  the logits, an assignment lands, a tower block's `layer_output` write and a
  `patch_embeddings` edit move `image_features`;
- `patch_embeddings` is the convolution's output flattened, `tower_output` the last block's
  stream after `vision.norm` where there is one;
- a text-only trace reads the text values with the mask all false; a text-only checkpoint
  of the family (and Gemma 3's `text-generation` load) has no `vision` host in `support()`.

Every `FamilySuite` test also runs on the wrappers loaded under `image-text-to-text`
(`TestLlavaWrapper`, `TestIdefics3Wrapper`, `TestGemma3ImageTextToText`,
`TestMistral3Wrapper`, `TestLlavaInterleaveWrapper`, `TestQwen3_5Wrapper`,
`TestQwen3_5MoeWrapper`), so the text side is checked as loaded with the processor.
`WrapperSuite` checks the text names, `support()` and the text identity on a wrapper with
no tiny checkpoint of its own, built from the family's tiny text config with random
weights (Mistral 3 around Mistral, Aya Vision, EXAONE 4.5, LightOnOCR), and on PaliGemma.

The pinned checkpoints, all loadable offline once cached:

| family | checkpoint | note |
|---|---|---|
| `gemma3_text` | `yujiepan/gemma-3-tiny-random` | `trl-internal-testing/tiny-Gemma3ForConditionalGeneration` (the wrapper test's) has a projector that outputs exact zeros, so no edit upstream of it shows |
| `llama` | `trl-internal-testing/tiny-LlavaForConditionalGeneration` | real check: `llava-hf/llava-1.5-7b-hf` |
| `qwen2` | `llava-hf/llava-interleave-qwen-0.5b-hf` (Llava with SigLIP) | |
| `qwen3_5_text`, `qwen3_5_moe_text` | `yujiepan/qwen3.5-tiny-random`, `yujiepan/qwen3.5-moe-tiny-random` | real check: `Qwen/Qwen3.5-0.8B` |
| `ministral3` | `yujiepan/mistral-3-tiny-random` | |
| `gemma` | `trl-internal-testing/tiny-PaliGemmaForConditionalGeneration` | |
| `gemma4_text` | `trl-internal-testing/tiny-Gemma4ForConditionalGeneration`; `yujiepan/gemma-4-e-tiny-random` (with audio) | phase 2 |
| `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text` | `yujiepan/qwen2-vl-tiny-random`, `trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration`, `yujiepan/qwen3-vl-tiny-random` | phase 2 |
| `llama4_text` | `yujiepan/llama-4-tiny-random`, config-patched as its text test does | phase 2 |
| `kimi_k2` | `hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration`, config-patched plus `image_token_id: 163602` | its processor uses 14-pixel patches merged 2x2 where its tower config says 8 and 1x1, so the image path needs a matching tiny before it can be tested |

## Phases

**Phase 0, the wrappers' text stacks.** Families whose checkpoints load as wrappers under
`image-text-to-text` carry the wrapper spellings of their text stack: `qwen3_5_text`,
`qwen3_5_moe_text`, `llama` (`model.language_model.*` and Idefics 3's `model.text_model.*`),
`qwen2`, `mistral`, `ministral3`, `gemma`, `cohere2`, `exaone4`, `qwen3`. Done.

**Phase 1, the per-image towers.** `components/vision.py` with the image values on the tower,
`VisionSuite`; SigLIP on `gemma3_text`, CLIP on `llama` (Llava 1.5). Done.

**Phase 2, the packed and the remaining towers.**

- New text families, re-exporting their base's components on the VL classes (M-RoPE), as
  `kimi_k2` re-exports DeepSeek-V3's: `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text`
  (with `deepstack_output`), `qwen3_vl_moe_text`. Today these wrappers resolve to a
  `model_type` with no family and raise `UnsupportedFamily`.
- Qwen's ViT for `qwen3_5_text`, `qwen3_5_moe_text` and the new Qwen-VL families (packed:
  the interior `Unavailable`); MoonViT for `kimi_k2`; Pixtral for `mistral` and
  `ministral3`; Llama 4's ViT for `llama4_text`; Gemma 4's ViT for `gemma4_text`; Gemma 4
  unified's embedder (no tower: a blockless `Vision` with the image values only, read at the scatter).
- SigLIP on the other families that host it (`gemma` for PaliGemma, `qwen2` for
  LLaVA-OneVision and llava-interleave, `cohere2` for Aya Vision), with their wrappers in
  `IMAGE_WRAPPERS` once the suite checks the scatter.
- From nnsight: the task derived from the config (`image-text-to-text` where
  `AutoModelForImageTextToText` maps it), after which nnterp drops its `text-generation`
  default; batching several image-carrying
  invokes in one trace; `mm_token_type_ids` padded as a row field, so two text invokes on
  Gemma 4's wrapper batch; the image-text-to-text mapping checked first when inferring a
  pre-loaded wrapper module's task.

**Phase 3, the rest.** Video and audio values (`video_token_mask`, `video_features`,
`audio_token_mask`, `audio_features`), Mllama's family (`mllama_text_model`) with a
`CrossAttention` component, and the long tail (InternViT, Idefics 3's ViT, LLaVA-NeXT's
packing read at the scatter).
