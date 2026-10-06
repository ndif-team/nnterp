---
title: Vision design
one_liner: How nnterp standardizes the vision side of image-text-to-text checkpoints — the tower under model.vision with its blocks and the image values, the projector, where the code lives, how loading and the suite work, what does not fit, and the phases.
tags: [developing, design, vision, multimodal, families]
related: [docs/usage/vision.md, docs/developing/architecture.md, docs/developing/eproperty-internals.md, docs/developing/testing.md, docs/extending/adding-a-family.md]
sources: [nnterp/components/vision.py, nnterp/components/standard.py, nnterp/standardized.py, nnterp/components/eproperty.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, nnterp/families/llama4_text.py, nnterp/families/gemma4_text.py, nnterp/families/gemma4_unified_text.py, tests/families/vision_suite.py, tests/families/suite.py]
sources: [nnterp/components/vision.py, nnterp/components/standard.py, nnterp/standardized.py, nnterp/components/eproperty.py, nnterp/families/gemma3_text.py, nnterp/families/llama.py, nnterp/families/gemma.py, nnterp/families/qwen2.py, nnterp/families/cohere2.py, nnterp/families/mistral.py, nnterp/families/ministral3.py, tests/families/vision_suite.py, tests/families/suite.py]
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
  inline where the family needs them, plus one `ENVOYS` entry per wrapper keying
  `ImageScatter` on the wrapper's model, where the image features enter the text stream.

## The vocabulary

### Names

| standard name | what it is | Gemma 3 | Llava | Qwen3.5 / Qwen-VL | Llama 4 | Mistral 3 | Kimi K2.5 | Gemma 4 |
|---|---|---|---|---|---|---|---|---|
| `vision` | the tower's root module | `model.vision_tower` | `model.vision_tower` | `model.visual` | `vision_model` | `model.vision_tower` | `model.vision_tower` | `model.vision_tower` |
| `vision.patch_embed` | the patch embedding | `embeddings.patch_embedding` | `embeddings.patch_embedding` | `patch_embed` (native) | `patch_embedding` | `patch_conv` | `patch_embed` (native) | `patch_embedder.input_proj` |
| `vision.layers` | the tower's blocks | `encoder.layers` | `encoder.layers` | `blocks` | `model.layers` | `transformer.layers` | `layers` (native) | `encoder.layers` |
| `vision.layers[i].self_attn`, `.mlp`, `.input_layernorm`, `.post_attention_layernorm` | the block's sublayers and norms | `self_attn`, `mlp`, `layer_norm1`, `layer_norm2` | same as Gemma 3 | `attn`, `mlp`, `norm1`, `norm2` | native | `attention`, `feed_forward`, `attention_norm`, `ffn_norm` | `attn`, `mlp`, `norm1`, `norm2` | native (sandwich) |
| `vision.norm` | the tower's final norm over the patches, where it has one | `post_layernorm` | none (CLIP's `post_layernorm` norms the pooled CLS only) | none (the merger norms) | `layernorm_post` | none | `final_layernorm` | none |
| `projector` | the module whose output is scattered into the text stream | `model.multi_modal_projector` | `model.multi_modal_projector` | `model.visual.merger` | `multi_modal_projector` | `model.multi_modal_projector` | `model.mm_projector` | `model.embed_vision` |

Gemma 3, Llava, Llama 4 and Gemma 4 are implemented; the other columns are what the later phases bind. The
Gemma 3, Llava and the Qwen column are implemented; the other columns are what the later phases bind. The
Gemma 3, Llava and Mistral 3 are implemented, and SigLIP and CLIP on their other hosts
(below); the other columns are what the later phases bind. The
text names (`embed_tokens`, `layers`, `norm`, `lm_head`) keep their meaning; on a wrapper
they alias `model.language_model.*` (Llava's family included) or `model.text_model.*`
(Idefics 3, SmolVLM). Native names keep working everywhere.

The other hosts of the same towers:

| wrapper (`model_type`) | family | tower, at | `projector` |
|---|---|---|---|
| PaliGemma (`paligemma`) | `gemma` | SigLIP, `model.vision_tower` | `model.multi_modal_projector` (one linear) |
| llava-interleave (`llava`), LLaVA-OneVision (`llava_onevision`) | `qwen2` | SigLIP, `model.vision_tower` | `model.multi_modal_projector` |
| Aya Vision (`aya_vision`), Cohere2-Vision (`cohere2_vision`) | `cohere2` | SigLIP, `model.vision_tower` | `model.multi_modal_projector` (pixel shuffle inside) |
| DeepSeek-VL (`deepseek_vl`) | `llama` | SigLIP, `model.vision_model` | `model.aligner` |
| Idefics 3 (`idefics3`), SmolVLM (`smolvlm`) | `llama` | their SigLIP-shaped ViT, `model.vision_model` | `model.connector` (pixel shuffle inside) |
| VipLlava (`vipllava`), LLaVA-NeXT (`llava_next`) | `llama` | CLIP, `model.vision_tower` | `model.multi_modal_projector` |
| LLaVA-NeXT (`llava_next`, `llava-v1.6-mistral`), BakLLaVA (`llava`) | `mistral` | CLIP, `model.vision_tower` | `model.multi_modal_projector` |
| Mistral 3 (`mistral3`) | `mistral`, `ministral3` | Pixtral, `model.vision_tower` | `model.multi_modal_projector` (`patch_merger` inside) |
| Pixtral-12B (`llava`) | `mistral` | Pixtral, `model.vision_tower` | `model.multi_modal_projector` |

Where one family hosts a tower at two paths (`llama`: `model.vision_tower` and
`model.vision_model`), both spellings are in `RENAME`, as the text spellings are. Where one
family hosts two towers whose inner names differ (`mistral`: CLIP and Pixtral), both sets
are keyed; each binds only on its own tower. In `llama` CLIP's `post_layernorm` norms the
pooled CLS token and SigLIP's norms the patches; a rename key cannot tell them apart, so
`llama` keys `post_layernorm` on neither and its `SiglipVision` (on SigLIP, Idefics 3's and
SmolVLM's towers) serves `vision.norm` as a property.

`projector` names the last module before the scatter. A pooling or token-merging step
between the tower and the projector keeps its native name (Gemma 3's pooling is inside the
projector; Llama 4's pixel-shuffle is `vision.vision_adapter`; Kimi's temporal merge is a
method of the tower; Idefics 3's pixel shuffle is inside `model.connector`).
`projector.input` is therefore what the host feeds its projector, which is not always the
tower's output: Llava feeds `vision.layers[-2].layer_output` without the CLS token. And
`projector.output` is not always what the text model receives: LLaVA-NeXT and
LLaVA-OneVision unpad it and add a newline token per row, which is why `image_features` is
read at the scatter.

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
| `tower_output` | `vision` | `Patches` | the last block's stream after the tower's final norm, before any pooling or adapter (the tower's `last_hidden_state` on SigLIP and CLIP; `layernorm_post`'s output, CLS included, on Llama 4; the encoder's output on Gemma 4, whose tower pools after it) |
| `image_token_mask` | `vision` | `ImageTokenMask` `[batch seq]` | `input_ids == config.image_token_id`, off the model's inputs; read-only |
| `image_features` | `vision` | `ImageFeatures` `[image_tokens hidden]` | what the wrapper scatters into the token embeddings, flat over every image token of the batch, in scatter order: `layers[0].input[vision.image_token_mask] == vision.image_features` |

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
| Gemma 4 ViT | one row per image | patches padded to `max_soft_tokens * pooling_kernel_size**2` rows (2520 by default), whatever the image; the padded rows run through every block (masked as keys only) and the pooler zeroes and strips them |
| Gemma 4 unified's embedder (no blocks) | one row per image | `patch_embeddings` and `tower_output` padded to `max_soft_tokens` rows (280) the same way; the wrapper strips the padded rows after the projector |
| Pixtral (Mistral 3, LightOnOCR, Pixtral-12B's Llava wrapper) | 1 | every image's patches concatenated (block-diagonal mask) |
| Qwen2-VL, Qwen3-VL, Qwen3.5 ViT; MoonViT (Kimi K2.5) | 1 | every image's patches concatenated (`cu_seqlens`); natively `[patches, hidden]`, served with a leading 1; on the Qwen ViT each image's patches are in the processor's merge-block order (each `spatial_merge_size` x `spatial_merge_size` block consecutive), not raster order |
| Qwen2.5-VL ViT (also EXAONE 4.5) | 1 | as Qwen2-VL, but in *window order*: the tower permutes the merge blocks into attention windows at entry and restores the merge-block order after the merger |
| Gemma 4 ViT | one row per image | patches padded to the batch's longest; the padding is masked and stripped by the pooler |
| Pixtral (Mistral 3, LightOnOCR, Pixtral-12B's Llava wrapper) | 1 | every image's patches concatenated, each image's grid in raster order (block-diagonal mask) |
| Qwen2-VL, Qwen3-VL, Qwen3.5 ViT; MoonViT (Kimi K2.5) | 1 | every image's patches concatenated (`cu_seqlens`); natively `[patches, hidden]`, served with a leading 1 |
| Qwen2.5-VL ViT (also EXAONE 4.5) | 1 | as Qwen2-VL, but in *window order*: the tower permutes the patches into attention windows at entry and restores raster order after the merger |

A video enters as frames on the per-image towers and as temporal patches on the packed
ones. A user splits a packed row per image with the processor's `image_grid_thw` (Qwen,
Kimi) or `image_sizes` (Pixtral).

### What does not fit, and how it is handled

- **Attention interior on a packed tower.** Under eager attention, Qwen's ViT and MoonViT
  call the attention interface once per image, inside a list comprehension (op
  `attention_interface_2`; `_1` is the flash branch). A value at that call is one image's: a
  trace with two images served the first image's 256x256 pattern while the block held 536
  patches. The interior values are `Unavailable` on those towers, with the op named in the
  reason; `attention_output` and the block values are whole and available.
  `PackedVisionAttention.off_interface` returns that reason whatever `attn_implementation`
  is (sdpa runs the same per-image loop, flash one varlen call), ahead of `needs_eager`, so
  the reason never tells the user to load eager.
  reason; `attention_output` and the block values are whole and available. Pixtral is packed
  too but calls the interface once over the whole row with a block-diagonal mask, so its
  interior is whole: `attention_probabilities` is `[1, heads, all patches, all patches]`,
  zero between images, and every interior value is available.
- **Pixtral's patch embedding.** The convolution runs on the batch padded to its largest
  image and each image is cropped to its own grid before the grids are concatenated, so the
  convolution's output is not the tower's stream. On `PixtralVision`, `patch_embeddings` is
  the packed row entering `ln_pre`, `[1, all patches, vision_hidden]`; `patch_embed` still
  names the convolution.
- **Qwen2.5-VL's window order.** The block values are served in the tower's own (window)
  order and documented as such; the order back needs the window index, which the tower
  computes inside its forward, so nnterp does not reorder. The tower restores the order
  *after* the merger (`merger(hidden)[reverse_indices]`, an indexing op, not a module), so
  `projector.output` is in window order too and is not what the wrapper scatters once an
  image spans more than one window (`window_size` 112 pixels: a 256x320 image does).
  `image_features` on the Qwen ViT is therefore read at the tower's own output,
  `vision.output.pooler_output`, which is the merger's output in scatter order on all four
  towers (the same tensor as `projector.output` on Qwen2-VL, Qwen3-VL and Qwen3.5), and
  `IMAGE_WRAPPERS` lists each Qwen wrapper on that reading.
- **Qwen3-VL's DeepStack.** Three tower blocks (`deepstack_visual_indexes`) each feed a
  `deepstack_merger_list[k]` whose output the text model adds at the image positions after
  text block `k`, outside the block. So on `qwen3_vl_text`,
  `layers[k+1].input != layers[k].layer_output` at image positions for `k < 3`. The family
  serves it as a value of its own `Layer`: `layers[k].deepstack_output`, `[image_tokens
  hidden]`, read at the text model's `_deepstack_process` call (third argument), with
  `layers[k+1].input[mask] == layers[k].layer_output[mask] + deepstack_output`; unavailable
  on the other blocks. The tower side needs no new name: the taps are
  `vision.layers[i].layer_output` and the mergers keep their native path. The call runs
  from one line of the text model's loop, so every block's location is the same op and
  block `k`'s is its `k`-th occurrence: the value is a `DeepstackEProperty`, which pins the
  read or write to that occurrence (`pinned(k)`), and the text model's envoy sets
  `sourced = True` so the op exists before the blocks run. The value is also what an
  ablation of the image has to reach: on `Qwen/Qwen3-VL-4B-Instruct`, zeroing
  `image_features` alone leaves the answer to "what color is the square?" at "Red"; zeroing
  `deepstack_output` on blocks 0-2 as well makes it "White".
- **Mllama (Llama 3.2 Vision).** Its text model (`mllama_text_model`) interleaves
  cross-attention blocks that attend to `cross_attention_states` (the projector's output) and
  are skipped entirely on a text-only input; nothing is scattered. It needs its own family
  with a `CrossAttention` component; `image_features` is unavailable there with that reason.
  Its tower has two encoders (`transformer`, `global_transformer` with gated blocks) over
  tiles and concatenates intermediate layers' outputs.
- **An encoder-free wrapper** (Gemma 4 unified, `Gemma4UnifiedForConditionalGeneration`;
  Fuyu) has no tower: raw patches go through one embedder (`model.embed_vision`) into the
  text stream. `vision` names that embedder, a `Vision` with no `layers` (`num_layers` 0;
  the attention and MLP sizes raise `Unavailable`), and `projector` names the embedder's
  last stage, `embed_vision.multimodal_embedder`, the RMS norm and linear onto the text
  width: the same projection Gemma 4's own `model.embed_vision` is, and the one module a
  `RENAME` key can give that name (a key maps to one alias, so one module cannot be both).
  `vision.patch_embed` is the embedder's `patch_dense`; `patch_embeddings`, `tower_output`
  (the states the projector receives) and `image_token_mask` are served as on a tower, and
  `vision.image_features` is read at the scatter (the first `inputs_embeds.masked_scatter`
  of `Gemma4UnifiedModel.forward`, keyed `"/model.source.inputs_embeds_masked_scatter_0.inputs"`,
  select 1), since the projector's output still holds the padding patches the forward
  strips. The read follows the embedder's values inside the same forward, so the family
  keys a `Standard` with `sourced = True` on `Gemma4UnifiedModel`. Fuyu is not bound.
  text stream. `projector` names that embedder and so does `vision`, a `Vision` with no
  blocks that serves only the two image values; `vision.image_features` is read at the
  scatter as everywhere, which matters here, since the embedder's output still holds the
  padding patches the forward strips.
- **A wrapper that scatters in its own top-level forward** (Llama 4's
  `Llama4ForConditionalGeneration`) has no inner model to key `ImageScatter` on: the root
  envoy is the `StandardizedTransformer`. The root's forward can be instrumented ahead of a
  run (`model.source` before the trace makes its operations reachable after the tower's
  values are read); serving the scatter there needs the root to be instrumented at build
  when the family asks for it.
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
                     VisionLayer, VisionAttention, VisionMlp (Layer/Attention/Mlp with Patches stream values);
                     PackedVision, PackedVisionLayer, PackedVisionAttention, PackedVisionMlp (a packed tower:
                     the leading 1, the interior unavailable with PACKED); QwenVision (the Qwen ViT's sizes,
                     image_features at the tower's pooler_output)
                     ImageScatter (the wrapper's model: where image_features is read); PixtralVision
    standard.py      blocks_support: the support walk over a block list, shared by model.support()
                     and model.vision.support()
    eproperty.py     root-anchored keys ("/inputs", "/model.source.<op>.inputs"), walked from envoy.root
  standardized.py    support() lists a Standard child of the root (the tower) under its name
  families/
    gemma3_text.py   SigLIP's paths in RENAME, its module types in ENVOYS, IMAGE_WRAPPERS = ("gemma3",)
    llama.py         CLIP's paths in RENAME, its module types in ENVOYS, IMAGE_WRAPPERS = ("llava",)
    llama4_text.py   Llama 4's ViT; a Vision subclass whose tower_output is layernorm_post's output
    gemma4_text.py   Gemma 4's ViT; a Vision subclass (tower_output at the encoder, no image_size) and
                     VisionAttention/VisionMlp subclasses pointing at the sandwich's post-norms
    gemma4_unified_text.py  the encoder-free embedder: a blockless Vision reading image_features at the
                     scatter, and a sourced Standard on Gemma4UnifiedModel
    qwen2_vl_text.py, qwen2_5_vl_text.py, qwen3_vl_text.py, qwen3_vl_moe_text.py, qwen3_5_text.py,
    qwen3_5_moe_text.py
                     the Qwen ViT's paths in RENAME, its module types keyed to the packed classes in ENVOYS,
                     the family's wrapper in IMAGE_WRAPPERS; qwen3_vl_text.py: deepstack_output
    gemma3_text.py   SigLIP's paths in RENAME; its module types and Gemma3Model: ImageScatter in ENVOYS
    gemma.py         the same for PaliGemma
    qwen2.py         the same for llava-interleave and LLaVA-OneVision
    cohere2.py       the same for Aya Vision and Cohere2-Vision
    llama.py         CLIP (Llava, VipLlava, LLaVA-NeXT), SigLIP (DeepSeek-VL), Idefics 3's and SmolVLM's ViT
                     (SiglipVision, InputsMerger)
    mistral.py       Pixtral (Mistral 3, Pixtral-12B) and CLIP (LLaVA-NeXT, BakLLaVA)
    ministral3.py    Pixtral (Mistral 3)
```

A tower whose blocks are plain pre-norm attention + MLP on the shared attention interface
(SigLIP, CLIP) needs no class of its own: the family keys `VisionLayer`, `VisionAttention`,
`VisionMlp` and `Vision` on the tower's module types. A tower that differs (a sandwich
block, a packed attention, a scaled residual) gets a small subclass in its family file,
named as the base (`class VisionAttention(VisionAttention)`, as the text classes are), and
the family keys that one; a subclass that several families need moves to
`components/vision.py`, beside the base. A tower hosted by
several families (SigLIP under Gemma 3, PaliGemma, LLaVA-OneVision, Aya Vision) repeats its
five or so `RENAME` lines and four `ENVOYS` entries in each family file.

What `gemma3_text.py` carries for the tower:

```python
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import ImageScatter, Vision, VisionAttention, VisionLayer, VisionMlp

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
    Gemma3Model: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
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
the names bind. A tower whose projector is inside it and whose output is the scattered
tensor reads `image_features` there: the Qwen ViT returns the merger's output as
`pooler_output`, after Qwen2.5-VL's window restore, so `QwenVision.image_features` is keyed
`"output"` (the tower's own) and serves `pooler_output`.
- `image_features` is keyed at the scatter: `"/model.source.inputs_embeds_masked_scatter_0.inputs"`
  with `select=1`, the features argument of `inputs_embeds.masked_scatter(image_mask,
  image_features)` in the wrapper model's forward (the key and the selection are functions
  of the host that read the scatter's host off the root). It returns that argument flattened
  to `[image_tokens, hidden]`, a view, so in-place edits land; an assignment is reshaped back
  to the argument's shape.

What the tower reads outside itself (the wrapper's config, its processor, the scatter's
host, the root's `projector` alias) it reads through `self.root`, nnsight's walk up the
envoys' parent links to the `StandardizedTransformer`.

**Where `image_features` is read.** At the scatter, on every wrapper: the tensor the
wrapper's forward writes into the token embeddings at the image tokens, so it is what the
text model receives whatever the wrapper did after its projector. The family keys
`ImageScatter` on the wrapper's model type in `ENVOYS` (`LlavaModel: ImageScatter`); that
envoy is `sourced`, so its forward is instrumented at build and the scatter is served after
the tower's values, which run inside the same forward (instrumenting on first read is too
late there: the read after a tower value raises `OutOfOrderError`). `ImageScatter.scatter`
names the operation (`inputs_embeds_masked_scatter_0` on every wrapper with a
`masked_scatter`) and `scatter_argument` the argument (1). Idefics 3 and SmolVLM write the
features in through a helper, `self.inputs_merger(input_ids=..., inputs_embeds=...,
image_hidden_states=...)`; SmolVLM's helper gathers rather than scatters. Their family keys
a subclass, `InputsMerger`, naming the helper call (`self_inputs_merger_0`) and its keyword
argument (`image_hidden_states`), whose flat row-major order is the order the helper places
the rows in. A wrapper whose family keys no `ImageScatter` binds the tower's names but
`image_features` is `Unavailable` there, with that reason.

The scatter is right where the projector's output is not what enters the text model:
LLaVA-NeXT and LLaVA-OneVision unpad the projector's output per image and append a newline
token per row (`image_newline`): on `trl-internal-testing/tiny-LlavaNextForConditionalGeneration`
with a 64x64 image the projector returns `[3, 576, 16]` (the base image and two crops,
1728 rows) and the text model receives 1176 rows, 24 of them `image_newline`. On the
other wrappers the scattered tensor is the projector's output, reshaped. Reading at the
scatter costs nothing measurable: an instrumented `LlavaModel` forward traced
`llava-hf/llava-1.5-7b-hf` in 111 ms against 110 ms without.

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
`TestLlavaVision`, `TestLlama4Vision`, `TestGemma4Vision`, `TestGemma4UnifiedVision`), loads
the pinned tiny wrapper with `task="image-text-to-text"`, eager, in float32 (its `DTYPE`;
Llama 4's is bfloat16), and traces a fixed random image through the processor's chat
template:
`TestLlavaVision`, ...), loads the pinned tiny wrapper with `task="image-text-to-text"`,
eager, in float32, and traces a fixed random image through the processor's chat template
(the processor's image token before the text where the tiny has no usable template). A
subclass's `fix_processor` sets a tiny checkpoint's processor to its model where the two
disagree (the hf-tiny-v2 checkpoints' processors carry the full-size models' patch sizes,
token counts and crop grids, and their configs' `image_token_id` is not the tokenizer's
image token); `PixtralSuite` replaces the patch-embedding checks for a packed tower:

- the tower names alias the native modules; the tower's envoys are `Vision`, `VisionLayer`,
  `VisionAttention`, `VisionMlp`; the wrapper's model is an `ImageScatter`; no tower alias
  binds on a text block;
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
- `patch_embeddings` is the patch embedding's output (a convolution's flattened),
  `tower_output` the last block's stream after `vision.norm` where there is one;
- a text-only trace reads the text values with the mask all false; a text-only checkpoint
  of the family (and Gemma 3's `text-generation` load) has no `vision` host in `support()`;
- on Pixtral (`PixtralSuite`): `patch_embeddings` is `[1, all patches, vision_hidden]`, the
  row entering `ln_pre`, with each image's patch count off the processor's `image_sizes`;
  two images of different shapes make one row whose pattern is zero between the images
  and sums to one within each, and the scatter identity holds;
- on LLaVA-NeXT (both families) and LLaVA-OneVision, `image_features` is not the projector's
  output: it has another row count and holds `image_newline` rows.

A family's test class adds its tower's own facts: Llama 4's CLS row is the last and is
dropped before the adapter; Gemma 4's padded rows are rows of every block's stream and
the pooler leaves `image_features` unchanged when they are zeroed, its contributions are
the post-norms, and an audio prompt still runs under both tasks; the encoder-free
embedder has no blocks, its `image_features` is `projector.output` with the padded rows
stripped, and is served after the embedder's own values.

Every `FamilySuite` test also runs on the wrappers loaded under `image-text-to-text`
(`TestLlavaWrapper`, `TestIdefics3Wrapper`, `TestGemma3ImageTextToText`,
`TestMistral3Wrapper`, `TestLlavaInterleaveWrapper`, `TestQwen3_5Wrapper`,
`TestQwen3_5MoeWrapper`, `TestLlama4ImageTextToText`, `TestGemma4ImageTextToText`), so the text side is checked as loaded with the processor.
`WrapperSuite` checks the text names, `support()` and the text identity on a wrapper with
no tiny checkpoint of its own, built from the family's tiny text config with random
weights (Mistral 3 around Mistral, Aya Vision, EXAONE 4.5, LightOnOCR), and on PaliGemma;
a wrapper built without a processor lists no `vision` rows, PaliGemma's lists its tower.

`QwenVisionSuite` (`tests/families/qwen_vision_suite.py`) is `VisionSuite` on the packed
Qwen ViT, subclassed in the six families that host it: the sizes off the Qwen vision
config, the stream values `[1, patches, vision_hidden]` equal to the native packed tensors,
the interior `Unavailable` with `PACKED` under eager and sdpa, `image_features` equal to the
tower's `pooler_output`, the merger's output in scatter order except on Qwen2.5-VL, and two
images in one invoke (the patches of both in one row, both images' tokens in the mask, the
scatter exact). The text families' `FamilySuite` subclasses mix in `MRopeSuite` (the queries
and keys at the interface are `q_proj`/`k_proj` rotated by the `cos`/`sin` the model folds
from three position streams) and, on Qwen3-VL, `DeepstackSuite`.

The pinned checkpoints, all loadable offline once cached:

| family | checkpoint | note |
|---|---|---|
| `gemma3_text` | `yujiepan/gemma-3-tiny-random` | `trl-internal-testing/tiny-Gemma3ForConditionalGeneration` (the wrapper test's) has a projector that outputs exact zeros, so no edit upstream of it shows |
| `llama` | `trl-internal-testing/tiny-LlavaForConditionalGeneration` | real check: `llava-hf/llava-1.5-7b-hf` |
| `qwen2` | `llava-hf/llava-interleave-qwen-0.5b-hf` (Llava with SigLIP) | |
| `qwen3_5_text`, `qwen3_5_moe_text` | `yujiepan/qwen3.5-tiny-random`, `yujiepan/qwen3.5-moe-tiny-random` | real check: `Qwen/Qwen3.5-0.8B` (text side) |
| `ministral3` | `yujiepan/mistral-3-tiny-random` | |
| `gemma` | `trl-internal-testing/tiny-PaliGemmaForConditionalGeneration` | |
| `gemma4_text` | `yujiepan/gemma-4-e-tiny-random` (with audio, so the audio path is checked on the same load); the text suite also runs on `trl-internal-testing/tiny-Gemma4ForConditionalGeneration` under `image-text-to-text` | real check: `google/gemma-4-E2B` |
| `gemma4_unified_text` | a tiny wrapper the test builds once into the temp dir: `google/gemma-4-12B`'s config (config only is cached) with `hf-tiny-v2/tiny-random-Gemma4UnifiedForCausalLM`'s text config, a 16-wide embedder, random weights, and a processor from that checkpoint's tokenizer plus the default image processor | no tiny wrapper is published; 12B's own names are checked on meta |
| `llama` | `hf-tiny-v2/tiny-random-VipLlavaForConditionalGeneration`, `hf-tiny-v2/tiny-random-LlavaNextForConditionalGeneration`, `hf-tiny-v2/tiny-random-DeepseekVLForConditionalGeneration` | processors set to the model (`fix_processor`) |
| `llama` | `trl-internal-testing/tiny-Idefics3ForConditionalGeneration`, `trl-internal-testing/tiny-SmolVLMForConditionalGeneration` | SmolVLM's processor needs `num2words` |
| `qwen2` | `llava-hf/llava-interleave-qwen-0.5b-hf` (Llava with SigLIP); `hf-tiny-v2/tiny-random-LlavaOnevisionForConditionalGeneration` | llava-interleave's `VisionSuite` runs only where CUDA is: real weights, a trace per tower block |
| `cohere2` | `hf-tiny-v2/tiny-random-AyaVisionForConditionalGeneration`, `hf-tiny-v2/tiny-random-Cohere2VisionForConditionalGeneration` | |
| `mistral` | `trl-internal-testing/tiny-LlavaNextForConditionalGeneration` (LLaVA-NeXT); `hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration`; Pixtral-12B's Llava wrapper built small from `mistral-community/pixtral-12b`'s config with its processor | no tiny Pixtral-12B or BakLLaVA checkpoint exists |
| `qwen3_5_text`, `qwen3_5_moe_text` | `yujiepan/qwen3.5-tiny-random`, `yujiepan/qwen3.5-moe-tiny-random` | real check: `Qwen/Qwen3.5-0.8B` |
| `ministral3` | `yujiepan/mistral-3-tiny-random` | |
| `gemma` | `hf-tiny-v2/tiny-random-PaliGemmaForConditionalGeneration` | `trl-internal-testing/tiny-PaliGemmaForConditionalGeneration` (the text test's) projects to 2048 where its text model is 16 wide, so its image path fails in transformers' own forward |
| `gemma4_text` | `trl-internal-testing/tiny-Gemma4ForConditionalGeneration`; `yujiepan/gemma-4-e-tiny-random` (with audio) | phase 2 |
| `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text` | `yujiepan/qwen2-vl-tiny-random`, `trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration`, `yujiepan/qwen3-vl-tiny-random` | phase 2 |
| `llama4_text` | `yujiepan/llama-4-tiny-random`, config-patched as its text test does, in bfloat16 (its image processor returns bfloat16 pixels, which a float32 tower refuses); the text suite also runs on it under `image-text-to-text` | |
| `gemma4_text` | `trl-internal-testing/tiny-Gemma4ForConditionalGeneration`; `yujiepan/gemma-4-e-tiny-random` (with audio) | phase 2 |
| `qwen2_vl_text` | `yujiepan/qwen2-vl-tiny-random` | real check: `Qwen/Qwen2-VL-2B-Instruct` |
| `qwen2_5_vl_text` | `trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration` | the window order shows only on an image wider than one window: the suite's second image is 256x320 |
| `qwen3_vl_text` | `yujiepan/qwen3-vl-tiny-random` | real check: `Qwen/Qwen3-VL-4B-Instruct`; two text blocks for three taps, so the blocks past the taps are checked on a four-block model built from its config |
| `qwen3_vl_moe_text` | `yujiepan/qwen3-vl-moe-tiny-random` | |
| `llama4_text` | `yujiepan/llama-4-tiny-random`, config-patched as its text test does | phase 2 |
| `kimi_k2` | `hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration`, config-patched plus `image_token_id: 163602` | its processor uses 14-pixel patches merged 2x2 where its tower config says 8 and 1x1, so the image path needs a matching tiny before it can be tested |

## Phases

**Phase 0, the wrappers' text stacks.** Families whose checkpoints load as wrappers under
`image-text-to-text` carry the wrapper spellings of their text stack: `qwen3_5_text`,
`qwen3_5_moe_text`, `llama` (`model.language_model.*` and Idefics 3's `model.text_model.*`),
`qwen2`, `mistral`, `ministral3`, `gemma`, `cohere2`, `exaone4`, `qwen3`. Done.

**Phase 1, the per-image towers.** `components/vision.py` with the image values on the tower,
`VisionSuite`; SigLIP on `gemma3_text`, CLIP on `llama` (Llava 1.5). Done.

**Phase 2, part B.** SigLIP on `gemma` (PaliGemma), `qwen2` (llava-interleave,
LLaVA-OneVision), `cohere2` (Aya Vision, Cohere2-Vision) and `llama` (DeepSeek-VL, and
Idefics 3's and SmolVLM's ViT); CLIP on `llama` (VipLlava, LLaVA-NeXT) and `mistral`
(LLaVA-NeXT, BakLLaVA); Pixtral on `mistral` and `ministral3`; `image_features` read at the
scatter on every wrapper (`ImageScatter`). Done.

**Phase 2, the packed and the remaining towers.**

- New text families on the VL classes (M-RoPE; the base `Attention` holds, since the
  rotation is applied before the interface): `qwen2_vl_text`, `qwen2_5_vl_text`,
  `qwen3_vl_text` (with `deepstack_output`), `qwen3_vl_moe_text` (with `Moe`). Done.
- Qwen's ViT for `qwen3_5_text`, `qwen3_5_moe_text` and the new Qwen-VL families (packed:
  the interior `Unavailable`); MoonViT for `kimi_k2`; Pixtral for `mistral` and
  `ministral3`. Llama 4's ViT for `llama4_text`, Gemma 4's ViT for `gemma4_text` and Gemma 4
  unified's embedder (a blockless `Vision`, `image_features` read at the scatter): done.
  the interior `Unavailable`; `image_features` at the tower's `pooler_output`). Done.
- MoonViT for `kimi_k2`; Pixtral for `mistral` and
  `ministral3`; Llama 4's ViT for `llama4_text`; Gemma 4's ViT for `gemma4_text`; Gemma 4
  unified's embedder (no tower: a blockless `Vision` with the image values only, read at the scatter).
- SigLIP on the other families that host it (`gemma` for PaliGemma, `qwen2` for
  LLaVA-OneVision and llava-interleave, `cohere2` for Aya Vision), with their wrappers in
  `IMAGE_WRAPPERS` once the suite checks the scatter.
  the interior `Unavailable`); MoonViT for `kimi_k2`; Llama 4's ViT for `llama4_text`;
  Gemma 4's ViT for `gemma4_text`; Gemma 4 unified's embedder (no tower: a blockless
  `Vision` with the image values only). Each wrapper's model is keyed `ImageScatter`.
- From nnsight: the task derived from the config (`image-text-to-text` where
  `AutoModelForImageTextToText` maps it), after which nnterp drops its `text-generation`
  default; batching several image-carrying
  invokes in one trace; `mm_token_type_ids` padded as a row field, so two text invokes on
  Gemma 4's wrapper batch; the image-text-to-text mapping checked first when inferring a
  pre-loaded wrapper module's task.

**Phase 3, the rest.** Video and audio values (`video_token_mask`, `video_features`,
`audio_token_mask`, `audio_features`), Mllama's family (`mllama_text_model`) with a
`CrossAttention` component, and the long tail (InternViT, Janus, Video-Llava's separate
image and video towers, LLaVA-NeXT-Video).
