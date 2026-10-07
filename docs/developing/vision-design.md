---
title: Vision design
one_liner: How nnterp standardizes the vision side of image-text-to-text checkpoints — the tower under model.vision with its blocks and the image values, the projector, where image_features is read, where the code lives, how loading and the suite work, what does not fit, and what is left.
tags: [developing, design, vision, multimodal, families]
related: [docs/usage/vision.md, docs/developing/architecture.md, docs/developing/eproperty-internals.md, docs/developing/testing.md, docs/extending/adding-a-family.md]
sources: [nnterp/components/vision.py, nnterp/components/standard.py, nnterp/standardized.py, nnterp/components/eproperty.py, nnterp/families/gemma3_text.py, nnterp/families/gemma.py, nnterp/families/llama.py, nnterp/families/qwen2.py, nnterp/families/cohere2.py, nnterp/families/mistral.py, nnterp/families/ministral3.py, nnterp/families/llama4_text.py, nnterp/families/gemma4_text.py, nnterp/families/gemma4_unified_text.py, nnterp/families/qwen2_vl_text.py, nnterp/families/qwen2_5_vl_text.py, nnterp/families/qwen3_vl_text.py, nnterp/families/qwen3_vl_moe_text.py, nnterp/families/qwen3_5_text.py, nnterp/families/qwen3_5_moe_text.py, tests/families/vision_suite.py, tests/families/qwen_vision_suite.py, tests/families/suite.py]
---

# Vision design

## What this is for

An image-text-to-text checkpoint is a text model plus a vision tower, a projector, and a
step that puts the projected features into the text stream. nnterp standardizes the text
model already; this page is how the vision side is standardized so the same names and
values hold on every wrapper: what the tower and projector are called, which values the
tower serves, where `image_features` is read, where the code lives, and how loading and the
suite work. The names, values and per-tower layouts themselves are in
[docs/usage/vision.md](../usage/vision.md). Facts quoted here were run on transformers 5.17
and nnsight `dev` on the tiny checkpoints named in [the suite section](#the-suite).

The rules:

- **One family per text `model_type`, wrapper included.** The family is chosen from
  `text_config.model_type`. Its `RENAME` carries the wrapper's spellings of the text stack,
  the tower and the projector, and the tower's own names, keyed so that on a text-only
  checkpoint none of them resolve.
- **The root stays the text model's.** `model.num_layers`, `model.hidden_size`,
  `model.layers` are the language model's; the tower's sizes are on `model.vision`, and so
  are the values where the image meets the text model. Vision is a component, like `Moe`:
  the root's only vision code instruments its own forward where the family says the root
  scatters (`ROOT_SCATTER`, below).
- **A value means the same thing on every tower.** Where a tower differs (CLS first or
  last, tiles as rows, padded rows, a packed row, window order), the difference is a
  documented fact about the value, not another definition of it.
- **What cannot be served is `Unavailable`, with a reason that names the fix**; never a
  wrong number, never an out-of-order error in place of a reason.
- **Shared tower classes live in `nnterp/components/vision.py`; per-family paths live in the
  family file.** A tower's names are a handful of `RENAME` lines and four `ENVOYS` entries,
  plus one `ENVOYS` entry per wrapper keying `ImageScatter` on the wrapper's model.

## How the tower keys bind

nnsight resolves a `rename` key from every envoy and binds the alias on the envoy it
resolves from. So the tower root and the projector are keyed from the model root
(`"model.vision_tower": "vision"`), and the tower's inner names are keyed relative to the
tower: multi-component keys no text model has (`"encoder.layers": "layers"`,
`"embeddings.patch_embedding": "patch_embed"`, `"patch_embedder.input_proj": "patch_embed"`)
and single names no text block has (`"post_layernorm": "norm"`,
`"layer_norm1": "input_layernorm"`). A bare name that a text block also has is never a tower
key. Where a family hosts the same tower spelled the same way under another wrapper
(DeepSeek-VL's `model.vision_model`), the inner keys bind there too, on that native envoy;
only the root keys decide what `model.vision` is.

Where one family hosts a tower at two paths (`llama`: `model.vision_tower` and
`model.vision_model`), both spellings are in `RENAME`, as the text spellings are. Where one
family hosts two towers whose inner names differ (`mistral`: CLIP and Pixtral), both sets
are keyed; each binds only on its own tower. In `llama` CLIP's `post_layernorm` norms the
pooled CLS token and SigLIP's norms the patches; a rename key cannot tell them apart, so
`llama` keys `post_layernorm` on neither and its `SiglipVision` (on SigLIP, Idefics 3's and
SmolVLM's towers) serves `vision.norm` as a property.

`projector` names the last module before the scatter. A pooling or token-merging step
between the tower and the projector keeps its native name (Gemma 3's pooling is inside the
projector; Llama 4's pixel-shuffle is `vision.vision_adapter`; Idefics 3's pixel shuffle is
inside `model.connector`). `projector.input` is therefore what the host feeds its
projector, which is not always `tower_output`, and `projector.output` is not always what
the text model receives, which is why `image_features` is read at the scatter.

## The values

The layouts are `jaxtyping` types in `nnterp/components/vision.py`:

```python
from jaxtyping import Bool, Float
from torch import Tensor

#: A tower's stream: the tower's own batch (images, crops, tiles; 1 on a packed tower) by its tokens.
Patches = Float[Tensor, "images patches vision_hidden"]
#: What the text model receives at the image positions, flat in scatter (row-major) order.
ImageFeatures = Float[Tensor, "image_tokens hidden"]
#: Which positions of the text batch hold image tokens.
ImageTokenMask = Bool[Tensor, "batch seq"]
```

The tower's blocks are `Layer`, `Attention` and `Mlp` with the stream values re-annotated as
`Patches` (`VisionLayer`, `VisionAttention`, `VisionMlp`). A tower that runs on
`[patches, vision_hidden]` (the Qwen ViT) is served through the same classes: every stream
value reads a 2-D native tensor with a leading images axis of 1 (`as_patches`, a view) and
puts a write back in the module's shape (`as_native`), so there is no packed variant of the
classes. `tower_output` has one definition, the last block's stream after the final norm
where there is one, before any pooling, CLS dropping or adapter: the tower's
`last_hidden_state` by default, read elsewhere on a tower that returns something after those
(Llama 4 at `layernorm_post`, Gemma 4 at the encoder's output). Gemma 4 unified's embedder
has no blocks, so its `tower_output` is the states before the projection.

The sizes are the tower's, read off its own config on `model.vision`. A tower whose config
spells one its own way overrides the property on a `Vision` subclass (`QwenVision`); a tower
with no fixed resolution sets `image_size = property(variable_resolution)`, one reason on
every such tower.

Every tower value (`patch_embeddings`, `tower_output`, each block value, the attention
interior through `VisionAttention.off_interface`) is gated on `no_tower_run`, which walks up
to the `Vision` and asks `Vision.no_images()`: on a load no image reaches, the read raises
`Unavailable` saying how to load, instead of waiting in a trace for a tower that never runs.

## The image values

`image_token_mask` and `image_features` are `EProperty`s on `Vision`, so they read as
`model.vision.image_token_mask` and `model.vision.image_features`. Neither is read inside
the tower, so each key is anchored at the model's root with a leading `/` and walked from
`vision.root` by standard names, aliases included (see
[docs/extending/overriding-values.md](../extending/overriding-values.md)).

- `image_token_mask` is keyed `"/inputs"`: the root's inputs, the location
  `model.input_ids` reads, so it is read before anything else in the invoke. It returns
  `input_ids == image_token_id(config)` (`image_token_id`, else `image_token_index`, off the
  wrapper's config). Assigning raises.
- `image_features` is keyed at the scatter: on a Llava wrapper
  `"/model.source.inputs_embeds_masked_scatter_0.inputs"` with `select=1`, the features
  argument of `inputs_embeds.masked_scatter(image_mask, image_features)` in the wrapper
  model's forward. The key and the selection are functions of the host that read the
  scatter's host off the root (`scatter_host`, `scatter_call`). It returns that argument
  flattened to `[image_tokens, hidden]`, a view, so in-place edits land; an assignment is
  reshaped back to the argument's shape.

What the tower reads outside itself (the wrapper's config, its processor, the scatter's
host, the root's `projector` alias) it reads through `self.root`, nnsight's walk up the
envoys' parent links to the `StandardizedTransformer`.

**Where `image_features` is read.** At the scatter, on every wrapper: the tensor the
wrapper's forward writes into the token embeddings at the image tokens, so it is what the
text model receives whatever the wrapper did after its projector. The family keys
`ImageScatter` on the wrapper's model type in `ENVOYS` (`LlavaModel: ImageScatter`,
`Qwen2VLModel: ImageScatter`, `Gemma4UnifiedModel: ImageScatter`); that envoy is `sourced`,
so its forward is instrumented at build and the scatter is served after the tower's values,
which run inside the same forward (instrumenting on first read is too late there: the read
after a tower value raises `OutOfOrderError`). `ImageScatter.scatter` names the operation
(`inputs_embeds_masked_scatter_0` on every wrapper with a `masked_scatter`) and
`scatter_argument` the argument (1). Idefics 3 and SmolVLM write the features in through a
helper, `self.inputs_merger(input_ids=..., inputs_embeds=..., image_hidden_states=...)`;
SmolVLM's helper gathers rather than scatters. Their family keys a subclass,
`InputsMerger`, naming the helper call (`self_inputs_merger_0`) and its keyword argument
(`image_hidden_states`), whose flat row-major order is the order the helper places the rows
in. A wrapper whose family keys no `ImageScatter` binds the tower's names but
`image_features` is `Unavailable` there, with that reason.

**The root as the scatter's host.** Llama 4's `Llama4ForConditionalGeneration` scatters in
its own top-level forward, so there is no inner model to key: the root envoy is the
`StandardizedTransformer`. The family sets `ROOT_SCATTER = "inputs_embeds_masked_scatter_0"`;
`scatter_host` then returns the root (path `""`), the key is
`"/source.inputs_embeds_masked_scatter_0.inputs"`, and `StandardizedTransformer` instruments
its own forward at build and again when dispatch swaps in the real weights, where the
family names `ROOT_SCATTER` and the load has a `projector` (a text-generation load of the
same family builds `Llama4ForCausalLM` and is not instrumented).

The scatter is right where the projector's output is not what enters the text model:
LLaVA-NeXT and LLaVA-OneVision unpad the projector's output per image and append a newline
token per row (`image_newline`): on `trl-internal-testing/tiny-LlavaNextForConditionalGeneration`
with a 64x64 image the projector returns `[3, 576, 16]` (the base image and two crops,
1728 rows) and the text model receives 1176 rows, 24 of them `image_newline`. Gemma 4
unified strips the padded rows of the projector's output; Qwen2.5-VL restores the
merge-block order after the merger (`merger(hidden)[reverse_indices]`, an indexing op, not
a module), so its `projector.output` is in window order once an image spans more than one
window. On the other wrappers the scattered tensor is the projector's output, reshaped.
Reading at the scatter costs nothing measurable: an instrumented `LlavaModel` forward traced
`llava-hf/llava-1.5-7b-hf` in 111 ms against 110 ms without.

**Availability.** `Vision.no_images()` is the one place that decides whether an image can
reach the model: not where the family names no `projector`, nor on a load with no processor
(`task="text-generation"`, [Loading](#loading)). There every tower value is `Unavailable`
with that reason, `model.vision.support()` is empty (the tower never runs), and
`model.support()` has no `vision` rows; a text-only checkpoint has no `model.vision` at all.
Otherwise `model.vision.support()` reports the tower's values and its block values over
`vision.layers` the way `model.support()` reports the text blocks', and `model.support()`
carries the same rows under the `vision` host (`"vision.image_features"`,
`"vision.self_attn.attention_probabilities"`): the root lists every `Standard` child it has
an alias for, as it lists the `Standard` children of a block.

## What does not fit, and how it is handled

- **The Qwen ViT's attention.** The module splits the queries, keys and values per image
  (`torch.split` at `cu_seqlens`) and calls the interface once per image inside a list
  comprehension (op `attention_interface_2`; `_1` is the flash branch), under eager and
  sdpa alike; a value read at that call is one image's. `QwenVisionAttention` reads the
  interior around the calls instead: the queries, keys and values at the
  `unsqueeze_0..2` that end their preparation, whole `[1, heads, patches, head_dim]`, and the
  head outputs at the `torch.cat` that joins the calls' outputs, `[1, patches, heads,
  head_dim]` (not on the flash branch, which makes one varlen call). The suite checks on
  every block of the six tinies, with two images, that these are the calls' arguments and
  returns concatenated and that the output projection of the head outputs is
  `attention_output`. The scores and the pattern exist only per call, so they are
  `Unavailable` with `PER_IMAGE`, which says to split the queries and keys at `cu_seqlens`
  and never tells the user to load eager. Pixtral is packed too but calls the interface
  once over the whole row with a block-diagonal mask, so its interior is whole.
- **Pixtral's patch embedding.** The convolution runs on the batch padded to its largest
  image and each image is cropped to its own grid before the grids are concatenated, so the
  convolution's output is not the tower's stream. On `PixtralVision`, `patch_embeddings` is
  the packed row entering `ln_pre`, `[1, all patches, vision_hidden]`; `patch_embed` still
  names the convolution.
- **Qwen2.5-VL's window order.** The block values are served in the tower's own (window)
  order and documented as such; the order back needs the window index, which the tower
  computes inside its forward, so nnterp does not reorder.
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
- **An encoder-free wrapper** (Gemma 4 unified, `Gemma4UnifiedForConditionalGeneration`;
  Fuyu) has no tower: raw patches go through one embedder (`model.embed_vision`) into the
  text stream. `vision` names that embedder, a `Vision` with no `layers` (`num_layers` 0;
  the attention and MLP sizes raise `Unavailable`), and `projector` names the embedder's
  last stage, `embed_vision.multimodal_embedder`, the RMS norm and linear onto the text
  width: the same projection Gemma 4's own `model.embed_vision` is, and the one module a
  `RENAME` key can give that name (a key maps to one alias, so one module cannot be both).
  `vision.patch_embed` is the embedder's `patch_dense`; `patch_embeddings`, `tower_output`
  (the states the projector receives) and `image_token_mask` are served as on a tower, and
  `image_features` is read at the scatter, keyed on `Gemma4UnifiedModel`. Fuyu is not bound.
- **Mllama (Llama 3.2 Vision).** Its text model (`mllama_text_model`) interleaves
  cross-attention blocks that attend to `cross_attention_states` (the projector's output) and
  are skipped entirely on a text-only input; nothing is scattered. It needs its own family
  with a `CrossAttention` component; `image_features` is unavailable there with that reason.
  Its tower has two encoders (`transformer`, `global_transformer` with gated blocks) over
  tiles and concatenates intermediate layers' outputs.
- **Gemma 4's audio tower** is a Conformer; its blocks are not `Layer`/`Attention`/`Mlp`.
  The audio tower's component gets `audio_token_mask` and `audio_features`, as `Vision`
  carries the image values (read at the audio scatter, since the embedder's output still
  holds padding); the tower's blocks stay native.
- **InternVL's ViT** scales each sublayer by a learned `lambda_1`/`lambda_2` before the
  residual add, so its `attention_output`/`mlp_output` point at those products, the way
  Gemma's point at the post-norms.
- **MoonViT (Kimi K2.5)** is packed like the Qwen ViT (`model.vision_tower`, blocks
  `layers`, `final_layernorm`, projector `model.mm_projector`); its tiny checkpoint's
  processor uses 14-pixel patches merged 2x2 where its tower config says 8 and 1x1, so it
  waits for a matching tiny.

## Where the code lives

```
nnterp/
  components/
    vision.py        Patches, ImageTokenMask, ImageFeatures; Vision (the tower root: sizes, image_token_mask,
                     patch_embeddings, tower_output, image_features, no_images, support); VisionLayer,
                     VisionAttention, VisionMlp (Layer/Attention/Mlp with Patches stream values, a 2-D native
                     stream served with a leading 1); ImageScatter (the wrapper's model: where image_features
                     is read); scatter_host, scatter_call (the scatter's host, the root included); no_tower_run,
                     variable_resolution; QwenVision (the Qwen ViT's sizes), QwenVisionAttention and PER_IMAGE
                     (its interior around the per-image calls); PixtralVision (patch_embeddings at ln_pre)
    standard.py      blocks_support: the support walk over a block list, shared by model.support()
                     and model.vision.support()
    eproperty.py     root-anchored keys ("/inputs", "/model.source.<op>.inputs"), walked from envoy.root
  standardized.py    support() lists a Standard child of the root (the tower) under its name; the root's forward
                     instrumented where the family names ROOT_SCATTER
  families/
    gemma3_text.py   SigLIP's paths in RENAME; its module types and Gemma3Model: ImageScatter in ENVOYS
    gemma.py         the same for PaliGemma
    qwen2.py         the same for llava-interleave and LLaVA-OneVision
    cohere2.py       the same for Aya Vision and Cohere2-Vision
    llama.py         CLIP (Llava, VipLlava, LLaVA-NeXT), SigLIP (DeepSeek-VL), Idefics 3's and SmolVLM's ViT
                     (SiglipVision, InputsMerger)
    mistral.py       Pixtral (Mistral 3, Pixtral-12B) and CLIP (LLaVA-NeXT, BakLLaVA)
    ministral3.py    Pixtral (Mistral 3)
    llama4_text.py   Llama 4's ViT: a Vision subclass whose tower_output is layernorm_post's output; ROOT_SCATTER
    gemma4_text.py   Gemma 4's ViT: a Vision subclass (tower_output at the encoder) and VisionAttention/VisionMlp
                     subclasses pointing at the sandwich's post-norms; Gemma4Model: ImageScatter
    gemma4_unified_text.py
                     the encoder-free embedder: a blockless Vision; Gemma4UnifiedModel: ImageScatter
    qwen2_vl_text.py, qwen2_5_vl_text.py, qwen3_vl_text.py, qwen3_vl_moe_text.py, qwen3_5_text.py,
    qwen3_5_moe_text.py
                     the Qwen ViT's paths in RENAME; QwenVision, VisionLayer, QwenVisionAttention, VisionMlp and
                     the wrapper's model: ImageScatter in ENVOYS; qwen3_vl_text.py: deepstack_output
```

A tower whose blocks are plain pre-norm attention + MLP on the shared attention interface
(SigLIP, CLIP, Llama 4's ViT) needs no class of its own: the family keys `VisionLayer`,
`VisionAttention`, `VisionMlp` and `Vision` on the tower's module types. A tower that
differs (a sandwich block, a scaled residual) gets a small subclass in its family file,
named as the base (`class VisionAttention(VisionAttention)`, as the text classes are), and
the family keys that one; a subclass that several families need (`QwenVision`,
`QwenVisionAttention`, `PixtralVision`) lives in `components/vision.py`, beside the base.
A tower hosted by several families (SigLIP under Gemma 3, PaliGemma, LLaVA-OneVision, Aya
Vision) repeats its five or so `RENAME` lines and four `ENVOYS` entries in each family file.

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

## Loading

`StandardizedTransformer` loads for `task="text-generation"` unless the caller passes a
task; `task="image-text-to-text"` loads the wrapper with its processor. What each task
builds, run on the tiny checkpoints:

| config | `text-generation` | `image-text-to-text` |
|---|---|---|
| Gemma 3, Gemma 4, Gemma 4 unified | the wrapper (transformers maps the composite config to it under causal LM too), tokenizer only | the wrapper, with its processor |
| Llama 4, Qwen3.5, Qwen3.5-MoE, Mllama | the text-only class (`Llama4ForCausalLM`, `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, `MllamaForCausalLM`), weights read out of the wrapper checkpoint | the wrapper, with its processor |
| Llava, Mistral 3, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Kimi K2.5, PaliGemma, Idefics 3, DeepSeek-VL | refused on the lazy (meta) load: no causal-LM class for the config; `dispatch=True` falls back through the pipeline and loads the wrapper with a warning, tokenizer only | the wrapper, with its processor |

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
it); several images or prompts go in one invoke, as lists. Two text-only invokes work. Chat-message
inputs with the image embedded batch through the pipeline, but the scatter runs once over the batch,
so `image_features` is the whole batch's rows in every invoke (slicing it per invoke is open).

## The suite

`tests/families/vision_suite.py` holds two bases.

`VisionSuite`, subclassed per wrapper in the family's test file (`TestGemma3Vision`,
`TestLlavaVision`, `TestLlama4Vision`, `TestGemma4Vision`, `TestGemma4UnifiedVision`, ...),
loads the pinned tiny wrapper with `task="image-text-to-text"`, eager, in its `DTYPE`
(float32; Llama 4's is bfloat16), and traces a fixed random image through the processor's
chat template (the processor's image token before the text where the tiny has no usable
template). A subclass's `fix_processor` sets a tiny checkpoint's processor to its model
where the two disagree (the hf-tiny-v2 checkpoints' processors carry the full-size models'
patch sizes, token counts and crop grids, and their configs' `image_token_id` is not the
tokenizer's image token). Where a tower differs, a subclass sets a hook rather than
replacing a test: `patches_of(model, images)` (the length of the patches axis: the
configured grid by default, `image_grid_thw` on Qwen, `image_sizes` on Pixtral,
`image_position_ids` on Gemma 4), `PATCHES_AT` (where `patch_embeddings` is read:
`ln_pre.input` on Pixtral) and `EXPECTED_VISION_UNAVAILABLE` (the block values a tower does
not serve, with a substring of the reason). Every subclass checks:

- the tower names alias the native modules; the tower's envoys are `Vision`, `VisionLayer`,
  `VisionAttention`, `VisionMlp`; the scatter's host is an `ImageScatter` or the root
  (`ROOT_SCATTER`); no tower alias binds on a text block;
- the tower's sizes are what its modules run with (the first block's norm width, the
  attention's heads and head width, the MLP's width); the root's sizes are the text
  config's;
- on every tower block, `input + attention_output + mlp_output == layer_output`, and under
  eager `attention_probabilities` rows sum to one;
- `vision.support()` lists the tower's four values and the block values, available except
  `EXPECTED_VISION_UNAVAILABLE`, and `model.support()` carries them as `vision.*` rows; the
  root has no image values;
- `image_token_mask == (input_ids == image_token_id)`, and its count is
  `image_features.shape[0]`;
- `layers[0].input[vision.image_token_mask] == vision.image_features` exactly, with one
  image and with two images of different shapes in one invoke (a packed tower then holds
  both in one row, and Pixtral's pattern is zero between them);
- writes are causal: zeroing `image_features` lands at the image positions only and moves
  the logits, an assignment lands, a tower block's `layer_output` write and a
  `patch_embeddings` edit move `image_features`;
- `patch_embeddings` is the rows at `PATCHES_AT`, `tower_output` the last block's stream
  after `vision.norm` where there is one;
- a text-only trace reads the text values with the mask all false; a text-only checkpoint
  of the family has no `vision` host in `support()`; a `text-generation` load that builds
  the wrapper (dispatched) has none either, and every tower and block value raises
  `Unavailable` with the text-only reason, inside a trace too.

A family's test class adds its tower's own facts: Llama 4's CLS row is the last and is
dropped before the adapter, whose output is `projector.input`; Gemma 4's position embedding
comes after `patch_embeddings`, its padded rows are rows of every block's stream and the
pooler leaves `image_features` unchanged when they are zeroed, its contributions are the
post-norms, and an audio prompt still runs under both tasks; the encoder-free embedder has
no blocks, its `image_features` is `projector.output` with the padded rows stripped, and is
served after the embedder's own values; on LLaVA-NeXT (both families) and LLaVA-OneVision
`image_features` is not the projector's output: it has another row count and holds
`image_newline` rows.

`QwenVisionSuite` (`tests/families/qwen_vision_suite.py`) is `VisionSuite` on the Qwen ViT,
subclassed in the six families that host it: the stream values `[1, patches,
vision_hidden]` equal to the native packed tensors, the interior whole around the
per-image calls (above), the pattern `Unavailable` with `PER_IMAGE` under eager and sdpa,
one image token per merge block, the merger's output in scatter order except on Qwen2.5-VL,
and the Qwen sizes (`spatial_merge_size`, `window_size`). The text families' `FamilySuite`
subclasses mix in `MRopeSuite` (the queries and keys at the interface are `q_proj`/`k_proj`
rotated by the `cos`/`sin` the model folds from three position streams) and, on Qwen3-VL,
`DeepstackSuite`.

Every `FamilySuite` test also runs on the wrappers loaded under `image-text-to-text`
(`TestLlavaWrapper`, `TestIdefics3Wrapper`, `TestGemma3ImageTextToText`,
`TestLlavaInterleaveWrapper`, `TestQwen3_5Wrapper`, `TestQwen3_5MoeWrapper`,
`TestLlama4ImageTextToText`, `TestGemma4ImageTextToText`, ...), so the text side is checked
as loaded with the processor. `WrapperSuite` checks the text names, `support()` and the text
identity on a wrapper with no tiny checkpoint of its own, built from the family's tiny text
config with random weights (Mistral 3 around Mistral, Aya Vision, EXAONE 4.5, LightOnOCR),
and on PaliGemma; a wrapper built without a processor lists no `vision` rows, PaliGemma's
lists its tower.

The pinned checkpoints, all loadable offline once cached:

| family | checkpoint | note |
|---|---|---|
| `gemma3_text` | `yujiepan/gemma-3-tiny-random` | `trl-internal-testing/tiny-Gemma3ForConditionalGeneration` (the wrapper test's) has a projector that outputs exact zeros, so no edit upstream of it shows |
| `gemma` | `hf-tiny-v2/tiny-random-PaliGemmaForConditionalGeneration` | `trl-internal-testing/tiny-PaliGemmaForConditionalGeneration` (the text test's) projects to 2048 where its text model is 16 wide, so its image path fails in transformers' own forward |
| `llama` | `trl-internal-testing/tiny-LlavaForConditionalGeneration` | real check: `llava-hf/llava-1.5-7b-hf` |
| `llama` | `hf-tiny-v2/tiny-random-VipLlavaForConditionalGeneration`, `hf-tiny-v2/tiny-random-LlavaNextForConditionalGeneration`, `hf-tiny-v2/tiny-random-DeepseekVLForConditionalGeneration` | processors set to the model (`fix_processor`) |
| `llama` | `trl-internal-testing/tiny-Idefics3ForConditionalGeneration`, `trl-internal-testing/tiny-SmolVLMForConditionalGeneration` | SmolVLM's processor needs `num2words` |
| `qwen2` | `llava-hf/llava-interleave-qwen-0.5b-hf` (Llava with SigLIP); `hf-tiny-v2/tiny-random-LlavaOnevisionForConditionalGeneration` | llava-interleave's `VisionSuite` runs only where CUDA is: real weights, a trace per tower block |
| `cohere2` | `hf-tiny-v2/tiny-random-AyaVisionForConditionalGeneration`, `hf-tiny-v2/tiny-random-Cohere2VisionForConditionalGeneration` | |
| `mistral` | `trl-internal-testing/tiny-LlavaNextForConditionalGeneration` (LLaVA-NeXT); `hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration`; Pixtral-12B's Llava wrapper built small from `mistral-community/pixtral-12b`'s config with its processor | no tiny Pixtral-12B or BakLLaVA checkpoint exists |
| `ministral3` | `yujiepan/mistral-3-tiny-random` | |
| `qwen2_vl_text` | `yujiepan/qwen2-vl-tiny-random` | real check: `Qwen/Qwen2-VL-2B-Instruct` |
| `qwen2_5_vl_text` | `trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration` | the window order shows only on an image wider than one window: the suite's wide image is 256x320 |
| `qwen3_vl_text` | `yujiepan/qwen3-vl-tiny-random` | real check: `Qwen/Qwen3-VL-4B-Instruct`; two text blocks for three taps, so the blocks past the taps are checked on a four-block model built from its config |
| `qwen3_vl_moe_text` | `yujiepan/qwen3-vl-moe-tiny-random` | |
| `qwen3_5_text`, `qwen3_5_moe_text` | `yujiepan/qwen3.5-tiny-random`, `yujiepan/qwen3.5-moe-tiny-random` | real check: `Qwen/Qwen3.5-0.8B` (text side) |
| `llama4_text` | `yujiepan/llama-4-tiny-random`, config-patched as its text test does, in bfloat16 (its image processor returns bfloat16 pixels, which a float32 tower refuses); the text suite also runs on it under `image-text-to-text` | |
| `gemma4_text` | `yujiepan/gemma-4-e-tiny-random` (with audio, so the audio path is checked on the same load); the text suite also runs on `trl-internal-testing/tiny-Gemma4ForConditionalGeneration` under `image-text-to-text` | real check: `google/gemma-4-E2B` |
| `gemma4_unified_text` | a tiny wrapper the test builds once into the temp dir: `google/gemma-4-12B`'s config (config only is cached) with `hf-tiny-v2/tiny-random-Gemma4UnifiedForCausalLM`'s text config, a 16-wide embedder, random weights, and a processor from that checkpoint's tokenizer plus the default image processor | no tiny wrapper is published; 12B's own names are checked on meta |

## What is done, what is left

The wrappers' text stacks carry their wrapper spellings (`model.language_model.*`, Idefics 3's
`model.text_model.*`) on `qwen3_5_text`, `qwen3_5_moe_text`, `llama`, `qwen2`, `mistral`,
`ministral3`, `gemma`, `cohere2`, `exaone4`, `qwen3`. Every tower with a tiny checkpoint is
named and served: SigLIP on `gemma3_text`, `gemma` (PaliGemma), `qwen2` (llava-interleave,
LLaVA-OneVision), `cohere2` (Aya Vision, Cohere2-Vision) and `llama` (DeepSeek-VL, Idefics 3's
and SmolVLM's ViT); CLIP on `llama` (Llava 1.5, VipLlava, LLaVA-NeXT) and `mistral` (LLaVA-NeXT,
BakLLaVA); Pixtral on `mistral` and `ministral3`; the Qwen ViT on `qwen2_vl_text`,
`qwen2_5_vl_text`, `qwen3_vl_text` (with `deepstack_output`), `qwen3_vl_moe_text`,
`qwen3_5_text`, `qwen3_5_moe_text`; Llama 4's ViT on `llama4_text` (the root as the scatter's
host); Gemma 4's ViT on `gemma4_text`; Gemma 4 unified's embedder on `gemma4_unified_text`.
`image_features` is read at the scatter on every one, and `VisionSuite` runs on each.

Left, in rough order of value:

- From nnsight, the task derived from the config (`image-text-to-text` where
  `AutoModelForImageTextToText` maps it; nnsight #755), after which nnterp drops its
  `text-generation` default; batching several image-carrying invokes in one trace (the batcher
  has to collate `pixel_values`); `mm_token_type_ids` padded as a row field, so two text invokes
  on Gemma 4's wrapper batch; the image-text-to-text mapping checked first when inferring a
  pre-loaded wrapper module's task.
- MoonViT for `kimi_k2` (needs a matching tiny checkpoint); the towers of EXAONE 4.5 and
  LightOnOCR, whose text names bind but whose towers are native-only.
- Video and audio values (`video_token_mask`, `video_features`, `audio_token_mask`,
  `audio_features`; Gemma 4's audio tower as `model.audio` by the same pattern), Mllama's
  family (`mllama_text_model`) with a `CrossAttention` component, and the long tail (InternViT,
  Janus, Video-Llava's separate image and video towers, LLaVA-NeXT-Video).
- `scatter_host` walks nnsight's private `_named_children`; a public way to enumerate a root's
  children would remove that.
