# Vision

On a checkpoint loaded with `task="image-text-to-text"`, the family's `RENAME` names the vision tower `vision` and the last module before the scatter `projector`, and its `ENVOYS` wrap the tower in `Vision` and the tower's blocks in `VisionLayer`, `VisionAttention` and `VisionMlp`, the text classes with their stream values re-annotated as `Patches`. The two values where the image meets the text model, `image_token_mask` and `image_features`, are on `Vision` and read outside the tower: the mask off the model's inputs, the features at the scatter in the forward of the wrapper model the family keys `ImageScatter` on. The classes are in `nnterp/components/vision.py` ([vision](../../documentation/components/vision.md)), and the user-facing page is [Vision-language](../../guide/architectures/vision-language.md).

```python
import numpy as np
import torch
from PIL import Image
import nnterp
from nnterp import StandardizedTransformer
from nnterp.components.vision import scatter_call, scatter_host

model = StandardizedTransformer("llava-hf/llava-interleave-qwen-0.5b-hf", task="image-text-to-text", dispatch=True, attn_implementation="eager", device="cpu")
print(model.family.__name__, type(model._module).__name__)   # nnterp.families.qwen2 LlavaForConditionalGeneration
print(model._aliases)
# {'embed_tokens': 'model.language_model.embed_tokens', 'layers': 'model.language_model.layers', 'norm': 'model.language_model.norm', 'vision': 'model.vision_tower', 'projector': 'model.multi_modal_projector'}
print(model.vision._aliases, model.vision.layers[0]._aliases)
# {'patch_embed': 'embeddings.patch_embedding', 'layers': 'encoder.layers', 'norm': 'post_layernorm'} {'input_layernorm': 'layer_norm1', 'post_attention_layernorm': 'layer_norm2'}
print(model.vision.num_layers, model.vision.vision_hidden_size, model.vision.patch_size, model.num_layers, model.hidden_size)   # 26 1152 14 24 1024
print(scatter_host(model)[0], type(scatter_host(model)[1]).__name__, scatter_call(model))   # model ImageScatter ('model.source.inputs_embeds_masked_scatter_0', 1)
print(type(model.vision).image_features.path(model.vision), type(model.vision).image_token_mask.key)   # /model.source.inputs_embeds_masked_scatter_0.inputs /inputs

image = Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8"))
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in this image?"}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    tower = model.vision.tower_output.save()
    proj = model.projector.output.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()
print(tuple(mask.shape), int(mask.sum()), tuple(tower.shape), tuple(proj.shape), tuple(features.shape))   # (1, 744) 729 (1, 729, 1152) (1, 729, 1024) (729, 1024)
print(torch.equal(first[mask], features))   # True
```

The family is `qwen2`, chosen from `text_config.model_type`, and the wrapper's spellings of the text stack and the SigLIP tower's paths are in that module's `RENAME` beside the plain Qwen2 ones. The reads follow the forward: the mask comes off the inputs, the tower and projector run, the scatter writes the features into the embeddings, and block 0 receives them.

## How the tower's keys bind

nnsight resolves every rename key from every envoy and binds the alias on the envoy it resolves from ([Architecture](architecture.md)). The tower root and the projector are keyed from the model root (`"model.vision_tower": "vision"`), and the tower's inner names are keyed relative to the tower: multi-component keys no text model has (`"encoder.layers": "layers"`, `"embeddings.patch_embedding": "patch_embed"`) and single names no text block has (`"post_layernorm": "norm"`, `"layer_norm1": "input_layernorm"`). A bare name a text block also has is never a tower key. Where a family hosts the same tower under two wrappers at two paths (`llama`: `model.vision_tower` and `model.vision_model`) both are keyed, and where two towers spell their insides differently (`mistral`: CLIP and Pixtral) both sets are keyed and each binds only on its own tower. A rename key cannot tell CLIP's `post_layernorm`, which norms the pooled CLS token, from SigLIP's, which norms the patches, so `llama` keys neither and its `SiglipVision` serves `norm` as a property.

Because `projector` is the last module before the scatter, `projector.input` is what the host feeds its projector, not always `tower_output` (Llava 1.5 feeds block -2's stream, llava-interleave the last block's stream before `post_layernorm`), and `projector.output` is not always what the text model receives (LLaVA-NeXT unpads it and appends newline rows). That is why `image_features` is read at the scatter.

## The blocks and the tower's values

`Patches` is `[images, patches, vision_hidden_size]`, the tower's own batch by its tokens. `VisionLayer`, `VisionAttention` and `VisionMlp` redefine the stream values of `Layer`, `Attention` and `Mlp` with that annotation and inherit the attention interior unchanged, so `attention_probabilities` is a `Pattern` over the patches with no causal mask. A tower that runs on `[patches, vision_hidden_size]` (the Qwen ViT) is served through the same classes: `as_patches` gives the 2-D native tensor a leading images axis of 1 as a view, and `as_native` drops it on a write, which leaves no need for a packed variant of the classes. Every block value is gated on `no_tower_run`, which walks up the parent links to the `Vision` and asks `no_images()`.

`Vision` serves `patch_embeddings` (the patch embedding's output, a convolution's grid flattened to rows as a view) and `tower_output` (the last block's stream after the tower's final norm where it has one, before any pooling, CLS dropping or adapter: the output's `last_hidden_state` by default, which a tower that returns something after those reads elsewhere, as `llama4_text`'s subclass does). Its sizes are read off the tower's own config and shadow the root's on purpose, since a tower value's axes are the tower's sizes. `vision_hidden_size` is named apart from `hidden_size` because `image_features` is the text model's width served on the tower, and `axis_size` walks past the tower to the root for it. A tower with no fixed resolution sets `image_size = property(variable_resolution)`, which raises `Unavailable` naming where each image's grid is.

## Where `image_features` is read

The key is a function of the host: `image_scatter` returns `f"/{path}.inputs"` and `scattered_argument` the `select`, both from `scatter_call(vision.root)`. `scatter_host` finds the root's child the family keyed `ImageScatter` on (`LlavaModel`, `Qwen2VLModel`, `Gemma3Model`), or the root itself where the family sets `ROOT_SCATTER` because the wrapper scatters in its own top-level forward (Llama 4, NemotronH Omni). `ImageScatter.scatter` names the operation, `inputs_embeds_masked_scatter_0` on every wrapper with a `masked_scatter`, and `scatter_argument` the argument, 1. Idefics 3 and SmolVLM write the features in through a helper call, and `llama`'s `InputsMerger` subclass names `self_inputs_merger_0` and its `image_hidden_states` keyword. The preprocess flattens the argument to `[image_tokens, hidden_size]` as a view, so in-place edits land, and the postprocess reshapes an assignment back to the argument's shape.

`ImageScatter` sets `sourced = True`: its forward is instrumented at build, because the tower runs inside that forward and a drill on first read, after a tower value, would ask for a location the model passed already ([Descriptors](descriptors.md)). For the root case `StandardizedTransformer._source_root_scatter` touches the root's own `source` at build and again after dispatch, when the family names `ROOT_SCATTER` and the load has a `projector`.

## Availability

`Vision.no_images()` is the one place that decides whether an image reaches the model: the family names no `projector` on this wrapper, or the load has no processor. Then every tower value is `Unavailable` with that reason, `model.vision.support()` is empty, and `model.support()` has no `vision` rows. A text-only checkpoint has no `model.vision` at all. Otherwise the root lists the tower's rows the way it lists a block's, under the alias (`"vision.image_features"`, `"vision.self_attn.attention_probabilities"`), through the same `standard_children` walk.

```python
text_only = StandardizedTransformer("llava-hf/llava-interleave-qwen-0.5b-hf", dispatch=True, device="cpu")
print(type(text_only._module).__name__, text_only.vision.no_images())
# LlavaForConditionalGeneration a text-only load: no processor, so no image reaches the model; load with task='image-text-to-text'
print([name for name in text_only.support() if name.startswith("vision")])   # []
```

## Loading

`task` defaults to `"text-generation"`, and `task="image-text-to-text"` is the only switch: it loads the wrapper with its processor, and `model.tokenizer` is then the processor's tokenizer, which keeps the text helpers working. Under the default task transformers builds whatever causal-LM class the config maps to: the text-only class where there is one (Llama 4's `Llama4ForCausalLM`, with the weights read out of the wrapper checkpoint), and on a config with no causal-LM class, as above, the lazy build is refused and `dispatch=True` falls back through the pipeline to the wrapper with a warning and no processor. A trace on a wrapper takes `model.trace(prompt, images=[image])` with the placeholder the processor's chat template puts in the prompt, or an encoding built with `model.processor`. One image-carrying invoke goes in a trace, since nnsight does not batch `pixel_values` across invokes.

## Which file holds what

`nnterp/components/vision.py` holds everything shared: the three layouts, `Vision`, the three block classes, `ImageScatter`, `scatter_host` and `scatter_call`, `no_tower_run`, `variable_resolution`, and the subclasses several families need (`QwenVision`, `QwenVisionAttention`, `PixtralVision`). A family file holds the paths. A tower on the shared attention interface (SigLIP, CLIP, Llama 4's ViT) is about five `RENAME` lines and four `ENVOYS` entries keying the base classes on the tower's module types, plus one `ENVOYS` entry per wrapper keying `ImageScatter` on the wrapper's model, repeated in each family that hosts the tower. A tower that differs gets a small subclass named as the base in its family file, such as `gemma4_text`'s `VisionAttention`, which points the contributions at a sandwich block's post-norms.

## What does not fit

The Qwen ViT splits the queries, keys and values per image and calls the interface once per image, under eager and sdpa alike, so no one tensor is the block's pattern. `QwenVisionAttention` serves the queries, keys and values whole at the `unsqueeze` operations that end their preparation and the head outputs at the `torch.cat` that joins the calls, and marks the scores and pattern `Unavailable` with `PER_IMAGE`, which says to split at `cu_seqlens` and never to load eager. Pixtral's convolution runs on the batch padded to its largest image and the grids are cropped and concatenated afterwards, and `PixtralVision.patch_embeddings` is the packed row entering `ln_pre`. Qwen2.5-VL serves its block values in the tower's window order and does not reorder them. Qwen3-VL's three tower taps re-enter the text stream after text blocks 0 to 2, and `qwen3_vl_text` adds `layers[k].deepstack_output`, a `DeepstackEProperty` pinned to block k's occurrence of the one `_deepstack_process` call, with the text model's envoy `sourced`. Gemma 4 unified has no tower: `vision` is its embedder, a `Vision` with no `layers`. Mllama's cross-attention blocks and Gemma 4's audio tower have no family and no values yet ([Contributing](../contributing.md)).
