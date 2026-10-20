# Vision-language

An image-text-to-text checkpoint is a text model plus a vision tower, a projector, and a step that scatters the projected image features into the text stream at the image tokens. Load it with `task="image-text-to-text"` and pass the image to the trace. The text model keeps its standard names and values, `model.vision` and `model.projector` name the vision side, and two values of the tower say where the image enters the text model: `vision.image_token_mask` and `vision.image_features`. GPT-2 has no tower, so this page runs `llava-hf/llava-interleave-qwen-0.5b-hf`, a SigLIP tower on a Qwen2 text model.

```python
import torch
from PIL import Image
from nnterp import StandardizedTransformer

model = StandardizedTransformer("llava-hf/llava-interleave-qwen-0.5b-hf", task="image-text-to-text", device="cpu", attn_implementation="eager")
image = Image.new("RGB", (64, 64), "red")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Answer in one word: what color is this?"}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()                                  # first: it comes off the inputs
    pattern = model.vision.layers[0].self_attn.attention_probabilities.save()
    patches = model.vision.layers[0].layer_output.save()
    projected = model.projector.output.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()
    logits = model.logits.save()

print(tuple(mask.shape), mask.sum().item())
# (1, 748) 729
print(tuple(pattern.shape), tuple(patches.shape), tuple(projected.shape), tuple(features.shape))
# (1, 16, 729, 729) (1, 729, 1152) (1, 729, 1024) (729, 1024)
print(torch.equal(first[mask], features))
# True
```

The processor's chat template puts the image placeholder in the prompt, and the processor expands it to one token per projected patch, 729 here. A tower block's values have the `Patches` layout, `[images, patches, vision_hidden_size]`, and its attention interior has the same names as a text block's ([Attention](../values/attention.md)). The tower's sizes are on `model.vision` (`num_layers` 26, `vision_hidden_size` 1152), apart from the text model's (`model.num_layers` 24, `model.hidden_size` 1024). `image_token_mask` is `[batch, seq]` bool. `image_features` is `[image_tokens, hidden_size]`, the tensor the wrapper writes into the token embeddings at the image tokens, in row-major order, so the equality above is exact and holds on every wrapper. It is where to ablate, patch or steer the image as the text model sees it.

```python
with model.trace(prompt, images=[image]):
    model.vision.image_features[:] = 0
    ablated = model.logits.save()

print(model.probs_to_dict(logits[0, -1].softmax(-1), k=2), model.probs_to_dict(ablated[0, -1].softmax(-1), k=2))
# {'Red': 0.984375, 'Orange': 0.004852294921875} {'Answer': 0.56640625, 'The': 0.2080078125}
```

What a row of a tower value is differs by encoder. SigLIP has one row per image, in raster order. CLIP, Llava 1.5's tower, puts the CLS token first on the patch axis, and Llama 4's encoder puts it last. Pixtral, the Qwen ViT and RADIO pack every image's patches into one row, `[1, patches, vision_hidden_size]`, and Gemma 4 pads the patch axis to a fixed row count. On the Qwen ViT and RADIO the attention runs once per image, so their `attention_probabilities` and `attention_scores` are `Unavailable` and the queries and keys are served whole.

## Image rows and text rows

Every value of the text model is over the whole sequence, image tokens and text tokens alike. The mask splits them, and boolean indexing flattens the batch: `out[mask]` is `[image_tokens, hidden_size]` in the order of `image_features`, and `out[~mask]` the text rows. To keep the batch shape, mask instead of indexing (`out.masked_fill(~mask[..., None], 0)`).

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    out = model.layers[6].layer_output.save()

print(tuple(out[mask].shape), tuple(out[~mask].shape))
# (729, 1024) (19, 1024)

with model.trace(prompt, images=[image]):
    positions = model.vision.image_token_mask
    model.layers[6].layer_output[positions] = 0                                 # the image positions only
    edited = model.logits.save()

print(model.probs_to_dict(edited[0, -1].softmax(-1), k=2))
# {'Black': 0.169921875, 'White': 0.0966796875}
```

The in-place write through the mask is a one-sided edit, which leaves the text rows untouched. [Image pathway](../../tutorials/image-pathway.md) builds the ablation and patching recipes on it.

## The projector

`model.projector` is the last module before the scatter, and `projector.input` is what it receives, which is not `tower_output` on every host. Here the projector reads the last block's `layer_output` before the tower's final norm (`vision.norm`), so a write to `tower_output` never reaches the text model. Llava 1.5 reads block `-2` without its CLS token, and Gemma 3 pools `tower_output`. On the Qwen ViT the projector is the tower's merger and runs inside the tower, so read `projector.input` and `projector.output` before `tower_output`.

```python
with model.trace(prompt, images=[image]):
    last_block = model.vision.layers[-1].layer_output.save()
    tower = model.vision.tower_output.save()
    received = model.projector.input.save()

print(torch.equal(received, last_block), torch.equal(received, tower))
# True False
```

## Read order and one image per trace

Reads follow the forward: `image_token_mask` first, then the tower's values in order (a block's attention interior before its `layer_output`), the projector's, `image_features`, and then the text model's. A read out of that order raises `OutOfOrderError`, as in any trace. A trace carries one image-carrying invoke, and a second raises `NotImplementedError: Can't batch these inputs`. Several images go in that one invoke as a list, and each becomes a row of the tower's values:

```python
blue = Image.new("RGB", (64, 64), "blue")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare them."}]}]
two = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

with model.trace(two, images=[image, blue]):
    mask = model.vision.image_token_mask.save()
    patches = model.vision.layers[0].layer_output.save()
    features = model.vision.image_features.save()

print(mask.sum().item(), tuple(patches.shape), tuple(features.shape))
# 1458 (2, 729, 1152) (1458, 1024)
```

Under `generate` the tower runs once, for the prompt: `image_token_mask` is `[batch, prompt_len]` on step 0 and `[batch, 1]`, all false, on every decode step, and the tower's values and `image_features` occur on step 0 only, so read them in `tracer.iter[0]` ([Generation](../generation.md)). `model.support()` lists the tower's values under `vision.`, such as `"vision.image_features"` and `"vision.self_attn.attention_probabilities"`, and `model.vision.support()` lists them alone. A load without `task="image-text-to-text"` builds the wrapper with no processor, so no image can reach it: `model.vision.support()` is then `{}` and reading an image value raises `Unavailable` saying so.
