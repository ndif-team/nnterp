---
title: The image pathway
one_liner: "Ablate the image where it enters the text model (`vision.image_features`) or inside the tower, patch one image's features into another image's run, measure each head's attention onto the image with `vision.image_token_mask`, and edit the image positions of a text block without touching the text, on any wrapper nnterp names the tower of."
tags: [patterns, vision, multimodal, ablation, activation-patching, attention, image_features, image_token_mask]
related: [docs/usage/vision.md, docs/patterns/ablation.md, docs/patterns/activation-patching.md, docs/patterns/attention-patterns.md, docs/usage/availability.md, docs/usage/attention-interior.md]
sources: [nnterp/components/vision.py, nnterp/standardized.py, nnterp/families/llama.py, tests/families/vision_suite.py]
---

# The image pathway

## What this is for

An image enters a vision-language model once: the tower turns it into patches, the
projector maps them into the text model's width, and the wrapper scatters the result into
the token embeddings at the image tokens. From there on the image is a run of positions in
the text model's sequence, and every text-block value covers it. nnterp gives the pathway
three handles that mean the same thing on every wrapper
([vision.md](../usage/vision.md)):

- `vision.image_features`, `[image_tokens, hidden]`: what the text model receives at the
  image tokens, read at the scatter, so `layers[0].input[mask] == features` exactly.
  Ablate or patch here to change *the image as the text model sees it*.
- `vision.image_token_mask`, `[batch, seq]`: which positions are the image. Index any
  text-block value with it to split the image rows from the text rows.
- `vision.layers[i]`: the tower's blocks, with `layer_output`, `attention_output`,
  `mlp_output` and the attention interior over the patches. Ablate here to change *what the
  tower computes*.

The recipes run on `llava-hf/llava-1.5-7b-hf` (CLIP tower, Llama 2 text model, 14.6 GB in
float16 under eager with `torch.no_grad()`), with a red square on white and the question
"What color is the square? Answer with one word.", answered `Red` at 0.99. The same code
runs on every wrapper in [vision.md](../usage/vision.md#which-wrappers); for a CPU dry run
use `trl-internal-testing/tiny-LlavaForConditionalGeneration` and look at shapes.

## Canonical pattern

Load with `task="image-text-to-text"`, build the prompt with the processor's chat template
(it places the image token), and read the mask before anything else: it comes off the
inputs.

```python
import torch
from PIL import Image, ImageDraw
from nnterp import StandardizedTransformer

model = StandardizedTransformer("llava-hf/llava-1.5-7b-hf", task="image-text-to-text", dispatch=True, dtype=torch.float16, attn_implementation="eager")

def square(color):
    image = Image.new("RGB", (336, 336), "white")
    ImageDraw.Draw(image).rectangle([84, 84, 252, 252], fill=color)
    return image

red, blue = square("red"), square("blue")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the square? Answer with one word."}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def token(word):
    ids = model.tokenizer(word, add_special_tokens=False).input_ids
    assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or the target is not the word
    return ids[0]
RED, BLUE = token("Red"), token("Blue")                                  # Llama's tokenizer: '▁Red', '▁Blue'

def report(name, probs):
    print(f"{name:34s} top={model.tokenizer.decode([probs[0].argmax()])!r:8s} P(Red)={probs[0, RED]:.3f} P(Blue)={probs[0, BLUE]:.3f}")

with torch.no_grad(), model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask.save()           # [1, 599], 576 true
    red_features = model.vision.image_features.save()     # [576, 4096]
    clean = model.next_token_probs.save()
report("clean (red)", clean)                              # top='Red'  P(Red)=0.990

with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.image_features[:] = 0                    # the text model receives zeros at the image tokens
    report("image_features zeroed", model.next_token_probs.save())   # top=''  P(Red)=0.010 P(Blue)=0.011
```

With the image gone the model has nothing to answer about: `Red` falls from 0.99 to 0.01
and no color token takes its place. One image-carrying invoke per trace: the clean and the
ablated runs are two traces, not two invokes of one, and `red_features` is an ordinary
tensor by the time the second runs.

## Ablate inside the tower

A tower block's contribution is zeroed like a text block's. Which block reaches the text
model depends on what the wrapper feeds the projector: Llava takes block -2's stream
(`vision_feature_layer=-2`), so the last block and `tower_output` are computed and
discarded, while Gemma 3 pools `tower_output` itself.

```python
with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.tower_output[:] = 0
    report("tower_output zeroed", model.next_token_probs.save())     # top='Red'  P(Red)=0.990: Llava never reads it

with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.layers[-2].layer_output[:] = 0
    report("tower block -2 zeroed", model.next_token_probs.save())   # top='</s>' P(Red)=0.001: this is what the projector reads

with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.layers[0].mlp.mlp_output[:] = 0
    report("tower block 0 MLP zeroed", model.next_token_probs.save())  # top='Red'  P(Red)=0.932
```

`model.projector.input` is what the projector actually receives, on every wrapper; when a
tower edit does nothing, compare it with the value you edited.

## Patch one image into another's run

Activation patching across images: save `image_features` from the red run (done above) and
assign it in the blue run. The text model then sees the red image under the blue image's
prompt, and answers for the red one.

```python
with torch.no_grad(), model.trace(prompt, images=[blue]):
    report("clean (blue)", model.next_token_probs.save())           # top='Blue' P(Blue)=0.989

with torch.no_grad(), model.trace(prompt, images=[blue]):
    model.vision.image_features[:] = red_features                   # same shape: both images give 576 tokens here
    report("blue run, red features", model.next_token_probs.save()) # top='Red'  P(Red)=0.990
```

The two runs must have the same number of image tokens for the assignment to fit; on a
fixed-resolution tower (CLIP, SigLIP) every image does, on a variable-resolution one (the
Qwen ViT, Pixtral, Gemma 4) use images of the same size. To patch part of the image, index
the features: `features[rows] = red_features[rows]`, where `rows` are patch indices in the
tower's row order ([vision.md](../usage/vision.md#the-values) says what the order is per
tower; CLIP's and SigLIP's are raster order, 24 x 24 patches here).

## Attention onto the image

The text blocks' pattern is `[batch, heads, query, key]` over the whole sequence, so the
mass a head puts on the image is the pattern summed over the image key columns. Per block,
from the last token:

```python
masses = []                                   # made outside the block: a name bound inside does not survive it
with torch.no_grad(), model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask       # first in the trace: it comes off the inputs
    for i in range(model.num_layers):
        probabilities = model.layers[i].self_attn.attention_probabilities
        masses.append(probabilities[0, :, -1, mask[0]].sum(-1).save())   # [heads]: the last token's mass on the image

mean = torch.stack(masses).float().mean(-1)   # [layers]
peak = torch.stack(masses).float().amax(-1)   # [layers]: the head that looks most
```

On the red square the mean mass is 0.73 at block 0 and 0.42 at block 1, under 0.1 from
block 3 on, then back between 0.1 and 0.26 across blocks 10 to 24, where single heads put
0.75 to 0.95 of their mass on the image (blocks 12, 14, 18, 19, 22). Those are the heads to
look at in [attention-patterns.md](attention-patterns.md); a head's mass restricted to one
image's tokens, in a two-image prompt, is the same expression with that image's slice of the
mask.

## Edit the image positions of a text block

Boolean indexing with the mask writes the image rows and leaves the text rows alone, so an
edit at one text block asks when the text positions have finished reading the image:

```python
for k in (4, 8, 16, 24):
    with torch.no_grad(), model.trace(prompt, images=[red]):
        mask = model.vision.image_token_mask
        model.layers[k].layer_output[mask] = 0                      # the image rows leaving block k; text rows untouched
        report(f"image rows zeroed after block {k}", model.next_token_probs.save())
```

```
image rows zeroed after block 4    top=''      P(Red)=0.012 P(Blue)=0.008
image rows zeroed after block 8    top='Black' P(Red)=0.114 P(Blue)=0.086
image rows zeroed after block 16   top='Red'   P(Red)=0.384 P(Blue)=0.013
image rows zeroed after block 24   top='Red'   P(Red)=0.994 P(Blue)=0.000
```

The image is read out of its positions by the middle of the stack: after block 24 the
image rows no longer matter. Mean-ablating instead (`h[mask] = h[mask].mean(0)`, every image
row replaced by their average, the image's identity removed but not its presence) gives
`White` at 0.27 after block 4 and `Red` at 0.62 after block 16. The same indexing reads the
two sides apart: `model.layers[16].layer_output[mask]` is `[576, 4096]` and `[~mask]` is
`[23, 4096]`, the batch flattened in both.

## Gotchas

- Read `vision.image_token_mask` first in every trace that uses it, and `image_features`
  after the tower's values and before the text model's; a later read raises `OutOfOrderError`
  in a plain trace.
- One image-carrying invoke per trace, so clean and ablated runs are separate traces; several
  images go in one invoke as a list, and the mask covers them all.
- `image_features` is zeroed before block 0, but it is not always the only way in: Qwen3-VL
  re-adds the image after text blocks 0 to 2 (`layers[k].deepstack_output`), so zero those
  too ([vision.md](../usage/vision.md#the-qwen-vit)).
- A tower edit that changes nothing is usually at a block the wrapper does not read (Llava's
  last block, `tower_output`); check `model.projector.input`.
- Eager attention over a big tower holds every block's pattern for autograd: wrap the trace
  in `torch.no_grad()` (Gemma 3's 4096 patches ran out of a 48 GB card without it).
- On a text-only load (the default `task="text-generation"`) every tower value raises
  `Unavailable` and `model.support()` has no `vision.` rows: load with
  `task="image-text-to-text"` ([availability.md](../usage/availability.md)).

## Related

- [docs/usage/vision.md](../usage/vision.md): the names, the values, the per-tower facts.
- [ablation.md](ablation.md), [activation-patching.md](activation-patching.md): the same moves on the text model.
- [attention-patterns.md](attention-patterns.md): head metrics once you know which heads look at the image.
