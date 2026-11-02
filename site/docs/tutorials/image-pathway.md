# Image pathway

An image enters the text model once, as `vision.image_features` scattered into the positions that `vision.image_token_mask` marks, and from then on it is a run of positions that every text-block value covers. The recipe ablates the image at that entry and inside the tower, patches one image's features into another image's run, measures each head's attention onto the image, and edits the image positions of a text block, on a checkpoint asked for the color of a square ([Vision-language](../guide/architectures/vision-language.md)).

## The image as the text model sees it

```python
import torch
from PIL import Image, ImageDraw
from nnterp import StandardizedTransformer

model = StandardizedTransformer("llava-hf/llava-interleave-qwen-0.5b-hf", task="image-text-to-text", dispatch=True, device="cuda", attn_implementation="eager")


def square(color):
    image = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(image).rectangle([96, 96, 288, 288], fill=color)
    return image


red, blue = square("red"), square("blue")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the square? Answer with one word."}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def token(word):
    ids = model.tokenizer(word, add_special_tokens=False).input_ids
    assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or the target is not the word
    return ids[0]


RED, BLUE = token("Red"), token("Blue")


def report(name, probs):
    top = probs[0].argmax()
    print(f"{name:32s} top={model.tokenizer.decode([top])!r:8s} P(Red)={probs[0, RED]:.3f} P(Blue)={probs[0, BLUE]:.3f}")


with torch.no_grad(), model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask.save()           # first: it comes off the inputs
    red_features = model.vision.image_features.save()
    clean = model.next_token_probs.save()
print(mask.shape, int(mask.sum()), red_features.shape)   # torch.Size([1, 749]) 729 torch.Size([729, 1024])
report("clean (red)", clean)
# clean (red)                      top='Red'    P(Red)=0.984 P(Blue)=0.000

with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.image_features[:] = 0                    # the text model receives zeros at the image tokens
    report("image_features zeroed", model.next_token_probs.save())
# image_features zeroed            top='The'    P(Red)=0.001 P(Blue)=0.004
```

The load is `task="image-text-to-text"` (the default text task has no processor, and `model.vision` then serves nothing), the processor's chat template places the image token, and the mask is read first in every trace because it comes off the inputs. `image_features` is `[image_tokens, hidden_size]`, flat over the image tokens of the batch, and `layers[0].input[mask] == image_features` exactly. One image-carrying invoke per trace, so the clean and the ablated runs are two traces, and `red_features` is an ordinary tensor by the time the second runs. With the image gone the model has nothing to answer about, and no color token takes `Red`'s place.

## Inside the tower

```python
with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.tower_output[:] = 0
    report("tower_output zeroed", model.next_token_probs.save())
# tower_output zeroed              top='Red'    P(Red)=0.984 P(Blue)=0.000

with torch.no_grad(), model.trace(prompt, images=[red]):
    model.vision.layers[-1].layer_output[:] = 0
    report("tower block -1 zeroed", model.next_token_probs.save())
# tower block -1 zeroed            top='White'  P(Red)=0.088 P(Blue)=0.077
```

!!! warning "A tower edit the wrapper never reads changes nothing"
    Which tower value reaches the projector is the wrapper's choice. llava-interleave feeds the projector the last block's `layer_output` before the tower's final norm, so `tower_output` is computed and discarded and zeroing it leaves `Red` at 0.984. Llava 1.5 reads block -2, and Gemma 3 pools `tower_output`. `model.projector.input` is what the projector receives on every wrapper. Compare it with the value you edited when an edit has no effect.

## One image's features in another's run

```python
with torch.no_grad(), model.trace(prompt, images=[blue]):
    report("clean (blue)", model.next_token_probs.save())
# clean (blue)                     top='Blue'   P(Red)=0.000 P(Blue)=0.980

with torch.no_grad(), model.trace(prompt, images=[blue]):
    model.vision.image_features[:] = red_features         # both images give 729 tokens on this tower
    report("blue run, red features", model.next_token_probs.save())
# blue run, red features           top='Red'    P(Red)=0.984 P(Blue)=0.000
```

The assignment is activation patching across images, the method of nnsight.net's [Activation Patching](https://nnsight.net/tutorials/tutorials/causal_mediation_analysis/activation_patching/) tutorial: The text model sees the red image under the blue image's run and answers for the red one, at the clean run's probability. The assignment needs the same number of image tokens in both runs, which every image gives on a fixed-resolution tower such as this SigLIP. On a variable-resolution tower, such as the Qwen ViT or Pixtral, use images of the same size. To patch part of the image, index the rows, which are the patches in raster order here.

## Attention onto the image

```python
masses = []                                               # made outside the trace
with torch.no_grad(), model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask
    for i in range(model.num_layers):
        p = model.layers[i].self_attn.attention_probabilities
        masses.append(p[0, :, -1, mask[0]].sum(-1).save())   # [num_heads]: the last token's mass on the image
masses = torch.stack(masses)                              # [layers, num_heads]
print([round(float(m), 2) for m in masses.mean(-1)])
# [0.36, 0.12, 0.14, 0.12, 0.11, 0.09, 0.22, 0.2, 0.15, 0.12, 0.19, 0.11, 0.21, 0.21, 0.22, 0.24, 0.24, 0.26, 0.29, 0.29, 0.26, 0.25, 0.2, 0.28]
for flat in masses.flatten().topk(3).indices:
    layer, head = divmod(int(flat), masses.shape[1])
    print(f"block {layer} head {head}: {float(masses[layer, head]):.2f}")
# block 20 head 9: 0.98
# block 17 head 14: 0.96
# block 19 head 13: 0.91
```

The pattern is `[batch, num_heads, query, key]` over the whole sequence, so a head's mass on the image is its row summed over the image key columns. The last token puts 0.36 of its attention on the image at block 0 and between 0.1 and 0.3 afterwards, while single heads in blocks 17 to 20 put over 0.9 of theirs there. Those are the heads to score in [Attention patterns](attention-patterns.md).

## The image positions of a text block

```python
for k in (4, 8, 12, 16, 20):
    with torch.no_grad(), model.trace(prompt, images=[red]):
        mask = model.vision.image_token_mask
        model.layers[k].layer_output[mask] = 0            # the image rows leaving block k; the text rows untouched
        report(f"image rows zeroed after block {k}", model.next_token_probs.save())
# image rows zeroed after block 4  top='The'    P(Red)=0.006 P(Blue)=0.005
# image rows zeroed after block 8  top='The'    P(Red)=0.000 P(Blue)=0.000
# image rows zeroed after block 12 top='White'  P(Red)=0.004 P(Blue)=0.011
# image rows zeroed after block 16 top='No'     P(Red)=0.062 P(Blue)=0.005
# image rows zeroed after block 20 top='Red'    P(Red)=0.895 P(Blue)=0.005
```

Boolean indexing with the mask writes the image rows and leaves the text rows alone, so the edit asks when the text positions have finished reading the image. The answer survives at 0.895 once the image rows are zeroed after block 20, and not before. The same indexing reads the two sides apart, `layer_output[mask]` as `[729, hidden_size]` and `layer_output[~mask]` as the text rows, the batch flattened in both.

## On another family

`vision.image_features`, `vision.image_token_mask` and `vision.layers[i]` mean the same on every wrapper where nnterp names the tower, and `attention_probabilities` is the same pattern on every text family, so the code changes only in the checkpoint, the prompt template and the target words. Zeroing `image_features` is not the whole image on Qwen3-VL, where the first three text blocks re-add it through `layers[k].deepstack_output`. Eager attention over a large tower holds every block's pattern for autograd, so keep the traces under `torch.no_grad()`.
