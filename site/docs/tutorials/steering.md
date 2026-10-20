# Steering

Adding a direction to the residual stream.

`model.steer(layer, vector, factor, token_positions=-1)` adds `factor * vector` in place to `layer_output` of one block, and the direction is a difference of `layer_output` between two prompts, so deriving and applying a steering vector are the same two snippets on every family. The method is activation addition (Turner et al., 2023, [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)). What the recipe adds is reading the factor against the stream's norm at the block you steer, since that norm differs several-fold between checkpoints and across a model's blocks.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu")
LAYER = 6

with model.trace() as tracer:
    with tracer.invoke("I love this so much"):
        positive = model.layers[LAYER].layer_output[0, -1].save()      # [hidden_size]
    with tracer.invoke("I hate this so much"):
        negative = model.layers[LAYER].layer_output[0, -1].save()
direction = positive - negative
direction = direction / direction.norm()                             # unit norm: factors compare across directions

prompt = "I went to the bakery and"
with model.trace(prompt):
    scale = model.layers[LAYER].layer_output[0, -1].norm().save()    # the stream's norm where you steer
print(float(scale))
factor = 0.25 * float(scale)                                         # a fraction of it; sweep the fraction

with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.next_token_probs.save()
    with tracer.invoke(prompt):
        model.steer(LAYER, direction, factor=factor, token_positions=-1)
        steered = model.next_token_probs.save()
print(model.probs_to_dict(baseline[0], k=3))
print(model.probs_to_dict(steered[0], k=3))
# 92.03986358642578
# {' bought': 0.09450755268335342, ' I': 0.08870173245668411, ' asked': 0.07155062258243561}
# {' I': 0.1267288476228714, ' they': 0.055561766028404236, ' bought': 0.05142388120293617}
```

The stream leaving GPT-2's block 6 has norm 92 at the last position, so a factor of 23 adds a quarter of that along the direction, and the top token moves from ` bought` to ` I`. The add is in place on the served tensor, so it reaches the model, and the invoke that does not call `steer` is untouched. A raw factor means nothing on its own. On `Qwen/Qwen2.5-0.5B` the same prompt's stream has norm 13 at block 6 and 79 at block 22, against 57 to 465 across GPT-2's blocks, so measure the norm where you steer and sweep the fraction, not the factor. `steer` is one line that you could write yourself, `out[:, -1] += factor * direction.to(out)` on `layer_output`. Use the hand form when the edit is not an add, such as projecting the direction out.

## At every step of a generate

Steering a generation means steering every decode step, not only the prompt. `tracer.iter[:N]` runs the body on the prefill and on each of the `N - 1` decode steps, and `min_new_tokens=N` holds the run to that bound:

```python
N = 15
for f in (0.0, factor, -factor):
    with model.generate(prompt, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
        for step in tracer.iter[:N]:                                 # the prefill and every decode step
            model.steer(LAYER, direction, factor=f, token_positions=-1)
        ids = tracer.result.save()
    print(f"{f:+.1f}: {model.tokenizer.decode(ids[0][-N:])!r}")
# +0.0: ' bought a bag of cookies. I was so excited. I was so excited'
# +23.0: ' I was able to get some bread and a little bit of butter. I'
# -23.0: ' asked the owner if he could make a cake for me. He said he'
```

Adding and subtracting the direction give two different continuations, and both stay fluent at a quarter of the norm. `token_positions=-1` is the right index across steps. On the prefill it is the prompt's last token and on each decode step the one token that step processes, where an absolute position would be out of range from the first decode step on. nnsight's [multiple token generation](https://nnsight.net/features/4_multiple_token/) page covers `tracer.iter` and `tracer.result`.

!!! warning "A bare `steer` in a `generate` body runs once"
    Outside `tracer.iter`, the call fires on the prefill only. The continuation still changes, because the prompt was read differently (on this prompt and factor it becomes ` I was like, 'Oh my God, I'm so sorry. I`), which makes the mistake look like a modest success. Put the call inside the loop.

## On another family

`layer_output` and `steer` do the per-family work. The stream leaving a block is `layer_output` on every block of every family, a DeltaNet block of a hybrid included, and `steer` moves the vector to the stream's dtype and device, so a CPU float32 direction steers a bf16 GPU model unchanged. The one thing that changes with the checkpoint is the norm, which is why the recipe measures it rather than fixing a factor. `steer`'s arguments, including `batch_index` for one row of a batch and a list of ascending blocks, are in [Methods](../guide/methods.md).
