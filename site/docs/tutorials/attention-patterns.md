# Attention patterns

Reading and scoring attention heads.

`model.layers[i].self_attn.attention_probabilities` is the attention pattern after the softmax, `[batch, num_heads, query, key]`, on every family that runs transformers' eager attention. Load with `attn_implementation="eager"` and the per-head scores below are three lines on the saved tensor. The pattern is served in the model's dtype, and small Pythia and GPT-NeoX checkpoints load in float16, where on the CPU the row sums miss the tolerance below and the entropy is NaN, so pass `dtype=torch.float32` there. nnsight's [attention tutorial](https://nnsight.net/tutorials/tutorials/attention/attention_heads/) explains the metrics and the head types.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", attn_implementation="eager")
prompt = "When Mary and John went to the store, John gave a drink to Mary"
LAYER = 4

with model.trace(prompt):
    pattern = model.layers[LAYER].self_attn.attention_probabilities.save()   # [batch, num_heads, query, key]

print(pattern.shape, torch.equal(pattern.tril(), pattern), torch.allclose(pattern.sum(-1).float(), torch.ones(pattern.shape[:-1]), atol=1e-4))

entropy = -(pattern * (pattern + 1e-12).log()).sum(-1).mean(-1)[0]      # [num_heads]: low means sharp
previous = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)[0]     # mass on key i-1
first = pattern[0, :, 1:, 0].mean(-1)                                    # mass on key 0 from every later query
for h in range(model.num_heads):
    print(f"head {h:2d}  entropy {entropy[h]:.2f}  previous {previous[h]:.2f}  first {first[h]:.2f}")
# torch.Size([1, 12, 15, 15]) True True
# head  0  entropy 0.79  previous 0.36  first 0.41
# head  1  entropy 1.01  previous 0.23  first 0.56
# head  2  entropy 1.01  previous 0.11  first 0.72
# head  3  entropy 1.14  previous 0.29  first 0.36
# head  4  entropy 1.11  previous 0.15  first 0.65
# head  5  entropy 0.85  previous 0.16  first 0.73
# head  6  entropy 1.26  previous 0.16  first 0.58
# head  7  entropy 1.17  previous 0.19  first 0.31
# head  8  entropy 0.86  previous 0.11  first 0.76
# head  9  entropy 1.11  previous 0.16  first 0.63
# head 10  entropy 0.60  previous 0.09  first 0.86
# head 11  entropy 0.00  previous 1.00  first 0.07
```

The two checks say that the pattern is causal (nothing above the diagonal) and that every row sums to one. Head 11 puts all of its mass on the previous token, with entropy zero: GPT-2's block-4 previous-token head. Head 10 puts 0.86 of its mass on the first token, which is what a head does when it has nothing to attend to on this prompt. The rows of a GPT-OSS pattern sum to less than one, by the mass a learned sink column took before it was dropped. The pattern's meaning, mass on real keys, is unchanged.

## Editing the pattern

The pattern is the matrix that the attention's values are mixed with, so an in-place write reaches the model. Zero one head's rows and the head reads nothing:

```python
with model.trace(prompt):
    clean = model.next_token_probs.save()
with model.trace(prompt):
    model.layers[LAYER].self_attn.attention_probabilities[:, 11] = 0      # head 11 reads nothing
    ablated = model.next_token_probs.save()
print(model.probs_to_dict(clean[0], k=3))
print(model.probs_to_dict(ablated[0], k=3))
# {' and': 0.3203795254230499, ',': 0.24333427846431732, '.': 0.1615239828824997}
# {' and': 0.34432530403137207, ',': 0.21988408267498016, '.': 0.14424201846122742}
```

Zeroing the pattern and zeroing the head's slice of `attention_head_outputs` give the same logits. The head-level recipes, with the clean and ablated runs in one trace, are on the [Ablation](ablation.md) page.

## Without eager attention

A checkpoint loads `sdpa` unless you ask for eager, and the pattern is then unavailable. `support()` reports that before anything runs, and reading the value anyway raises `nnterp.Unavailable` with the same text at that line:

```python
sdpa = StandardizedTransformer("openai-community/gpt2", device="cpu")
print(sdpa.support()["self_attn.attention_probabilities"][0])
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

## On another family

The layout is `[batch, num_heads, query, key]` everywhere, so the scores above do not change. What changes is which blocks have a pattern. On a hybrid (Qwen3-Next, Qwen3.5) three blocks in four carry `linear_attn` instead of `self_attn`, with no pattern, so pick the blocks from the module list outside the trace (`[i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]`) rather than from `num_layers // 2`. The value's shape and the eager requirement are described in [Attention](../guide/values/attention.md).
