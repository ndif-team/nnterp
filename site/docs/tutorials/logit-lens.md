# Logit lens

Where in depth a prediction emerges, on any family.

`model.project_on_vocab(layer.layer_output)` reads the residual stream leaving a block through the model's own final norm, its unembedding and whatever the model does after the head, so one loop over `model.layers` is the logit lens on every family. The method is the one in nnsight's [logit lens tutorial](https://nnsight.net/tutorials/tutorials/probing/logit_lens/). What nnterp changes is that the norm, the head and any softcap are no longer yours to find per architecture.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu")
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]

lens = []                                   # bound outside the trace: a list made inside it is gone afterwards
with model.trace(prompt):
    for layer in model.layers:
        lens.append(model.project_on_vocab(layer.layer_output)[0, -1].save())   # [vocab_size]
    logits = model.logits[0, -1].save()

for i, row in enumerate(lens):
    print(f"block {i:2d}  top {model.tokenizer.decode(row.argmax())!r:>10}  P(Paris) {row.softmax(-1)[paris]:.3f}")
print(torch.equal(lens[-1], logits))
# block  0  top     ' the'  P(Paris) 0.000
# block  1  top     ' the'  P(Paris) 0.000
# block  2  top     ' the'  P(Paris) 0.000
# block  3  top     ' the'  P(Paris) 0.000
# block  4  top     ' the'  P(Paris) 0.000
# block  5  top     ' the'  P(Paris) 0.000
# block  6  top    ' East'  P(Paris) 0.001
# block  7  top     ' Ing'  P(Paris) 0.003
# block  8  top    ' Rome'  P(Paris) 0.025
# block  9  top  ' London'  P(Paris) 0.248
# block 10  top   ' Paris'  P(Paris) 0.183
# block 11  top   ' Paris'  P(Paris) 0.070
# True
```

GPT-2 decodes ` the` for the first six blocks, a city from block 8 (` Rome`, then ` London`) and ` Paris` from block 10. The probability says more than the argmax: P(Paris) peaks at block 9, where ` London` is still the top token, and falls through the last two blocks to the 0.070 the model outputs. The loop reads the blocks in forward order and `logits` after them, which is the order one trace requires.

On many newer checkpoints the early blocks decode to unrelated tokens and the answer appears only in the last few blocks; `Qwen/Qwen2.5-0.5B` puts ` Paris` on top from block 21 of 24 and nothing recognizable before it. That is not a wiring fault, and the wiring check is what decides.

The final `True` is the wiring check. At the last block the lens is the model's own computation, so it must equal `model.logits` exactly. Run it first on a new checkpoint. A lens that differs there is reading through the wrong norm or head, and every earlier block is then on the wrong scale. Decoding `model.lm_head(hidden)` without the norm is not a cheaper lens. A norm is not affine, so the curve comes out confident and wrong rather than as an error.

## The step after the head

`model.logits` is what the model returns and `model.lm_head.output` the raw projection. On Gemma-2 the two differ, because the model applies `cap * tanh(logits / cap)` after the head (`final_logit_softcapping`, 30.0 on `google/gemma-2-2b`, where the raw projection and the logits differ by up to 7.8 at the last position of this prompt). `project_on_vocab` reads that key from the text config and applies the same cap, so the wiring check passes there too and every block is read on the model's scale rather than an uncapped one. Cohere's `logit_scale` and Granite's `logits_scaling` are the same kind of step, and those families define their own `project_on_vocab` with it.

## On another family

The recipe runs unchanged. `layer_output` is a tensor on every family, including the families with a tuple-returning block (BLOOM, MPT, Falcon), and on DeepSeek-V4, where it carries an extra stream axis, the family's `project_on_vocab` collapses the streams the way the model does. The lens reads module boundaries, so it does not need `attn_implementation="eager"`. The per-family work is all in `project_on_vocab`: the norm, the head and the step after it.

[Contribution decomposition](contribution-decomposition.md) reads each sublayer's addition through the same head, in a form that sums.
