# Contribution decomposition

Direct logit attribution over the standard contributions.

`layer_input + attention_output + mlp_output == layer_output` on every family, so direct logit attribution reads `attention_output` and `mlp_output` from each block, pushes each term through the final norm frozen at the real stream and the unembedding, and the terms sum to the model's logit. `attention_head_outputs` times the output projection's weight splits a block's attention term per head. The method is from Elhage et al. (2021), [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]

parts = {}                                              # bound outside the trace
with model.trace(prompt):
    base = model.layers[0].layer_input.save()           # the stream entering block 0
    for i, layer in enumerate(model.layers):            # block i's attention, then its MLP: forward order
        parts["attn", i] = layer.self_attn.attention_output.save()
        parts["mlp", i] = layer.mlp.mlp_output.save()
    final = model.layers[-1].layer_output.save()
    logit = model.lm_head.output[0, -1, paris].save()   # the head's output, before any step after it

print(torch.allclose(base + sum(parts.values()), final, atol=1e-3))   # the stream is the sum of its contributions

norm, head = model.norm._module, model.lm_head._module
x = final[0, -1]                                        # the stream the final norm reads at this position
with torch.no_grad():
    shift = norm(torch.zeros_like(x))                   # a LayerNorm's bias; zeros on an RMSNorm
    centers = torch.allclose(norm(torch.ones_like(x)), shift)         # a LayerNorm subtracts the mean first
    center = (lambda t: t - t.mean(-1, keepdim=True)) if centers else (lambda t: t)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0))
    probe = torch.ones_like(x); probe[1::2] = -1        # mean 0, mean square 1
    gain = (norm(probe) - shift) / probe * (1 + eps) ** 0.5           # the norm's weight, or 1 + weight on Gemma
    scale = (center(x).pow(2).mean() + eps).rsqrt()     # the one nonlinear step, frozen at the real stream
    frozen = lambda t: gain * center(t) * scale         # the final norm as a linear map at this position
    row = head.weight[paris]
    dla = {key: float(frozen(part[0, -1]) @ row) for key, part in parts.items()}
    dla["base"] = float(frozen(base[0, -1]) @ row)
    constant = float(shift @ row) + (float(head.bias[paris]) if head.bias is not None else 0.0)

print(f"{sum(dla.values()) + constant:.4f} {float(logit):.4f}")
for i in range(model.num_layers):
    print(f"block {i:2d}  attn {dla['attn', i]:+.3f}  mlp {dla['mlp', i]:+.3f}")
# True
# -92.0873 -92.0873
# block  0  attn -0.264  mlp -0.320
# block  1  attn -0.291  mlp +0.049
# block  2  attn +0.190  mlp -0.232
# block  3  attn +0.051  mlp -0.049
# block  4  attn -0.230  mlp -0.080
# block  5  attn +0.303  mlp -0.476
# block  6  attn -0.480  mlp -1.090
# block  7  attn +0.303  mlp -0.239
# block  8  attn -0.346  mlp -1.012
# block  9  attn +1.006  mlp -1.404
# block 10  attn -1.263  mlp -17.002
# block 11  attn -36.726  mlp -24.635
```

The attributions plus the constant sum to the model's own ` Paris` logit, -92.0873, to four decimals. The last two blocks dominate the logit's scale (block 11's attention -36.7 and MLP -24.6, block 10's MLP -17.0), and block 9's attention is the largest positive term, +1.006. An attribution to one logit includes what a term adds to every token at once. `row = head.weight[paris] - head.weight[rome]` attributes the contrast between two tokens instead, with the same code.

The norm has to be frozen because it rescales each position by that position's own statistics, so it does not distribute over the sum. Holding the scale at the real stream's value makes it a linear map (a centering, a gain, a scale), and the LayerNorm's bias plus any `lm_head` bias enters once, as `constant`. The recipe reads the centering, the gain and the bias off the norm module itself (its output on zeros is its bias, on ones says whether it centers, on a ±1 vector gives its gain), so an RMSNorm (Llama), a LayerNorm (GPT-2) and Gemma's `1 + weight` gain need no separate cases. `model.project_on_vocab(contribution)` is the other reading, the full norm applied to one term alone. It answers what the term would predict by itself and does not sum.

## Per head

`attention_head_outputs` is each head's output before concatenation and the output projection, `[batch, seq, num_heads, head_dim]`. The projection is linear, so a head's share of `attention_output` is its slice times the matching rows of the projection's weight. The projection is `o_proj` on every family (GPT-2's `c_proj` is aliased), but its weight layout differs: `torch.nn.Linear` stores `[out, in]`, GPT-2's `Conv1D` stores `[in, out]`.

```python
LAYER = 9
attention = model.layers[LAYER].self_attn

with model.trace(prompt):
    heads = attention.attention_head_outputs.save()        # [batch, seq, num_heads, head_dim]
    contribution = attention.attention_output.save()       # [batch, seq, hidden_size]

proj = attention.o_proj._module
W = proj.weight.t() if isinstance(proj, torch.nn.Linear) else proj.weight   # [num_heads * head_dim, hidden_size]
H, D = heads.shape[2], heads.shape[3]
per_head = torch.einsum("bshd,hdo->bsho", heads, W.reshape(H, D, -1))      # [batch, seq, num_heads, hidden_size]
print(torch.allclose(per_head.sum(2) + proj.bias, contribution, atol=1e-3))   # the heads sum to the contribution

with torch.no_grad():
    head_dla = [float(frozen(per_head[0, -1, h]) @ row) for h in range(H)]
    bias_dla = float(frozen(proj.bias) @ row)
print([round(v, 3) for v in head_dla])
print(round(sum(head_dla) + bias_dla, 3), round(dla["attn", LAYER], 3))
# True
# [-0.084, -0.132, -0.066, 0.791, 0.116, -0.153, -0.0, -0.072, 0.872, 0.048, 0.008, 0.064]
# 1.006 1.006
```

Block 9's +1.006 is two heads: head 8 (+0.872) and head 3 (+0.791), with the other heads near zero and the projection's bias taking 0.39 back off. Head 8 of block 9 is also the head that [Ablation](ablation.md) finds by removing it.

The per-head sum equals `attention_output` where the projection's output is what the block adds, as on GPT-2 and Llama. On a post-norm family (Gemma-2, Gemma-3, OLMo-2) a norm sits between the projection and the residual add, so the heads sum to `o_proj.output` and not to `attention_output`, and the split is of the pre-norm tensor. On Granite the contribution is the projection's output times `residual_multiplier`. Under grouped-query attention the head axis is still the query heads, so the slicing does not change.

## On another family

The identity does the per-family work. `attention_output` and `mlp_output` are what each sublayer adds whether the block is sequential, parallel or post-norm, so the recipe runs unchanged there. Not every family is a plain sum. On Gemma-4 each block multiplies its sum by `layer_scalar`, so a term enters the final stream weighted by the product of the scalars of its block and every later block. Doge and ZAYA gate the stream per channel, and DeepSeek-V4 keeps parallel streams. Weight each term accordingly before the unembedding. The frozen norm then applies to the weighted terms unchanged.
