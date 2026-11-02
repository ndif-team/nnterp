# Activation patching

Patching a value from a clean run into a corrupt one.

`layer_output` is the residual stream leaving a block and `attention_output` and `mlp_output` are what its sublayers add, on every family, so a patch is one assignment at one of those values at a position, and the same line runs on a sequential, parallel or post-norm block. The method is nnsight's [activation patching tutorial](https://nnsight.net/tutorials/tutorials/causal_mediation_analysis/activation_patching/). The checkpoint is `Qwen/Qwen2.5-0.5B`, where the clean prompt gives P(Paris) 0.797 and the corrupt one 0.000 (GPT-2 gives 0.070 and 0.003, too little to read).

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen2.5-0.5B", device="cpu", dtype=torch.float32)
clean = "The Eiffel Tower is in the city of"
corrupt = "The Colosseum is in the city of"
assert len(model.tokenizer(clean).input_ids) == len(model.tokenizer(corrupt).input_ids)   # positions must correspond
paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]
LAYER, POS = 21, -1

with model.trace(clean):
    clean_resid = model.layers[LAYER].layer_output.save()           # [1, seq, hidden_size]

with model.trace() as tracer:
    with tracer.invoke(corrupt):
        baseline = model.next_token_probs[0, paris].save()
    with tracer.invoke(corrupt):
        model.layers[LAYER].layer_output[:, POS] = clean_resid[:, POS]
        patched = model.next_token_probs[0, paris].save()

print(f"P(Paris)  corrupt {float(baseline):.3f}   patched {float(patched):.3f}")
# P(Paris)  corrupt 0.000   patched 0.778
```

Pasting the clean stream into the last position at block 21 restores 0.778 of the clean 0.797. The write is in place on the served tensor, so it reaches the model, and it touches only its invoke's rows. The unpatched invoke in the same trace is the baseline. `clean_resid` is an ordinary tensor by the time the second trace runs, so nothing has to be synchronized.

## Across invokes

The clean and corrupt rows can share one forward. A write that uses another invoke's value then needs a [barrier](https://nnsight.net/documentation/intervention/barrier/), because the assignment evaluates its right-hand side before its own invoke waits:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(clean):
        source = model.layers[LAYER].layer_output[:, POS]
        barrier()                                                    # source is read
    with tracer.invoke(corrupt):
        barrier()                                                    # wait for it
        model.layers[LAYER].layer_output[:, POS] = source
        patched = model.next_token_probs[0, paris].save()

print(f"across invokes {float(patched):.3f}")
# across invokes 0.778
```

## Sweeping the layers

Cache the clean run once, then one invoke per block, at the position you ask about:

```python
cache = []
with model.trace(clean):
    for layer in model.layers:
        cache.append(layer.layer_output.save())

for pos in (2, -1):                                                 # the subject's second token, the last token
    sweep = []
    with model.trace() as tracer:
        for i in range(model.num_layers):
            with tracer.invoke(corrupt):
                model.layers[i].layer_output[:, pos] = cache[i][:, pos]
                sweep.append(model.next_token_probs[0, paris].save())
    print(f"pos {pos:2d}:", " ".join(f"{float(p):.2f}" for p in sweep))
# pos  2: 0.38 0.61 0.64 0.55 0.50 0.52 0.53 0.52 0.25 0.23 0.30 0.32 0.30 0.30 0.27 0.33 0.17 0.06 0.05 0.05 0.01 0.00 0.00 0.00
# pos -1: 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.32 0.78 0.80 0.80
```

Position 2 is `iff` in the clean prompt and `os` in the corrupt one, the first token where the two differ. A patch there restores most of the answer through block 7 (0.5 to 0.6) and fades to nothing by block 20. At the last position nothing happens until block 19, then 0.32 at block 20 and 0.78 from block 21 on. The fact leaves the subject's position in the early blocks and arrives at the last position in blocks 20 and 21. The last entry of a last-position sweep is fixed by construction, since patching the last block there reproduces the clean prediction. That row says how far back a patch reaches, and the position axis is what localizes.

## Patching a contribution

Patching `layer_output` at block `i` replaces everything computed at that position up to block `i`. Patching a contribution replaces one sublayer's addition and nothing else:

```python
with model.trace(clean):
    clean_attn = model.layers[LAYER].self_attn.attention_output.save()
    clean_mlp = model.layers[LAYER].mlp.mlp_output.save()

with model.trace() as tracer:
    with tracer.invoke(corrupt):
        model.layers[LAYER].self_attn.attention_output[:, POS] = clean_attn[:, POS]
        attn = model.next_token_probs[0, paris].save()
    with tracer.invoke(corrupt):
        model.layers[LAYER].mlp.mlp_output[:, POS] = clean_mlp[:, POS]
        mlp = model.next_token_probs[0, paris].save()

print(f"attention_output {float(attn):.3f}   mlp_output {float(mlp):.3f}")
# attention_output 0.151   mlp_output 0.018
```

Block 21's attention brings 0.151 of the answer to the last position on its own and its MLP 0.018. The rest of the 0.778 was already in the stream from block 20, where the `layer_output` patch gives 0.32. One head is a slice of `attention_head_outputs`, `[..., HEAD, :]`, patched the same way under `attn_implementation="eager"`.

## On another family

The identity `layer_input + attention_output + mlp_output == layer_output` is the per-family work. A contribution patch means "the block adds the clean sublayer's contribution instead of the corrupt one's" on every block type, and `layer_output` is a tensor even where the block returns a tuple (BLOOM, MPT, Falcon), so there is no `[0]` and no tuple to rebuild. What does change is the tokenizer. The two prompts must tokenize to the same length for positions to correspond, which the `assert` checks, and `[:, -1]` survives a length difference where an absolute index does not. On Granite a patch at one position moves the other positions by rounding in bf16, because the block adds a scaled copy. Load in float32 there.
