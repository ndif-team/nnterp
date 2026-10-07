"""Mamba (Mamba-1): a pure state-space model, every block one norm and one selective-scan mixer."""

MODEL_TYPE = "mamba"
TITLE = "Mamba"
SUBTITLE = (
    "No attention and no MLP: each block is one norm and a selective-scan mixer whose state is per channel, "
    "and the mixer's values need its CUDA kernels routed to transformers' torch ones."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "state-spaces/mamba-130m-hf"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MambaForCausalLM"
CHECKPOINTS = [
    "state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf", "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf", "state-spaces/mamba-2.8b-hf",
]

#: Set by hues.py (lineage: Mamba).
PALETTE = {"hue": 359}
VLLM = False
QUIRKS = ["mamba1", "fp32-residual"]

#: Every real-value number below was run on mamba-130m-hf (and, where it says so, mamba-370m-hf),
#: routed to the torch kernels, in float32, the checkpoints' dtype; the kernel comparison ran on an RTX A6000.

#: One sublayer: the mixer. Its chips are in the order the scan meets them on a prompt.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Selective scan",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "The block's module is named norm; nnterp calls it input_layernorm. model.norm is the final norm_f.",
            "contribution": "attention_output",
            "interior": [
                "state_input", "attention_values", "betas", "decays", "attention_keys", "attention_queries",
                "states", "attention_head_outputs", "state_output",
            ],
            "detail": "{intermediate_size} channels × state {state_size}",
        },
    ],
    "identity_note": "Exact: the block adds the mixer's output to the stream in float32, and nothing else.",
}

STRIP = {
    "embed": "No position embedding and no scale: token_embeddings equals layers[0].input. 50280 rows for "
             "GPT-NeoX's 50277 tokens.",
    "layers": "residual_in_fp32: each block adds in float32, so layer_output is float32 whatever the load dtype.",
    "head": "lm_head is embed_tokens' weight (tied); the checkpoint stores no lm_head.",
    "logits": "lm_head.output cast to float32; the stream is cast to lm_head's dtype first. project_on_vocab does the same.",
}

NOTES = """
## The block, in order

```
out = x + mixer(norm(x))                           # added in float32 (residual_in_fp32)
mixer(h): u, z = in_proj(h);  u = silu(conv1d(u))  # causal, width 4; u is attention_values
          dt, B, C = x_proj(u);  dt = dt_proj(dt)
          y = scan(u, dt, A, B, C) + D * u         # y is attention_head_outputs
          return out_proj(y * silu(z))             # attention_output
```

Two modules are called `norm` natively: the block's own, which nnterp also serves as
`input_layernorm`, and the model's final `norm_f`, which is `model.norm`. So
`model.layers[i].norm` still resolves, to the block's pre-norm, not to the final norm.

## The contribution is the whole block

`attention_output` is the mixer's own output (`out_proj`'s), and it is all the block adds, so
`layers[i].input + attention_output == layer_output` exactly, and ablating `attention_output`
ablates the block. The add is in float32: in a bfloat16 load `attention_output` is bfloat16 and
`layer_output` float32, and the identity holds after the cast:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].linear_attn.attention_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn.float(), out)
```

The stream grows through the stack: on 130m its norm at the last token of a ten-token prompt is
7.4 entering block 0 and 188 entering block 23, and block 23's contribution (262) is larger than
the stream it joins. A steering vector sized for one depth is small or large at another.

## Loading: route the kernels

With `mamba_ssm` installed transformers dispatches the scan and the decode step to its CUDA
kernels. On CPU every trace then fails inside the kernel, even one that reads only
`layer_output` (`RuntimeError: Expected u.is_cuda() to be true`). On GPU, unrouted, `layer_output`,
`attention_output` and `logits` read normally and every other `linear_attn` value raises
`nnterp.Unavailable`. Route before the trace that reads the mixer's values; an earlier unrouted
trace (one that read only `layer_output`, one that failed on CPU, one whose read was refused)
does not pin the kernels:

```python
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.mamba, "torch")
model = StandardizedTransformer("state-spaces/mamba-130m-hf", device="cpu")
```

The two paths compute the same model. In float32 on GPU the kernel and the routed torch scan
give logits within 2e-4 of each other (130m and 370m, logits up to 98), with the same top token
everywhere. In bfloat16 they differ by up to 2.25 on 130m, about what the torch scan on CPU and
on GPU differ by (2.0), so the gap is bfloat16 rounding. The torch scan is a token loop: on a 22-token
prompt it took 83 ms against the kernel's 46 ms on 130m, 286 ms against 46 ms on 370m.

`model.train()` sends the forward down a third path, the fused `mamba_inner_fn`, which
`route_kernels` does not reroute: with `mamba_ssm` installed and `causal-conv1d` not, every trace
in training mode fails with `AssertionError: causal_conv1d_cuda is not available`, on CPU and
GPU, routed or not. Gradients do not need it; `backward()` runs through the routed scan in eval
mode.

## The mixer's values

Each channel of the mixer's width (1536 on 130m) keeps its own state of `state_size` (16)
numbers. `attention_queries` and `attention_keys` are `C` and `B`, `[batch, seq, 1, 16]`: one
vector per token, shared by every channel. `attention_values` is `x`, `[batch, seq, 1536]`, after
the convolution and the `silu`. `betas` is the step size `softplus(dt + dt_bias)`, one per token
and channel; `decays` is `betas[..., None] * A`, `[batch, seq, 1536, 16]`, the log of how much of
each state entry a token keeps. Both are read-only.

The rates are far apart on real weights. `A = -exp(A_log)` runs from -2e-6 to -3e8 on 130m
(median -12.6), so over every block and a ten-token prompt `exp(decays)` has a median of 0.48,
28% of entries keep less than 1% and 12% keep more than 99% (on 370m: 0.55, 15%, 9%). The pinned
tiny checkpoint has the initialization's `A`, -1 to -16 on every channel.

## The hidden attention

The scan is a causal linear map over tokens, one per channel: `y_t = sum_s alpha[t, s] x_s + D x_t`
with `alpha[t, s] = C_t . (exp(decays_{s+1} + ... + decays_t) * betas_s B_s)`. The served values
rebuild it exactly (float32, as loaded):

```python
mix = model.layers[1].linear_attn
D = mix._module.D
with model.trace(prompt):
    x = mix.attention_values[0].save()            # [seq, channels]
    beta = mix.betas[0].save()                    # [seq, channels]
    decay = mix.decays[0].save()                  # [seq, channels, state_dim]
    B = mix.attention_keys[0, :, 0].save()        # [seq, state_dim]
    C = mix.attention_queries[0, :, 0].save()     # [seq, state_dim]
    y = mix.attention_head_outputs[0].save()      # [seq, channels]

cum = decay.cumsum(0)
causal = torch.ones(len(x), len(x), dtype=torch.bool).tril()[..., None, None]
kept = (cum[:, None] - cum[None, :]).masked_fill(~causal, -torch.inf).exp()
alpha = torch.einsum("tn,tscn,sn->tsc", C, kept, B) * beta     # [query t, key s, channel]
torch.testing.assert_close(torch.einsum("tsc,sc->tc", alpha, x) + D * x, y)
```

`kept` is `[seq, seq, channels, state_dim]`, 10 MB for ten tokens on 130m and growing with the
square of the prompt: slice the channels on a long one.

## The state

`state_output` is the state after the call's last token, `[batch, 1536, 16]`, and equals
`states[:, -1]`; `states` holds every token's and needs the routing. A write to the state at
token `t` is not all the block carries forward: the width-4 convolution mixes each token's
input with the three before it. On 130m, zeroing every block's state after token 4 of a
ten-token prompt and running the last five tokens alone give logits up to 87 apart and
different top tokens at three of the five positions.

On a decode step the update comes before the read, the reverse of the prompt's scan, so under
`generate` read `state_output` before `attention_head_outputs` on every step after the first.
The wrong order raises nothing: step 1's `state_output` comes back as step 2's.

```python
heads, leaving = [], []
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        if step == 0:                        # the prompt's scan: y, then the state
            heads.append(mix.attention_head_outputs.save())
            leaving.append(mix.state_output.save())
        else:                                # a decode step: the state, then y
            leaving.append(mix.state_output.save())
            heads.append(mix.attention_head_outputs.save())
```

## Edit C and B by assignment

`attention_queries` and `attention_keys` are views of `torch.split`'s outputs, and an in-place edit under
autograd raises `RuntimeError: Output 0 of Select is a view and is being modified inplace`.
Assign a copy, or edit in place under `torch.no_grad()`; both reach the scan and give the same
logits. `attention_values` takes in-place edits.

```python
with model.trace(prompt):
    q = mix.attention_queries.clone()
    q[:, -1] = 0                  # the last token reads nothing from the state
    mix.attention_queries = q
```

## The readout and the embeddings

`lm_head` is `embed_tokens`' weight; the checkpoint stores no `lm_head`, so an edit to one is an
edit to the other. `project_on_vocab` at the last block's `layer_output` equals `logits`
exactly. The final norm's gain is its weight (mean 1.05, up to 5.2 on 130m). Nothing is added
after `embed_tokens`, so `token_embeddings` equals `layers[0].input`.

## The tokenizer is GPT-NeoX's

The checkpoints use GPT-NeoX-20B's BPE (`GPTNeoXTokenizer`, 50277 tokens; the embedding has 50280
rows). It prepends no BOS, and `<|endoftext|>` (id 0) is BOS, EOS and padding. The mixer zeroes
padded positions before and after the convolution, so the state stays zero across left padding:
a padded prompt's logits match the prompt alone to 2e-4. On 370m `config.d_inner` reads 160; the
mixer's width is `intermediate_size`, 2048.

## The family's checkpoints

This family is the `-hf` conversions of the original Mamba, 130m to 2.8b. Falcon-Mamba is its
own family, `falcon_mamba`, with RMS norms on `B`, `C` and `dt` inside the mixer; Mamba-2 is
`mamba2`, a different mixer (SSD, one state per head); Jamba's Mamba blocks carry this mixer.

## Published work

- *The Hidden Attention of Mamba Models* (Ali, Zimerman and Wolf, 2024) reads Mamba's scan as
  the attention matrix above, built from `attention_queries`, `attention_keys`, `betas` and
  `decays`.
- *Locating and Editing Factual Associations in Mamba* (Sen Sharma, Atkinson and Bau, 2024)
  localizes factual recall with causal tracing over Mamba's hidden states, `layer_output` here,
  and inserts facts with rank-one edits.
"""
