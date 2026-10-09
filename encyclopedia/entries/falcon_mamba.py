"""Falcon-Mamba: Mamba-1's block and selective scan, with weightless RMS norms on the scan's B, C and step size."""

MODEL_TYPE = "falcon_mamba"
TITLE = "Falcon-Mamba / Falcon3-Mamba"
SUBTITLE = (
    "Mamba's block, one norm and a selective-scan mixer, with no attention and no MLP; inside the mixer, "
    "weightless RMS norms sit on B, C and the step-size projection before the scan."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tiiuae/falcon-mamba-7b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-FalconMambaForCausalLM"
CHECKPOINTS = [
    "tiiuae/falcon-mamba-7b", "tiiuae/falcon-mamba-7b-instruct", "tiiuae/falcon-mamba-7b-pre-decay",
    "tiiuae/Falcon3-Mamba-7B-Base", "tiiuae/Falcon3-Mamba-7B-Instruct",
]

#: Set by hues.py (lineage: Mamba).
PALETTE = {"hue": 5}
VLLM = False
QUIRKS = ["mamba1", "fp32-residual"]

#: Every checkpoint is 7B: shapes and identities below were checked on the pinned tiny checkpoint, sizes are read
#: from the released configs and a meta build of falcon-mamba-7b; no real-weight number is given.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Selective scan",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "The block's module is named norm; nnterp calls it input_layernorm. model.norm is the "
                             "final norm_f. The mixer's dt_layernorm, b_layernorm and c_layernorm sit inside it.",
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
    "embed": "No position embedding and no scale: token_embeddings equals layers[0].input.",
    "layers": "residual_in_fp32: each block adds in float32, so layer_output is float32 whatever the load dtype.",
    "head": "Its own weight, not tied to embed_tokens.",
    "logits": "lm_head.output cast to float32; the stream is cast to lm_head's dtype first. project_on_vocab does the same.",
}

NOTES = """
## The block, in order

```
out = x + mixer(norm(x))                   # added in float32
mixer(h): u, z = in_proj(h)
          u = silu(conv1d(u))              # attention_values
          dt, B, C = x_proj(u)
          dt, B, C = rms(dt), rms(B), rms(C)   # weightless
          y = scan(u, dt_proj(dt), A, B, C) + D * u
          return out_proj(y * silu(z))     # attention_output
```

The tree, the block and the values are Mamba's (the `mamba` page): the block's norm is
`input_layernorm` (native `norm`), the mixer is `linear_attn`, and `model.norm` is the final
`norm_f`. What is this family's own sits inside the mixer: `dt_layernorm`, `b_layernorm` and
`c_layernorm`, RMS norms with no gain (their `weight` is a buffer of ones the torch path never
reads), with `mixer_rms_eps` (`1e-6` on falcon-mamba-7b, `1e-5` on Falcon3-Mamba).

## Where the norms sit

`x_proj` maps `attention_values` (the convolved `x`) to a low-rank step, `B` and `C`; the three
norms apply to those three outputs, and the scan reads what they return.

- `attention_keys` and `attention_queries` are the normed `B` and `C`: each token's vector of
  `state_size` (16) entries has a root-mean-square of one.
- `betas` is `softplus(dt_proj(rms(dt)) + dt_bias)`: the norm sits on the low-rank step
  (`time_step_rank`, 256) before `dt_proj`, so `betas` itself carries no fixed scale, and
  `decays` is `betas * A` as on Mamba.
- The un-normed `B` and `C` are `x_proj`'s output, columns `time_step_rank:` of it. Read it in a
  trace of its own: in one trace with the kernel's values, `x_proj` runs after the point where
  the values decide which kernel the call runs, and the trace fails with `OutOfOrderError`.

```python
mix = model.layers[1].linear_attn
with model.trace(prompt):
    raw = mix.x_proj.output.save()          # [batch, seq, rank + 2 * state_size]
with model.trace(prompt):
    B = mix.attention_keys.save()           # [batch, seq, 1, state_size]

r, n = model.config.time_step_rank, model.config.state_size
b = raw[..., r:r + n].float()
normed = b * torch.rsqrt(b.pow(2).mean(-1, keepdim=True) + model.config.mixer_rms_eps)
torch.testing.assert_close(normed.to(B.dtype), B[:, :, 0])
```

A write to `attention_keys` or `attention_queries` replaces the normed vector the scan reads,
so a written vector need not have unit scale; a write to `x_proj.output` passes through the
norms first.

## The contribution is the whole block

`attention_output` is `out_proj`'s output and all the block adds:
`layers[i].input + attention_output == layer_output`, exactly, in float32.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].linear_attn.attention_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn.float(), out)
```

## Loading: route the kernels

With `mamba_ssm` installed every trace on CPU fails inside the scan kernel
(`RuntimeError: Expected u.is_cuda() to be true`), and on GPU every `linear_attn` value but
`attention_output` reports the kernel in `support()`. Routing binds the scan and the decode
step to transformers' torch functions in Falcon-Mamba's own modeling module; after it
`support()` reports every value, `state` and `states` included.

```python
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.falcon_mamba, "torch")
model = StandardizedTransformer("tiiuae/falcon-mamba-7b")
```

## The mixer's values

On falcon-mamba-7b the mixer is 8192 channels wide, each with a state of 16:
`attention_values`, `betas` and `attention_head_outputs` are `[batch, seq, 8192]`,
`attention_keys` and `attention_queries` `[batch, seq, 1, 16]`, `decays` and `states`
`[batch, seq, 8192, 16]`. `config.expand` reads 16 on the falcon-mamba-7b checkpoints and 2 on Falcon3-Mamba;
the mixer's width is `intermediate_size`, 8192 on both. The scan is Mamba's, so the hidden
attention on the `mamba` page rebuilds `attention_head_outputs` here unchanged, with the normed
`B` and `C`.

## The state

`state_output` is `[batch, 8192, 16]` and equals `states[:, -1]`. On a decode step read
`state_output` before `attention_head_outputs`: the step updates the state before it reads it.

```python
heads, leaving = [], []
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        if step == 0:
            heads.append(mix.attention_head_outputs.save())
            leaving.append(mix.state_output.save())
        else:
            leaving.append(mix.state_output.save())
            heads.append(mix.attention_head_outputs.save())
```

## The readout and the embeddings

`lm_head` has its own weight. `project_on_vocab` casts the normed stream to the head's dtype
and the logits to float32, as the model does, so at the last block's `layer_output` it equals
`logits`. `token_embeddings` equals `layers[0].input`.

## The checkpoints and the tokenizer

`falcon-mamba-7b` and its `-instruct` and `-pre-decay` checkpoints, and Falcon3-Mamba-7B
`-Base` and `-Instruct`, all 64 blocks of width 4096. They tokenize with Falcon's 65024-token
BPE and prepend nothing. `<|endoftext|>` (id 11) ends a sequence, and the tokenizer sets no
padding token.
"""
