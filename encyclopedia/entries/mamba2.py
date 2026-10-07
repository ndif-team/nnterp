"""Mamba-2: a pure state-space model, every block one norm and one SSD mixer whose state is per head."""

MODEL_TYPE = "mamba2"
TITLE = "Mamba-2"
SUBTITLE = (
    "No attention and no MLP: each block is one norm and a Mamba-2 (SSD) mixer whose heads each keep a "
    "state matrix, with one scalar decay per head and token."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "AntonV/mamba2-130m-hf"
#: The Hub org the index files the family under: the conversion is AntonV's; the model is state-spaces'.
ORG = "state-spaces"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/mamba2-tiny-random"
#: The `-hf` conversions of the original Mamba-2 release, Mamba-Codestral, and the originals: state-spaces/mamba2-*
#: store a config without `model_type`, so they are listed greyed.
CHECKPOINTS = [
    "AntonV/mamba2-130m-hf", "AntonV/mamba2-370m-hf", "AntonV/mamba2-780m-hf",
    "AntonV/mamba2-1.3b-hf", "AntonV/mamba2-2.7b-hf",
    "mistralai/Mamba-Codestral-7B-v0.1",
    "state-spaces/mamba2-130m", "state-spaces/mamba2-370m", "state-spaces/mamba2-780m",
    "state-spaces/mamba2-1.3b", "state-spaces/mamba2-2.7b",
]

#: Set by hues.py (lineage: Mamba).
PALETTE = {"hue": 2}
VLLM = False
QUIRKS = ["mamba2", "one-sublayer-blocks", "fp32-residual"]

#: Every real-value number below was run on mamba2-130m-hf, routed to the torch kernels, in float32 on an RTX A6000
#: unless it says bfloat16.

#: One sublayer: the mixer. Its chips are the SSD values, as the nemotron_h page draws them.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "norm",
            "pre_norm_note": "The block's one RMSNorm keeps its native name, norm: the mixer has a gated norm of its own, "
                             "linear_attn.norm, so nnterp gives the block's no input_layernorm alias. model.norm is the "
                             "final norm_f.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD, {num_heads} heads × {head_dim}, state {state_size}",
        },
    ],
    "identity_note": "Exact: the block adds the mixer's output to the stream in float32, and nothing else.",
}

STRIP = {
    "embed": "No position embedding and no scale: token_embeddings equals layers[0].input.",
    "layers": "residual_in_fp32: each block adds in float32, so layer_output is float32 whatever the load dtype.",
    "head": "Tied to embed_tokens on the 130m to 2.7b conversions; Mamba-Codestral has its own lm_head.",
    "logits": "lm_head.output cast to float32; the stream is cast to lm_head's dtype first. project_on_vocab does the same.",
}

NOTES = """
## The block, in order

```
out = x + mixer(norm(x))                  # added in float32
mixer(h): z, xBC, dt = in_proj(h)
          x, B, C = silu(conv1d(xBC))     # causal, width 4
          y = ssd(x, dt, A, B, C) + D * x # attention_head_outputs
          return out_proj(norm(y * silu(z)))
```

The block's norm keeps its native name, `norm`, and `model.layers[i].norm` is the norm before
the mixer; `model.layers[i].linear_attn.norm` is the mixer's own gated norm and `model.norm` the
final `norm_f`. The convolution runs over `x`, `B` and `C` together. The gated norm multiplies `y` by `silu(z)` and then normalizes over the mixer's
whole width (1536 on 130m), whatever `n_groups` is; `attention_head_outputs` is read before both.

## The contribution is the whole block

`attention_output` is the mixer's output (`out_proj`'s), and it is all the block adds:
`layers[i].input + attention_output == layer_output` exactly, so ablating `attention_output`
ablates the block. The add is in float32: `attention_output` is in the load dtype and
`layer_output` is float32.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].linear_attn.attention_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn.float(), out)
```

The stream grows through the stack: on 130m, at the last token of a 14-token prompt, its norm
is 7.0 entering block 0 and 1316 entering block 23, and block 23 adds a vector of norm 1376.

## Loading: route the kernels

With `mamba_ssm` installed transformers sends both scans to its Triton kernels. On a CPU every
trace then fails inside the chunk scan, even one that reads only `layer_output`
(`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`, or
`RuntimeError: invalid argument to exchangeDevice` with no GPU visible). On a GPU, unrouted,
`layer_output`, `attention_output` and `logits` read normally and every other `linear_attn`
value reports the kernel in `support()`. Route before the first trace:

```python
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.mamba2, "torch")
model = StandardizedTransformer("AntonV/mamba2-130m-hf")
```

On 130m in float32 the kernel and the routed torch scan give logits within 0.06 of each other
(logits up to 120), with the same top token at every position; in bfloat16 within 2.25. The
torch scan took 146 ms on a 14-token prompt against the kernel's 23 ms.

## The mixer's values

On 130m each mixer has 24 heads of 64 channels and a state of 128 per head, in one group:
`attention_queries` (`C`) and `attention_keys` (`B`) are `[batch, seq, 1, 128]`, shared by all
24 heads, and `attention_values` (`x`) and `attention_head_outputs` (`y`) are
`[batch, seq, 24, 64]`. `betas` is the step `softplus(dt + dt_bias)` and `decays` is
`betas * A`, both `[batch, seq, 24]`: one scalar decay per head and token. Mamba-Codestral has 128 heads in 8 groups, so `C` and `B` are
`[batch, seq, 8, 128]` and each serves 16 heads.

The rates are far apart on real weights. `A = -exp(A_log)` runs from -4e-4 to -36316 over the
576 heads of 130m (median -0.21). Over every block of a 14-token prompt, `exp(decays)` has a
median of 0.89; 5% of entries keep less than 1% of the state and 19% keep more than 99%.

## A zero step skips a token exactly

Every Mamba-2 checkpoint here has `time_step_limit` `(0.0, inf)`, so the chunk scan clamps
nothing, and a written zero runs as zero: the token writes nothing and the state does not decay.
On 130m, with `chunk_per_token`, the state after token 3 then equals the state after token 2 bit
for bit, and a read of `betas` after the write returns the zero.

```python
nnterp.chunk_per_token(model)
mix = model.layers[0].linear_attn
with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0
    mix.betas = betas
    states = mix.states.save()

assert torch.equal(states[:, 3], states[:, 2])
```

Writing every block's `betas` back unchanged moves 130m's float32 logits by up to 1.2e-4.

## The hidden attention

Per head the scan is a causal linear map over tokens: `y_t = sum_s alpha[t, s] x_s + D x_t` with
`alpha[t, s] = (C_t . B_s) * exp(decays_{s+1} + ... + decays_t) * betas_s`, one scalar per head
and query-key pair, so the map is a `[seq, seq]` matrix per head. The served values rebuild `y`
exactly (float32):

```python
mix = model.layers[1].linear_attn
with model.trace(prompt):
    C = mix.attention_queries[0].save()         # [seq, groups, state_dim]
    B = mix.attention_keys[0].save()            # [seq, groups, state_dim]
    x = mix.attention_values[0].save()          # [seq, heads, head_dim]
    beta = mix.betas[0].save()                  # [seq, heads]
    decay = mix.decays[0].save()                # [seq, heads]
    y = mix.attention_head_outputs[0].save()    # [seq, heads, head_dim]

D = mix._module.D
per_group = x.shape[1] // C.shape[1]
C, B = C.repeat_interleave(per_group, 1), B.repeat_interleave(per_group, 1)
cum = decay.cumsum(0)
causal = torch.ones(len(x), len(x), dtype=torch.bool, device=x.device).tril()[..., None]
kept = (cum[:, None] - cum[None, :]).masked_fill(~causal, -torch.inf).exp()
alpha = torch.einsum("thn,shn->tsh", C, B) * kept * beta     # [query t, key s, head]
torch.testing.assert_close(torch.einsum("tsh,shd->thd", alpha, x) + D[:, None] * x, y)
```

## The state

`state_output` is the state after the call's last token, `[batch, 24, 128, 64]` on 130m, key
side first. `states`, the state after every token, needs `nnterp.chunk_per_token(model)`: the
chunk scan keeps the state only every `chunk_size` tokens (256). Then `states[:, -1]` equals
`state_output`.

```python
nnterp.chunk_per_token(model)
with model.trace(prompt):
    states = mix.states.save()           # [batch, seq, 24, 128, 64] on 130m
    final = mix.state_output.save()

assert torch.equal(states[:, -1], final)
```

On a decode step the update comes before the read, so under `generate` read `state_output`
before `attention_head_outputs` on every step after the first:

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

The width-4 convolution runs over `x`, `B` and `C`, so a state written through `state_input`
is not all a block carries into the next tokens.

## The readout and the embeddings

On the 130m to 2.7b conversions `lm_head` is `embed_tokens`' weight, so an edit to one is an edit
to the other; Mamba-Codestral's head has its own. `project_on_vocab` at the last block's
`layer_output` equals `logits` exactly. The final norm's gain is its weight (mean 0.94, up to 4.9
on 130m). Nothing is added after `embed_tokens`, so `token_embeddings` equals `layers[0].input`.

## The checkpoints and the tokenizer

`state-spaces/mamba2-130m` to `-2.7b` are the original release, stored for the `mamba_ssm`
package with no `model_type`; transformers loads the `AntonV/mamba2-*-hf` conversions, which
nnterp's docs use. They tokenize with GPT-NeoX-20B's BPE (`GPTNeoXTokenizer`, 50277 tokens; the
embedding has 50288 rows), prepend no BOS, and use `<|endoftext|>` (id 0) as BOS, EOS and
padding. Mamba-Codestral-7B is a code model with a 32768-token vocabulary.

## Published work

- *Transformers are SSMs* (Dao and Gu, 2024) defines SSD as the masked attention above:
  `C`, `B` and `x` in the roles of `attention_queries`, `attention_keys` and
  `attention_values`, with the decay mask built from `decays`.
"""
