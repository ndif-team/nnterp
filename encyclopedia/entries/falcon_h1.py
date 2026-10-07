"""Falcon-H1: a parallel hybrid, a Mamba-2 mixer and attention side by side in every block, then an MLP."""

MODEL_TYPE = "falcon_h1"
TITLE = "Falcon-H1"
SUBTITLE = (
    "Every block runs a Mamba-2 mixer and attention side by side on one normed input and adds both, each times "
    "its own multiplier, then a gated MLP; the embeddings and the logits carry multipliers too."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tiiuae/Falcon-H1-0.5B-Base"
#: The tiny checkpoint the test suite builds the page from (the released multipliers; loaded in float32 by the suite).
PINNED = "yujiepan/falcon-h1-tiny-random"
CHECKPOINTS = [
    "tiiuae/Falcon-H1-0.5B-Base", "tiiuae/Falcon-H1-0.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Base", "tiiuae/Falcon-H1-1.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Deep-Base", "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
    "tiiuae/Falcon-H1-3B-Base", "tiiuae/Falcon-H1-3B-Instruct",
    "tiiuae/Falcon-H1-7B-Base", "tiiuae/Falcon-H1-7B-Instruct",
    "tiiuae/Falcon-H1-34B-Base", "tiiuae/Falcon-H1-34B-Instruct",
    "tiiuae/Falcon-H1R-7B",
    "tiiuae/Falcon-H1-Tiny-90M-Base", "tiiuae/Falcon-H1-Tiny-90M-Instruct",
]

#: Set by hues.py (lineage: Falcon).
PALETTE = {"hue": 14}
VLLM = False
QUIRKS = ["parallel-mixers", "mamba2", "scaled-residual-adds", "embedding-multiplier", "scaled-logits", "tuple-blocks"]

#: Every real-value number below was run on Falcon-H1-0.5B-Base, routed to the torch kernels, in float32 with
#: attn_implementation="eager", on an RTX A6000, unless it says bfloat16.

NORM_NOTE = ("One module, input_layernorm, drawn on both mixers' rows: the Mamba-2 mixer and the attention read the "
             "same output tensor (the attention times attention_in_multiplier, 1.0 on every released checkpoint).")

#: The block in forward order: the Mamba-2 mixer and the attention both read input_layernorm's output and join the
#: stream at one add, then the MLP reads the stream after that add. `parallel_with_next` marks the mixer as the
#: first branch of that pair, so the page draws the two mixers side by side, joining at one add.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "input_layernorm",
            "pre_norm_note": NORM_NOTE,
            "parallel_with_next": True,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD, beside self_attn",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm: it reads the stream after both mixers' add.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, gated {hidden_act}",
        },
    ],
    "identity": "layers[i].input + (linear_attn.attention_output + self_attn.attention_output) + mlp.mlp_output == layer_output",
    "identity_note": "Exact with the two mixers summed first, as the block sums them; each mixer's attention_output is its "
                     "output times its multiplier, the tensor the block adds.",
}

STRIP = {
    "embed": "The model multiplies embed_tokens' output by embedding_multiplier before block 0: token_embeddings is the "
             "unscaled lookup, layers[0].input the scaled one.",
    "norm": "An RMSNorm whose gain is its weight as stored (mean 16.2, up to 106 on 0.5B); project_on_vocab applies it.",
    "head": "Its own weight on 0.5B to 34B; tied to embed_tokens on the Tiny 90M checkpoints.",
    "logits": "lm_head.output times lm_head_multiplier; project_on_vocab applies it, so a lens at the last block equals logits.",
}

NOTES = """
## The block, in order

```
n   = input_layernorm(x)
h   = x + (mamba(n) * ssm_out_multiplier
           + self_attn(n * attention_in_multiplier) * attention_out_multiplier)
out = h + feed_forward(pre_ff_layernorm(h))
```

Every block has both mixers: `linear_attn` (native `mamba`, a Mamba-2 mixer) and `self_attn`
read the same normed input, the Mamba-2 mixer runs first, and the block adds the two at once.
The MLP (native `feed_forward`, nnterp's `mlp`) then reads the stream after that add through
`pre_ff_layernorm`, which nnterp names `post_attention_layernorm`. The block returns a
one-element tuple; `layer_output` is the tensor. `config.layer_types` is `hybrid` on every block.

## The contributions are the scaled outputs

Each mixer's `attention_output` is the block's own product, the module's output times
`ssm_out_multiplier` or `attention_out_multiplier`: the tensor the block adds, so a read, an
assignment or an in-place edit needs no arithmetic. `linear_attn.output` and
`self_attn.output[0]` are the unscaled module outputs. The identity has four terms:

```python
L = model.layers[1]
with model.trace(prompt):
    x = L.input.save()
    ssm = L.linear_attn.attention_output.save()
    attn = L.self_attn.attention_output.save()
    mlp = L.mlp.mlp_output.save()
    out = L.layer_output.save()

torch.testing.assert_close(x + (ssm + attn) + mlp, out)
```

On 0.5B, over the positions after the first, the Mamba-2 mixer adds a larger vector than the
attention in 33 of 36 blocks (a median norm ratio of 2.2), and the MLP's is about 3.4 times the
attention's.

## Every multiplier, and where the values read it

The configs carry µP multipliers; nnterp reads each value where the forward has applied them.

- `ssm_out_multiplier`, `attention_out_multiplier`: inside each mixer's `attention_output`
  (0.236 and 0.9375 on 0.5B; 0.088 and 0.0375 on 34B).
- `attention_in_multiplier`: the attention reads `input_layernorm`'s output times it, 1.0 on
  every released checkpoint, so `self_attn.input` is the norm's output.
- `key_multiplier`: `k_proj`'s output is multiplied by it before the rotary, so
  `attention_keys` carry it (0.39 on 0.5B).
- `ssm_in_multiplier` and `ssm_multipliers`: the mixer scales its input, then each part of
  `in_proj`'s output (`z`, `x`, `B`, `C`, `dt`), before the convolution; `attention_queries`,
  `attention_keys`, `attention_values` and `betas` are read after them.
- `mlp_multipliers`: the gate's pre-activation and `down_proj`'s output are scaled inside the
  MLP, so `mlp_output` (the MLP's return) carries both, and `mlp.down_proj.output` does not.
- `embedding_multiplier` and `lm_head_multiplier`: on the strip above.

## Loading

The attention interior needs `attn_implementation="eager"`. The default load runs the same
function: on 0.5B the eager and `sdpa` logits differ by at most 2.3e-5 in float32 and 0.5 in
bfloat16, with the same top token at every position.

With `mamba_ssm` installed transformers sends both Mamba-2 scans to its Triton kernels: every
`linear_attn` value but `attention_output` reports the kernel in `support()`, and on a CPU no
trace runs. Route before the first trace. The routed and the default kernels give logits within
0.019 of each other in float32 (logits up to 46) and 0.5 in bfloat16 on 0.5B, with the same top
tokens.

```python
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.falcon_h1, "torch")
model = StandardizedTransformer("tiiuae/Falcon-H1-0.5B-Base", attn_implementation="eager")
```

## The Mamba-2 mixer

On 0.5B the mixer has 24 heads of 64 channels and a state of 128 per head, in one group:
`attention_queries` and `attention_keys` are `[batch, seq, 1, 128]`, `attention_values` and
`attention_head_outputs` `[batch, seq, 24, 64]`, and the state `[batch, 24, 128, 64]`. `states`
needs `nnterp.chunk_per_token(model)`: the scan keeps the state every `mamba_chunk_size` tokens
(128). The family module's docstring and `docs/usage/state-space.md` give the rest.

`mamba_rms_norm` decides what follows the scan. It is false on 0.5B and the Tiny checkpoints:
`linear_attn.norm` is an identity, `y` is gated by `silu(z)` and goes to `out_proj`, and on a
decode step the gate is applied inside the update kernel, so a decode step's
`attention_head_outputs` is gated where a prompt's is not. It is true on 1.5B, 1.5B-Deep, 3B,
7B, 34B and H1R-7B: a gated RMSNorm follows the scan, and both are ungated.

## Position 0 carries a very large norm

On 0.5B the stream at the first position has a norm of 570 to 920 after block 9, where the
other positions' median is about 60; the gap closes in the last blocks (214 to 545 at position 0
after block 27, about 250 elsewhere). Block 18's attention puts 0.64 to 0.85 of its mass on position 0, over heads
and queries, on four test texts. The tokenizer prepends nothing, so position 0 is the prompt's own
first token. Leave it out before averaging activations for steering vectors or probes.

```python
with model.trace(prompt):
    resid = model.layers[9].layer_output.save()

resid[0].norm(dim=-1)        # 0.5B: 570 to 920 at position 0, about 60 elsewhere
```

## Attention

Every block's attention applies rotary embeddings with `rope_theta` 1e11: of the 32 frequency
pairs of a 64-wide head, about ten turn once within 0.5B's 16384-token context, and the rest
barely move. 0.5B has 8 query heads over 2 key/value heads of 64: query head `h` reads key/value
head `h // 4`. `attention_keys` are scaled by `key_multiplier` before the rotary.

## The readout and the embeddings

`logits` is `lm_head.output * lm_head_multiplier` (0.039 on 0.5B), and `project_on_vocab`
applies the multiplier, so at the last block's `layer_output` it equals `logits`. The final
norm's gain is large (mean 16.2, up to 106 on 0.5B) and the head's raw output reaches 1179 where
`logits` stops at 46. `layers[0].input` is `token_embeddings * embedding_multiplier` (5.66 on
0.5B to 34B; 0.08 to 0.11 on the Tiny checkpoints).

## The checkpoints and the tokenizer

0.5B, 1.5B, 1.5B-Deep (66 blocks of width 1280), 3B, 7B and 34B, each Base and Instruct;
Falcon-H1R-7B, a reasoning model on 7B's shape; and the Falcon-H1-Tiny 90M checkpoints, whose
mixers' and attention's multipliers are all 1. The vocabulary is 32768 tokens on 0.5B and grows
with size (261120 on 34B). The tokenizer prepends no BOS on a plain prompt; `<|end_of_text|>`
(id 11) ends a sequence. The Instruct chat template opens with `<|begin_of_text|>` and uses
`<|im_start|>` and `<|im_end|>` turns.
"""
