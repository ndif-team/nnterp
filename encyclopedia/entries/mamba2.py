"""Mamba-2: a pure state-space model, every block one norm and one SSD mixer whose state is per head."""

MODEL_TYPE = "mamba2"
TITLE = "Mamba-2"

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
QUIRKS = ["mamba2", "one-sublayer-blocks"]

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
            "pre_norm_note": "The block's norm; linear_attn.norm is the mixer's gated norm and model.norm the final one.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD, {num_heads} heads × {head_dim}, state {state_size}",
        },
    ],
}

STRIP = {
    "layers": "Each block adds in float32, so layer_output is float32 whatever the load dtype.",
}

NOTES = """
## A gated norm over every head follows `attention_head_outputs`

```
out = h + mixer(norm(h))
mixer: y = ssd(x, dt, A, B, C) + D * x       # x: attention_values; y: attention_head_outputs
       return out_proj(norm(y * silu(z)))    # one norm over the mixer's whole width
```

The mixer gates `y` with `silu(z)` and then normalizes it over all heads together, whatever
`n_groups` is. A head's term in `attention_output` depends on every other head, so zeroing one head
of `attention_head_outputs` rescales the rest.

## `C` and `B` are shared by the heads of a group

`attention_queries` (`C`) and `attention_keys` (`B`) have one row per group of heads, and the 130m to
2.7b checkpoints have one group, so an edit to either reaches every head. Mamba-Codestral has eight
groups.

## The pattern is rebuilt from the scan's values

There is no `attention_probabilities`. Per head the scan is a causal linear map over tokens, and the
served values give its weights exactly:

```
y_t        = sum_s alpha[t, s] * x_s + D * x_t
alpha[t, s] = (C_t . B_s) * exp(decays_{s+1} + ... + decays_t) * betas_s
```

Repeat `C` and `B` from groups to heads first.
"""
