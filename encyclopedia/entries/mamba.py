"""Mamba (Mamba-1): a pure state-space model, every block one norm and one selective-scan mixer."""

MODEL_TYPE = "mamba"
TITLE = "Mamba"

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
QUIRKS = ["mamba1"]

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
            "pre_norm_note": "Natively layers[i].norm; model.norm is the final norm_f.",
            "contribution": "attention_output",
            "interior": [
                "state_input", "attention_values", "betas", "decays", "attention_keys", "attention_queries",
                "states", "attention_head_outputs", "state_output",
            ],
            "detail": "{intermediate_size} channels × state {state_size}",
        },
    ],
}

STRIP = {
    "layers": "Each block adds in float32, so layer_output is float32 whatever the load dtype.",
}

NOTES = """
## `attention_head_outputs` is read before the gate

```
out = h + mixer(norm(h))
mixer: y = scan(x, dt, A, B, C) + D * x     # x: attention_values; y: attention_head_outputs
       return out_proj(y * silu(z))         # attention_output
```

A channel's term in `attention_output` is `out_proj` applied to `y * silu(z)`, so attribute through
`out_proj.input`, not `attention_head_outputs`.

## `C` and `B` are shared by every channel

`attention_queries` (`C`) and `attention_keys` (`B`) are one vector per token, `[batch, seq, 1,
state_size]`, read by every channel of the mixer, so an edit to either reaches all of them.

## The pattern is rebuilt from the scan's values

There is no `attention_probabilities`. Per channel the scan is a causal linear map over tokens, and
the served values give its weights exactly:

```
y_t        = sum_s alpha[t, s] * x_s + D * x_t
alpha[t, s] = C_t . (exp(decays_{s+1} + ... + decays_t) * betas_s * B_s)
```

The decay term is `[seq, seq, channels, state_size]`; on a long prompt, slice the channels.
"""
