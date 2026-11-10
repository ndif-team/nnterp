"""Falcon-Mamba: Mamba-1's block and selective scan, with weightless RMS norms on the scan's B, C and step size."""

MODEL_TYPE = "falcon_mamba"
TITLE = "Falcon-Mamba / Falcon3-Mamba"

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
QUIRKS = ["mamba1"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Selective scan",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Native name norm, the block's own; model.norm is the final norm_f.",
            "contribution": "attention_output",
            "interior": [
                "state_input", "attention_values", "betas", "decays", "attention_keys", "attention_queries",
                "states", "attention_head_outputs", "state_output",
            ],
            "detail": "{intermediate_size} channels × state {state_size}",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = ""
