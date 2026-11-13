"""Gemma 1: Llama's block with Gemma's scaled embeddings, 1 + weight norms and a GeGLU MLP; PaliGemma wraps it."""

MODEL_TYPE = "gemma"
TITLE = "Gemma / PaliGemma"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-GemmaForCausalLM"
CHECKPOINTS = [
    "google/gemma-2b", "google/gemma-2b-it",
    "google/gemma-7b", "google/gemma-7b-it",
    "google/gemma-1.1-2b-it", "google/gemma-1.1-7b-it",
    "google/codegemma-2b", "google/codegemma-7b",
    # Vision-language wrappers around a Gemma 1 text model (text_config.model_type gemma): model_type paligemma.
    # PaliGemma 2 wraps a Gemma 2 text model, so it is not this family.
    "google/paligemma-3b-pt-224", "google/paligemma-3b-pt-448", "google/paligemma-3b-pt-896",
    "google/paligemma-3b-mix-224", "google/paligemma-3b-mix-448",
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "paligemma": {
        "title": "PaliGemma",
        "pinned": "hf-tiny-v2/tiny-random-PaliGemmaForConditionalGeneration",
        "projector": "multi_modal_projector: one linear with a bias, from the vision encoder's width to the text "
                     "model's, over vision.tower_output",
        "projector_input": "`vision.tower_output`",
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.

## The image and the prompt attend in both directions

The processor marks the image and the prompt as a prefix, and every prefix position attends to every
other, later ones included. Attention patterns over the prompt are not causal, and a patch at one
prompt position reaches earlier ones. A text-only trace, with the tokenizer's encoding, is causal.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 153}
QUIRKS = ["scaled-embeddings", "gain-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` is formatted with the sizes and the config keys.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by √hidden_size inside the embedding module, so token_embeddings equals layers[0].input on text.",
}

NOTES = ""
