"""Gemma 2: the first entry, and the reference for writing the others; PaliGemma 2 wraps it."""

import os
import tempfile

MODEL_TYPE = "gemma2"
TITLE = "Gemma 2 / PaliGemma 2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-2-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Gemma2ForCausalLM"
CHECKPOINTS = [
    "google/gemma-2-2b", "google/gemma-2-2b-it",
    "google/gemma-2-9b", "google/gemma-2-9b-it",
    "google/gemma-2-27b", "google/gemma-2-27b-it",
    # Vision-language wrappers around a Gemma 2 text model (text_config.model_type gemma2): model_type paligemma.
    "google/paligemma2-3b-pt-224", "google/paligemma2-3b-pt-448", "google/paligemma2-3b-pt-896",
    "google/paligemma2-3b-mix-224", "google/paligemma2-3b-mix-448", "google/paligemma2-3b-ft-docci-448",
    "google/paligemma2-10b-pt-224", "google/paligemma2-10b-pt-448", "google/paligemma2-10b-pt-896",
    "google/paligemma2-10b-mix-224", "google/paligemma2-10b-mix-448", "google/paligemma2-10b-ft-docci-448",
    "google/paligemma2-28b-pt-224", "google/paligemma2-28b-pt-448", "google/paligemma2-28b-pt-896",
    "google/paligemma2-28b-mix-224", "google/paligemma2-28b-mix-448",
]


def _tiny_wrapper() -> str:
    """The tiny PaliGemma 2 ``tests/families/test_gemma2.py`` builds (``_paligemma2_checkpoint``): no tiny PaliGemma 2
    is published, so the suite writes one into the temp dir, named after the snapshot of the tiny text checkpoint it
    wraps. The same path, so the wrapper's ``pinned`` is the suite's ``REPO``."""
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(PINNED, "config.json")  # the configured cache's snapshot, as the suite resolves it
    snapshot = os.path.basename(os.path.dirname(config)) if isinstance(config, str) else "missing"
    return os.path.join(tempfile.gettempdir(), f"nnterp-paligemma2-{snapshot}")


#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "paligemma": {
        "title": "PaliGemma 2",
        "pinned": _tiny_wrapper(),
        "projector": "multi_modal_projector: one linear with a bias, from the vision encoder's width to the text "
                     "model's, over vision.tower_output",
        "projector_input": "`vision.tower_output`",
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.

## The image and the prompt attend in both directions

The processor marks the image and the prompt as a prefix, and every prefix position attends to every
other, later ones included, on sliding and full blocks alike. Attention patterns over the prompt are
not causal, and a patch at one prompt position reaches earlier ones. A text-only trace, with the
tokenizer's encoding, is causal.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 156, "paper": "#EFE8CE"}
QUIRKS = ["sandwich-norms", "softcapped-logits", "sliding-window", "scaled-embeddings", "gain-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads, head_dim {head_dim}; "
                      "scores softcapped at {attn_logit_softcapping} under eager",
            "variants": {
                "sliding_attention": "sliding window of {sliding_window} tokens",
                "full_attention": "full causal attention",
            },
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_feedforward_layernorm",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, gelu_pytorch_tanh",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by √hidden_size inside the embedding module, so token_embeddings equals layers[0].input on text.",
    "logits": "logits = 30 · tanh(lm_head.output / 30), which project_on_vocab applies too; PaliGemma 2's wrapper "
              "applies no cap.",
}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. RMSNorm is scale-invariant, so halving `mlp.output` does nothing to the stream. Scale,
steer and decompose through `attention_output` and `mlp_output`.

## The logits are softcapped

`model.logits` is `30 · tanh(lm_head.output / 30)` and `project_on_vocab` applies the cap, so a lens
at the last block equals `logits`. Take KL divergences on `logits`, not on `lm_head.output`.
"""
