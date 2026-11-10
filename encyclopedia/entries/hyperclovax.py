"""HyperCLOVA X: a sandwich block (post_norm1, post_norm2) with Granite's four config multipliers around it."""

MODEL_TYPE = "hyperclovax"
TITLE = "HyperCLOVA X"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HyperCLOVAXForCausalLM"
#: Think-32B is a HyperCLOVA X Vision V2 wrapper (model_type hyperclovax_vision_v2, a key of WRAPPERS), which
#: transformers builds from 5.18 on; on an earlier release its config does not read and the page greys it out.
CHECKPOINTS = ["naver-hyperclovax/HyperCLOVAX-SEED-Think-14B", "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"]

#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. Its vision encoder is the
#: Qwen ViT (encyclopedia/vision/qwen_vit.py). Shapes, orders and identities were checked on the pinned tiny wrapper
#: under transformers 5.19 (float32, eager, its processor set to the model as tests/families/test_hyperclovax.py
#: does); the sizes are Think-32B's config.
WRAPPERS = {
    "hyperclovax_vision_v2": {
        "title": "HyperCLOVA X Vision V2",
        "pinned": "hf-tiny-v2/tiny-random-HyperCLOVAXVisionV2ForConditionalGeneration",
        "projector": "a linear (model.projector, with a bias) from the merger's out_hidden_size to the text model's width, "
                     "after the vision encoder; the merger inside the encoder folds each 2 × 2 block of patches first",
        "projector_input": "the merger's output, back in the image tokens' order",
        "notes": """
## The projector reads the merger's output in token order

`vision.merger.output` is in window order and a write to `vision.tower_output` does not reach the
text model, so edit the image at `model.projector.input` or `vision.image_features`.
""",
    },
}

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 273}
QUIRKS = [
    # use_post_norm is true on Think-14B and false on Think-32B, whose post_norm1 and post_norm2 are nn.Identity
    {"slug": "sandwich-norms", "when": lambda config: config.use_post_norm},
    # every multiplier is 1.0 on Think-32B; residual_multiplier is 1.0 on Think-14B too
    {"slug": "scaled-residual-adds", "when": lambda config: config.residual_multiplier != 1.0},
    {"slug": "embedding-multiplier", "when": lambda config: config.embedding_multiplier != 1.0},
    {"slug": "scaled-logits", "when": lambda config: config.logits_scaling != 1.0},
]

#: No checkpoint of this family is small enough to run here: the identities, read orders and the write
#: round trip were checked on the pinned tiny checkpoint and on the suite's copy of it with
#: residual_multiplier 0.22, logits_scaling 4 and embedding_multiplier 3, in float32 and bfloat16; the
#: numbers for Think-14B come from its config.

_ATTENTION = {
    "host": "self_attn",
    "kind": "attention",
    "label": "Attention",
    "pre_norm": "input_layernorm",
    "contribution": "attention_output",
    "interior": [
        "attention_queries", "attention_keys", "attention_values",
        "attention_scores", "attention_probabilities", "attention_head_outputs",
    ],
    "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
}
_MLP = {
    "host": "mlp",
    "kind": "mlp",
    "label": "MLP",
    "pre_norm": "post_attention_layernorm",
    "contribution": "mlp_output",
    "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
    "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
}

#: With use_post_norm (Think-14B) each sublayer has a norm after it; without it (Think-32B) post_norm1 and
#: post_norm2 are nn.Identity and the block is Llama's.
BLOCK = [
    (lambda config: config.use_post_norm, {
        "topology": "sequential",
        "sublayers": [
            {**_ATTENTION, "post_norm": "post_norm1"},
            {**_MLP, "post_norm": "post_norm2"},
        ],
    }),
    (lambda config: True, {
        "topology": "sequential",
        "sublayers": [_ATTENTION, _MLP],
    }),
]

STRIP = {
    "embed": "The model multiplies it by embedding_multiplier before block 0, so layers[0].input is "
             "token_embeddings · embedding_multiplier.",
    "logits": "logits = lm_head.output · logits_scaling, and project_on_vocab multiplies too.",
}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_norm1(self_attn(input_layernorm(x)))
out = h + post_norm2(mlp(post_attention_layernorm(h)))
```

On Think-14B `attention_output` and `mlp_output` are the outputs of `post_norm1` and `post_norm2`, the
tensors the block adds. RMSNorm is scale-invariant, so halving `mlp.output` does nothing to the stream.
Scale, steer and decompose through `attention_output` and `mlp_output`. On Think-32B the post-norms
are identities and the block is Llama's.

## The logits are multiplied by `logits_scaling`

`model.logits` is `lm_head.output` times `logits_scaling` (0.125 on Think-14B), and
`project_on_vocab` multiplies too. A softmax over `lm_head.output`, or a lens written by hand as
`lm_head(norm(h))`, runs at an eighth of the model's temperature.
"""
