"""Nemotron-H: the hybrid line (NemotronHForCausalLM), Nemotron-H, Nemotron Nano 2 and Nemotron 3."""

MODEL_TYPE = "nemotron_h"
TITLE = "Nemotron-H / Nemotron Nano\u00a02 / Nemotron 3"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
#: The tiny checkpoint the test suite builds the page from (the suite renames its embedding weight in a copy).
PINNED = "hf-tiny-v2/tiny-random-NemotronHForCausalLM"
CHECKPOINTS = [
    "nvidia/Nemotron-H-4B-Base-8K", "nvidia/Nemotron-H-4B-Instruct-128K",
    "nvidia/Nemotron-H-8B-Base-8K", "nvidia/Nemotron-H-8B-Reasoning-128K",
    "nvidia/Nemotron-H-47B-Base-8K", "nvidia/Nemotron-H-47B-Reasoning-128K",
    "nvidia/Nemotron-H-56B-Base-8K",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
]

def _omni_processor(repo: str):
    """The checkpoint's processor, or, where it ships no processor files (the pinned tiny wrapper), one built from its
    tokenizer as tests/families/test_nemotron_h.py builds it."""
    from transformers import AutoProcessor, AutoTokenizer, ParakeetFeatureExtractor
    from transformers.models.nemotron_h_omni import (
        image_processing_nemotron_h_omni as image, processing_nemotron_h_omni as processing,
        video_processing_nemotron_h_omni as video,
    )

    try:
        return AutoProcessor.from_pretrained(repo)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        return processing.NemotronH_Omni_Reasoning_V3Processor(
            image_processor=image.NemotronH_Omni_Reasoning_V3ImageProcessor(min_num_patches=16, max_num_patches=24),
            video_processor=video.NemotronH_Omni_Reasoning_V3VideoProcessor(), feature_extractor=ParakeetFeatureExtractor(),
            tokenizer=tokenizer, chat_template=tokenizer.chat_template,
        )


#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type (transformers 5.18 and later).
#: Its RADIO vision encoder is hosted by this family alone, so it is inline (``tower``). Shapes, orders and identities
#: were checked on the pinned tiny wrapper under transformers 5.19 (float32, eager, pure-torch kernels), with the
#: processor tests/families/test_nemotron_h.py builds (the tiny checkpoint ships none, and leaves image_token_id unset).
#: The released Omni checkpoint keeps its original config (model_type NemotronH_Nano_Omni_Reasoning_V3, remote code),
#: which transformers' AutoConfig does not read, so the page lists it greyed out with that reason.
WRAPPERS = {
    "nemotron_h_omni": {
        "title": "NemotronH Omni",
        "pinned": "hf-tiny-v2/tiny-random-NemotronH_Omni_Reasoning_V3",
        "processor": _omni_processor,
        "projector": "multi_modal_projector: each image's patch rows (the CLS and register tokens dropped) pixel-shuffled "
                     "2 × 2 into one row 4 × vision_hidden_size wide, then an RMSNorm, linear_1, squared ReLU and linear_2",
        "projector_input": "the pixel shuffle of each image's patch rows of the last block's output",
        "quirks": ["pooled-projector"],
        "tower": {
            "TITLE": "RADIO",
            "VISION_CONFIG_TYPES": ["radio"],
            "MODULE_CLASSES": ["RadioModel"],
            "BLOCK": {
                "topology": "sequential",
                "sublayers": [
                    {
                        "host": "self_attn",
                        "kind": "attention",
                        "label": "Attention",
                        "pre_norm": "input_layernorm",
                        "contribution": "attention_output",
                        # one interface call per image for several images: no pattern, so scores, probabilities and head outputs are Unavailable
                        "interior": ["attention_queries", "attention_keys", "attention_values"],
                        "detail": "{num_heads} heads × {head_dim}, per image, layer-scaled",
                    },
                    {
                        "host": "mlp",
                        "kind": "mlp",
                        "label": "MLP",
                        "pre_norm": "post_attention_layernorm",
                        "contribution": "mlp_output",
                        "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, layer-scaled",
                    },
                ],
            },
            "ROWS": ("One row, packed: every image of the invoke in one row, `[1, tokens, vision_hidden_size]`. Each image is its "
                     "`num_cls_tokens + num_registers` prefix tokens, then its patches in raster order; the processor's "
                     "`image_grid_hw` (patches per side) splits the row."),
            "MASKING": ("No causal mask. With one image the attention runs once over the row; with several, once per image "
                        "(split at `cu_seqlens`), so no token attends to another image's."),
            "POSITIONS": ("`patch_embeddings` is `patch_projection`'s output, before the prefix tokens and the position "
                          "embeddings. A learned position table, resampled to each image's grid, is added to the patches; "
                          "the prefix tokens have no position. No rotary."),
            "NORM": ("None after the last block: `vision.tower_output` is the last block's stream, prefix tokens included. "
                     "A checkpoint whose language model has multi-token-prediction layers adds `vision_final_layernorm` "
                     "before the pixel shuffle, outside the vision encoder."),
            "QUIRKS": ["packed-tower", "variable-resolution", "cls-token"],
            "NOTES": """
## The block scales what each sublayer adds

```
h   = x + layer_scale1(self_attn(input_layernorm(x)))
out = h + layer_scale2(mlp(post_attention_layernorm(h)))      # fc1, hidden_act (GELU), fc2
```

Both norms are LayerNorms (native `norm1`, `norm2`). `layer_scale1` and `layer_scale2` multiply
each channel by a learned `lambda1`, so `attention_output` and `mlp_output` are the scales'
outputs, and `vision.layers[i].input + attention_output + mlp_output == layer_output` holds.
`self_attn.output[0]` is the attention before its scale.

## Every image in one row, behind its prefix tokens

The row holds each image's `num_cls_tokens` CLS tokens and `num_registers` register tokens, then
its `h * w` patches. On the pinned tiny checkpoint (2 CLS, 1 register) an image the processor
puts on a 4 × 4 grid is `[1, 19, 32]` at every block, and `patch_embeddings` is its 16 patches,
`[1, 16, 32]`. The prefix tokens reach the text model only through the attention: the wrapper
drops them before the pixel shuffle, so a write to the last block's `layer_output[:, :3]` leaves
the logits unchanged and a write to its patch rows changes them.

## Queries, keys and values whole; no pattern

With several images the attention makes one call per image, so no single call holds the block's
pattern: `attention_scores`, `attention_probabilities` and `attention_head_outputs` are
`Unavailable`. `attention_queries`, `attention_keys` and `attention_values` are served whole,
`[1, num_heads, tokens, head_dim]`; split them at `self_attn.inputs[1]["cu_seqlens"]` (absent for one
image) to compute one image's pattern.
""",
        },
        "notes": """
## Writes to `vision.tower_output` do not reach the text model

The wrapper reads the vision encoder's `features`, the patch rows the encoder takes from its last
block before it returns, so a write to `vision.tower_output` changes nothing downstream. Edit the
image at `vision.layers[-1].layer_output`, `model.projector.input` or `vision.image_features`.
""",
    },
}

#: Set by hues.py (lineage: Nemotron).
PALETTE = {"hue": 226}
QUIRKS = ["one-sublayer-blocks", "hybrid", "mamba2", "nope-blocks", "squared-relu", "mixture-of-experts"]

#: The sublayers in forward order. Each block holds one of them, the ``mixer`` its entry in
#: ``config.layer_types`` names, so each block draws one. The reference is dense (Mamba-2, attention,
#: MLP blocks); the mixture of Nemotron 3's 30B, Super and Ultra checkpoints is described in the notes.
NORM_NOTE = "The block's one RMSNorm, layers[i].norm; the final norm is model.norm."

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD, {mamba_num_heads} heads × {mamba_head_dim}, {n_groups} groups",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, no rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {mlp_hidden_act}, no gate",
        },
        {
            # The mixture layout (Nemotron 3 30B, Super, Ultra; the pinned checkpoint): drawn where `mlp` is a mixture.
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "mlp_output",
            "interior": ["router_logits", "expert_weights", "expert_indices", "expert_outputs", "routed_output", "shared_expert_output"],
            "detail": "{num_experts} experts, top {top_k}, plus a shared expert",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Each block holds one sublayer

```
out = x + mixer(norm(x))     # mixer: a Mamba-2 mixer, attention, an MLP or a mixture
```

A block is a Mamba-2 mixer (`linear_attn`), attention (`self_attn`), or an MLP or a mixture (`mlp`),
as `config.layer_types` lists, and has no attribute under the other names. Choose blocks from
`layer_types` outside the trace. A logit lens or a patch per block is one per sublayer.

## The prompt's first token carries a very large norm

The tokenizer adds no beginning-of-sequence token, and an early Mamba-2 mixer writes a vector at the
prompt's first token many times the norm it writes elsewhere, on which the attention blocks put most of
their attention. Leave position 0 out of activation statistics and steering-vector norms.

## The expert weights sum to `routed_scaling_factor`

On the mixture checkpoints (Nemotron 3 Nano 30B-A3B, Super, Ultra, Nemotron 3.5 Lightning) the router
picks experts by their sigmoid plus a selection bias, `e_score_correction_bias`, so the top
`router_logits` are not always the chosen experts. `expert_weights` are the chosen sigmoids without
the bias, renormalized and multiplied by `routed_scaling_factor`, so they sum to it at every token.
"""
