"""Llama: the block the standard names are taken from."""

MODEL_TYPE = "llama"
TITLE = "Llama"
SUBTITLE = (
    "The reference block: one RMSNorm before each sublayer and none after, so what reaches the "
    "residual stream is the attention's and the MLP's own output, with rotary positions applied to "
    "queries and keys inside the attention."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "meta-llama/Llama-3.1-8B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CHECKPOINTS = [
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    # Vision-language wrappers around a Llama text model, each a key of WRAPPERS by its config's model_type.
    "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf",
    "llava-hf/vip-llava-7b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "deepseek-community/deepseek-vl-1.3b-chat",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "HuggingFaceTB/SmolVLM-Instruct",  # model_type idefics3
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",  # model_type smolvlm
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "llava": {
        "title": "Llava 1.5",
        "pinned": "trl-internal-testing/tiny-LlavaForConditionalGeneration",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2) over block -2's patches, the CLS token dropped",
        "projector_input": "`vision.layers[-2].layer_output[:, 1:]`",
        "notes": """
## Writes to `vision.tower_output` do not reach the text model

The projector reads block -2's patches without the CLS token, `vision.layers[-2].layer_output[:, 1:]`,
so the last block and `vision.tower_output` are computed and discarded. Ablate or patch the image at
block -2 or at `vision.image_features`.
""",
    },
    "vipllava": {
        "title": "VipLlava",
        "pinned": "hf-tiny-v2/tiny-random-VipLlavaForConditionalGeneration",
        "projector": "a LayerNorm and a two-layer MLP over five blocks' patches concatenated, the CLS token dropped from each",
        "projector_input": "five blocks' `layer_output[:, 1:]`, concatenated",
        "notes": """
## The projector reads five blocks' streams

On `vip-llava-7b-hf` it concatenates the patches of blocks 22, 19, 16, 13 and 5, each without its
CLS token. The config's `vision_feature_layers` counts hidden states, so its `k > 0` is block `k - 1`.
The last block and `vision.tower_output` are computed and discarded.
""",
    },
    "llava_next": {
        "title": "LLaVA-NeXT",
        "pinned": "hf-tiny-v2/tiny-random-LlavaNextForConditionalGeneration",
        "projector": "a two-layer MLP over each crop's block -2 patches, the CLS token dropped; the wrapper unpads its output and adds newline tokens",
        "projector_input": "each crop's `vision.layers[-2].layer_output[:, 1:]`",
        "quirks": ["tiled-images", "unpadded-features"],
        "notes": """
## Writes to `vision.tower_output` do not reach the text model

As on Llava 1.5, the projector reads each crop's block -2 patches without the CLS token, so the last
block and `vision.tower_output` are discarded. The wrapper then unpads the projector's output and
adds newline tokens, so edit `vision.image_features` for what the text model receives.
""",
    },
    "deepseek_vl": {
        "title": "DeepSeek-VL",
        "pinned": "hf-tiny-v2/tiny-random-DeepseekVLForConditionalGeneration",
        "projector": "aligner, a two-layer MLP (linear1, GELU, linear2) over vision.tower_output",
        "projector_input": "`vision.tower_output`",
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.
""",
    },
    "idefics3": {
        "title": "Idefics 3",
        "pinned": "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",
        "projector": "connector: a pixel shuffle folding each scale_factor × scale_factor block of patches into one token, then modality_projection, a linear",
        "projector_input": "`vision.tower_output`",
        "quirks": ["tiled-images", "pooled-projector"],
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.
""",
    },
    "smolvlm": {
        "title": "SmolVLM",
        "pinned": "trl-internal-testing/tiny-SmolVLMForConditionalGeneration",
        "projector": "connector: a pixel shuffle folding each scale_factor × scale_factor block of patches into one token, then modality_projection, a linear",
        "projector_input": "`vision.tower_output`",
        "quirks": ["tiled-images", "pooled-projector"],
        "notes": """
## The projector reads `vision.tower_output`

A write to `tower_output`, or to the last block through `vision.norm`, reaches the text model.
""",
    },
}

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 241}
QUIRKS: list[str] = []

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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## circuit-tracer's transcoders read `layer_mid`

The Llama-3.2-1B transcoders (`mntss/transcoder-Llama-3.2-1B`, `mntss/clt-llama-3.2-1b-524k`) read
`hook_resid_mid`, which is `layers[i].layer_mid`, the stream before the MLP's norm, and write
`mlp_output`.
"""
