"""Kimi K2: DeepSeek-V3's block under Moonshot's model type, a dense first block and mixtures after."""

MODEL_TYPE = "kimi_k2"
TITLE = "Kimi K2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "moonshotai/Kimi-K2-Instruct"
#: The tiny checkpoint the test suite builds the page from (a Kimi K2.5 wrapper; the suite rewrites its text config).
PINNED = "hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration"
CHECKPOINTS = [
    "moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.7-Code",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 337}
QUIRKS = ["latent-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]


def load(checkpoint, **kwargs):
    """The page's model: transformers' ``DeepseekV3ForCausalLM`` built on the meta device from the
    checkpoint's config (a ``kimi_k25`` wrapper's ``text_config``) and handed over as a module, the
    way a text-only K2 checkpoint loads. ``kimi_k2`` at the top of a config is a type ``AutoConfig``
    maps only through the checkpoint's remote code. The page reads no tokens, so the pinned
    checkpoint's tokenizer stands in for Moonshot's."""
    import json
    import os

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM

    from nnterp import StandardizedTransformer

    path = os.path.join(checkpoint, "config.json") if os.path.isdir(checkpoint) else hf_hub_download(checkpoint, "config.json")
    raw = json.load(open(path))
    raw = raw.get("text_config", raw)
    config = DeepseekV3Config(**{k: v for k, v in raw.items() if k not in ("model_type", "auto_map", "quantization_config")})
    config.model_type = "kimi_k2"
    if "attn_implementation" in kwargs:
        config._attn_implementation = kwargs.pop("attn_implementation")
    with torch.device("meta"):
        module = DeepseekV3ForCausalLM(config)
    return StandardizedTransformer(module, tokenizer=AutoTokenizer.from_pretrained(PINNED), device="meta", **kwargs)


#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on block 0, a mixture after).
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
            "detail": "latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Load the text-only checkpoints through `DeepseekV3ForCausalLM`

`AutoConfig` maps `kimi_k2` only through the checkpoints' remote code, which builds its own copy of
DeepSeek-V3's classes, and no nnterp family covers those. Load `DeepseekV3ForCausalLM.from_pretrained(repo)`
and pass the module to `StandardizedTransformer` with the checkpoint's tokenizer, which needs
`trust_remote_code=True`.

## The expert weights sum to `routed_scaling_factor`

The router picks experts by the sigmoid of `router_logits` plus a selection bias, so the top
`router_logits` are not always the chosen experts. `expert_weights` are the chosen sigmoids without the
bias, renormalized and multiplied by `routed_scaling_factor` (2.827), so they sum to it at every token.
Every token also runs through a shared expert, and `mlp_output` is
`routed_output + shared_expert_output`.
"""
