"""Kimi K2: DeepSeek-V3's block under Moonshot's model type, a dense first block and mixtures after."""

MODEL_TYPE = "kimi_k2"
TITLE = "Kimi K2"
SUBTITLE = (
    "DeepSeek-V3's architecture: latent attention, whose queries and keys are wider than its values, "
    "and a mixture of 384 experts with a shared one on every block but the first."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "moonshotai/Kimi-K2-Instruct"
#: The tiny checkpoint the test suite builds the page from (a Kimi K2.5 wrapper; the suite rewrites its text config).
PINNED = "hf-tiny-v2/tiny-random-Kimi_K25ForConditionalGeneration"
CHECKPOINTS = [
    "moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.7-Code",
]

#: Kimi lineage: set here; kimi_linear sits at 342.
PALETTE = {"hue": 330}
VLLM = False
QUIRKS = ["latent-attention", "mixture-of-experts", "dense-first-blocks"]

#: Every number on this page comes from the reference's config or a run on the pinned tiny checkpoint;
#: none has been checked on real weights, which are not cached (about a trillion parameters). The
#: loading snippet in the notes runs the module path of tests/families/test_kimi_k2.py
#: (test_a_text_only_checkpoint_loads_from_its_module) on a model built from the pinned config, not from_pretrained.


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

STRIP = {
    "head": "lm_head has its own weight (tie_word_embeddings is false).",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))          # multi-head latent attention
out = h + mlp(post_attention_layernorm(h))       # a dense MLP on block 0, a mixture of experts after
```

DeepSeek-V3's pre-norm block, Llama's names. `first_k_dense_replace` is 1: block 0's `mlp` is
a dense MLP and the other 60 blocks' a mixture. The block adds each sublayer's output to the
stream and returns a tensor, so the identity is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Loading a text-only K2 checkpoint

`Kimi-K2-Instruct`, `-Instruct-0905`, `-Base` and `-Thinking` say `kimi_k2` at the top of their
config, which `AutoConfig` maps only through the checkpoints' remote code, and the remote code
builds Moonshot's own classes, which no family covers. transformers' `DeepseekV3ForCausalLM`
reads the same config; load it and hand the module over, with the checkpoint's tokenizer:

```python
from transformers import DeepseekV3ForCausalLM

module = DeepseekV3ForCausalLM.from_pretrained("moonshotai/Kimi-K2-Instruct", attn_implementation="eager")
model = StandardizedTransformer(module, tokenizer=tokenizer)
```

The family is found from the config's `model_type`, `kimi_k2`, and is DeepSeek-V3's
(`nnterp/families/deepseek_v3.py`) under that name. K2.5, K2.6 and K2.7-Code wrap the same
text model in `Kimi_K25ForConditionalGeneration`, under `model.language_model`; their
`text_config` says `kimi_k2`, and the standard names are the same.

## Latent attention: queries and keys are wider than values

Queries and keys are `qk_head_dim` wide (`qk_nope_head_dim + qk_rope_head_dim`, 128 + 64),
values `head_dim` (`v_head_dim`, 128); the root publishes both. Rotary turns only the last
`qk_rope_head_dim` dimensions of each query and key, and that part of the key comes from one
projection per token, shared by every head. Keys and values are projected for every head, so
`attention_keys` has `num_heads` (64) heads, as many as the queries. The attention interior
needs `attn_implementation="eager"`.

## The mixture of experts

384 routed experts, 8 per token, and one shared expert every token runs through.
`router_logits` are the logits before the sigmoid; the router picks 8 experts by the sigmoid plus
a selection bias, in one expert group (`n_group` is 1), and `expert_weights` are the chosen
sigmoids without the bias, renormalized and multiplied by `routed_scaling_factor` (2.827), so they
sum to 2.827 at every token. The mixture's output is `routed_output + shared_expert_output`:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    weights = moe.expert_weights.save()                     # [batch, seq, top_k]: 8 on K2
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

torch.testing.assert_close(routed + shared, out)
```

Block 0's MLP is dense: `support()` reports every mixture value missing there.

## The readout and the embeddings

`logits` is `lm_head.output`: no softcap and no scale. `lm_head` has its own weight. The
embedding is not scaled, so `token_embeddings` equals `layers[0].input`.
"""
