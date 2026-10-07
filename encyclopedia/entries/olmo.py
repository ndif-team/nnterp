"""OLMo 1: Llama's pre-norm block with LayerNorms that have no weight and no bias, and a clamp on queries, keys and values on some checkpoints."""

MODEL_TYPE = "olmo"
TITLE = "OLMo / OLMo 0424 / OLMo 0724"
SUBTITLE = (
    "Llama's pre-norm block with LayerNorm in place of RMSNorm, and every norm without weight or bias, so a "
    "sublayer reads the stream standardized to mean 0 and variance 1; the 0424 and 0724 base checkpoints "
    "clamp queries, keys and values to ±8 inside the attention."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/OLMo-1B-hf"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "katuni4ka/tiny-random-olmo-hf"
CHECKPOINTS = [
    "allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf", "allenai/OLMo-7B-Twin-2T-hf",
    "allenai/OLMo-7B-SFT-hf", "allenai/OLMo-7B-Instruct-hf",
    "allenai/OLMo-7B-0424-hf", "allenai/OLMo-7B-0424-SFT-hf",
    "allenai/OLMo-1B-0724-hf", "allenai/OLMo-7B-0724-hf",
    "allenai/OLMo-7B-0724-SFT-hf", "allenai/OLMo-7B-0724-Instruct-hf",
]

#: OLMo lineage (olmo2 58, olmoe 70, olmo_hybrid 64); 55 sits between jetmoe's 52 and olmo2's 58.
PALETTE = {"hue": 55}
VLLM = False
QUIRKS = ["layernorm", "weightless-norms"]

#: Shapes, identities and read places are from the pinned tiny checkpoint (float32, CPU); the clip_qkv
#: trap also from the tiny loaded with clip_qkv=0.05. Real numbers are from allenai/OLMo-1B-0724-hf in
#: float32 on CPU, over four prompts of prose and Python; sizes and clip_qkv values from the configs.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "A LayerNorm with no weight and no bias: the attention reads the stream standardized "
                             "to mean 0 and variance 1 per position.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, no bias",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add; a LayerNorm "
                             "with no weight and no bias.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "norm": "A LayerNorm with no weight and no bias, eps 1e-5, computed in float32: no gain to fold into the "
            "unembedding. project_on_vocab applies it.",
    "head": "Tied to embed_tokens on OLMo-1B-hf (tie_word_embeddings); its own weight on every other checkpoint. "
            "Nothing follows it: logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's block and Llama's names, with LayerNorm where Llama has RMSNorm. `post_attention_layernorm`
is named for its place on the stream, after the attention's add: it is the MLP's input norm. Nothing
norms a sublayer's output.

## The norms have no weight and no bias

`OlmoLayerNorm` is `F.layer_norm` over the hidden axis in float32, `eps` 1e-5, with no weight and no
bias, cast back to the input dtype. The module holds no parameters: `list(model.norm._module.parameters())`
is empty, and so is every block's `input_layernorm` and `post_attention_layernorm`. What a sublayer
reads is the stream standardized per position, mean 0 and variance 1:

```python
import torch.nn.functional as F

with model.trace(prompt):
    x = model.layers[1].input.save()
    attn_in = model.layers[1].self_attn.input.save()

torch.testing.assert_close(attn_in, F.layer_norm(x, x.shape[-1:], eps=1e-5))   # float32
```

A shift along the all-ones direction never reaches a sublayer or the logits. On 1B-0724 in float32,
adding 10 to every coordinate of `layers[8].input`, 27 times the stream's RMS there, moves the logits
by at most 1.3e-4. The final norm has no gain either, so `project_on_vocab(h)` is
`lm_head(layer_norm(h))`, and a direction's logit effect is read off `lm_head` alone.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream with
nothing in between. The identity is the plain sum, exact in float32:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

Scaling a module's output scales its contribution. The inputs are the norms' outputs:
`self_attn.input` is `input_layernorm.output` and `mlp.input` is `post_attention_layernorm.output`.

## clip_qkv clamps queries, keys and values on 0424 and 0724

`clip_qkv` is 8.0 on OLMo-7B-0424-hf, OLMo-1B-0724-hf and OLMo-7B-0724-hf, and `null` on the other
checkpoints, the 0424 and 0724 SFT and Instruct ones included. Where it is set, the attention clamps
the outputs of `q_proj`, `k_proj` and `v_proj` to `[-8, 8]`, before the heads are split and before
the rotary embedding. `attention_queries`, `attention_keys` and `attention_values` are read after the
clamp; the rotary turns pairs of query and key dimensions, so those two can reach `8·√2`.

On 1B-0724 every block clamps some query or key entries: at most 0.11% of query entries and 0.32% of
key entries per block, with raw values up to 13.6. The values reach the bound on block 1 only.

The clamp is in place on the projection's output tensor. A saved `q_proj.output` therefore holds the
clamped values once the trace ends, and an edit written to `q_proj.output` is clamped after it:
clone to keep the raw projection, and edit at `attention_queries`, which no clamp follows.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    raw = attn.q_proj.output.clone().save()     # before the clamp
    clamped = attn.q_proj.output.save()         # the same tensor, clamped in place
```

Every checkpoint has as many key/value heads as query heads, so there is no grouping.

## Load with eager for the attention interior only

The attention interior needs `attn_implementation="eager"`. The default `sdpa` load computes the same
function: on 1B-0724 in float32 the two loads' logits differ by at most 1.5e-5, with the same top
token at every position. There is no softcap, sink or window; the rotary is the plain one, `rope_theta`
10000.

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. OLMo-1B-hf ties `lm_head` to `embed_tokens` (one parameter), so an
edit to `embed_tokens.weight` is an edit to the unembedding there; every other checkpoint has its own
`lm_head`. The tokenizer is GPT-NeoX's: it prepends nothing, so `model.input_ids` is the prompt's
tokens alone, and `<|endoftext|>` (id 50279) ends a text. The Instruct chat templates start with
`eos_token`.

## Intermediate checkpoints

The base repositories keep their pretraining checkpoints as Hub branches: 351 on OLMo-1B-hf, from
`step1000-tokens4B` to `step738020-tokens3094B`; 1,446 on OLMo-1B-0724-hf, from `step0-tokens0B` to
`step1454000-tokens3048B`; about 820 each on OLMo-7B-0424-hf and OLMo-7B-0724-hf. Each loads with
`revision=`:

```python
model = StandardizedTransformer("allenai/OLMo-1B-hf", revision="step10000-tokens41B")
```
"""
