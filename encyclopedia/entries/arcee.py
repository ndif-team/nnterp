"""Arcee AFM: Llama's block with a two-projection MLP around a squared ReLU."""

MODEL_TYPE = "arcee"
TITLE = "Arcee AFM"
SUBTITLE = (
    "Llama's block whose MLP has no gate: up_proj, a squared ReLU and down_proj, so a neuron is exactly zero "
    "wherever its pre-activation is negative."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "arcee-ai/AFM-4.5B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-ArceeForCausalLM"
CHECKPOINTS = [
    "arcee-ai/AFM-4.5B-Base", "arcee-ai/AFM-4.5B",
    "arcee-ai/AFM-4.5B-Preview", "arcee-ai/AFM-4.5B-Base-Pre-Anneal",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 235}
VLLM = False
QUIRKS = ["squared-relu"]

#: Every checkpoint is 4.5B parameters, above the size a real run here allows, so no real weights were run:
#: shapes, identities and the activation ran on the pinned tiny checkpoint (whose hidden_act is gelu; the
#: activation note loads it with hidden_act="relu2"), the activation class on a meta build of AFM-4.5B-Base,
#: and the rest is read off the configs and tokenizers.

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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends "
             "<|begin_of_text|> (id 128000).",
    "head": "lm_head has its own weight on every checkpoint (tie_word_embeddings is false). "
            "logits is lm_head.output, with no cap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
mlp(z) = down_proj(relu(up_proj(z)) ** 2)
```

Llama's names and Llama's two pre-norms. The MLP has two projections where Llama's has three:
there is no `gate_proj`, and `up_proj` widens the normed stream to `intermediate_size` (18432 on
4.5B, 7.2 × `hidden_size`).

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream
with nothing in between. The identity holds exactly in float32 on the pinned checkpoint:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## A neuron is `relu(x)²`, exactly zero below zero

`hidden_act` is `relu2` on every checkpoint, so `mlp.act_fn` is a `ReLUSquaredActivation` and a
neuron's activation is `relu(up_proj(z))²`. It is exactly zero wherever the pre-activation is
negative, so a neuron that is off on a token has no effect on it and receives no gradient: the
derivative of `relu(x)²` is `2 · relu(x)`. Where it is on, it grows with the square of the
pre-activation. `mlp.down_proj.input` holds the activations and `mlp.up_proj.output` the
pre-activations:

```python
mlp = model.layers[1].mlp
with model.trace(prompt):
    pre = mlp.up_proj.output.save()
    acts = mlp.down_proj.input.save()

torch.equal(acts, torch.relu(pre) ** 2)   # True
```

## Grouped-query attention, 128-wide heads

The 4.5B checkpoints have 20 query heads over 4 key/value heads, each 128 wide (`head_dim`), so
`num_heads × head_dim` equals `hidden_size` (2560). `attention_keys` and `attention_values` are
read before `repeat_kv`, `[batch, 4, seq, 128]`, and an edit to key/value head `j` reaches query
heads `5j` to `5j + 4`. No projection has a bias (`attention_bias` and `mlp_bias` are false).

## Loading and positions

The six attention interior values need `attn_implementation="eager"`. There is no softcap,
window or sink, so the default `sdpa` load computes the same function. AFM-4.5B, Base and Preview
stretch the rotary with YaRN, `factor` 20 over 4096 original positions, to 65536;
transformers warns that `max_position_embeddings / original_max_position_embeddings` is 16 and uses
the explicit 20. AFM-4.5B-Base-Pre-Anneal has the plain rotary and 4096 positions. `rope_theta`
is 10000 throughout.

## Tokenizer and templates

Every tokenizer prepends `<|begin_of_text|>` (id 128000), and `" Paris"` is one token. The
vocabularies differ: 128004 rows on Base, 128005 on AFM-4.5B, 128064 on Preview, 128256 on
Pre-Anneal. AFM-4.5B's chat template is ChatML (`<|im_start|>`, `<|im_end|>`) and writes a default
system turn naming the model when the messages have none; its `eos_token` is `<|im_end|>`, Base's
`<|end_of_text|>`.

## What loads as this family

`AFM-4.5B`, its Base, Preview and Base-Pre-Anneal are `arcee`. The `AFM-4.5B-Base-KDA-Only` and
`-KDA-NoPE` conversions are `arcee_kda`, a remote-code architecture, and Arcee's Trinity models
are the `afmoe` family.
"""
