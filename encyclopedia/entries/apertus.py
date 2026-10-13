"""Apertus: Llama's block under other norm names, per-head query/key norms and a gateless MLP with a learned xIELU."""

MODEL_TYPE = "apertus"
TITLE = "Apertus"
SUBTITLE = (
    "Llama's block with its norms named attention_layernorm and feedforward_layernorm, every query and key head "
    "RMS-normed before the rotary embedding, and an MLP with no gate whose activation, xIELU, has two parameters "
    "learned per block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "swiss-ai/Apertus-8B-2509"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-ApertusForCausalLM"
CHECKPOINTS = [
    "swiss-ai/Apertus-8B-2509", "swiss-ai/Apertus-8B-Instruct-2509",
    "swiss-ai/Apertus-70B-2509", "swiss-ai/Apertus-70B-Instruct-2509",
    "swiss-ai/Apertus-v1.1-0.5B", "swiss-ai/Apertus-v1.1-0.5B-Instruct",
    "swiss-ai/Apertus-v1.1-1.5B", "swiss-ai/Apertus-v1.1-1.5B-Instruct",
    "swiss-ai/Apertus-v1.1-4B", "swiss-ai/Apertus-v1.1-4B-Instruct",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 238}
VLLM = False
QUIRKS = ["qk-norm", "learned-activation"]

#: Real values in the notes were measured on Apertus-v1.1-0.5B (the family's smallest checkpoint) on a GPU in
#: float32; shapes, identities and read orders also ran on the pinned tiny checkpoint, whose hidden_act is gelu.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "attention_layernorm",
            "pre_norm_note": "nnterp also answers to input_layernorm here: the two names are one module.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "feedforward_layernorm",
            "pre_norm_note": "nnterp also answers to post_attention_layernorm here: the two names are one module, "
                             "the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends <s> (id 1).",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "Untied on 8B, 70B, 1.5B and 4B; tied to embed_tokens on v1.1-0.5B and its Instruct (tie_word_embeddings). "
            "logits is lm_head.output, with no cap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(attention_layernorm(x))
out = h + mlp(feedforward_layernorm(h))
mlp(z) = down_proj(xielu(up_proj(z)))
```

Llama's two pre-norms under their own names: `attention_layernorm` feeds the attention and
`feedforward_layernorm` the MLP. nnterp aliases them `input_layernorm` and
`post_attention_layernorm`, so either name reads the same module. The config carries
`post_norm: false` and `qk_norm: true`; the modeling code reads neither and builds this block on
every checkpoint.

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

## The MLP has no gate, and its activation is learned per block

`up_proj` widens the normed stream to `intermediate_size` (21504 on 8B), `act_fn` applies xIELU,
and `down_proj` narrows it back. xIELU is `alpha_p·x² + beta·x` for `x > 0` and
`alpha_n·(expm1(min(x, eps)) − x) + beta·x` otherwise, with `beta` 0.5 and two learned scalars per
block, `mlp.act_fn.alpha_p` and `alpha_n` (stored before a softplus). On v1.1-0.5B the effective
`alpha_p` runs from 17.75 at block 1 down to 0.54 at block 12, and `alpha_n` lies between 0.85 and 6.28, so
one pre-activation value means a different activation in each block. `act_fn` is a module, so
`mlp.act_fn.input` is the pre-activation and `mlp.act_fn.output` the activation:

```python
mlp = model.layers[1].mlp
with model.trace(prompt):
    pre = mlp.act_fn.input.save()        # equals mlp.up_proj.output
    acts = mlp.act_fn.output.save()      # equals mlp.down_proj.input
```

## Negative pre-activations are not switched off

The negative branch dips below zero and then grows again: as `x` falls, the output grows like
`(alpha_n − beta)·|x|`. At block 8 of v1.1-0.5B its minimum is `−0.14`, near `x = −0.63`; it gives `−0.107` at `−1`, `1.78` at
`−5` and `10.3` at `−20`. There 93% of the pre-activations are negative, and under 0.4% of the
activations lie within `1e-3` of zero. A neuron is not sparse: a large negative pre-activation
writes its `down_proj` column with a positive weight, as a large positive one does. Read
`act_fn.output`, not the sign of `act_fn.input`, to tell whether a neuron fires.

## Queries and keys are normed per head, before the rotary

```
q = rope(q_norm(q_proj(x).view(..., heads, head_dim)))     # one gain, every head
k = rope(k_norm(k_proj(x).view(..., kv_heads, head_dim)))
```

`q_norm` and `k_norm` are RMSNorms over one head's `head_dim`, one gain vector for every head,
applied before the rotary. `attention_queries` and `attention_keys` are read after both; at position 0,
where the rotation is the identity, they equal `q_norm.output` and `k_norm.output`. On v1.1-0.5B
the gains reach 81.5 (`q_norm`) and 87.0 (`k_norm`) at block 0. An edit to one head's slice of
`q_proj.output` is renormalized away: doubling head 3's slice at block 8 of v1.1-0.5B moves the
logits by `8e-5`, while doubling head 3 of `attention_queries` moves them by 1.26 and tripling a
key/value head's slice of `v_proj.output` (no norm) by 23.8.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    attn.attention_queries[:, 1] *= 2     # head 1's scores x2
```

## Grouped-query attention

8B has 32 query heads over 8 key/value heads, 70B 64 over 8, both 128 wide. The v1.1 sizes: 16
over 4 at 64 wide (0.5B), 32 over 8 at 64 wide (1.5B), 24 over 8 at 128 wide (4B).
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, kv_heads, seq,
head_dim]`, and an edit to key/value head `j` reaches query heads `4j` to `4j + 3` on 8B.

## The default load runs the same attention as eager

The six attention interior values need `attn_implementation="eager"`. There is no softcap, window
or sink, and on v1.1-0.5B in float32 the default `sdpa` load's logits differ from the eager
load's by at most `5e-5`. The 2509 checkpoints use `rope_type` `llama3` (`rope_theta` 12,000,000,
`factor` 8 over 8192 original positions, 65536 in all); the v1.1 checkpoints the plain rotary
(`rope_theta` 500,000, 4096 positions).

## Position 0 is `<s>`, and it is the sink

The tokenizer prepends `<s>` (id 1). On v1.1-0.5B its `layer_output` at block 8 has a norm of
19,000 against 140 to 240 at the other positions of a test prompt. A mean over positions is
dominated by it unless `[:, 1:]` is dropped first.

The instruct templates write `<s>` themselves, so a templated string traced as is starts with two;
pass `add_special_tokens=False`. The 2509 instruct template also writes a system turn with
`Current date:` set to the day it runs, so the same messages tokenize differently from one day to
the next:

```python
text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
with model.trace(text, add_special_tokens=False):
    probs = model.next_token_probs.save()
```

## The readout

`model.logits` is `model.lm_head.output`: no cap and no scale, and `project_on_vocab` is
`lm_head(norm(hidden))`. `lm_head` and `embed_tokens` are one matrix on v1.1-0.5B and its Instruct, and separate on
every other checkpoint listed here. Apertus v1.5 (`apertus1p5`) is another architecture, a
multimodal wrapper, and is not this family.
"""
