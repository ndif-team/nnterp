"""Nemotron: the Nemotron-4 block as Minitron and Nemotron-Mini ship it (NemotronForCausalLM)."""

MODEL_TYPE = "nemotron"
TITLE = "Nemotron-4 / Minitron"
SUBTITLE = (
    "Llama's tree with Nemotron-4's block: the norms are LayerNorms with a bias whose gain is stored minus one, "
    "the MLP has no gate and squares a ReLU, and rotary positions turn half of each query and key head."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "nvidia/Nemotron-Mini-4B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-NemotronForCausalLM"
CHECKPOINTS = [
    "nvidia/Minitron-4B-Base", "nvidia/Nemotron-Mini-4B-Instruct",
    "nvidia/Minitron-8B-Base",
    "nvidia/Nemotron-4-Mini-Hindi-4B-Base", "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct",
]

VLLM = False
QUIRKS = ["layernorm", "gain-norm", "partial-rotary", "squared-relu"]

#: What the visualization draws: Llama's sequential block, a LayerNorm before each sublayer
#: and none after; the MLP is two projections around a squared ReLU.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "NemotronLayerNorm1P: a LayerNorm with a bias whose gain is 1 + weight, not the stored weight.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads; "
                      "rotary on {partial_rotary_factor} of each {head_dim}-wide head",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm, "
                             "a NemotronLayerNorm1P like input_layernorm.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding is added, so token_embeddings equals "
             "layers[0].input. Positions enter as a rotation of half of each query and key head.",
    "norm": "A LayerNorm with a bias whose gain is 1 + norm.weight (a mean of 4.18 on Mini-4B, where the stored "
            "weight averages 3.18). project_on_vocab applies it, bias included.",
    "head": "Its own weight, not tied to embed_tokens (tie_word_embeddings is false), and no bias. Nothing follows "
            "it: logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's order and Llama's names: one norm before each sublayer, none after, and
`post_attention_layernorm` is the MLP's input norm. The norms are `NemotronLayerNorm1P`, the
attention and the MLP have no biases, and the MLP is `down_proj(relu2(up_proj(x)))`, two
projections and no `gate_proj`.

## The norms are LayerNorms whose gain is 1 + weight

`NemotronLayerNorm1P` subtracts the mean, divides by the standard deviation, multiplies by
`1 + weight` and adds `bias`. The stored weight is the gain minus one, and on Mini-4B it is far
from zero: block 0's `input_layernorm.weight` averages `-0.85` (a gain of `0.15`, some entries
negative) and the final norm's `3.18` (a gain of `4.18`, up to `13.9`). Computing a norm, or
folding one into the next projection, with the stored weight gives a different tensor; use
`1 + weight` and keep the bias:

```python
import torch.nn.functional as F

norm = model.layers[5].input_layernorm
with model.trace(prompt):
    x = model.layers[5].input.save()
    y = norm.output.save()

gain = 1 + norm.weight                    # not norm.weight
ref = F.layer_norm(x, x.shape[-1:], gain, norm.bias, model.config.norm_eps)
torch.testing.assert_close(ref, y)
```

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the
stream with nothing in between. The identity holds exactly on Mini-4B, in float32 and in
bfloat16:

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

Scaling `mlp.output` and scaling `mlp_output` are the same edit and give the same logits.
`self_attn.input` is `input_layernorm`'s output and `mlp.input` is `post_attention_layernorm`'s.

## The MLP squares a ReLU, and most neurons are exactly zero

The MLP has one up projection, `hidden_size → intermediate_size` (`3072 → 9216` on Mini-4B, so
`mlp.intermediate_size` is `9216`), and the activation `relu(x)²`. A neuron's activation is the
input of `down_proj`; its pre-activation is the output of `up_proj`. Across 39 Pile documents
(up to 256 tokens each, position 0 left out) 70% to 96% of the 9216 activations at a token are exactly
zero, depending on the block (fewest zeros in blocks 13 to 15, most in blocks 0, 1 and 23 to 25),
about 1250 active neurons per token averaged over blocks. The active ones are heavy-tailed: the
top 1% of neurons at a token carry 27% to 87% of the summed activation, again by block. Almost no neuron is dead: 16 of
the 294,912 never fired on that sample.

```python
mlp = model.layers[5].mlp
with model.trace(prompt):
    pre = mlp.up_proj.output.save()
    acts = mlp.down_proj.input.save()      # [batch, seq, 9216]

assert torch.equal(acts, torch.relu(pre) ** 2)
```

For neuron-level work this has three consequences. Zero-ablating a neuron that is already zero
on a token changes nothing. The derivative of `relu(x)²` is `2 · relu(x)`, so a gradient-based
attribution gives exactly zero to a neuron that is off in the run the gradient is taken on,
however large its effect when patched in. And an activation is quadratic in its pre-activation:
scaling `up_proj.output` by `c` scales an active neuron by `c²`. The pinned tiny checkpoint sets
`hidden_act` to `gelu`, so none of this shows on it.

## Rotary turns half of each head

`partial_rotary_factor` is `0.5` on every released checkpoint (and `NemotronConfig`'s default
when a config leaves it out): the first 64 of each head's 128 dimensions are rotated, as two
halves of 32 (`rotate_half`), with `rope_theta` 10000. The other 64 dimensions of every query
and key are the projection's output unchanged, so they add the same term to a score at any
distance. Neither half dominates: at block 5 of Mini-4B the scores from the rotated dimensions
and from the others have standard deviations of 2.2 and 2.4 on a test prompt.
`attention_queries` and `attention_keys` are read after the rotation; the projection is one
`source` read away:

```python
attn = model.layers[5].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()   # before the rotary
    q = attn.attention_queries.save()                 # after it

b, s, _ = q_raw.shape
q_raw = q_raw.view(b, s, model.num_heads, model.head_dim).transpose(1, 2)
rot = model.head_dim // 2
assert torch.equal(q_raw[..., rot:], q[..., rot:])    # the unrotated half
```

## Grouped-query attention, three or six query heads per key/value head

The 4B checkpoints have 24 query heads over 8 key/value heads, Minitron-8B 48 over 8.
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, 8, seq, 128]`, and
query head `h` reads key/value head `h // 3` on 4B (`h // 6` on 8B): an edit to key/value head
`j` reaches query heads `3j` to `3j + 2`.

```python
attn = model.layers[5].self_attn
with model.trace(prompt):
    attn.attention_values[:, 1] = 0             # key/value head 1
    heads = attn.attention_head_outputs.save()  # only query heads 3 to 5 change on 4B
```

The head width is 128 on every checkpoint, and on Minitron-8B it is not `hidden_size /
num_heads`: 48 heads of 128 are 6144 wide against a `hidden_size` of 4096, so `q_proj` widens
and `o_proj` narrows (its config sets `head_dim`). The query scale is `128 ** -0.5`.

## The default load runs the same attention as eager

A default load runs `sdpa` and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. There is no softcap, window or sink, so the two paths
compute the same function: on Mini-4B their logits differ by at most `1e-4` in float32, and by up
to `1.4` in bfloat16 with the same top token at every position of a test prompt. The checkpoints
are stored in bfloat16, and a load without `dtype` keeps it.

## Position 0 and one or two other tokens carry very large norms

From block 1 to block 28 of Mini-4B the residual stream at position 0 has a norm near 2150,
against a median of 3 to 290 over the other positions, and on most prompts one or two later
tokens reach 4000 to 6600 (in different prompts, the first `the`, `is`, `,` or newline). They
live mostly in dimensions 8 and 13, and heads put 0.53 to 0.90 of their attention on these positions
(blocks 2 to 25, on a 72-token sample). By block 30 none stands out. The tokenizer adds no
beginning-of-sequence token, so position 0 is the prompt's own first token. Leave these
positions out before averaging activations for steering vectors, mean ablation or probes.

```python
with model.trace(prompt):
    resid = model.layers[5].layer_output.save()

resid[0].float().norm(dim=-1)        # about 2150 at position 0 on Mini-4B
```

## The tokenizer and the chat template

The vocabulary is a 256000-token SentencePiece model with byte fallback (a newline is
`<0x0A>`). No `<s>` (id 2) is prepended. The word marker is added by the tokenizer, so `"Paris"`
and `" Paris"` both encode to `▁Paris` (`8045`).

The Instruct checkpoint's template opens every conversation with a system turn, empty or not,
and marks turns with `<extra_id_0>` (id 4) and `<extra_id_1>` (id 5); the reply ends with `</s>`
(id 3). Those two markers are not special tokens to the tokenizer, so
`skip_special_tokens=True` leaves them in decoded text.

```python
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(prompt)
# <extra_id_0>System
#
# <extra_id_1>User
# What is the capital of France?
# <extra_id_1>Assistant
```

## The readout

`logits` equals `lm_head.output`, and `project_on_vocab` applied to the last block's
`layer_output` equals `logits` exactly. The final norm's bias adds `lm_head.weight @ norm.bias`
to every lens readout; on Mini-4B that vector is small (a standard deviation of 0.23 logits,
against 2.3 for the model's own logits on a test prompt). `embed_tokens` and `lm_head` are
separate matrices.

## What this family module covers

Every checkpoint that loads as `NemotronForCausalLM`: Minitron-4B-Base and Minitron-8B-Base,
pruned and distilled from Nemotron-4 15B; Nemotron-Mini-4B-Instruct, fine-tuned from
Minitron-4B-Base; and Nemotron-4-Mini-Hindi-4B-Base, continued from Minitron-4B-Base on Hindi and
English, with its Instruct version. Nemotron-4
340B and Nemotron-3 8B are published as NeMo checkpoints with no transformers config, and do not
load here. Llama-3.1-Minitron and Mistral-NeMo-Minitron are `llama` and `mistral` checkpoints.
Nemotron-H, Nemotron Nano 2 and Nemotron 3 are the hybrid `nemotron_h` family.
"""
