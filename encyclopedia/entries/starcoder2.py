"""StarCoder2: BigCode's code models, which load as Starcoder2ForCausalLM."""

MODEL_TYPE = "starcoder2"
TITLE = "StarCoder2"
SUBTITLE = (
    "Llama's block with LayerNorms, a bias on every projection, a two-matrix GELU MLP (c_fc, c_proj) "
    "in place of SwiGLU, and a sliding window of 4096 positions on every block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "bigcode/starcoder2-3b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-Starcoder2ForCausalLM"
CHECKPOINTS = [
    "bigcode/starcoder2-3b",
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-15b-instruct-v0.1",
]

#: Beside codegen's 234, another code model.
PALETTE = {"hue": 238}
VLLM = False
QUIRKS = ["layernorm", "qkv-bias", "sliding-window"]

#: What the visualization draws: Llama's sequential block, one LayerNorm before each sublayer.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; sliding window",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm, "
                             "a LayerNorm with a bias.",
            "contribution": "mlp_output",
            "detail": "GELU: {hidden_size} → {intermediate_size} → {hidden_size}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding (an embedding dropout, the identity in eval), so "
             "token_embeddings equals layers[0].input.",
    "norm": "A LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "Tied to embed_tokens on 3B and 7B (no lm_head weight is stored); its own weight on 15B and "
            "15B-instruct. No bias, and nothing follows it: logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))     # LayerNorm, biased q, k, v, o
out = h + mlp(post_attention_layernorm(h))  # LayerNorm, c_fc -> GELU -> c_proj
```

Llama's tree and Llama's names, with LayerNorms (a mean subtracted, a weight and a bias) where Llama
has RMSNorms. Every linear map carries a bias (`use_bias`): `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`c_fc` and `c_proj`. The MLP has no gate: `c_fc` widens the normed stream four times, GELU (the tanh
approximation, `gelu_pytorch_tanh`) acts on it, and `c_proj` maps it back, so a neuron is one
column of `c_fc` and one row of `c_proj`. The attention and the MLP each end in a residual dropout
inside the module, the identity in eval.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream
with nothing in between. The identity holds exactly, in float32 on the pinned checkpoint and in
bfloat16 on starcoder2-3b (block 5):

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.equal(x + attn + mlp, out)   # True
```

With the biases, a zeroed sublayer input does not give a zero contribution: the attention falls
back to its value and output biases and the MLP to `c_proj(gelu(c_fc.bias))` plus `c_proj.bias`.
To remove a sublayer's effect, ablate `attention_output` or `mlp_output` itself.

## Every block attends over a window of 4096

`config.sliding_window` is `4096` on every StarCoder2 checkpoint, against a
`max_position_embeddings` of 16384, and the config has no `layer_types`: every block takes the
window. A query attends to its own position and the 4095 before it, so on a prompt shorter than
4096 tokens the mask is the plain causal mask. The window is in the mask the attention receives,
so under `attn_implementation="eager"` it shows in `attention_scores` (the largest negative float
outside it) and in `attention_probabilities` (zero outside it). On the pinned checkpoint loaded
with `sliding_window=4`, each query has at most 4 nonzero keys, itself and the 3 before:

```python
model = StandardizedTransformer(repo, attn_implementation="eager", sliding_window=4)
with model.trace(prompt):
    p = model.layers[1].self_attn.attention_probabilities.save()

(p[0, 0] > 0).sum(-1)   # 1, 2, 3, 4, 4, 4, ...
```

The pinned checkpoint itself has `sliding_window` `null`.

## Grouped-query attention, few key/value heads

Every size has `head_dim` 128. 3B has 24 query heads over 2 key/value heads, 7B 36 over 4, 15B 48
over 4. On 3B `attention_keys` and `attention_values` are `[batch, 2, seq, 128]` and query head
`h` reads key/value head `h // 12`: an edit to key/value head `j` reaches twelve query heads,
`12j` to `12j + 11`, half of the block's heads. The query scale is `head_dim ** -0.5`; the rotary
is the plain one over the whole head, with `rope_theta` near 1000000 on 3B and 7B and 100000 on
15B.

## Load with eager for the interior

A default load runs `sdpa`, and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. Both paths apply the same window, so they compute the
same function.

## The readout and the tokenizer

`model.logits` is `model.lm_head.output`, and `project_on_vocab` is `lm_head(norm(hidden))`, the
final LayerNorm's bias included. That bias adds the fixed vector `lm_head.weight @ norm.bias` to
every lens readout; on 3B it is small, a standard deviation of 0.23 logits against 2.15 for a whole
readout at block 14 on a code prompt. On 3B and 7B `lm_head` is tied to `embed_tokens`
(`tie_word_embeddings` defaults to `true` and the files store no `lm_head` weight), so an edit to
`embed_tokens.weight` is an edit to the unembedding; 15B sets it to `false`.

The tokenizer has 49152 ids and prepends nothing: position 0 is the text's first token. It is a
code tokenizer, and common English words split: `" Paris"` is two tokens, `" Par"` and `"is"`.
`<fim_prefix>`, `<fim_suffix>` and `<fim_middle>` are special tokens for fill-in-the-middle prompts.
"""
