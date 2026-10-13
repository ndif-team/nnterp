"""StableLM: Stability AI's StableLM-3B-4E1T and StableLM 2, which load as StableLmForCausalLM."""

MODEL_TYPE = "stablelm"
TITLE = "StableLM 2 / StableLM-3B"
SUBTITLE = (
    "Llama's block with LayerNorms and rotary on the first quarter of each head; StableLM-2-12B "
    "makes it parallel (use_parallel_residual) and norms each query and key head (qk_layernorm)."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "stabilityai/stablelm-2-1_6b"
#: The tiny checkpoint the test suite builds the page from (StableLM-2-12B's shape: parallel, qk_layernorm).
PINNED = "stabilityai/tiny-random-stablelm-2"
CHECKPOINTS = [
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "stabilityai/stablelm-2-12b",
    "stabilityai/stablelm-3b-4e1t",
    "stabilityai/stablelm-zephyr-3b",
]

#: Set by hues.py (lineage: GPT-NeoX).
PALETTE = {"hue": 69}
VLLM = False
QUIRKS = [
    "layernorm", "partial-rotary",
    {"slug": "qkv-bias", "when": lambda config: config.use_qkv_bias},  # StableLM 2 1.6B and its zephyr
    {"slug": "parallel-blocks", "when": lambda config: config.use_parallel_residual},  # StableLM-2-12B
    {"slug": "qk-norm", "when": lambda config: config.qk_layernorm},  # StableLM-2-12B
]

ATTENTION_INTERIOR = [
    "attention_queries", "attention_keys", "attention_values",
    "attention_scores", "attention_probabilities", "attention_head_outputs",
]
MLP_DETAIL = "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}"

#: StableLM-2-1.6B and StableLM-3B (use_parallel_residual false): input_layernorm before the attention,
#: post_attention_layernorm before the MLP, each sublayer adding to the stream in turn.
SEQUENTIAL = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "A LayerNorm with a bias.",
            "contribution": "attention_output",
            "interior": ATTENTION_INTERIOR,
            "detail": "{num_heads} heads, {num_kv_heads} kv; rotary on ¼",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "A LayerNorm with a bias, of the stream after the attention's add.",
            "contribution": "mlp_output",
            "detail": MLP_DETAIL,
        },
    ],
}

#: StableLM-2-12B and the pinned checkpoint (use_parallel_residual true): one input_layernorm whose output both
#: sublayers read (drawn once per branch, one node), per-head q/k LayerNorms inside the attention, one add.
PARALLEL_NORM_NOTE = ("One LayerNorm with a bias, drawn on both branches: the attention and the MLP read the same "
                      "output tensor. This block has no post_attention_layernorm.")
PARALLEL = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": PARALLEL_NORM_NOTE,
            "contribution": "attention_output",
            "interior": ATTENTION_INTERIOR,
            "detail": "{num_heads} heads, {num_kv_heads} kv; q/k norms",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": PARALLEL_NORM_NOTE,
            "contribution": "mlp_output",
            "detail": MLP_DETAIL,
        },
    ],
}

#: What the visualization draws, by the checkpoint's config: parallel where use_parallel_residual is set
#: (StableLM-2-12B, the pinned checkpoint), sequential otherwise (the configs that leave it out take false).
BLOCK = [
    (lambda config: getattr(config, "use_parallel_residual", False), PARALLEL),
    (lambda config: True, SEQUENTIAL),
]

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding, so token_embeddings equals layers[0].input. "
             "The tokenizer prepends no BOS.",
    "norm": "A LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight, not tied to embed_tokens, and no bias. Nothing follows it: logits "
            "equals lm_head.output.",
}

NOTES = """
## The block is sequential on 1.6B and 3B, parallel on 12B

```
# StableLM-2-1.6B, StableLM-3B-4E1T (use_parallel_residual false)
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))

# StableLM-2-12B (use_parallel_residual true)
n   = input_layernorm(x)
out = x + self_attn(n) + mlp(n)
```

The configs differ: only `stablelm-2-12b` sets `use_parallel_residual: true`, and the others leave
it out and take the default, `false`. On the parallel block `post_attention_layernorm` does not
exist, both sublayers read the one `input_layernorm` output, and an edit to `attention_output`
leaves that block's `mlp.input` bit-identical. The diagram draws each checkpoint's own block: the
parallel one on 12B, the sequential one on 1.6B and 3B. Read `model.config.use_parallel_residual`
before reusing a recipe across sizes. The pinned checkpoint has 12B's shape.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output` (after the block's
`dropout`, the identity in eval), and the identity holds in both shapes: exactly in float32 on the
pinned parallel checkpoint, and on StableLM-2-1.6B in float32 (block 5), where the stream between
the adds, `post_attention_layernorm.input`, equals `layers[i].input + attention_output`.

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.equal(x + attn + mlp, out)   # True
```

## Rotary turns a quarter of each head

Every config sets `partial_rotary_factor: 0.25`. On 1.6B, `head_dim` is 64, so the first 16
dimensions of each query and key head are rotated, as two halves of 8 (`rotate_half`), and the
other 48 are the projection's output unchanged; only those 16 make a score depend on the distance
between two tokens. On StableLM-2-1.6B the queries past dimension 16 equal `q_proj`'s output split
into heads, and the first 16 do not:

```python
attn = model.layers[5].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()
    q = attn.attention_queries.save()

b, s, _ = q_raw.shape
q_pre = q_raw.view(b, s, -1, model.head_dim).transpose(1, 2)
rot = int(model.head_dim * model.config.rope_parameters["partial_rotary_factor"])   # 16
torch.equal(q_pre[..., rot:], q[..., rot:])   # True
```

The query scale is `head_dim ** -0.5`, and `rope_theta` is 10000.

## Per-head LayerNorms on queries and keys on 12B

`stablelm-2-12b` sets `qk_layernorm: true`: the attention has `q_layernorm` and `k_layernorm`, each
a list of LayerNorms without bias, one per query head (32) and one per key/value head (8), applied
to the split heads before the rotary. `attention_queries` and `attention_keys` are read after both,
so past the rotary dimensions they equal `q_layernorm.output` and `k_layernorm.output`, not the
projections (checked on the pinned checkpoint, which has them). The other checkpoints have no such
modules.

## Biased queries, keys and values on StableLM 2 1.6B

`use_qkv_bias` is `true` on `stablelm-2-1_6b` and its zephyr, `false` on 12B and the 3B models.
Where set, `q_proj`, `k_proj` and `v_proj` add a bias before the rotary; `o_proj` and the MLP have
none. On 1.6B the biases are large: the key bias has a larger norm than the input-dependent part
`W_k·x` (median over the tokens of a prompt) on most blocks, 14.5 times on block 0, and the query
bias is the larger on blocks 0 to 4. So a zeroed attention input does not silence the attention;
ablate `attention_output` itself.

## Grouped-query attention on 12B

1.6B and the 3B models have one key/value head per query head (32 and 32). 12B has 32 query heads
over 8 key/value heads, so an edit to key/value head `j` reaches query heads `4j` to `4j + 3`.

## Load with eager for the interior

A default load runs `sdpa`, and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. There is no window, softcap or sink, so the two compute
the same function. The 1.6B files are `float16`; pass `dtype=torch.float32` for exact comparisons.

## The readout and the tokenizer

`model.logits` is `model.lm_head.output`, and `lm_head` has its own weight on every checkpoint.
The final norm is a LayerNorm with a bias, so a logit lens adds `lm_head.weight @ norm.bias` at
every block; on 1.6B its standard deviation is 0.31 logits and its top tokens are ` the`, ` and`,
` a`, ` to`, ` that`. StableLM 2 uses a 100289-token tokenizer (`vocab_size` 100352), StableLM-3B
GPT-NeoX's (50277 tokens, `vocab_size` 50304); neither prepends a BOS, so position 0 is the text's
first token.
"""
