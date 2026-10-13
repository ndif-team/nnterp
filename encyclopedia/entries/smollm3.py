"""SmolLM3: Hugging Face's 3B models, which load as SmolLM3ForCausalLM."""

MODEL_TYPE = "smollm3"
TITLE = "SmolLM3"
SUBTITLE = (
    "Llama's block in which every fourth attention applies no rotary embedding (NoPE): there queries and "
    "keys are the projections as split, and only the causal mask orders the tokens."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "HuggingFaceTB/SmolLM3-3B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/smollm3-tiny-random"
CHECKPOINTS = [
    "HuggingFaceTB/SmolLM3-3B-Base",
    "HuggingFaceTB/SmolLM3-3B",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 247}
VLLM = False
QUIRKS = ["nope-blocks"]

#: What the visualization draws: Llama's sequential block, one RMSNorm before each sublayer.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; NoPE 1 in {no_rope_layer_interval}",
            "variants": {
                "full_attention": "{num_heads} heads, {num_kv_heads} kv; NoPE 1 in {no_rope_layer_interval}",
                "sliding_attention": "{num_heads} heads, {num_kv_heads} kv; sliding window",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding, so token_embeddings equals layers[0].input. "
             "The tokenizer prepends no BOS.",
    "layers": "Every fourth block (3, 7, ..., 35 on 3B) applies no rotary embedding; config.no_rope_layers "
              "marks them with 0. The slider's ticks follow layer_types, which does not say this.",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "Tied to embed_tokens (tie_word_embeddings): one parameter. Nothing follows it: logits equals "
            "lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's block and Llama's names: two RMSNorms, each before its sublayer, none after.
`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, and the identity
holds exactly, in float32 on the pinned checkpoint and in bfloat16 on SmolLM3-3B (blocks 2 and 3):

```python
with model.trace(prompt):
    x = model.layers[3].input.save()
    attn = model.layers[3].self_attn.attention_output.save()
    mlp = model.layers[3].mlp.mlp_output.save()
    out = model.layers[3].layer_output.save()

torch.equal(x + attn + mlp, out)   # True
```

## Every fourth block has no rotary

`config.no_rope_layers` holds one flag per block, `1` where the attention applies the rotary and `0`
where it does not. On both 3B checkpoints it is `[1, 1, 1, 0]` repeated, so blocks 3, 7, 11, ...,
35 are NoPE blocks, 9 of 36; `SmolLM3-3B-Base` leaves the list out of its config and transformers
builds the same one from `no_rope_layer_interval` (4). The pinned checkpoint has two blocks and
`no_rope_layers` `[1, 0]`, so its block 1 is the NoPE one. Read the list before choosing a block:

```python
nope = [i for i, rope in enumerate(model.config.no_rope_layers) if not rope]
# [3, 7, 11, 15, 19, 23, 27, 31, 35] on 3B
```

`layer_types` says nothing about this: on 3B every block is `full_attention`, so the slider's
ticks are all one colour and the NoPE blocks are not marked on it.

## On a NoPE block the queries and keys are the projections

On a NoPE block `attention_queries` and `attention_keys` equal `q_proj` and `k_proj`'s outputs split
into heads, at every position; on a rotary block they agree only at position 0, where the rotation
is the identity. Checked on the pinned checkpoint (block 1 against block 0) and on SmolLM3-3B
(block 3 against block 2):

```python
attn = model.layers[3].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()
    q = attn.attention_queries.save()

b, s, _ = q_raw.shape
torch.equal(q_raw.view(b, s, -1, model.head_dim).transpose(1, 2), q)   # True on a NoPE block
```

So on a NoPE block a query–key score depends on the two tokens' contents alone: a key patched in
from another position scores the same as it did there, and what orders the tokens is the causal
mask, plus whatever position information earlier rotary blocks wrote into the stream.

## Grouped-query attention

3B has 16 query heads over 4 key/value heads, `head_dim` 128. `attention_keys` and
`attention_values` are `[batch, 4, seq, 128]`, and query head `h` reads key/value head `h // 4`: an
edit to key/value head `j` reaches query heads `4j` to `4j + 3`. The rotary blocks use the plain
rotary with `rope_theta` 5000000; the query scale is `head_dim ** -0.5`.

## A sliding window the 3B checkpoints leave off

The config can put a sliding window on some blocks (`use_sliding_window`, `sliding_window`,
`layer_types`); both 3B checkpoints set `use_sliding_window` to `false` and `sliding_window` to
`null`, so every block attends over the whole prefix. The pinned checkpoint turns it on, with a
window of 128 on its block 1, which is why its slider shows a second kind of block.

## Load with eager for the interior

A default load runs `sdpa`, and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. With no window, softcap or sink on 3B, the two compute
the same function.

## The readout, the embeddings and the tokenizer

`model.logits` is `model.lm_head.output`, and `lm_head` is tied to `embed_tokens` (one parameter),
so an edit to `embed_tokens.weight` is an edit to the unembedding. The tokenizer has Llama-3's
vocabulary, 128256 ids, and prepends no BOS (`bos_token` is `None`): position 0 is the text's first
token. `" Paris"` (`12366`) and `"Paris"` (`60704`) are two whole-word tokens, as on Llama-3.
"""
