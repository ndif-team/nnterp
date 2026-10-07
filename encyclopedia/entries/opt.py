"""OPT: Meta's OPT-125m to OPT-66b, a LayerNorm block whose MLP is fc1/fc2 on the block itself."""

MODEL_TYPE = "opt"
TITLE = "OPT"
SUBTITLE = (
    "A LayerNorm block with no MLP module: fc1, the activation and fc2 sit on the block itself, so "
    "fc2.output is the block's second term, and a learned position embedding, offset by 2, is added after embed_tokens."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "facebook/opt-125m"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-OPTForCausalLM"
CHECKPOINTS = [
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
    "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b",
]

#: OPT and XGLM share the block (fairseq's decoder layer); OPT sets the lineage's hue.
PALETTE = {"hue": 31}
VLLM = False
QUIRKS = ["no-mlp", "position-embeddings", "layernorm", "qkv-bias"]

#: What the visualization draws: the attention and its pre-norm. The MLP path (final_layer_norm,
#: fc1, activation_fn, fc2) has no module, so it is no sublayer; the identity names fc2.output.
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
            "detail": "{num_heads} heads × {head_dim}, biased q/k/v",
            "pre_norm_note": "Native name self_attn_layer_norm, a LayerNorm with a bias. On opt-350m "
                             "(do_layer_norm_before false) it follows the attention's add instead, and the "
                             "attention reads the raw stream.",
        },
    ],
    "identity": "layers[i].input + self_attn.attention_output + fc2.output.view_as(layer_output) == layer_output",
    "identity_note": "fc2.output is the MLP path's term: the block has no mlp module, and it flattens the stream to "
                     "[batch * seq, hidden] before final_layer_norm and fc1. Exact on every pre-norm checkpoint; on "
                     "opt-350m a post-norm follows each add, and no sum of these terms is layer_output.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is embed_tokens' lookup alone. The learned position embedding "
             "model.decoder.embed_positions is added after it, so layers[0].input = token_embeddings + "
             "embed_positions.output. On opt-350m the lookup is 512 wide and project_in maps it to 1024 before the add.",
    "layers": "Each block runs the attention, then final_layer_norm, fc1, activation_fn and fc2, all on the block: "
              "there is no mlp module, and layers[i].fc2.output is the block's second term.",
    "norm": "final_layer_norm is a LayerNorm with a bias; project_on_vocab applies it. opt-350m has none, "
            "so model.norm does not exist there.",
    "head": "lm_head has no bias and shares its weight with embed_tokens. Its output is logits, unchanged. "
            "On opt-350m project_out maps the stream from 1024 to 512 before lm_head.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(self_attn_layer_norm(x))           # self_attn_layer_norm = input_layernorm
f   = h.reshape(-1, hidden)                            # [batch * seq, hidden]
out = (f + fc2(relu(fc1(final_layer_norm(f))))).view_as(x)
```

The block returns a bare tensor. `attention_output` is the attention module's output after
`out_proj`. The block's own `final_layer_norm` is the MLP path's input norm; it keeps its native name,
since the decoder's last norm has the same name and is `model.norm`. Every norm is a `LayerNorm` with a
weight and a bias, and every projection, `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1` and `fc2`, has
a bias.

## The MLP is fc1 and fc2 on the block

No block has an `mlp` module, so `mlp`, `mlp_output` and their kin do not exist on OPT and
`support()` lists no `mlp.*` key. What the MLP path adds is `layers[i].fc2.output`, and
`layers[i].fc1.input` is `final_layer_norm`'s output. The block flattens the stream before that path:
`final_layer_norm.output`, `fc1`'s and `fc2`'s outputs and the neurons, `activation_fn.output`, are
`[batch * seq, ...]`, row `b * seq + t` for prompt `b`, position `t`. Unflatten them to read or edit a
position:

```python
with model.trace(prompts):
    batch, seq = model.input_size
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    neurons = model.layers[1].activation_fn.output.save()   # [batch * seq, ffn_dim]
    fc2 = model.layers[1].fc2.output
    mlp_out = fc2.unflatten(0, (batch, seq)).save()
    out = model.layers[1].layer_output.save()

torch.equal(x + attn + mlp_out, out)                        # True
```

The unflattened tensor is a view, so an in-place edit lands:
`model.layers[1].fc2.output.unflatten(0, (batch, seq))[:, -1] = 0` removes the MLP path's term at the
last position, and that position's `layer_output` is then `input + attention_output`. `intermediate_size`
is the config's `ffn_dim`, four times the width on every released size, and the activation is ReLU.

## The queries arrive scaled

OPT multiplies the projected queries by `head_dim ** -0.5` and calls the attention with a scale of 1,
so `attention_queries` is `q_proj`'s output times that scale, and `attention_scores` is
`attention_queries @ attention_keys.transpose(-1, -2)` plus the mask. A query-key product computed from
`q_proj.output` is `head_dim ** 0.5` times the score. The interior values need
`attn_implementation="eager"`; the default load runs `sdpa`. Every head has its own keys and values.

## Learned positions, offset by 2

`model.decoder.embed_positions` is an embedding of `max_position_embeddings + 2` rows (2050), and
position `t` reads row `t + 2`. It is added to the token embeddings before
block 0:

```python
with model.trace(prompt):
    tokens = model.token_embeddings.save()
    positions = model.model.decoder.embed_positions.output.save()
    x0 = model.layers[0].input.save()

W = model.model.decoder.embed_positions.weight
torch.equal(tokens + positions, x0)                         # True
torch.equal(positions[0], W[2 : 2 + x0.shape[1]])           # True
```

Positions follow the attention mask, so a left-padded prompt reads the same rows as the prompt alone, and its logits match (largest difference 6e-8 on the pinned
checkpoint). A prompt longer than 2048 tokens runs past the table.

## opt-350m is post-norm and projects the embedding

Of the released sizes only `opt-350m` sets `do_layer_norm_before` to false. Each of its norms follows
the add: the attention reads the raw stream, `self_attn_layer_norm` normalizes `input +
attention_output`, and `layer_output` is the block's `final_layer_norm` output. The identity does not
hold there: on `opt-350m`, `input + attention_output + fc2.output` misses `layer_output` by 10 to 19 in
its largest coordinate, on a stream whose norm is 32. The decoder has no final norm, so `model.norm`
does not exist and `project_on_vocab` raises `AttributeError`. The embedding is 512 wide:
`project_in` maps `token_embeddings` to the stream's 1024 before the positions are added, and
`project_out` maps the stream back to 512 for the tied `lm_head`. The logit lens there is
`lm_head(project_out(x))`, on `opt-350m`:

```python
with model.trace(prompt):
    last = model.layers[-1].layer_output.save()
    logits = model.logits.save()

project_out = model.model.decoder.project_out._module
lm_head = model.lm_head._module
torch.equal(lm_head(project_out(last)), logits)             # True
```

## The first position is a sink in one coordinate

The tokenizer prepends `</s>` (id 2) to every prompt. On `opt-125m` the stream at that position has a
norm of 120 after block 0 and 311 to 379 from block 2 to block 10, nearly all of it in coordinate 422,
against medians of 2 to 8 at the other positions. From block 1 on, 63% to 85% of the attention pattern,
averaged over heads and later queries, lands on key 0. Block 11 brings that position's norm back to 9.
Leave position 0 out of norms, steering scales and activation statistics.

## The readout: LayerNorm with a bias, tied weights

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` is
`lm_head(final_layer_norm(x))`, so a logit lens at the last block equals `logits` on every pre-norm
size. `lm_head` has no bias and its weight is `embed_tokens.weight`, the same tensor, so an edit to the
embedding matrix is an edit to the unembedding.

## The family's checkpoints

Blocks × width: 12 × 768 (`opt-125m`), 24 × 1024 (`opt-350m`), 24 × 2048 (`opt-1.3b`), 32 × 2560
(`opt-2.7b`, `head_dim` 80), 32 × 4096 (`opt-6.7b`), 40 × 5120 (`opt-13b`), 48 × 7168 (`opt-30b`),
64 × 9216 (`opt-66b`). `head_dim` is 64 up to 1.3b and 128 from 6.7b. Every size has ReLU, an MLP four
times the width, 2048 positions and the same 50272-token vocabulary, and stores float16 weights.
"""
