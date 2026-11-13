"""OPT: Meta's OPT-125m to OPT-66b, a LayerNorm block whose MLP is fc1/fc2 on the block itself."""

MODEL_TYPE = "opt"
TITLE = "OPT"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "facebook/opt-125m"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-OPTForCausalLM"
CHECKPOINTS = [
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
    "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b",
]

#: Set by hues.py (lineage: OPT).
PALETTE = {"hue": 78}
QUIRKS = [
    "no-mlp", "position-embeddings", "layernorm", "qkv-bias",
    {"slug": "post-ln", "when": lambda config: not config.do_layer_norm_before},  # opt-350m
]

ATTENTION_INTERIOR = [
    "attention_queries", "attention_keys", "attention_values",
    "attention_scores", "attention_probabilities", "attention_head_outputs",
]
MLP_PATH_NOTE = "fc2.output is flattened to [batch * seq, hidden_size]: row b * seq + t is prompt b, position t."

#: What the visualization draws on every pre-norm checkpoint: the attention and its pre-norm, then the MLP path,
#: which has no module: final_layer_norm, fc1, activation_fn and fc2 sit on the block, so the sublayer is hosted
#: on the native fc2.
PRE_NORM = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": ATTENTION_INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, biased q/k/v",
        },
        {
            "host": "fc2",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "final_layer_norm",
            "reads": "fc1",
            "pre_norm_note": "The block's own norm; the decoder's last norm has the same native name and is model.norm.",
            "contribution": "fc2.output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
            "host_note": MLP_PATH_NOTE,
        },
    ],
    "identity": "layers[i].input + self_attn.attention_output + fc2.output.view_as(layer_output) == layer_output",
}

#: opt-350m (do_layer_norm_before false): each LayerNorm follows its add, on the stream. The attention reads the
#: raw stream, the MLP path reads self_attn_layer_norm's output, and layer_output is final_layer_norm's output.
POST_NORM = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "contribution": "attention_output",
            "interior": ATTENTION_INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, biased q/k/v",
            "stream_norm": "self_attn_layer_norm",
        },
        {
            "host": "fc2",
            "kind": "mlp",
            "label": "MLP",
            "contribution": "fc2.output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
            "host_note": MLP_PATH_NOTE,
            "stream_norm": "final_layer_norm",
        },
    ],
    "identity": "final_layer_norm((self_attn_layer_norm(layers[i].input + self_attn.attention_output).flatten(0, 1) "
                "+ fc2.output).view_as(layer_output)) == layer_output",
}

#: What the visualization draws, by the checkpoint's config: opt-350m is the one post-norm size.
BLOCK = [
    (lambda config: not config.do_layer_norm_before, POST_NORM),
    (lambda config: True, PRE_NORM),
]

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The learned position embedding, which reads row t + 2 for position t, is added after it; on opt-350m "
             "project_in first widens the lookup from 512 to 1024.",
    "head": "On opt-350m project_out narrows the stream from 1024 to 512 before lm_head; project_on_vocab applies it too.",
}

NOTES = """
## The MLP's term is `fc2.output`, one row per token

```
h   = x + self_attn(input_layernorm(x))
f   = h.reshape(-1, hidden_size)                        # [batch * seq, hidden_size]
out = (f + fc2(relu(fc1(final_layer_norm(f))))).view_as(x)
```

The block has no `mlp` module, so `mlp_output` does not exist; the MLP's term is `layers[i].fc2.output`.
`fc2.output`, `fc1.output` and the neurons at `activation_fn.output` are `[batch * seq, ...]`, row
`b * seq + t` for prompt `b`, position `t`, so `fc2.output[:, -1]` is a hidden dimension, not the last
position. Read or edit a position through `fc2.output.unflatten(0, (batch, seq))`; it is a view, so an
in-place edit lands.

## On opt-350m the embeddings and the lens pass through projections

`opt-350m` has a 1024-wide stream and 512-wide embeddings: `project_in` widens `token_embeddings`
before the positions are added, and `project_out` narrows the stream before `lm_head`, with no final
norm between. `project_on_vocab` there is `lm_head(project_out(h))`.
"""
