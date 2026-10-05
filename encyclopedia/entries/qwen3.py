"""Qwen3: the dense checkpoints, and every other line that loads as ``Qwen3ForCausalLM``."""

MODEL_TYPE = "qwen3"
TITLE = "Qwen3"
SUBTITLE = (
    "Llama's block with an RMSNorm on each query and key head before the rotary embedding, so their "
    "size comes from the norms' gains, not the input's scale; there is no BOS, and position 0 becomes "
    "the attention sink."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-0.6B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Qwen3ForCausalLM"
CHECKPOINTS = [
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-SafeRL",
    "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B", "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3Guard-Gen-0.6B", "Qwen/Qwen3Guard-Gen-4B", "Qwen/Qwen3Guard-Gen-8B",
    "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

#: The Qwen lineage sits at 285 (qwen2); qwen3 takes a hue 12 degrees from it.
PALETTE = {"hue": 297}
VLLM = True
QUIRKS = ["qk-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k RMS-normed",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no biases",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input. No BOS is prepended, "
             "so position 0 holds the text's first token.",
    "head": "lm_head is embed_tokens' weight on 0.6B, 1.7B and 4B (tie_word_embeddings) and a matrix of its own "
            "on 8B, 14B and 32B. logits is lm_head.output, with no softcap or scale.",
}

NOTES = """
## Queries and keys are normed per head, before the rotary

```
q = rope(q_norm(q_proj(x).view(..., heads, 128)))     # one gain, every head
k = rope(k_norm(k_proj(x).view(..., kv_heads, 128)))
v = v_proj(x).view(..., kv_heads, 128)                # no norm
out = o_proj(softmax(q·kᵀ / √128) · v)
```

`q_norm` and `k_norm` are RMSNorms over one head's 128 dimensions, applied to every head with the
same gain vector. `attention_queries` and `attention_keys` are read at the attention interface,
after both the norm and the rotary: on 0.6B they equal `rope(q_norm(...))` of `q_proj.output`
exactly. `q_norm.output` is `[batch, seq, heads, 128]`, before the transpose and the rotary.
Because each head is rescaled to a fixed RMS, the size of a query or key comes from the gains
and the head's direction, not from the input's scale: on 0.6B the `k_norm` gains reach 96.5 on
block 0 and the `q_norm` gains 11.8.

## An edit to q_proj.output is renormalized

A head's slice of `q_proj.output` or `k_proj.output` passes through its own norm before it is
used, so the edit reaches that head only, and its size is lost. On 0.6B, doubling head 3's slice
of `q_proj.output` moves the logits by under 1e-4, tripling a key/value head's slice of
`k_proj.output` by about 1e-4, while tripling the same slice of `v_proj.output` (no norm) moves
them by 5.8. Zeroing a slice gives a zero query, and that head attends uniformly over the prefix.
To change how sharply a head attends, edit `attention_queries` (or `attention_keys`), which come
after the norm:

```python
d = model.head_dim
with model.trace(prompt):
    model.layers[1].self_attn.q_proj.output[:, :, 3 * d:4 * d] *= 2  # undone

with model.trace(prompt):
    model.layers[1].self_attn.attention_queries[:, 3] *= 2  # head 3's scores x2
```

On 0.6B the second edit moves the logits by up to 1.8 and changes only head 3's pattern. An
additive edit to a slice survives only as a change of the head's direction.

## No projection has a bias

`attention_bias` is false on every checkpoint listed here, so `q_proj`, `k_proj`, `v_proj` and
`o_proj` are bias-free, as are the MLP's three projections. Zeroing `self_attn.input` gives zero
keys, values and queries and an `attention_output` of exactly zero.

## Grouped-query attention, with 128-wide heads on every size

0.6B has 16 query heads over 8 key/value heads: `attention_keys` and `attention_values` are
`[batch, 8, seq, 128]`, and an edit to key/value head `j` reaches query heads `2j` and `2j + 1`.
The other sizes: 16 over 8 (1.7B), 32 over 8 (4B, 8B), 40 over 8 (14B), 64 over 8 (32B). The
score scale is `128 ** -0.5`. Since `head_dim` is 128 throughout, `num_heads × head_dim` is not
`hidden_size` on 0.6B (2048 against 1024), 4B (4096 against 2560) or 32B (8192 against 5120):
there `q_proj` widens the stream, `o_proj` narrows it back, and `attention_head_outputs`
flattened is wider than `layer_output`.

## The default load computes the same model

Every checkpoint here sets `use_sliding_window: false`, so `config.layer_types` is
`full_attention` on every block and `self_attn._module.sliding_window` is `None`. With no window
and no softcap, the default `sdpa` load and an eager one agree: on 0.6B in float32 their logits
differ by at most 3e-5. `attn_implementation="eager"` is needed only to read the six attention
interior values.

## Position 0 is the first token, and it becomes the sink

Qwen's tokenizers set `bos_token` to `None` and prepend nothing; the config's `bos_token_id`,
151643, is `<|endoftext|>`. Whatever token sits at position 0 becomes an attention sink. On 0.6B,
from block 2 to block 26, position 0's `layer_output` has a norm of 5600 to 6600 on every prompt,
98 to 99% of it (squared) on coordinate 35, pointing the same way on every prompt (cosine
1.000); up to block 15 every other position is under 80. The last block shrinks it to 1650 to
1850. From block 3 on, queries put on average 0.58 to 0.91 of their attention on key 0. 8B (from
block 6, coordinate 2276) and 4B-Thinking-2507 (from block 6, coordinate 4) do the same.

A probe, a patch or a logit lens at position 0 reads the sink, and a mean over positions is
dominated by it unless it is dropped. Prepending `<|endoftext|>` moves the sink onto that token
and gives the first word its own position; on 0.6B `' Paris'` goes from 0.885 to 0.976.

```python
with model.trace("<|endoftext|>The Eiffel Tower is in the city of"):
    resid = model.layers[10].layer_output.save()

resid[0].norm(dim=-1)        # position 0 about 6500, the others under 55
```

## The chat template opens a thinking block

The template writes no system turn of its own, so a templated prompt starts with
`<|im_start|>user` and its position 0, `<|im_start|>`, takes the sink. On the hybrid checkpoints
(0.6B to 32B; the Base ones ship a template with the same ending) `add_generation_prompt=True`
ends the prompt at `<|im_start|>assistant\\n`, and the next token is `<think>` (151667): on 0.6B with
probability 1.000. A next-token metric at the end of that prompt measures the start of a
reasoning trace, not the answer. `enable_thinking=False` appends an empty
`<think>\\n\\n</think>\\n\\n`, after which 0.6B puts 0.873 on `'Paris'`:

```python
text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
with model.trace(text):
    probs = model.next_token_probs.save()
```

The answer at the start of the assistant's turn is `'Paris'`, with no leading space;
`get_first_tokens("Paris", model)` returns both `'Paris'` and `' Paris'`. The template also strips
the `<think>` part from assistant turns before the last user message.

Qwen3-4B-Thinking-2507's template ignores `enable_thinking` and ends the prompt at `<think>\\n`;
closing the block yourself does not skip the reasoning (its next token is still `'Hmm'`, 0.805). Qwen3-4B-Instruct-2507 has
no thinking block, and puts 1.000 on `'Paris'` right after `<|im_start|>assistant\\n`.
Generation stops on `<|im_end|>` (151645) or `<|endoftext|>` (151643).

## The readout is the plain projection

`model.logits` equals `model.lm_head.output`, and `project_on_vocab` on the last block's
`layer_output` equals `logits` exactly. On 0.6B, 1.7B and 4B (the 2507 releases included) the
embedding and the unembedding are one matrix, so an edit to `embed_tokens.weight` edits
`lm_head`; on 8B, 14B and 32B they are separate.

## What loads as this family

The configs that say `Qwen3ForCausalLM` include the dense Qwen3 checkpoints from 0.6B to 32B and their
Base models (there is no 32B Base), Qwen3-4B-Instruct-2507 and Thinking-2507, Qwen3-4B-SafeRL,
Qwen3Guard-Gen, Qwen3-Reranker, and DeepSeek-R1-0528-Qwen3-8B. `rope_theta` is 10⁶ everywhere
but the 4B-2507 pair (5·10⁶, 262144 positions); the DeepSeek model adds YaRN scaling (factor 4).
Qwen3-30B-A3B, 235B-A22B and their 2507 releases are `qwen3_moe`. Qwen3-Next is `qwen3_next`,
and Qwen3.5's text model is `qwen3_5_text`, a hybrid with gated DeltaNet blocks.

## Sparse autoencoders

Qwen-Scope's residual-stream SAEs are trained on Qwen3-1.7B-Base
(`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` and `-L0_100`, 32768 features) and Qwen3-8B-Base
(`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` and `-L0_100`, 65536 features), one per block, TopK with
`k` 50 or 100. Their hook point is `resid_post`, a forward hook on the decoder block's output, which
is `model.layers[i].layer_output`. Each `layer<i>.sae.pt` holds `W_enc` `[d_sae, hidden]`,
`b_enc`, `W_dec` `[hidden, d_sae]` and `b_dec` in float32.
"""
