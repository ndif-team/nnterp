"""BitNet b1.58: ternary-weight projections with 8-bit inputs, and an RMSNorm before each output projection."""

MODEL_TYPE = "bitnet"
TITLE = "BitNet b1.58"
SUBTITLE = (
    "Llama's tree whose projections run ternary weights on 8-bit inputs, with an RMSNorm inside each sublayer "
    "before its output projection (attn_sub_norm, ffn_sub_norm), so the contributions are o_proj's and "
    "down_proj's outputs after a norm; the MLP gates with a squared ReLU."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/bitnet-b1.58-2B-4T"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-BitNetForCausalLM"
CHECKPOINTS = ["microsoft/bitnet-b1.58-2B-4T", "microsoft/bitnet-b1.58-2B-4T-bf16"]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 253}
VLLM = False
QUIRKS = ["squared-relu"]

#: Real values in the notes were measured on bitnet-b1.58-2B-4T-bf16 in bfloat16 on a GPU (its weights load as
#: bfloat16 and are ternarized in every forward); shapes, identities and read orders also ran on the pinned tiny
#: checkpoint, whose projections are plain nn.Linear and whose hidden_act is gelu.

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
            "detail": "{num_heads}/{num_kv_heads} heads, attn_sub_norm, o_proj",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm. "
                             "The MLP's second norm, ffn_sub_norm, is inside it, before down_proj.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act} gate, ffn_sub_norm",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, not quantized: token_embeddings equals layers[0].input. The tokenizer prepends "
             "<|begin_of_text|> (id 128000).",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "lm_head is embed_tokens' weight (tie_word_embeddings) and is not quantized. logits is "
            "lm_head.output, with no cap or scale.",
}

NOTES = """
## The block, in order

```
a   = attention(q_proj, k_proj, v_proj of input_layernorm(x))   # heads, before o_proj
h   = x + o_proj(attn_sub_norm(a))
m   = relu(gate_proj(z)) ** 2 * up_proj(z),  z = post_attention_layernorm(h)
out = h + down_proj(ffn_sub_norm(m))
```

Four RMSNorms per block. `input_layernorm` and `post_attention_layernorm` are Llama's pre-norms;
`self_attn.attn_sub_norm` (over `hidden_size`, all heads at once) and `mlp.ffn_sub_norm` (over
`intermediate_size`) sit inside the modules, each right before the output projection. Every
projection inside the block (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`) is quantized; `embed_tokens` and `lm_head` are not.

## The contributions are the output projections' outputs

`attention_output` is `self_attn.output[0]`, which is `o_proj.output`, and `mlp_output` is
`mlp.output`, `down_proj.output`: the sub-norms come before them, not after, so the contributions
are the modules' outputs and the identity holds exactly. `attention_head_outputs` is read at the
attention interface, before `attn_sub_norm`: flattened over its heads it equals
`attn_sub_norm.input`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    heads = attn.attention_head_outputs.save()   # [batch, seq, heads, head_dim]
    normed = attn.attn_sub_norm.input.save()

torch.equal(heads.flatten(-2), normed)   # True
```

## Scaling inside a sublayer is undone by the sub-norm

`attn_sub_norm` divides by the RMS over every head's output together, so scaling all of
`attention_head_outputs` changes nothing: on 2B-4T-bf16, doubling them at block 5 leaves
`attention_output` bit-identical, and so does doubling `ffn_sub_norm.input` for `mlp_output`. A
change to one head moves the others, which share the RMS: zeroing head 3 at block 5 changes the
other heads' slices of `attn_sub_norm.output` by 0.5%. Scale or ablate a sublayer's effect at
`attention_output` or `mlp_output`; a head's effect is not a separate term of `attention_output`.

```python
with model.trace(prompt):
    model.layers[1].self_attn.attention_head_outputs[:] *= 2   # no effect
```

## The projections quantize their inputs and weights

On a real load every projection in the block is an `AutoBitLinear`. Its forward rounds its input
per token to 255 levels (`round(x · 127 / max|x|)`) and multiplies by ternary weights: on the
bf16 checkpoint the weights are stored in bfloat16 and rounded to `{−1, 0, 1} · mean|W|` in every
forward; on the packed checkpoint they are stored ternary, four to a byte, with a `weight_scale`.
`q_proj.input` is the unrounded normed stream; on 2B-4T-bf16 `q_proj.output` equals
`F.linear(ActQuant(input), WeightQuant(weight))` exactly, and differs from
`F.linear(input, weight)` by 2.4 times its own norm. An edit at a projection's input is rounded
with the rest of the row, and a weight edit on the bf16 checkpoint is re-ternarized: adding
`0.1 · mean|W|` to one `q_proj` weight at block 1 leaves the logits bit-identical. The page's
printout comes from a meta build, where the projections are plain `Linear`: the swap happens when
the weights load.

## The stream is large, and the input norms' gains small

On 2B-4T-bf16 the residual stream at block 10 has a norm of 20,000 to 35,000 per position and
161,000 at position 0, `<|begin_of_text|>`. The pre-norms' gains are small to match:
`input_layernorm.weight` at block 5 averages 0.015, the final `norm.weight` 0.10. A steering
vector sized from another family's stream is lost in this one; take its scale from
`layer_output.norm(dim=-1)` on this checkpoint.

## The MLP gates with a squared ReLU

`mlp.act_fn` is `relu2` on the gate: a neuron is `relu(gate_proj(z))² · up_proj(z)`, exactly zero
wherever the gate's pre-activation is negative. At block 5 of 2B-4T-bf16, 62% of
`gate_proj.output` entries are at or below zero on a test prompt. `ffn_sub_norm.input` is the
gated product, `[batch, seq, 6912]`.

## Loading

The six attention interior values need `attn_implementation="eager"`. There is no softcap,
window or sink, and the default `sdpa` load computes the same function up to rounding: on 2B-4T-bf16
in bfloat16 the two loads' logits differ by up to 0.75 with the same top token at every position
(`" Paris"` 0.822 eager, 0.807 sdpa). 20 query heads share 5 key/value heads, 128 wide, so an edit to
key/value head `j` of `attention_keys` reaches query heads `4j` to `4j + 3`. The rotary is Llama's
`rotate_half` with `rope_theta` 500,000, over 4096 positions.

## The two checkpoints

`bitnet-b1.58-2B-4T` stores its weights packed (`quantization_mode` `offline`) and
`bitnet-b1.58-2B-4T-bf16` in bfloat16 for training (`online`); the configs differ only there.
The packed one multiplies by its stored ternary weights and `weight_scale`, the bf16 one
ternarizes its weights in each forward. The tokenizer is Llama 3's: `<|begin_of_text|>` is prepended and `" Paris"` is one
token. The chat template writes `User: ...<|eot_id|>Assistant: `.
"""
