"""Nemotron-H: the hybrid line (NemotronHForCausalLM), Nemotron-H, Nemotron Nano 2 and Nemotron 3."""

MODEL_TYPE = "nemotron_h"
TITLE = "Nemotron-H / Nemotron Nano\u00a02 / Nemotron 3"
SUBTITLE = (
    "Every block is one RMSNorm and one sublayer, a Mamba-2 mixer on about half the blocks, attention without "
    "rotary on a few, and a squared-ReLU MLP or, on Nemotron 3's mixture checkpoints, a mixture of experts on the rest."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
#: The tiny checkpoint the test suite builds the page from (the suite renames its embedding weight in a copy).
PINNED = "hf-tiny-v2/tiny-random-NemotronHForCausalLM"
CHECKPOINTS = [
    "nvidia/Nemotron-H-4B-Base-8K", "nvidia/Nemotron-H-4B-Instruct-128K",
    "nvidia/Nemotron-H-8B-Base-8K", "nvidia/Nemotron-H-8B-Reasoning-128K",
    "nvidia/Nemotron-H-47B-Base-8K", "nvidia/Nemotron-H-47B-Reasoning-128K",
    "nvidia/Nemotron-H-56B-Base-8K",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
]

#: Nemotron lineage, beside nemotron (262).
PALETTE = {"hue": 250}
VLLM = False
QUIRKS = ["one-sublayer-blocks", "hybrid", "mamba2", "nope-blocks", "squared-relu", "mixture-of-experts"]

#: The sublayers in forward order. Each block holds one of them, the ``mixer`` its entry in
#: ``config.layer_types`` names, so each block draws one. The reference is dense (Mamba-2, attention,
#: MLP blocks); the mixture of Nemotron 3's 30B, Super and Ultra checkpoints is described in the notes.
NORM_NOTE = ("The block's one RMSNorm, model.layers[i].norm, before whichever sublayer the block holds; "
             "the final norm is model.norm (native norm_f).")

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD, {mamba_num_heads} heads × {mamba_head_dim}, {n_groups} groups",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}, no rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "norm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {mlp_hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, no scale and no position embedding: token_embeddings equals layers[0].input.",
    "norm": "An RMSNorm whose gain is norm.weight as stored (a mean of 2.49 on Nano-4B, up to 8.4); project_on_vocab "
            "applies it.",
    "head": "Its own weight, not tied to embed_tokens.",
    "logits": "lm_head.output cast to float32; project_on_vocab casts too, so a lens at the last block equals logits.",
}

NOTES = """
## The block, in order

```
out = x + mixer(norm(x))     # mixer: the Mamba-2 mixer, attention, the MLP or the mixture
```

A block holds one sublayer, and `config.layer_types` (built from the checkpoint's
`hybrid_override_pattern`, one character per block: `M`, `*`, `-`, `E`) says which:
`linear_attention` for a Mamba-2 mixer, `full_attention` for attention, `mlp` for the MLP and
`moe` for a mixture of experts. transformers calls all four `mixer`; nnterp names it
`linear_attn`, `self_attn` or `mlp` by its class, and the other two names are `None` on that block.
On Nano-4B the 42 blocks are 21 Mamba-2, 17 MLP and 4 attention blocks (12, 17, 24 and 32). Pick
blocks outside the trace:

```python
kinds = model.config.layer_types
ssm = [i for i, t in enumerate(kinds) if t == "linear_attention"]
attn = [i for i, t in enumerate(kinds) if t == "full_attention"]
mlp = [i for i, t in enumerate(kinds) if t in ("mlp", "moe")]
```

The block's norm keeps its native name, `norm`, so `model.layers[i].norm` is the norm before the
block's sublayer and `model.norm` is the final norm (`norm_f`).

## A block adds one contribution

`layer_output` is the block's input plus its one sublayer's output: `attention_output` on a
Mamba-2 or attention block, `mlp_output` on an MLP or mixture block. The sum is exact on
Nano-4B in bfloat16 for each kind:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    mlp = model.layers[1].mlp.mlp_output.save()      # block 1 is an MLP block on Nano-4B
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + mlp, out)
```

Every sublayer ends a block, so `layers[i].layer_output` is the stream after each sublayer and
`layers[i].input` the stream before it: a logit lens or a patch at block granularity is one at
sublayer granularity here.

## Loading

The attention interior needs `attn_implementation="eager"`. There is no softcap, window or sink,
and the default load (`sdpa`) computes the same function: on Nano-4B the two loads' logits differ
by at most `7e-4` in float32 and `0.16` in bfloat16, with the same top token at every position of
a test prompt.

With `mamba_ssm` installed, transformers runs the Mamba-2 scans in its CUDA kernels, and every
`linear_attn` value but `attention_output` is unavailable until
`nnterp.route_kernels(model.family, "torch")` is called before the first trace. On a CPU the
unrouted model does not run at all. The routed and the default kernels give logits that differ by
`0.013` in float32 and `0.2` in bfloat16 on Nano-4B, with the same top tokens. `states` and
`state_after` also need `nnterp.chunk_per_token(model)`: the scan keeps the state every
`chunk_size` tokens, 256 on Nano-4B and Nemotron-H 4B and 128 on the other released checkpoints.
In bfloat16 that change of chunking moves Nano-4B's logits by up to `0.19`.

```python
import nnterp

nnterp.route_kernels(nnterp.families.nemotron_h, "torch")     # before the first trace
model = StandardizedTransformer(
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", attn_implementation="eager"
)
```

## The Mamba-2 mixer

On Nano-4B each mixer has 96 heads of 80 channels in 8 groups: `attention_queries` (`C`) and
`attention_keys` (`B`) are `[batch, seq, 8, 128]`, one per group of 12 heads, and
`attention_values` (`x`) and `attention_head_outputs` are `[batch, seq, 96, 80]`. The state is
`[batch, 96, 128, 80]`. A width-4 convolution runs over `x`, `B` and `C` before the scan. After
it, a gated RMSNorm multiplies `y` by `silu(z)` and then normalizes it in 8 groups of 960
channels; `attention_head_outputs` is read before both.

The mixer clamps `betas` (`dt`, after the softplus) to at least `time_step_min`, `0.001`, on a
prompt. A token's `betas` therefore never reaches zero: written zeros run as `0.001`, and a read
after the write returns `0.001`. On Nano-4B in float32, `betas[:, 3] = 0` leaves the state after
token 3 within `1e-4` of the state after token 2, close to a skip but not exact, and `decays = 0`
leaves a state of norm `0.97` where the unedited prompt's is `2059`. A decode step does not clamp.

```python
mix = model.layers[0].linear_attn
with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0
    mix.betas = betas
    written = mix.betas.save()          # 0.001 at token 3, not 0
```

Writing back unchanged `betas` moves Nano-4B's bfloat16 logits by up to `0.25`.

## Attention has no position encoding

The attention blocks apply no rotary embedding: `attention_queries` is `q_proj`'s output split into
heads, unchanged, and the model has no position embedding. Order reaches the attention only
through the causal mask and the Mamba-2 blocks before it, whose recurrence and convolution are
causal.

```python
kinds = model.config.layer_types
attn = model.layers[kinds.index("full_attention")].self_attn     # block 12 on Nano-4B
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()
    q = attn.attention_queries.save()

b, s, _ = q_raw.shape
assert torch.equal(q_raw.view(b, s, -1, model.head_dim).transpose(1, 2), q)
```

Nano-4B has 40 query heads over 8 key/value heads of 128: query head `h` reads key/value head
`h // 5`, so an edit to `attention_values[:, j]` reaches query heads `5j` to `5j + 4`. The heads
are 5120 wide against a `hidden_size` of 3136, so `q_proj` widens and `o_proj` narrows. The query
scale is `128 ** -0.5`. Nemotron 3's 30B, Super and Ultra checkpoints have 2 key/value heads.

## Position 0 carries a very large norm

On Nano-4B the Mamba-2 mixer of block 6 adds a vector of norm about 770 at position 0 and about 5
elsewhere. Through the later blocks the stream at position 0 keeps a norm of 760 to 2000 while the
other positions grow from about 6 to about 1300, so the gap is widest in the middle (about 810
against 70 at block 20) and small in the last blocks. At block 20 four dimensions (357, 137, 63
and 105) hold 97% of position 0's squared norm. The four attention blocks put 0.54 to 0.82 of
their attention on position 0, averaged over heads and queries, on three test texts. The
tokenizer prepends nothing, so position 0 is the prompt's own first token, and the same happens
whatever that token is (`The`, `def`, `In`, `Paris` or `<s>` on five test texts). Leave it out before averaging activations
for steering vectors, mean ablation or probes.

```python
with model.trace(prompt):
    resid = model.layers[20].layer_output.save()

resid[0].float().norm(dim=-1)        # Nano-4B: about 810 at position 0, 70 elsewhere
```

## The MLP squares a ReLU and has no gate

An MLP block is `down_proj(relu(up_proj(x)) ** 2)`, `3136 → 12544 → 3136` on Nano-4B. A neuron is
exactly zero wherever its pre-activation is negative: on six short English and code texts
(position 0 left out), 79% to 98% of the 12544 activations at a token are zero, depending on the
block (fewest in blocks 15 and
18, most in blocks 1 and 3). A gradient through `relu(x) ** 2` is zero for every neuron that is
off, and scaling `up_proj.output` by `c` scales an active neuron by `c ** 2`.

```python
mlp = model.layers[1].mlp
with model.trace(prompt):
    pre = mlp.up_proj.output.save()
    acts = mlp.down_proj.input.save()

assert torch.equal(acts, torch.relu(pre) ** 2)
```

## The norms are RMSNorms with the stored gain

Every norm is an RMSNorm that normalizes in float32, casts back, and multiplies by `weight`, not
`1 + weight` (this family's norm is not Nemotron-4's `LayerNorm1P`). The stored gains are far
from one: the block norms of Nano-4B average 0.13 (block 1) to 2.65 (block 37), with 170 negative
entries among them, and the final norm averages 2.49, up to 8.4. Fold a norm into the next projection with `weight` as
stored.

## The mixture of experts on Nemotron 3

Nemotron 3 Nano 30B-A3B, Super 120B-A12B, Ultra 550B-A55B and Nemotron 3.5 Lightning 30B-A3B
replace the MLP blocks with mixture blocks (`moe` in `layer_types`), and their
`model.layers[i].mlp` is a `Moe`; the dense checkpoints, Nano-4B among them, have none, so the
diagram above draws no mixture. The router scores every expert by a sigmoid and picks the top
`num_experts_per_tok` by the sigmoid plus a selection bias (`e_score_correction_bias`);
`expert_weights` are the chosen sigmoids without the bias, renormalized and multiplied by
`routed_scaling_factor`, so they sum to it at every token: 128 experts, top 6 and 2.5 on the 30B
checkpoints, 512 experts, top 22 and 5.0 on Super and Ultra. Each expert is a squared-ReLU MLP
with no gate. One shared expert runs on every token after the routed ones, so read
`routed_output` before `shared_expert_output`:

```python
moe_blocks = [i for i, t in enumerate(model.config.layer_types) if t == "moe"]
moe = model.layers[moe_blocks[0]].mlp
with model.trace(prompt):
    weights = moe.expert_weights.save()       # [batch, seq, top_k]
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

torch.testing.assert_close(routed + shared, out)
```

Super and Ultra run the routed experts in a latent width (`moe_latent_size`, 1024 and 2048):
`fc1_latent_proj` projects each token down after the router has read it at full width, and
`fc2_latent_proj` projects the experts' sum back up. There `routed_output` is
`fc2_latent_proj`'s output, and `expert_outputs` is unavailable, since each slot's output is
latent-width. On the 30B checkpoints the latent projections are identities and `expert_outputs`
is served.

## The readout and the embeddings

The model casts the head's output to float32: `logits` is `lm_head.output.float()`, and
`project_on_vocab` applied to the last block's `layer_output` equals `logits` exactly.
`lm_head` has its own weight. The embedding is a plain lookup, so `token_embeddings` equals
`layers[0].input`.

## The tokenizer and the chat template

The vocabulary is a 131072-token byte-level BPE. No `<s>` (id 1) is prepended, `" Paris"` and
`"Paris"` are different tokens (6993 and 42572), and digits are split one per token. The chat
template uses `<|im_start|>` (id 10) and `<|im_end|>` (id 11, the tokenizer's end of sequence),
opens every conversation with a system turn, empty or not, and ends the generation prompt with
`<think>` and a newline, so the reply starts in a reasoning trace;
`enable_thinking=False` closes it at once (`<think></think>`). `skip_special_tokens=True` drops
`<|im_start|>` and `<|im_end|>` from decoded text and keeps `<think>` and `</think>`.

```python
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(prompt)
# <|im_start|>system
# <|im_end|>
# <|im_start|>user
# What is the capital of France?<|im_end|>
# <|im_start|>assistant
# <think>
```

## What this family module covers

Every checkpoint that loads as `NemotronHForCausalLM`: Nemotron-H 4B, 8B, 47B and 56B and Nemotron
Nano 2 9B and 12B, whose blocks are Mamba-2, attention and MLP blocks; Nemotron 3 Nano 4B, the
same dense kinds; and Nemotron 3 Nano 30B-A3B, Super, Ultra and Nemotron 3.5 Lightning, whose
blocks are Mamba-2, attention and mixture blocks. Nemotron-4, Minitron and Nemotron-Mini are the
`nemotron` family.
"""
