# Cross-family sweep

One trace body over several checkpoints: zero each block's mixer contribution and each block's MLP contribution in turn, record how far the next-token distribution moves, and tabulate the result per family. What the loop adds to a single-model script is a `support()` read, a guard on the MLP, and the mixer decided outside the trace. The trace body is the same on a GPT-2, a Pythia, a Llama and a Mamba.

## The loop

```python
import torch
import torch.nn.functional as F
import nnterp
from nnterp import StandardizedTransformer

REPOS = ["openai-community/gpt2", "EleutherAI/pythia-70m-deduped", "HuggingFaceTB/SmolLM2-135M", "state-spaces/mamba-130m-hf"]
prompt = "The Eiffel Tower is in the city of"


def mixer(layer):
    """The block's sequence mixer: self_attn, or linear_attn on a recurrent block."""
    attn = getattr(layer, "self_attn", None)
    return attn if attn is not None else layer.linear_attn


table = {}
for repo in REPOS:
    model = StandardizedTransformer(repo, dispatch=True, device="cpu", dtype=torch.float32, attn_implementation="eager")
    support = model.support()                                    # before any trace
    has_mlp = support.get("mlp.mlp_output", "absent") is None    # a module no block has (Mamba) has no key
    mixers = [mixer(layer) for layer in model.layers]            # decided outside the trace
    attention = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
    if len(attention) < model.num_layers:
        nnterp.route_kernels(model.family, "torch")              # a recurrent mixer: the torch kernels, before the first trace

    ablated = {}                                                 # containers made outside the trace
    with torch.no_grad(), model.trace() as tracer:
        with tracer.invoke(prompt):
            clean = model.logits[:, -1].log_softmax(-1).save()
        for i in range(model.num_layers):
            with tracer.invoke(prompt):
                mixers[i].attention_output[:] = 0
                ablated["mixer", i] = model.logits[:, -1].log_softmax(-1).save()
            if has_mlp:
                with tracer.invoke(prompt):
                    model.layers[i].mlp.mlp_output[:] = 0
                    ablated["mlp", i] = model.logits[:, -1].log_softmax(-1).save()

    entropy = {}
    with torch.no_grad(), model.trace(prompt):
        for i in attention:
            p = model.layers[i].self_attn.attention_probabilities
            entropy[i] = (-p * (p + 1e-12).log()).sum(-1).mean().save()

    kl = lambda logprobs: F.kl_div(logprobs, clean, log_target=True, reduction="sum")   # KL(clean || ablated)
    mixer_kl = torch.tensor([kl(ablated["mixer", i]) for i in range(model.num_layers)])
    mlp_kl = torch.tensor([kl(ablated["mlp", i]) for i in range(model.num_layers)]) if has_mlp else None
    table[model.family.__name__.split(".")[-1]] = {
        "blocks": f"{len(attention)}/{model.num_layers} attention",
        "top": model.tokenizer.decode(clean.argmax(-1)),
        "mixer": f"{float(mixer_kl.max()):.2f} nats at block {int(mixer_kl.argmax())}",
        "mlp": f"{float(mlp_kl.max()):.2f} nats at block {int(mlp_kl.argmax())}" if has_mlp else "no MLP",
        "entropy": f"{sum(float(e) for e in entropy.values()) / len(entropy):.2f}" if entropy else "-",
    }
    del model

print(f"{'family':9s} {'blocks':16s} {'top':8s} {'largest mixer ablation':24s} {'largest MLP ablation':24s} entropy")
for family, row in table.items():
    print(f"{family:9s} {row['blocks']:16s} {row['top']!r:8s} {row['mixer']:24s} {row['mlp']:24s} {row['entropy']}")
# family    blocks           top      largest mixer ablation   largest MLP ablation     entropy
# gpt2      12/12 attention  ' Paris' 3.04 nats at block 0     4.25 nats at block 0     0.63
# gpt_neox  6/6 attention    ' the'   8.42 nats at block 0     4.14 nats at block 1     0.61
# llama     30/30 attention  ' Paris' 5.97 nats at block 1     16.37 nats at block 0    0.60
# mamba     0/24 attention   ' Paris' 3.97 nats at block 0     no MLP                   -
```

`model.support()` returns, for every value, `None` when every block has it or `{block: reason}` where some do not. A module no block has (Mamba's `mlp`) has no key, which is why the guard uses `.get`. Guard on it rather than on `hasattr` inside the trace, which raises `nnterp.Unavailable` for an unavailable value ([Availability](../guide/values/availability.md)). A block has either `self_attn` or `linear_attn`, and the list comprehension outside the trace is the place to decide which. The KL is taken on log-probabilities because `next_token_probs` underflows to exact zeros, and `p * (p.log() - q.log())` is then NaN.

The loads are in float32 so that one dtype serves every checkpoint. Pythia's weights are stored in float16, and loaded at that dtype its KLs come back NaN. `attn_implementation="eager"` is what `attention_probabilities` needs on the attention families, and Mamba ignores it. `route_kernels` runs only where a block has no `self_attn`. A recurrent mixer's optimized kernels have no Python source and may not run on the CPU, and the torch kernels have to be bound before the family's first trace ([Mamba](../guide/architectures/mamba.md)).

## Reading the table

Zero ablation at block 0 or 1 moves the distribution by 3 to 8 nats for the mixer and 4 to 16 nats for the MLP on every checkpoint that has one. The stream leaving the first blocks is then unlike anything the later blocks were trained on, and these numbers measure the ablation more than the fact. The rows are not comparable to each other either, since vocabularies, depths and residual scales differ. Compare the shape of each checkpoint's curve (`mixer_kl` and `mlp_kl` hold it, one entry per block), or normalize per model. Pythia-70m does not predict `Paris`, and its row measures a different prediction. Mamba's row has no MLP column and no entropy, every block of it being one mixer.

## On another family

Add a repo id. `attention_output` is what a block's mixer adds to the stream whether the mixer is `self_attn` or `linear_attn`, `mlp_output` is the MLP's addition, and `mixer` picks the module per block, so a hybrid such as Qwen3.5 runs through the same body with its attention blocks alone in the entropy column. A `model_type` without a family loads with the best-effort `default` family and a warning, which serves `layer_output` but not `attention_output` or `mlp_output`, and its `support()` says so before the sweep reaches it.
