"""One rank of `test_tensor_parallel.py`, run under ``torch.distributed.run`` (or alone, for the reference).

Not a test module: the test launches it once in one process and once across
``--tp`` CPU ranks over gloo, and compares what each wrote. Every rank runs the
same block, as a sharded model requires; nothing here branches on rank.

    python -m torch.distributed.run --nproc_per_node=2 tests/tp_worker.py --tp 2 --repo REPO --out DIR
"""

from __future__ import annotations

import argparse
import os

import nnsight  # noqa: F401  nnsight before any transformers submodule
import torch

PROMPT = "The Eiffel Tower is in the city of"
LAYER = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from nnterp import StandardizedTransformer

    kwargs = {"dispatch": True, "dtype": torch.float32, "attn_implementation": "eager"}
    if args.tp > 1:
        from transformers.distributed import DistributedConfig

        kwargs["distributed_config"] = DistributedConfig(tp_size=args.tp)
    else:
        kwargs["device"] = "cpu"
    model = StandardizedTransformer(args.repo, **kwargs)
    attn, mlp = model.layers[LAYER].self_attn, model.layers[LAYER].mlp
    # Edits aim at the last head, the last key/value head and the neurons around the middle: the last rank's,
    # and the boundary between the ranks', where a value handed over as one rank's piece goes wrong.
    heads, kv_heads, neurons = attn.num_heads, attn.num_kv_heads, mlp.intermediate_size
    scale = torch.arange(1, heads + 1, dtype=torch.float32).view(1, heads, 1, 1)
    out = {}

    with torch.no_grad():
        with model.trace(PROMPT):
            out["attention_queries_prerope"] = attn.attention_queries_prerope.save()
            out["attention_keys_prerope"] = attn.attention_keys_prerope.save()
            out["attention_queries"] = attn.attention_queries.save()
            out["attention_keys"] = attn.attention_keys.save()
            out["attention_values"] = attn.attention_values.save()
            heads_again, pattern_again = attn.interface(out["attention_queries"], out["attention_keys"], out["attention_values"])
            out["interface_head_outputs"], out["interface_probabilities"] = heads_again.save(), pattern_again.save()
            out["attention_scores"] = attn.attention_scores.save()
            out["attention_probabilities"] = attn.attention_probabilities.save()
            out["attention_head_outputs"] = attn.attention_head_outputs.save()
            out["attention_premix"] = attn.attention_premix.save()
            out["attention_output"] = attn.attention_output.save()
            out["mlp_activation"] = mlp.mlp_activation.save()
            out["mlp_neurons"] = mlp.mlp_neurons.save()
            out["mlp_output"] = mlp.mlp_output.save()
            out["logits"] = model.logits.save()
        positions = torch.arange(out["attention_queries_prerope"].shape[2])
        out["rotated_queries"] = attn.rotate(out["attention_queries_prerope"], positions)

        with model.trace(PROMPT):
            attn.attention_head_outputs[:, :, heads - 1] = 0
            out["head_outputs_zeroed_in_place"] = model.logits.save()

        with model.trace(PROMPT):
            attn.attention_keys[:, kv_heads - 1] *= 2
            out["keys_scaled_in_place"] = model.logits.save()

        with model.trace(PROMPT):
            attn.attention_queries = attn.attention_queries * scale
            out["queries_scaled_by_assignment"] = model.logits.save()

        with model.trace(PROMPT):
            queries = attn.attention_queries_prerope.clone()
            queries[:, :, -1] = queries[:, :, 0]  # the first query, rotated for the last position
            attn.attention_queries_prerope = queries
            out["queries_prerope_moved"] = model.logits.save()

        with model.trace(PROMPT):
            values = attn.attention_values.clone()
            values[:, kv_heads - 1] += 1
            heads_patched, _ = attn.interface(attn.attention_queries, attn.attention_keys, values)
            attn.attention_head_outputs[:, -1, heads - 1] = heads_patched[:, -1, heads - 1]
            out["head_patched_through_interface"] = model.logits.save()

        with model.trace(PROMPT):
            activation = mlp.mlp_activation.clone()
            activation[..., neurons // 2 - 2 : neurons // 2 + 2] = 0
            mlp.mlp_activation = activation
            out["activation_assigned"] = model.logits.save()

    # With gradients on, a gathered value is a clone, which is what the in-place edit lands on.
    with model.trace(PROMPT):
        attn.attention_head_outputs[:, :, heads - 1] = 0
        out["head_outputs_zeroed_with_grad"] = model.logits.save()

    rank = int(os.environ.get("RANK", 0))
    torch.save({name: value.detach().cpu() for name, value in out.items()}, os.path.join(args.out, f"rank{rank}.pt"))


if __name__ == "__main__":
    main()
