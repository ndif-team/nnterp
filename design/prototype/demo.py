"""PROTOTYPE demo -- evidence for design/internals-addressing.md.

Run:  python design/prototype/demo.py
Uses the CPU checkpoints nnterp's own test suite names (nnterp/tests/test_config.yaml):
``Maykeye/TinyLLama-v0`` and ``gpt2``, plus ``hf-internal-testing/tiny-random-MistralForCausalLM``
for the grouped-query case (num_kv_heads=2 != num_heads=4).
"""

import sys, os, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as th
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError

from internals import (
    INERT_CONTROL,
    InternalsOrderError,
    attach,
    num_kv_heads_of,
    head_dim_of,
    verify,
)

PROMPT = "The Eiffel Tower is in the city of"
LLAMA = "Maykeye/TinyLLama-v0"
GPT2 = "gpt2"
GQA = "hf-internal-testing/tiny-random-MistralForCausalLM"


def head(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def load(repo):
    return StandardizedTransformer(
        repo, device_map="cpu", enable_attention_probs=True, check_renaming=True
    )


def show(model, tag):
    head(f"{tag}: availability, decided before the first trace")
    reg = attach(model, extra={"probs_at_mixer_return": INERT_CONTROL})
    print(reg.status())
    print(
        f"  heads={model.num_heads}  kv_heads={num_kv_heads_of(model)}  "
        f"head_dim={head_dim_of(model)}"
    )
    return reg


def five_in_one_trace(model, reg, tag):
    head(f"{tag}: five internals in one trace, named in the WRONG forward order")
    names = [
        "attention_z",
        "attention_probabilities",
        "attention_queries",
        "mlp_neuron_outputs",
        "attention_queries_pre_rope",
    ]
    print("  asked for:", names)
    print("  forward order is:", reg.order_hint(names))

    # (a) the manual spelling, in the order the user wrote them -> refused, loudly
    try:
        with model.trace(PROMPT):
            for name in names:
                _ = reg[name][0]
        print("  [!!] manual out-of-order read did NOT raise")
    except InternalsOrderError as error:
        print("  [ok] manual out-of-order read refused immediately:")
        print("       " + str(error).replace("\n", "\n       ")[:400])

    # (b) the spelling a user should reach for -> sorted for them
    import nnsight

    with model.trace(PROMPT):
        got = nnsight.save({})
        for key, value in reg.read(0, *names).items():
            got[key] = value
    print("  [ok] model.internals.read(0, *names) sorted it:")
    for name in names:
        print(f"         {name:28s} {tuple(got[name].shape)}  {reg.rows[name].layout.describe()}")


def writes(model, reg, tag):
    head(f"{tag}: writes")
    with model.trace(PROMPT):
        base = model.logits.save()

    # a per-head write, in place on a view of the native tensor
    with model.trace(PROMPT):
        z = reg["attention_z"][0]
        reg["attention_z"][0, 1] = th.zeros_like(z[:, :, 1])
        edited = model.logits.save()
    print(f"  [ok] attention_z[0, head=1] = 0 moved the logits: "
          f"{not th.allclose(base, edited)}  (max delta {(base - edited).abs().max():.4g})")

    # a write to one logical tensor that shares a native tensor with others:
    # on gpt2 q/k/v are three column blocks of one c_attn output, so the write
    # must land in its own columns and leave the others alone.
    name = "attention_queries_pre_rope"
    with model.trace(PROMPT):
        clean_v = reg["attention_values"][0].clone().save()
    with model.trace(PROMPT):
        reg[name][0] = th.zeros_like(reg[name][0])
        after_v = reg["attention_values"][0].clone().save()
        fused_edit = model.logits.save()
    fused = reg.rows[name].layout.fused_axis is not None
    print(f"  [ok] {name}[0] = 0 moved the logits: {not th.allclose(base, fused_edit)}; "
          f"attention_values unchanged by it: {th.allclose(clean_v, after_v)} "
          f"(shares a native tensor: {fused})")

    # the head bound comes from the component's own head space
    kv = num_kv_heads_of(model)
    if kv is not None and model.num_heads is not None and kv < model.num_heads:
        try:
            with model.trace(PROMPT):
                _ = reg["attention_keys"][0, model.num_heads - 1]
            print("  [!!] a query-space head index over a KV-space component did not raise")
        except RenamingError as error:
            print(f"  [ok] GQA: attention_keys[0, {model.num_heads - 1}] refused: {str(error)[:150]}")


def load_time_check(model, reg, tag):
    head(f"{tag}: the load-time / CI check, and the inert-write trap")
    import time

    t0 = time.perf_counter()
    structural = verify(reg, PROMPT, causal=False)
    t_struct = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = verify(reg, PROMPT, causal=True)
    t_causal = time.perf_counter() - t0
    for name in sorted(report):
        row = report[name]
        flag = "ok " if row["shape_ok"] and row.get("causal") is not False else "!! "
        print(f"  [{flag}] {name:28s} shape {str(row['shape']):22s} "
              f"declared {row['declared']}  causal={row['causal']}")
    dead = [n for n, r in report.items() if r["causal"] is False]
    print(f"  inert addresses caught: {dead}")
    print(f"  cost: structural (1 trace, all rows) {t_struct:.2f}s; "
          f"+ causal (1 trace per row, {len(report)} rows) {t_causal:.2f}s")
    return report


def act_vs_neuron(model, reg, tag):
    head(f"{tag}: mlp_act_fn_output vs mlp_neuron_outputs -- same tensor or not?")
    import nnsight

    with model.trace(PROMPT):
        got = nnsight.save({})
        for key, value in reg.read(0, "mlp_act_fn_output", "mlp_neuron_outputs").items():
            got[key] = value.clone()
    same = th.allclose(got["mlp_act_fn_output"], got["mlp_neuron_outputs"])
    print(f"  act_fn output {tuple(got['mlp_act_fn_output'].shape)} vs "
          f"neuron output {tuple(got['mlp_neuron_outputs'].shape)}: identical = {same}")


def remote_local(model, reg, tag):
    head(f"{tag}: remote='local' -- the accessor shipped by value")
    with model.trace(PROMPT, remote="local"):
        p = reg["attention_probabilities"][0].save()
        z = reg["attention_z"][0].save()
    print(f"  [ok] read under remote='local': attention_z {tuple(z.shape)}, "
          f"probs {tuple(p.shape)}")
    with model.trace(PROMPT):
        base = model.logits.save()
    with model.trace(PROMPT, remote="local"):
        z = reg["attention_z"][0]
        reg["attention_z"][0, 1] = th.zeros_like(z[:, :, 1])
        edited = model.logits.save()
    print(f"  [ok] write under remote='local' propagated: {not th.allclose(base, edited)}")


def pickles(reg, tag):
    import cloudpickle, pickle

    head(f"{tag}: serialization")
    import internals
    import nnterp.rename_utils as ru
    from internals import InternalAccessor, InternalsRegistry, Address, Layout

    payload = (InternalAccessor, InternalsRegistry, Address, Layout, internals.LLAMA_TREE)
    for label, mods in (("by reference", ()), ("by VALUE", (internals, ru))):
        for mod in mods:
            cloudpickle.register_pickle_by_value(mod)
        try:
            n = len(cloudpickle.dumps(payload))
            print(f"  [ok] accessor classes + table + layouts {label}: {n} bytes")
        except Exception as e:
            print(f"  [!!] accessor classes + table + layouts {label}: {type(e).__name__}: {e}")
        for mod in mods:
            cloudpickle.unregister_pickle_by_value(mod)
    print("     (an `eproperty` on any of these classes fails the by-VALUE column with")
    print("      TypeError: cannot pickle 'eproperty' object -- and nnsight.ndif.pull_env()")
    print("      registers every working-tree module by value before the first request)")


def match_op_demo():
    head("match_op: what a drifted forward looks like")
    from internals import match_op, AddressResolutionError

    names = ["attention_interface_0", "attention_interface_1", "attn_weights_1"]
    lines = {
        "attention_interface_0": "    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(",
        "attention_interface_1": "        attn_output, attn_weights = attention_interface(",
        "attn_weights_1": "        attn_weights = attn_weights + attention_mask",
    }
    print("  assigned-then-called ->", match_op("attention_interface", names, lines.get))
    try:
        match_op("nn_functional_softmax", names, lines.get)
    except AddressResolutionError as e:
        print("  a moved forward ->", str(e)[:200])


def main():
    for repo in (LLAMA, GPT2):
        tag = repo
        model = load(repo)
        reg = show(model, tag)
        five_in_one_trace(model, reg, tag)
        writes(model, reg, tag)
        act_vs_neuron(model, reg, tag)
        load_time_check(model, reg, tag)
        if repo == LLAMA:
            remote_local(model, reg, tag)
            pickles(reg, tag)
        del model

    model = load(GQA)
    reg = show(model, GQA)
    writes(model, reg, GQA)
    match_op_demo()


if __name__ == "__main__":
    main()
