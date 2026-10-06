"""Build the nnterp encyclopedia: one page per family entry and a searchable index.

    PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py            # every entry
    PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py gemma2     # one entry

A page merges two sources. What nnterp itself knows comes from the code: the
family module is imported, its reference checkpoint is built on the ``meta``
device (config only, no weights), and the page reads ``support()`` under an
eager load and under the default one, every standard value's key, layout and
description, the sizes, the aliases and the native paths. What a person knows
comes from ``entries/<model_type>.py``: the title and subtitle, the block schema
the visualization draws, the quirk tags, the palette and the notes. The output
is static HTML under ``site/``; ``static/`` is copied beside it.

Every page colours five roles, attention, the MLP, the norms, the residual
stream and the family's mark, from a palette ``palette.py`` generates for the
family, and the same role takes the same colour in the diagram, the ledgers,
the printout and the highlighted code.
"""

from __future__ import annotations

import datetime as dt
import html
import importlib
import inspect
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import markdown  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402
from markupsafe import Markup  # noqa: E402
from pygments import highlight  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402
from pygments.lexers import PythonLexer  # noqa: E402

import nnterp  # noqa: E402  (import nnterp before any transformers.models module)
from nnterp import StandardizedTransformer  # noqa: E402
from nnterp.components import Moe, RecurrentMixer, Standard  # noqa: E402
from nnterp.components.moe import SHARED_NAMES  # noqa: E402
from nnterp.components.standard import values as class_values  # noqa: E402

import entries  # noqa: E402
import palette as palettes  # noqa: E402

GITHUB = "https://github.com/ndif-team/nnterp/blob/0.8-refactor"

#: The design's ink and its default paper; an entry may tint the paper.
INK = "#3A2516"
PAPER = "#F1E6CB"

#: The role each host's values take, and the role of the names that are not values.
#: A sequence mixer is attention whether it is self_attn or linear_attn; the root and
#: the block belong to the stream.
HOST_ROLES = {"root": "stream", "layer": "stream", "self_attn": "attention", "linear_attn": "attention", "mlp": "mlp"}
NAME_ROLES = {
    "model": "stream", "layers": "stream", "embed_tokens": "stream", "lm_head": "stream", "logits": "stream",
    "norm": "norm", "self_attn": "attention", "linear_attn": "attention", "mlp": "mlp",
}

#: Quirk slugs an entry may carry, with the label and one line the index and the page show.
#: They follow the themes of docs/reference/families.md.
QUIRKS: dict[str, tuple[str, str]] = {
    "tuple-blocks": ("Tuple blocks", "The block returns (hidden_states, ...); layer_output is the first element."),
    "sandwich-norms": ("Sandwich norms", "Each sublayer is normed before and after; the stream receives the post-norm's output."),
    "post-norms": ("Post-norms only", "Each sublayer is normed after, not before: it reads the raw stream, and the stream receives the post-norm's output."),
    "residual-inside-module": ("Residual inside the module", "A sublayer adds the residual itself; the contribution is the tensor before the add."),
    "scaled-residual-adds": ("Scaled residual adds", "A multiplier sits between a sublayer and the stream, or scales the whole block."),
    "hyper-connections": ("Hyper-connections", "Several parallel residual streams; layer_output is [batch, seq, streams, hidden]."),
    "own-attention-arithmetic": ("Own attention arithmetic", "No shared attention interface; the interior is mapped onto the family's own ops."),
    "attention-sink": ("Attention sink", "A learned logit joins the softmax; pattern rows sum to less than one."),
    "latent-attention": ("Latent attention", "Queries and keys are qk_head_dim wide, values head_dim wide."),
    "sparse-attention": ("Sparse attention", "An indexer keeps the top keys per query; the pattern is sparse but served dense."),
    "mixture-of-experts": ("Mixture of experts", "layers[i].mlp is a Moe: router logits, expert weights and indices, expert outputs."),
    "borrowed-kv": ("Borrowed keys and values", "Later blocks attend with an earlier block's keys and values."),
    "gated-query": ("Gated query", "q_proj produces the query and a gate side by side."),
    "qkv-bias": ("Biased q, k, v", "The query, key and value projections add a bias, so each is W·x + b, not W·x."),
    "qk-norm": ("Query/key norms", "q_norm and k_norm normalize the projected queries and keys inside the attention; attention_queries and attention_keys are read after them."),
    "parallel-blocks": ("Parallel block", "Attention and MLP both read the block input, through one norm or two; the block sums x + attn + mlp."),
    "partial-rotary": ("Partial rotary", "Rotary embeddings turn only a leading fraction of each query and key head; the other dimensions carry no position."),
    "nope-blocks": ("Blocks without rotary", "Some blocks, or all, apply no rotary embedding (NoPE): there attention_queries and attention_keys carry no position, and only the causal mask orders the tokens."),
    "interleaved-rotary": ("Interleaved rotary", "Rotary turns adjacent pairs of dimensions (2i, 2i + 1) rather than i with i + rot/2 (rotate_half), so query and key dimensions are ordered differently from a rotate_half family's."),
    "hybrid": ("Hybrid", "Some blocks carry linear_attn (a recurrent mixer), others self_attn."),
    "mamba1": ("Selective scan (Mamba-1)", "The mixer is a selective scan; C/B/x as queries/keys/values."),
    "mamba2": ("State space (Mamba-2)", "The mixer is an SSD state-space block with a per-chunk state."),
    "fp32-residual": ("Float32 residual", "The blocks add in float32 (residual_in_fp32), so layer_output is float32 whatever the load dtype."),
    "no-mlp": ("No MLP module", "fc1/fc2 sit on the block, so layers[i].mlp does not exist."),
    "softcapped-logits": ("Softcapped logits", "logits is tanh-capped after lm_head; project_on_vocab applies the cap."),
    "scaled-logits": ("Scaled logits", "The head's output is multiplied or divided by a config scale."),
    "sliding-window": ("Sliding window", "Some blocks attend over a window; config.layer_types says which."),
    "scaled-embeddings": ("Scaled embeddings", "The embedding output is multiplied before block 0; token_embeddings is the scaled tensor."),
    "embedding-multiplier": ("Embedding multiplier", "The model multiplies the embedding module's output before block 0; token_embeddings is the unscaled tensor, layers[0].input the scaled one."),
    "gain-norm": ("1 + weight norm gain", "The norm multiplies by (1 + weight), so the gain is not norm.weight."),
    "position-embeddings": ("Position embeddings", "A position embedding is added after embed_tokens; token_embeddings is not layers[0].input."),
    "layernorm": ("LayerNorm", "Norms subtract the mean before scaling; a shift along the all-ones direction never reaches the next sublayer or the logits."),
    "squared-relu": ("Squared ReLU", "The MLP's activation is relu(x)²: a neuron is exactly zero wherever its pre-activation is negative, and grows with its square elsewhere."),
    "fused-qkv": ("Fused QKV", "One projection yields queries, keys and values together, in the family's own layout; split its output by that layout before reading a head."),
    "dense-first-blocks": ("Dense first blocks", "The first blocks have a dense MLP and the rest a mixture of experts, so the mixture's values are missing on the first blocks."),
}

ROOT_NAMES = ("embed_tokens", "layers", "norm", "lm_head")
SIZE_NAMES = ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "qk_head_dim", "vocab_size", "intermediate_size")
CONFIG_KEYS = (
    "architectures", "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "sliding_window", "hidden_act",
    "hidden_activation", "rms_norm_eps", "layer_norm_eps", "rope_theta", "query_pre_attn_scalar",
    "attn_logit_softcapping", "final_logit_softcapping", "tie_word_embeddings", "dtype",
    "use_parallel_residual",
    "n_layer", "n_embd", "n_head", "n_inner", "n_positions", "activation_function", "layer_norm_epsilon",
    "scale_attn_by_inverse_layer_idx", "reorder_and_upcast_attn",
    "rotary_dim",
    "embedding_multiplier", "residual_multiplier", "attention_multiplier", "logits_scaling",
    "norm_eps", "partial_rotary_factor",
    "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
    "linear_num_heads", "linear_head_dim", "linear_conv_kernel_dim",
    "first_k_dense_replace", "moe_intermediate_size", "num_experts", "n_routed_experts", "num_experts_per_token",
    "num_experts_per_tok", "num_shared_experts", "n_shared_experts", "routed_scaling_factor",
    "state_size", "expand", "conv_kernel", "residual_in_fp32",
)

STRUCTURAL = re.compile(r"^no \w+ (module|value) on this block")
VALUE_LINE = re.compile(r"^\((?P<name>\w+)\)(?: -> (?P<layout>\w+(?: \| None)?) \[(?P<dims>[^\]]*)\])?: (?P<desc>.*)$")


# -- introspection ----------------------------------------------------------------

def value_rows(host: Any, expr: str) -> list[dict[str, Any]]:
    """Every standard value on ``host`` (an envoy or the root), as the page shows it."""
    found = host.values() if isinstance(host, Standard) else class_values(type(host))
    rows = []
    for name, value in found.items():
        match = VALUE_LINE.match(str(value))
        assert match, str(value)
        key, select = getattr(value, "key", None), getattr(value, "select", None)
        rows.append({
            "name": name,
            "expr": f"{expr}.{name}",
            "layout": match["layout"],
            "dims": match["dims"],
            "description": match["desc"],
            "key": key if isinstance(key, str) else ("computed" if key is not None else None),
            "select": select,
            "where": humanize_key(key, select),
        })
    return rows


def humanize_key(key: Any, select: int | None) -> str:
    """Where a value is read, in words: the key is a path from the host envoy."""
    if not isinstance(key, str):
        return "computed from several served values" if key is not None else "derived"
    if callable(select):
        # a select chosen per call (a Mamba-1 kernel argument sits at a different position in each kernel)
        name = getattr(select, "__name__", "")
        element = f", argument `{name.split('.', 1)[1]}`" if name.startswith("argument.") else ", the element this call's kernel takes"
    else:
        element = "" if select is None else f", argument `{select}`" if isinstance(select, str) else f", element {select}"
    if key.startswith("<"):
        # a location the value finds per call: a recurrent mixer's kernel, whichever fires on this call
        name = key[1:-1]
        if name.startswith("kernel."):
            what = {"inputs": "the arguments of", "output": "the output of"}.get(name.split(".", 1)[1], name)
            return f"inside the forward: {what} the kernel this call runs (a prompt's chunked one or a decode step's recurrent one){element}"
        return f"inside the forward, at the operation `{name}` finds on each call"
    if key == "output":
        return "the module's own output" + element
    if key in ("input", "inputs"):
        return "the module's own input" + element
    if key.startswith("source."):
        ops = key.split(".")
        chain, what = [], ops[-1]
        for i in range(1, len(ops) - 1):
            if ops[i] != "source":
                chain.append(ops[i])
        inside = " inside ".join(f"`{op}`" for op in reversed(chain))
        what = {"inputs": "the arguments of", "input": "the input of", "output": "the output of"}.get(what, what + " of")
        return f"inside the forward: {what} {inside}{element}"
    if ".source." in key and not key.startswith("../"):
        module, inside = key.split(".source.", 1)
        return humanize_key("source." + inside, select).replace("inside the forward", f"inside `{module}`'s forward", 1)
    if key.startswith("../"):
        module, attr = key[3:].rsplit(".", 1)
        return f"the sibling module `{module}`'s {attr}{element}"
    module, attr = key.rsplit(".", 1)
    return f"the child module `{module}`'s {attr}{element}"


def summarize_reason(reason: Any, num_layers: int) -> str | None:
    """A support() entry as one line: None, a string, or a per-block dict."""
    if reason is None:
        return None
    if isinstance(reason, str):
        return reason
    # A block without the value's host (a hybrid's other mixer, a dense block's mixture) is the
    # block's shape, which the diagram draws, not a condition on the load.
    reason = {i: why for i, why in reason.items() if not STRUCTURAL.match(why)}
    if not reason:
        return None
    reasons = sorted(set(reason.values()))
    blocks = sorted(reason)
    where = "every block" if len(blocks) == num_layers else f"blocks {blocks}"
    return f"{where}: " + " / ".join(reasons)


def introspect(entry: ModuleType, reference: str | None = None) -> dict[str, Any]:
    """What nnterp knows about the entry's family, read off a meta build of ``reference``."""
    reference = reference or entry.REFERENCE
    load = getattr(entry, "load", StandardizedTransformer)
    eager = load(reference, attn_implementation="eager")
    default = load(reference)
    family = eager.family
    assert family.__name__.rsplit(".", 1)[1] == entry.MODEL_TYPE, (family.__name__, entry.MODEL_TYPE)
    num_layers = eager.num_layers
    block = eager.layers[0]

    # Each block's standard children, and whether each is a mixture; one block of each
    # combination of child classes stands for the rest, so a hybrid's ledgers list every host.
    block_hosts, shapes = [], {}
    for layer in eager.layers:
        found = StandardizedTransformer._standard_children(layer)
        block_hosts.append({alias: isinstance(child, Moe) for alias, child in found.items()})
        shapes.setdefault(tuple((alias, type(child._module)) for alias, child in found.items()), (layer, found))
    hosts_found: dict[str, list[tuple[int, Any]]] = {}
    for layer, found in shapes.values():
        natives = list(layer._module.children())
        for alias, child in found.items():
            seen = hosts_found.setdefault(alias, [])
            if all(type(c._module) is not type(child._module) for _, c in seen):
                position = next((k for k, m in enumerate(natives) if m is child._module), len(natives))
                seen.append((position, child))
    # in the block's native order: a hybrid's two mixers sit where its one native mixer does
    order = sorted(hosts_found, key=lambda alias: hosts_found[alias][0][0])
    children = {alias: [child for _, child in hosts_found[alias]] for alias in order}

    support_eager = eager.support()
    support_default = default.support()
    default_impl = default.config._attn_implementation
    support = []
    for name in support_eager:
        under_eager = summarize_reason(support_eager[name], num_layers)
        under_default = summarize_reason(support_default.get(name), num_layers)
        if under_eager is None and under_default is not None:
            condition = {"kind": "eager", "reason": under_default}
        elif under_eager is not None:
            condition = {"kind": "other", "reason": under_eager}
        else:
            condition = None
        support.append({"name": name, "condition": condition})
    conditions = {row["name"]: row["condition"] for row in support}

    hosts = [("root", "model", [eager]), ("layer", "model.layers[i]", [block])]
    hosts += [(alias, f"model.layers[i].{alias}", found) for alias, found in children.items()]
    values: dict[str, list[dict[str, Any]]] = {}
    for alias, expr, found in hosts:
        # a host of several classes (a dense MLP and a mixture) lists the union, the fullest class's order first
        merged: dict[str, dict[str, Any]] = {}
        for host in sorted(found, key=lambda h: -len(value_rows(h, expr))):
            for row in value_rows(host, expr):
                row["module"] = type(host._module).__name__ if alias != "root" else type(eager._module).__name__
                merged.setdefault(row["name"], row)
        prefix = "" if alias in ("root", "layer") else alias + "."
        for row in merged.values():
            row["condition"] = conditions.get(prefix + row["name"])
            row["host"] = alias
        values[alias] = list(merged.values())

    moe = next((child for found in children.values() for child in found if isinstance(child, Moe)), None)
    mixer = next((child for found in children.values() for child in found if isinstance(child, RecurrentMixer)), None)

    paths = []
    for name in ROOT_NAMES:
        paths.append((f"model.{name}", eager.get(name).path))
    paths.append(("model.layers[i]", eager.get("layers.0").path))
    for name, child in block._named_children():
        alias = next((a for a, found in children.items() if any(c is child for c in found)), None)
        shown = alias or name
        paths.append((f"model.layers[i].{shown}", child.path))

    text_config = eager.config.get_text_config()
    config = text_config.to_dict()
    config_rows = [(key, config[key]) for key in CONFIG_KEYS if key in config and config[key] is not None]
    layer_types = config.get("layer_types")

    doc = inspect.getdoc(family) or ""
    return {
        "family_module": family.__name__,
        "family_file": f"nnterp/families/{entry.MODEL_TYPE}.py",
        "architecture": type(eager._module).__name__,
        "module_classes": {alias: [type(child._module).__name__ for child in found] for alias, found in children.items()},
        "host_classes": {alias: [(type(child._module).__name__, isinstance(child, Moe), len(child.values())) for child in found]
                         for alias, found in children.items()},
        "block_hosts": block_hosts,
        "moe": moe_sizes(moe),
        "mixer": mixer_kernels(mixer),
        "block_class": type(block._module).__name__,
        "returns_tuple": type(block).returns_tuple,
        "reference": reference,
        "default_impl": default_impl,
        "num_layers": num_layers,
        "layer_types": layer_types,
        # a size the family has nothing to read for (no attention heads on a pure state-space model) is left out
        "sizes": [(name, getattr(eager, name)) for name in SIZE_NAMES if getattr(eager, name, None) is not None],
        "config": config_rows,
        "rename": list(family.RENAME.items()),
        "paths": paths,
        "values": values,
        "support": support,
        "repr": root_printout(eager),
        "docstring": doc,
        "children": list(children),
        "versions": {"nnterp": nnterp.__version__, "transformers": importlib.import_module("transformers").__version__},
    }


def moe_sizes(moe: Any) -> dict[str, Any] | None:
    """A mixture's sizes, its scoring and its parts' classes, off the first `Moe` the blocks have."""
    if moe is None:
        return None
    modules = moe._module._modules
    router = next((modules[name] for name in ("router", "gate") if modules.get(name) is not None), None)
    shared = next((modules[name] for name in SHARED_NAMES if modules.get(name) is not None), None)
    return {
        "num_experts": moe.num_experts, "top_k": moe.top_k, "scoring": type(moe).SCORING,
        "router": type(router).__name__ if router is not None else None,
        "experts": type(modules["experts"]).__name__ if modules.get("experts") is not None else None,
        "shared": type(shared).__name__ if shared is not None else None,
    }


def mixer_kernels(mixer: Any) -> dict[str, str] | None:
    """A recurrent mixer's two kernels, by the functions' names: the one a prompt runs and the one a decode step runs."""
    if mixer is None:
        return None
    cls = type(mixer)
    return {"chunk": re.sub(r"_\d+$", "", cls.CHUNK_KERNEL), "recurrent": re.sub(r"_\d+$", "", cls.RECURRENT_KERNEL)}


# -- the block schema ----------------------------------------------------------------

def node(eyebrow: str, expr: str, desc: str, *, layout: str | None = None, dims: str | None = None,
         where: str | None = None, condition: dict | None = None, extra: str | None = None) -> dict[str, Any]:
    return {"eyebrow": eyebrow, "expr": expr, "desc": desc, "layout": layout, "dims": dims, "where": where,
            "condition": condition, "extra": extra}


def value_node(row: dict[str, Any], eyebrow: str) -> dict[str, Any]:
    return node(eyebrow, row["expr"], row["description"], layout=row["layout"], dims=row["dims"], where=row["where"],
                condition=row["condition"])


#: The sublayer kinds a BLOCK may draw. A "moe" sublayer is drawn on the blocks whose host is a
#: `Moe` and an "mlp" one on the blocks whose host is not, so a family with dense first blocks
#: lists both; every kind is drawn only on the blocks that have its host.
KINDS = ("attention", "mixer", "mlp", "moe")
#: Where each of a mixture's values is drawn: in the router's box, the experts' or the shared expert's.
MOE_PARTS = {
    "router_logits": "router", "expert_weights": "router", "expert_indices": "router",
    "expert_outputs": "experts", "routed_output": "experts", "shared_expert_output": "shared",
}


def drawn(specs: list[dict[str, Any]], hosts: dict[str, bool]) -> tuple[int, ...]:
    """The sublayers one block draws, as indices into ``specs``: ``hosts`` maps each of the block's
    standard children to whether it is a mixture."""
    return tuple(k for k, spec in enumerate(specs) if spec["host"] in hosts
                 and (spec["kind"] not in ("mlp", "moe") or hosts[spec["host"]] == (spec["kind"] == "moe")))


def block_schema(entry: ModuleType, info: dict[str, Any]) -> dict[str, Any]:
    """The entry's BLOCK, checked against the family and enriched with every node's hover card.

    The sublayers are listed once, in forward order; each block draws the ones its children
    match (`drawn`), and the distinct combinations are the block's shapes. A family with one
    shape gets the schema as it always has; a hybrid gets ``shapes`` and ``shape_of`` too, and
    the diagram redraws when the slider crosses into another shape."""
    by_host = {alias: {row["name"]: row for row in rows} for alias, rows in info["values"].items()}
    sizes = dict(info["sizes"])
    moe = info["moe"] or {}
    fmt = {**sizes, **{k: v for k, v in info["config"]}, **{k: moe[k] for k in ("num_experts", "top_k") if k in moe}}
    specs = entry.BLOCK["sublayers"]
    hosts = [s["host"] for s in specs]
    # A host drawn by two sublayers (a dense MLP and a mixture) keys its nodes by kind as well.
    keys = [s["host"] if hosts.count(s["host"]) == 1 else f"{s['host']}-{s['kind']}" for s in specs]
    nodes: dict[str, dict[str, Any]] = {}
    sublayers = []
    for k, spec in enumerate(specs):
        host, key, kind = spec["host"], keys[k], spec["kind"]
        assert host in by_host, f"{entry.MODEL_TYPE}: BLOCK names host {host!r}; the block has {list(by_host)}"
        assert kind in KINDS, f"{entry.MODEL_TYPE}: kind {kind!r}; known: {KINDS}"
        contribution = by_host[host][spec["contribution"]]
        sub = {
            "host": host, "kind": kind, "label": spec["label"],
            "detail": spec.get("detail", "").format(**fmt),
            "variants": {k2: v.format(**fmt) for k2, v in spec.get("variants", {}).items()},
            "pre_norm": spec.get("pre_norm"), "post_norm": spec.get("post_norm"),
            "contribution": spec["contribution"], "interior": [],
        }
        if key != host:
            sub["key"] = key
        matched = [(c, n) for c, is_moe, n in info["host_classes"][host] if kind not in ("mlp", "moe") or is_moe == (kind == "moe")]
        classes, count = [c for c, _ in matched], max(n for _, n in matched)
        extra = f"{count} standard value{'s' if count != 1 else ''}; `.input` is what the sublayer reads."
        if kind == "mixer" and info["mixer"]:
            kernels = info["mixer"]
            extra += f" A prompt runs `{kernels['chunk']}`, a decode step `{kernels['recurrent']}`; the values are read at that call."
            routed = [row["name"] for row in info["support"] if row["name"].startswith(host + ".") and row["condition"]
                      and "route_kernels" in row["condition"]["reason"]]
            if routed:
                extra += f" `{'`, `'.join(name.split('.', 1)[1] for name in routed)}` need `nnterp.route_kernels(model.family, \"torch\")` before the first trace."
        nodes[f"sub.{key}"] = node(
            spec["label"], f"model.layers[i].{host}",
            f"{' / '.join(classes)} under its standard name. " + spec.get("detail", "").format(**fmt), extra=extra)
        for name in spec.get("interior", []):
            assert name in by_host[host], f"{entry.MODEL_TYPE}: {host} has no value {name!r}"
            chip = {"name": name, "short": INTERIOR_SHORT.get(name, name)}
            if kind == "moe":
                assert name in MOE_PARTS, f"{entry.MODEL_TYPE}: {name!r} is not a mixture's value; a moe sublayer draws {list(MOE_PARTS)}"
                chip["part"] = MOE_PARTS[name]
            sub["interior"].append(chip)
            nodes[f"interior.{key}.{name}"] = value_node(by_host[host][name], "inside the sublayer")
        if kind == "moe":
            parts = {chip["part"] for chip in sub["interior"]}
            assert "shared" not in parts or moe.get("shared"), f"{entry.MODEL_TYPE}: the mixture has no shared expert to draw"
            sub["moe"] = {"num_experts": moe["num_experts"], "top_k": moe["top_k"], "scoring": moe["scoring"]}
            expr = f"model.layers[i].{host}"
            if "router" in parts:
                nodes[f"moe.{key}.router"] = node(
                    "the router", f"{expr}.router",
                    f"`{moe['router']}`. It scores all {moe['num_experts']} experts for each token by `{moe['scoring']}` and picks "
                    f"{moe['top_k']}: `expert_indices`, each weighted by its `expert_weights`.")
            if "experts" in parts:
                nodes[f"moe.{key}.experts"] = node(
                    "the routed experts", f"{expr}.experts",
                    f"`{moe['experts']}`, {moe['num_experts']} experts. Each token runs through the {moe['top_k']} it is routed to; "
                    "`expert_outputs` holds each slot's weighted output and `routed_output` their sum.")
            if "shared" in parts:
                nodes[f"moe.{key}.shared"] = node(
                    "the shared expert", f"{expr}.shared_experts",
                    f"`{moe['shared']}`. Every token runs through it; `shared_expert_output` is what it adds beside `routed_output`.")
        nodes[f"contrib.{key}"] = value_node(contribution, "contribution")
        if spec.get("pre_norm"):
            nodes[f"norm.{spec['pre_norm']}"] = node(
                "pre-norm", f"model.layers[i].{spec['pre_norm']}",
                f"A native module under its own name. Its output is what `{host}` reads: `model.layers[i].{host}.input`.",
                extra=spec.get("pre_norm_note"))
        if spec.get("post_norm"):
            nodes[f"norm.{spec['post_norm']}"] = node(
                "post-norm", f"model.layers[i].{spec['post_norm']}",
                f"A native module under its own name. Its output is the contribution: `{contribution['expr']}`.",
                extra=spec.get("post_norm_note"))
        sublayers.append(sub)

    shape_of = [drawn(specs, hosts) for hosts in info["block_hosts"]]
    for i, (shape, block_hosts) in enumerate(zip(shape_of, info["block_hosts"])):
        named = [h for h in block_hosts if h in hosts]
        assert sorted(named) == sorted(specs[k]["host"] for k in shape), \
            f"{entry.MODEL_TYPE}: block {i} has {named} but BLOCK draws {[keys[k] for k in shape]} on it"
    shapes = list(dict.fromkeys(shape_of))
    for k in range(len(specs)):
        assert any(k in shape for shape in shapes), f"{entry.MODEL_TYPE}: no block has BLOCK's {keys[k]!r} sublayer"
    single = len(shapes) == 1
    assert single or "identity" not in entry.BLOCK, f"{entry.MODEL_TYPE}: a BLOCK with several shapes takes no identity"

    layer_output = by_host["layer"]["layer_output"]
    nodes["stream.input"] = node("residual stream", "model.layers[i].input",
                                 "The residual stream entering the block, a tensor on every family.",
                                 layout="Residual", dims="batch seq hidden")
    nodes["stream.output"] = value_node(layer_output, "residual stream")
    drawn_shapes = []
    for s, shape in enumerate(shapes):
        subs = [sublayers[k] for k in shape]
        mids = []
        # Between two sequential sublayers the stream has a value of its own; a parallel block has no such point.
        for k in range(len(subs) - 1 if entry.BLOCK.get("topology", "sequential") == "sequential" else 0):
            nxt = subs[k + 1]
            after = subs[k]["label"].lower()
            mid = f"stream.mid.{k}" if single else f"stream.mid.{s}.{k}"
            if nxt["pre_norm"]:
                nodes[mid] = node("residual stream", f"model.layers[i].{nxt['pre_norm']}.input",
                                  f"The stream after the {after} add, as the next pre-norm receives it. No standard value of its own.")
            else:
                nodes[mid] = node("residual stream", f"model.layers[i].{nxt['host']}.input",
                                  f"The stream after the {after} add, as the next sublayer receives it. No standard value of its own.")
            mids.append(mid)
        terms = " + ".join(f"{sub['host']}.{sub['contribution']}" for sub in subs)
        identity = entry.BLOCK.get("identity", f"layers[i].input + {terms} == layer_output")
        plus = "plus" if single else f"plus.{s}"
        nodes[plus] = node("the add", identity, entry.BLOCK.get("identity_note", "The contribution identity nnterp's suite checks on this family."))
        drawn_shapes.append({"subs": list(shape), "mids": mids, "plus": plus, "identity": identity,
                             "label": " + ".join(sub["label"] for sub in subs)})

    schema = {
        "topology": entry.BLOCK.get("topology", "sequential"),
        "sublayers": sublayers,
        "identity": drawn_shapes[0]["identity"],  # block 0's; a hybrid's page swaps it as the slider moves
        "num_layers": info["num_layers"],
        "layer_types": info["layer_types"],
        "nodes": nodes,
    }
    if not single:
        schema["shapes"] = drawn_shapes
        schema["shape_of"] = [shapes.index(shape) for shape in shape_of]
    return schema


INTERIOR_SHORT = {
    "attention_queries": "q", "attention_keys": "k", "attention_values": "v", "attention_scores": "scores",
    "attention_probabilities": "pattern", "attention_head_outputs": "heads",
    "betas": "beta", "decays": "decay", "state_input": "state in", "state_output": "state out",
    "state": "state", "states": "states",
    "router_logits": "logits", "expert_weights": "weights", "expert_indices": "indices",
    "expert_outputs": "per slot", "routed_output": "routed", "shared_expert_output": "shared",
}


def strip_schema(entry: ModuleType, info: dict[str, Any]) -> dict[str, Any]:
    """The model-level strip: embeddings, the blocks, the final norm, the head, the logits."""
    root = {row["name"]: row for row in info["values"]["root"]}
    strip = getattr(entry, "STRIP", {})
    nodes = {
        "strip.embed": value_node(root["token_embeddings"], "embeddings"),
        "strip.layers": node("the blocks", "model.layers", f"{info['num_layers']} blocks of {info['block_class']}; the one drawn below is any of them."),
        "strip.norm": node("final norm", "model.norm", "The final norm, under its standard name; `norm.output` is what the head reads."),
        "strip.head": node("unembedding", "model.lm_head.output", "The raw projection onto the vocabulary."),
        "strip.logits": value_node(root["logits"], "logits"),
    }
    for key, note in strip.items():
        nodes[f"strip.{key}"]["extra"] = note
    return {"nodes": nodes, "notes": strip}


# -- rendering ------------------------------------------------------------------------

def name_roles(entry: ModuleType, info: dict[str, Any]) -> dict[str, str]:
    """The role that colours each nnterp name in code and in the printout: a value takes its
    host's role, a sublayer's own name wins over the block's, the block's norms are norms."""
    roles = dict(NAME_ROLES)
    for host, rows in info["values"].items():  # root and layer first, so a sublayer's value wins
        if host in HOST_ROLES:
            roles.update({row["name"]: HOST_ROLES[host] for row in rows})
    for sub in entry.BLOCK["sublayers"]:
        roles.update({sub[key]: "norm" for key in ("pre_norm", "post_norm") if sub.get(key)})
    return roles


LEXER = PythonLexer()
FORMATTER = HtmlFormatter(nowrap=True)
NAME_SPAN = re.compile(r'<span class="n">(\w+)</span>')
FENCE = re.compile(r'<pre><code(?: class="language-(\w+)")?>(.*?)</code></pre>', re.S)
MODULE_LINE = re.compile(r"^\((?P<name>\w+)\): (?P<cls>[\w.]+)(?P<args>\(.*)?$")


def highlight_python(code: str, roles: dict[str, str]) -> Markup:
    """Pygments' Python tokens as short-class spans, with nnterp's names in their role's class."""
    out = highlight(code, LEXER, FORMATTER).rstrip("\n")
    out = NAME_SPAN.sub(lambda m: f'<span class="n role-{roles[m[1]]}">{m[1]}</span>' if m[1] in roles else m[0], out)
    return Markup(out)


def root_printout(model: Any) -> str:
    """``print(model)`` without the native containers: a root child that holds a module mounted on
    the root under its standard name (``model``, holding ``model.layers``) is left out, so the
    standard names and whatever else sits on the root are what shows."""
    containers = {path.split(".")[0] for path in model._aliases.values() if "." in path}
    lines = repr(model).splitlines()
    out, i = [], 0
    while i < len(lines):
        name = re.match(r"  \((\w+)\): ", lines[i])
        if name and name[1] in containers:
            if lines[i].endswith("("):
                i = lines.index("  )", i)
        else:
            out.append(lines[i])
        i += 1
    return "\n".join(out)


def highlight_repr(text: str, roles: dict[str, str]) -> Markup:
    """The model's printout as spans: a module's name in its role, a value's name in its host's,
    layouts in plain mono, dims, descriptions and constructor arguments dimmed."""
    lines = []
    for line in text.splitlines():
        body = line.strip()
        indent = html.escape(line[:len(line) - len(body)])
        value = VALUE_LINE.match(body)
        module = MODULE_LINE.match(body)
        if value and value["layout"]:
            role = roles.get(value["name"])
            name = f'<span class="rn{" role-" + role if role else ""}">({html.escape(value["name"])})</span>'
            lines.append(f'{indent}{name} -&gt; <span class="rl">{html.escape(value["layout"])}</span> '
                         f'<span class="rd">[{html.escape(value["dims"])}]</span>: '
                         f'<span class="rd">{html.escape(value["desc"])}</span>')
        elif module:
            role = roles.get(module["name"])
            name = f'<span class="rn{" role-" + role if role else ""}">({html.escape(module["name"])})</span>'
            args = f'<span class="rd">{html.escape(module["args"])}</span>' if module["args"] and module["args"] != "(" else html.escape(module["args"] or "")
            lines.append(f'{indent}{name}: {html.escape(module["cls"])}{args}')
        else:
            lines.append(indent + html.escape(body))
    return Markup("\n".join(lines))


def embed_json(data: Any) -> str:
    """JSON for a <script type="application/json"> block: a closing tag inside a string must not end the block."""
    return json.dumps(data).replace("</", "<\\/")


def md(text: str, rst: bool = False, roles: dict[str, str] | None = None) -> Markup:
    """Markdown to HTML; with ``rst``, a docstring's double-backtick literals become code spans
    first. Fenced blocks with no language or ``python`` are highlighted, with ``roles`` colouring
    nnterp's names; inline code stays plain."""
    if rst:
        text = re.sub(r"``([^`]+?)``", lambda m: "`" + " ".join(m[1].split()) + "`", text)
    out = markdown.markdown(text, extensions=["fenced_code", "tables"])

    def fence(m: re.Match) -> str:
        if m[1] not in (None, "python", "py"):
            return m[0]
        return f'<pre class="code"><code>{highlight_python(html.unescape(m[2]), roles or {})}</code></pre>'

    return Markup(FENCE.sub(fence, out))


def css_variables(fills: list[str], deeps: list[str], paper: str) -> str:
    """The ten colour variables and the paper, as an inline style."""
    pairs = [(f"--c{k + 1}", fill) for k, fill in enumerate(fills)] + [(f"--c{k + 1}-deep", deep) for k, deep in enumerate(deeps)]
    return "; ".join(f"{name}: {value}" for name, value in pairs + [("--paper", paper)])


def palette(entry: ModuleType) -> dict[str, Any]:
    """The entry's five colours in two tiers: ``PALETTE["colors"]`` as given (its ``deeps`` too,
    or deepened), else generated from ``PALETTE["hue"]`` or a hash of the model_type. A palette
    that fails ``palette.check`` fails the build."""
    given = getattr(entry, "PALETTE", {})
    paper = given.get("paper", PAPER)
    if "colors" in given:
        fills = list(given["colors"])
        deeps = list(given.get("deeps") or palettes.deepen(fills))
        assert len(fills) == 5 and len(deeps) == 5, f"{entry.MODEL_TYPE}: PALETTE needs five colours"
    else:
        generated = palettes.generate(given.get("hue", palettes.hue_of(entry.MODEL_TYPE)))
        fills, deeps = generated["fills"], generated["deeps"]
    problems = palettes.check(fills, deeps, paper, INK)
    if problems:
        raise ValueError(f"{entry.MODEL_TYPE}: the palette does not pass:\n  " + "\n  ".join(problems))
    return {"fills": fills, "deeps": deeps, "paper": paper, "css": css_variables(fills, deeps, paper)}


def site_palette() -> dict[str, Any]:
    """The index's own palette, generated like a family's from a fixed name."""
    generated = palettes.generate(palettes.hue_of("nnterp"))
    return {**generated, "paper": PAPER, "css": css_variables(generated["fills"], generated["deeps"], PAPER)}


def quirks(entry: ModuleType) -> list[dict[str, str]]:
    out = []
    for slug in entry.QUIRKS:
        assert slug in QUIRKS, f"{entry.MODEL_TYPE}: unknown quirk {slug!r}; known: {sorted(QUIRKS)}"
        label, blurb = QUIRKS[slug]
        out.append({"slug": slug, "label": label, "blurb": blurb})
    return out


def page_model(entry: ModuleType, info: dict[str, Any]) -> dict[str, Any]:
    block = block_schema(entry, info)
    strip = strip_schema(entry, info)
    nodes = {**block["nodes"], **strip["nodes"]}
    eager_only = [row["name"] for row in info["support"] if row["condition"] and row["condition"]["kind"] == "eager"]
    other = [row for row in info["support"] if row["condition"] and row["condition"]["kind"] == "other"]
    roles = name_roles(entry, info)
    for shape in block.get("shapes", []):
        shape["identity_html"] = str(highlight_python(shape["identity"], roles))
    return {
        **info,
        "model_type": entry.MODEL_TYPE,
        "title": entry.TITLE,
        "subtitle": entry.SUBTITLE,
        "palette": palette(entry),
        "checkpoints": [{"id": c, "url": f"https://huggingface.co/{c}"} for c in entry.CHECKPOINTS],
        "pinned": entry.PINNED,
        "vllm": getattr(entry, "VLLM", False),
        "quirks": quirks(entry),
        "notes": md(entry.NOTES, roles=roles),
        "docstring": md(info["docstring"], rst=True, roles=roles),
        "host_roles": {host: HOST_ROLES.get(host, "mlp") for host in info["values"]},
        "repr_html": highlight_repr(info["repr"], roles),
        "identity_html": highlight_python(block["identity"], roles),
        "block": block,
        "strip": strip,
        "nodes_json": embed_json(nodes),
        "block_json": embed_json({**{k: v for k, v in block.items() if k != "nodes"},
                                  "roles": {s.get("key", s["host"]): HOST_ROLES.get(s["host"], "mlp") for s in block["sublayers"]}}),
        "eager_only": eager_only,
        "other_conditions": other,
        "available": sum(1 for row in info["support"] if row["condition"] is None),
        "total": len(info["support"]),
        "source_url": f"{GITHUB}/{info['family_file']}",
        "test_url": f"{GITHUB}/tests/families/test_{entry.MODEL_TYPE}.py",
        "families_url": f"{GITHUB}/docs/reference/families.md",
        "built": dt.date.today().isoformat(),
    }


def environment() -> Environment:
    env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["md"] = md
    env.filters["code"] = lambda s: Markup(f"<code>{html.escape(str(s))}</code>")
    return env


def build_page(entry: ModuleType, reference: str | None = None) -> str:
    """One family page as HTML."""
    info = introspect(entry, reference)
    return environment().get_template("family.html.j2").render(**page_model(entry, info))


def index_model(built: list[dict[str, Any]]) -> dict[str, Any]:
    done = {page["model_type"] for page in built}
    stubs = [name for name in nnterp.families.known() if name not in done]
    return {"pages": built, "stubs": stubs, "total": len(nnterp.families.known()), "built": dt.date.today().isoformat(),
            "palette": site_palette(), "quirks": [{"slug": s, "label": l} for s, (l, _) in QUIRKS.items()]}


def build(only: list[str] | None = None, out: Path = HERE / "site") -> list[Path]:
    env = environment()
    out.mkdir(exist_ok=True)
    if (out / "static").exists():
        shutil.rmtree(out / "static")
    shutil.copytree(HERE / "static", out / "static")
    written, cards = [], []
    for entry in entries.load_all():
        if only and entry.MODEL_TYPE not in only:
            continue
        info = introspect(entry)
        model = page_model(entry, info)
        path = out / f"{entry.MODEL_TYPE}.html"
        path.write_text(env.get_template("family.html.j2").render(**model))
        written.append(path)
        cards.append({k: model[k] for k in ("model_type", "title", "subtitle", "palette", "checkpoints", "quirks", "vllm",
                                             "num_layers", "architecture", "family_module")})
        print(f"wrote {path.relative_to(HERE.parent)}")
    if not only:
        index = out / "index.html"
        index.write_text(env.get_template("index.html.j2").render(**index_model(cards)))
        written.append(index)
        print(f"wrote {index.relative_to(HERE.parent)}")
    return written


if __name__ == "__main__":
    build(sys.argv[1:] or None)
