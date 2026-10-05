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
from nnterp.components import Standard  # noqa: E402
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
    "parallel-blocks": ("Parallel block", "One norm feeds attention and MLP; the block sums x + attn + mlp."),
    "hybrid": ("Hybrid", "Some blocks carry linear_attn (a recurrent mixer), others self_attn."),
    "mamba1": ("Selective scan (Mamba-1)", "The mixer is a selective scan; C/B/x as queries/keys/values."),
    "mamba2": ("State space (Mamba-2)", "The mixer is an SSD state-space block with a per-chunk state."),
    "no-mlp": ("No MLP module", "fc1/fc2 sit on the block, so layers[i].mlp does not exist."),
    "softcapped-logits": ("Softcapped logits", "logits is tanh-capped after lm_head; project_on_vocab applies the cap."),
    "scaled-logits": ("Scaled logits", "The head's output is multiplied or divided by a config scale."),
    "sliding-window": ("Sliding window", "Some blocks attend over a window; config.layer_types says which."),
    "scaled-embeddings": ("Scaled embeddings", "The embedding output is multiplied before block 0; token_embeddings is the scaled tensor."),
    "gain-norm": ("1 + weight norm gain", "RMSNorm multiplies by (1 + weight), so the gain is not norm.weight."),
}

ROOT_NAMES = ("embed_tokens", "layers", "norm", "lm_head")
SIZE_NAMES = ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "qk_head_dim", "vocab_size", "intermediate_size")
CONFIG_KEYS = (
    "architectures", "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "sliding_window", "hidden_act",
    "hidden_activation", "rms_norm_eps", "layer_norm_eps", "rope_theta", "query_pre_attn_scalar",
    "attn_logit_softcapping", "final_logit_softcapping", "tie_word_embeddings", "dtype",
)

VALUE_LINE = re.compile(r"^\((?P<name>\w+)\)(?: -> (?P<layout>\w+) \[(?P<dims>[^\]]*)\])?: (?P<desc>.*)$")


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
    element = "" if select is None else f", element {select}"
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
    reasons = sorted(set(reason.values()))
    blocks = sorted(reason)
    where = "every block" if len(blocks) == num_layers else f"blocks {blocks}"
    return f"{where}: " + " / ".join(reasons)


def introspect(entry: ModuleType, reference: str | None = None) -> dict[str, Any]:
    """What nnterp knows about the entry's family, read off a meta build of ``reference``."""
    reference = reference or entry.REFERENCE
    eager = StandardizedTransformer(reference, attn_implementation="eager")
    default = StandardizedTransformer(reference)
    family = eager.family
    assert family.__name__.rsplit(".", 1)[1] == entry.MODEL_TYPE, (family.__name__, entry.MODEL_TYPE)
    num_layers = eager.num_layers
    block = eager.layers[0]
    children = StandardizedTransformer._standard_children(block)

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

    hosts = [("root", "model", eager), ("layer", "model.layers[i]", block)]
    hosts += [(alias, f"model.layers[i].{alias}", child) for alias, child in children.items()]
    values: dict[str, list[dict[str, Any]]] = {}
    for alias, expr, host in hosts:
        rows = value_rows(host, expr)
        prefix = "" if alias in ("root", "layer") else alias + "."
        for row in rows:
            row["condition"] = conditions.get(prefix + row["name"])
            row["host"] = alias
            row["module"] = type(host._module).__name__ if alias != "root" else type(eager._module).__name__
        values[alias] = rows

    paths = []
    for name in ROOT_NAMES:
        paths.append((f"model.{name}", eager.get(name).path))
    paths.append(("model.layers[i]", eager.get("layers.0").path))
    for name, _ in block._named_children():
        alias = next((a for a, child in children.items() if child is block.__dict__.get(name)), None)
        shown = alias or name
        paths.append((f"model.layers[i].{shown}", eager.get(f"layers.0.{name}").path))

    text_config = eager.config.get_text_config()
    config = text_config.to_dict()
    config_rows = [(key, config[key]) for key in CONFIG_KEYS if key in config and config[key] is not None]
    layer_types = config.get("layer_types")

    doc = inspect.getdoc(family) or ""
    return {
        "family_module": family.__name__,
        "family_file": f"nnterp/families/{entry.MODEL_TYPE}.py",
        "architecture": type(eager._module).__name__,
        "module_classes": {alias: type(child._module).__name__ for alias, child in children.items()},
        "block_class": type(block._module).__name__,
        "returns_tuple": type(block).returns_tuple,
        "reference": reference,
        "default_impl": default_impl,
        "num_layers": num_layers,
        "layer_types": layer_types,
        "sizes": [(name, getattr(eager, name)) for name in SIZE_NAMES],
        "config": config_rows,
        "rename": list(family.RENAME.items()),
        "paths": paths,
        "values": values,
        "support": support,
        "repr": elide_native_copies(repr(eager)),
        "docstring": doc,
        "children": list(children),
        "versions": {"nnterp": nnterp.__version__, "transformers": importlib.import_module("transformers").__version__},
    }


# -- the block schema ----------------------------------------------------------------

def node(eyebrow: str, expr: str, desc: str, *, layout: str | None = None, dims: str | None = None,
         where: str | None = None, condition: dict | None = None, extra: str | None = None) -> dict[str, Any]:
    return {"eyebrow": eyebrow, "expr": expr, "desc": desc, "layout": layout, "dims": dims, "where": where,
            "condition": condition, "extra": extra}


def value_node(row: dict[str, Any], eyebrow: str) -> dict[str, Any]:
    return node(eyebrow, row["expr"], row["description"], layout=row["layout"], dims=row["dims"], where=row["where"],
                condition=row["condition"])


def block_schema(entry: ModuleType, info: dict[str, Any]) -> dict[str, Any]:
    """The entry's BLOCK, checked against the family and enriched with every node's hover card."""
    by_host = {alias: {row["name"]: row for row in rows} for alias, rows in info["values"].items()}
    sizes = dict(info["sizes"])
    fmt = {**sizes, **{k: v for k, v in info["config"]}}
    nodes: dict[str, dict[str, Any]] = {}
    sublayers = []
    hosts = [s["host"] for s in entry.BLOCK["sublayers"]]
    for k, spec in enumerate(entry.BLOCK["sublayers"]):
        host = spec["host"]
        assert host in by_host, f"{entry.MODEL_TYPE}: BLOCK names host {host!r}; the block has {list(by_host)}"
        contribution = by_host[host][spec["contribution"]]
        sub = {
            "host": host, "kind": spec["kind"], "label": spec["label"],
            "detail": spec.get("detail", "").format(**fmt),
            "variants": {k2: v.format(**fmt) for k2, v in spec.get("variants", {}).items()},
            "pre_norm": spec.get("pre_norm"), "post_norm": spec.get("post_norm"),
            "contribution": spec["contribution"], "interior": [],
        }
        nodes[f"sub.{host}"] = node(
            spec["label"], f"model.layers[i].{host}",
            f"{info['module_classes'][host]} under its standard name. " + spec.get("detail", "").format(**fmt),
            extra=f"{len(by_host[host])} standard values; `.input` is what the sublayer reads.")
        for name in spec.get("interior", []):
            assert name in by_host[host], f"{entry.MODEL_TYPE}: {host} has no value {name!r}"
            sub["interior"].append({"name": name, "short": INTERIOR_SHORT.get(name, name)})
            nodes[f"interior.{host}.{name}"] = value_node(by_host[host][name], "inside the sublayer")
        nodes[f"contrib.{host}"] = value_node(contribution, "contribution")
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

    layer_output = by_host["layer"]["layer_output"]
    nodes["stream.input"] = node("residual stream", "model.layers[i].input",
                                 "The residual stream entering the block, a tensor on every family.",
                                 layout="Residual", dims="batch seq hidden")
    nodes["stream.output"] = value_node(layer_output, "residual stream")
    for k in range(len(sublayers) - 1):
        nxt = sublayers[k + 1]
        after = sublayers[k]["label"].lower()
        if nxt["pre_norm"]:
            nodes[f"stream.mid.{k}"] = node("residual stream", f"model.layers[i].{nxt['pre_norm']}.input",
                                            f"The stream after the {after} add, as the next pre-norm receives it. No standard value of its own.")
        else:
            nodes[f"stream.mid.{k}"] = node("residual stream", f"model.layers[i].{nxt['host']}.input",
                                            f"The stream after the {after} add, as the next sublayer receives it. No standard value of its own.")
    terms = " + ".join(f"{s['host']}.{s['contribution']}" for s in sublayers)
    identity = entry.BLOCK.get("identity", f"layers[i].input + {terms} == layer_output")
    nodes["plus"] = node("the add", identity, entry.BLOCK.get("identity_note", "The contribution identity nnterp's suite checks on this family."))

    return {
        "topology": entry.BLOCK.get("topology", "sequential"),
        "sublayers": sublayers,
        "identity": identity,
        "num_layers": info["num_layers"],
        "layer_types": info["layer_types"],
        "nodes": nodes,
    }


INTERIOR_SHORT = {
    "attention_queries": "q", "attention_keys": "k", "attention_values": "v", "attention_scores": "scores",
    "attention_probabilities": "pattern", "attention_head_outputs": "heads",
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


def elide_native_copies(text: str) -> str:
    """``print(model)`` with a subtree printed once: a module mounted on the root under its standard
    name (``layers``) prints there in full, so its copy inside the native tree closes on the line it opens."""
    lines = text.splitlines()
    mounted = {line.strip() for line in lines if line.startswith("  (") and line.endswith("(")}
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        if indent > 2 and line.strip() in mounted:
            i = lines.index(" " * indent + ")", i)
            line += "…)"
        out.append(line)
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
        text = re.sub(r"``([^`\n]+)``", r"`\1`", text)
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
    return {
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
                                  "roles": {s["host"]: HOST_ROLES.get(s["host"], "mlp") for s in block["sublayers"]}}),
        "eager_only": eager_only,
        "other_conditions": other,
        "available": sum(1 for row in info["support"] if row["condition"] is None),
        "total": len(info["support"]),
        "source_url": f"{GITHUB}/{info['family_file']}",
        "test_url": f"{GITHUB}/tests/families/test_{entry.MODEL_TYPE}.py",
        "families_url": f"{GITHUB}/docs/reference/families.md",
        "built": dt.date.today().isoformat(),
        **info,
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
