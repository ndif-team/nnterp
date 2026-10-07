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

A page has one selector over the entry's checkpoints, and every part that
depends on the checkpoint (sizes, ``support()``, the printout, the tower of a
vision-language wrapper) is built for each of them and swapped by the page's
script. A checkpoint is read from its config: a ``model_type`` transformers maps
to image-text-to-text, and the entry describes in ``WRAPPERS``, loads with
``task="image-text-to-text"`` and brings its tower, whose facts live in
``vision/<tower>.py``; a config that cannot be read is listed and said so.

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
from nnterp import StandardizedTransformer, Unavailable  # noqa: E402
from nnterp.components import Moe, RecurrentMixer, Standard  # noqa: E402
from nnterp.components.moe import SHARED_NAMES  # noqa: E402
from nnterp.components.standard import values as class_values  # noqa: E402
from nnterp.components.vision import Vision, scatter_host  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES  # noqa: E402

import entries  # noqa: E402
import palette as palettes  # noqa: E402
import vision as towers  # noqa: E402

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
    "vision": "stream", "patch_embed": "stream", "projector": "stream",
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
    "proportional-rotary": ("Proportional rotary", "Some blocks turn only the first dimensions of each half of a query and key head, at frequencies spaced over the whole head; the other dimensions carry no position."),
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
    "unnormalized-routing": ("Unnormalized routing", "The expert weights are the router's softmax entries for the chosen experts, not renormalized over them: a token's expert_weights sum to less than one."),
    "one-sublayer-blocks": ("One sublayer per block", "Each block is one norm and one sublayer, so it adds one contribution to the stream, and support() reports the hosts it does not hold missing on it."),
    # The vision side (docs/usage/vision.md). `vision` marks a family some of whose checkpoints carry a tower; a
    # vision-language checkpoint's page adds its tower's and its wrapper's slugs from the rest.
    "vision": ("Vision-language", "Some checkpoints carry a vision tower: model.vision, its blocks, the projector, and the image values where the image enters the text model."),
    "cls-token": ("CLS token", "The tower's stream carries a class token beside the patches (CLIP's first, Llama 4's last), so the patch axis is one longer than the patch grid."),
    "packed-tower": ("Packed tower", "Every image's patches run in one row, [1, all patches, vision_hidden]; the processor's grid sizes split it."),
    "variable-resolution": ("Variable resolution", "The tower takes images of any resolution, so vision.image_size raises Unavailable; each image's grid is in the processor's output."),
    "padded-patches": ("Padded patches", "Each image's patches are padded to a fixed row count; the padded rows are masked as keys but run through every block, so they are rows of layer_output."),
    "tiled-images": ("Tiled images", "The processor cuts an image into crops or tiles, each a row of the tower's batch."),
    "deepstack": ("DeepStack", "Tower blocks feed the text model again after its first blocks: layers[k].deepstack_output is added at the image positions, outside the block."),
    "unpadded-features": ("Unpadded features", "The wrapper unpads the projector's output and adds a newline token per row, so projector.output is not image_features."),
    "pooled-projector": ("Pooled projector", "The projector pools or pixel-shuffles neighbouring patches into one token, so an image has fewer tokens than patches."),
    "encoder-free": ("Encoder-free", "No tower blocks: raw patches go through one embedder into the text stream, so vision.num_layers is 0."),
}

#: The vision quirk slugs: a page shows them on a vision-language checkpoint, the index keeps them off its first filter row.
VISION_QUIRKS = ("cls-token", "packed-tower", "variable-resolution", "padded-patches", "tiled-images", "deepstack",
                 "unpadded-features", "pooled-projector", "encoder-free")
#: The task a vision-language checkpoint loads with, and the config model_types transformers maps to it.
IMAGE_TASK = "image-text-to-text"
IMAGE_TEXT_TO_TEXT = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
TOWER_SIZE_NAMES = ("num_layers", "hidden_size", "num_heads", "head_dim", "intermediate_size", "patch_size", "image_size")
#: What the strip's embed node adds on a vision-language checkpoint.
IMAGE_SENTENCE = ("On this checkpoint the projected image features replace the image tokens' embeddings before block 0: "
                  "layers[0].input[vision.image_token_mask] == vision.image_features.")

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
    "norm_topk_prob",
    "num_local_experts",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
    "full_attention_interval",
    "state_size", "expand", "conv_kernel", "residual_in_fp32",
    "num_kv_shared_layers",
    "mamba_num_heads", "mamba_head_dim", "n_groups", "ssm_state_size", "chunk_size", "mlp_hidden_act",
    "moe_shared_expert_intermediate_size", "moe_latent_size",
)

#: A reason that is the model's shape, not a condition on the load: the block lacks the host, or the mixture a part.
STRUCTURAL = re.compile(r"^(no \w+ (module|value) on this block|this mixture has no shared expert)")
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
        if isinstance(host, Vision) and getattr(value, "locate", None) is not None:
            # a location the tower finds from itself (image_features: the scatter, wherever the family keys
            # it), with the argument it selects there
            key = value.locate(host)
            select = select(host) if callable(select) else select
        # `unavailable(reason)`: a value the family declares it does not have; its key falls back to its
        # own name and names no location, so it is read nowhere.
        nowhere = isinstance(getattr(value, "unavailable", None), str)
        rows.append({
            "name": name,
            "expr": f"{expr}.{name}",
            "layout": match["layout"],
            "dims": match["dims"],
            "description": match["desc"],
            "key": None if nowhere else key if isinstance(key, str) else ("computed" if key is not None else None),
            "select": select,
            "where": "nowhere: the family does not serve it" if nowhere else humanize_key(key, select),
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
        # a select resolved per call (a Mamba-2 value): the argument the kernel that fires takes it as,
        # or the element of its output that call returns it in
        named = re.fullmatch(r"argument\((\w+)\)", select.__name__)
        element = f", argument `{named[1]}`" if named else ", the element that call returns it in"
    else:
        element = "" if select is None else f", argument `{select}`" if isinstance(select, str) else f", element {select}"
    if key.startswith("/"):
        # a path from the model's root, not from the host
        inner = key[1:]
        if inner in ("input", "inputs"):
            return "the model's own inputs, at the root" + element
        return humanize_key(inner, select).replace("inside `", "inside `model.", 1)  # the root's child, as a user writes it
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


def runs_mixture(child: Any) -> bool:
    """Whether a block's child runs a mixture of experts: a `Moe` whose family does not say otherwise
    (Gemma-4's MLP is a `Moe` on every checkpoint and runs a mixture only under ``enable_moe_block``)."""
    return isinstance(child, Moe) and child.no_mixture() is None


def introspect(entry: ModuleType, reference: str | None = None, wrapper: str | None = None) -> dict[str, Any]:
    """What nnterp knows about the entry's family, read off a meta build of ``reference``.

    With ``wrapper`` (the checkpoint's ``model_type``, a key of the entry's ``WRAPPERS``), the checkpoint
    is a vision-language wrapper: it loads with ``task="image-text-to-text"`` and ``vision`` holds its tower."""
    reference = reference or entry.REFERENCE
    load = getattr(entry, "load", StandardizedTransformer)
    task = {"task": IMAGE_TASK} if wrapper else {}
    eager = load(reference, attn_implementation="eager", **task)
    default = load(reference, **task)
    family = eager.family
    assert family.__name__.rsplit(".", 1)[1] == entry.MODEL_TYPE, (family.__name__, entry.MODEL_TYPE)
    num_layers = eager.num_layers
    block = eager.layers[0]

    # Each block's standard children, and whether each is a mixture; one block of each
    # combination of child classes stands for the rest, so a hybrid's ledgers list every host.
    block_hosts, shapes = [], {}
    for layer in eager.layers:
        found = StandardizedTransformer._standard_children(layer)
        block_hosts.append({alias: runs_mixture(child) for alias, child in found.items()})
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
    # A value that no block of this checkpoint has, for a structural reason on every one, is not
    # a value of this model: it is left off the page rather than marked as conditional.
    absent = {name for name, why in support_eager.items()
              if isinstance(why, dict) and len(why) == num_layers and all(STRUCTURAL.match(w) for w in why.values())}
    # A tower's block values are per tower block: "every block" is every block of the tower.
    tower_layers = eager.vision.num_layers if wrapper else None
    support = []
    for name in support_eager:
        if name in absent:
            continue
        blocks = tower_layers if name.startswith("vision.") else num_layers
        under_eager = summarize_reason(support_eager[name], blocks)
        under_default = summarize_reason(support_default.get(name), blocks)
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
        values[alias] = [row for row in merged.values() if prefix + row["name"] not in absent]

    moe = next((child for found in children.values() for child in found if runs_mixture(child)), None)
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
        "host_classes": {alias: [(type(child._module).__name__, runs_mixture(child), len(child.values())) for child in found]
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
        "vision": vision_info(entry, eager, wrapper, conditions) if wrapper else None,
    }


def vision_info(entry: ModuleType, model: Any, wrapper: str, conditions: dict[str, Any]) -> dict[str, Any]:
    """A vision-language checkpoint's tower: which tower (by ``vision_config.model_type``), its sizes, its values
    with their ``support()`` conditions (the ``vision.`` rows), what the diagram needs to draw its block, the
    projector, and the envoy classes on the tower's modules."""
    vision = model.vision
    config = model.config.vision_config
    tower = towers.resolve(entry, wrapper, config.model_type)
    module_class = type(vision._module).__name__
    assert module_class in tower["module_classes"], \
        f"{entry.MODEL_TYPE}: {wrapper}'s tower is a {module_class}; {tower['slug']} lists {tower['module_classes']}"
    layer = vision.layers[0]
    children = StandardizedTransformer._standard_children(layer)
    hosts = [("vision", "model.vision", vision, "vision."), ("layer", "model.vision.layers[i]", layer, "vision.")]
    hosts += [(alias, f"model.vision.layers[i].{alias}", child, f"vision.{alias}.") for alias, child in children.items()]
    values: dict[str, list[dict[str, Any]]] = {}
    for alias, expr, host, prefix in hosts:
        rows = value_rows(host, expr)
        for row in rows:
            row.update(condition=conditions.get(prefix + row["name"]), host=alias, module=type(host._module).__name__)
        values[alias] = rows
    sizes = []
    for name in TOWER_SIZE_NAMES:
        try:
            sizes.append((name, getattr(vision, name)))
        except Unavailable:  # a tower that takes any resolution
            sizes.append((name, "varies"))
    vision_config = config.to_dict()
    scatter = scatter_host(model)
    envoys = [(type(envoy._module).__name__, type(envoy).__name__) for envoy in (vision, layer, *children.values())]
    if scatter is not None:
        envoys.append((type(scatter[1]._module).__name__, type(scatter[1]).__name__))
    return {
        "wrapper": wrapper,
        "wrapper_fields": entry.WRAPPERS[wrapper],
        "tower": tower,
        "module_class": module_class,
        "layer_class": type(layer._module).__name__,
        "path": vision.path,
        "sizes": sizes,
        "values": values,
        "has_norm": "norm" in vision._aliases or isinstance(getattr(type(vision), "norm", None), property),
        "projector": {"class": type(model.projector._module).__name__, "path": model.projector.path},
        "envoys": envoys,
        # what block_schema reads, for the tower's block
        "block": {
            "values": {"layer": values["layer"], **{alias: values[alias] for alias in children}},
            "sizes": [(name, value) for name, value in sizes],
            "config": [(key, vision_config[key]) for key in CONFIG_KEYS if vision_config.get(key) is not None],
            "moe": None, "mixer": None, "support": [], "layer_types": None,
            "num_layers": vision.num_layers,
            "host_classes": {alias: [(type(child._module).__name__, False, len(child.values()))] for alias, child in children.items()},
            "block_hosts": [{alias: False for alias in StandardizedTransformer._standard_children(block)} for block in vision.layers],
        },
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


#: The sublayer kinds a BLOCK may draw. A "moe" sublayer is drawn on the blocks whose host runs a
#: mixture (`runs_mixture`) and an "mlp" one on the blocks whose host does not, so a family with
#: dense first blocks, or with dense and mixture checkpoints, lists both; every kind is drawn only
#: on the blocks that have its host.
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


def block_schema(name: str, spec: dict[str, Any], info: dict[str, Any], base: str = "model.layers[i]",
                 stream: str = "residual stream") -> dict[str, Any]:
    """The entry's BLOCK, checked against the family and enriched with every node's hover card.

    The sublayers are listed once, in forward order; each block draws the ones its children
    match (`drawn`), and the distinct combinations are the block's shapes. A family with one
    shape gets the schema as it always has; a hybrid gets ``shapes`` and ``shape_of`` too, and
    the diagram redraws when the slider crosses into another shape.

    A host that runs a mixture on some checkpoints of the family and not on others (Gemma-4's
    ``mlp``) lists both sublayers; a checkpoint draws the one its blocks have, and the other is
    left out of the schema.

    ``name`` is what an assertion names (the entry's model_type), ``spec`` the BLOCK, ``base`` the block's
    expression (``model.vision.layers[i]`` for a tower's block) and ``stream`` what its stream is called."""
    by_host = {alias: {row["name"]: row for row in rows} for alias, rows in info["values"].items()}
    sizes = dict(info["sizes"])
    moe = info["moe"] or {}
    fmt = {**sizes, **{k: v for k, v in info["config"]}, **{k: moe[k] for k in ("num_experts", "top_k") if k in moe}}
    specs = spec["sublayers"]
    hosts = [s["host"] for s in specs]
    # A host drawn by two sublayers (a dense MLP and a mixture) keys its nodes by kind as well.
    keys = [s["host"] if hosts.count(s["host"]) == 1 else f"{s['host']}-{s['kind']}" for s in specs]

    shape_of = [drawn(specs, hosts) for hosts in info["block_hosts"]]
    for i, (shape, block_hosts) in enumerate(zip(shape_of, info["block_hosts"])):
        named = [h for h in block_hosts if h in hosts]
        assert sorted(named) == sorted(specs[k]["host"] for k in shape), \
            f"{name}: block {i} has {named} but BLOCK draws {[keys[k] for k in shape]} on it"
    shown = sorted({k for shape in shape_of for k in shape})
    for k in range(len(specs)):
        assert k in shown or any(hosts[j] == hosts[k] for j in shown), \
            f"{name}: no block has BLOCK's {keys[k]!r} sublayer"
    # Only the sublayers this checkpoint's blocks draw: the shapes index into what is left.
    position = {k: n for n, k in enumerate(shown)}
    shape_of = [tuple(position[k] for k in shape) for shape in shape_of]
    specs, keys = [specs[k] for k in shown], [keys[k] for k in shown]
    nodes: dict[str, dict[str, Any]] = {}
    sublayers = []
    for k, spec in enumerate(specs):
        host, key, kind = spec["host"], keys[k], spec["kind"]
        assert host in by_host, f"{name}: BLOCK names host {host!r}; the block has {list(by_host)}"
        assert kind in KINDS, f"{name}: kind {kind!r}; known: {KINDS}"
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
            spec["label"], f"{base}.{host}",
            f"{' / '.join(classes)} under its standard name. " + spec.get("detail", "").format(**fmt), extra=extra)
        for name in spec.get("interior", []):
            assert name in by_host[host], f"{name}: {host} has no value {name!r}"
            chip = {"name": name, "short": INTERIOR_SHORT.get(name, name)}
            if kind == "moe":
                assert name in MOE_PARTS, f"{name}: {name!r} is not a mixture's value; a moe sublayer draws {list(MOE_PARTS)}"
                chip["part"] = MOE_PARTS[name]
            sub["interior"].append(chip)
            nodes[f"interior.{key}.{name}"] = value_node(by_host[host][name], "inside the sublayer")
        if kind == "moe":
            parts = {chip["part"] for chip in sub["interior"]}
            assert "shared" not in parts or moe.get("shared"), f"{name}: the mixture has no shared expert to draw"
            sub["moe"] = {"num_experts": moe["num_experts"], "top_k": moe["top_k"], "scoring": moe["scoring"]}
            expr = f"{base}.{host}"
            if "router" in parts:
                nodes[f"moe.{key}.router"] = node(
                    "the router", f"{expr}.router",
                    f"`{moe['router']}`. One logit per expert ({moe['num_experts']}) for each token; the scoring is "
                    f"`{moe['scoring']}`, and {moe['top_k']} are picked: `expert_indices`, each weighted by its `expert_weights`.")
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
                "pre-norm", f"{base}.{spec['pre_norm']}",
                f"A native module under its own name. Its output is what `{host}` reads: `{base}.{host}.input`.",
                extra=spec.get("pre_norm_note"))
        if spec.get("post_norm"):
            nodes[f"norm.{spec['post_norm']}"] = node(
                "post-norm", f"{base}.{spec['post_norm']}",
                f"A native module under its own name. Its output is the contribution: `{contribution['expr']}`.",
                extra=spec.get("post_norm_note"))
        sublayers.append(sub)

    shapes = list(dict.fromkeys(shape_of))
    single = len(shapes) == 1
    assert single or "identity" not in spec, f"{name}: a BLOCK with several shapes takes no identity"

    layer_output = by_host["layer"]["layer_output"]
    entering = by_host["layer"].get("layer_output", {})
    nodes["stream.input"] = node(stream, f"{base}.input",
                                 "The residual stream entering the block, a tensor on every family." if stream == "residual stream"
                                 else f"The {stream} entering the block.",
                                 layout="Residual" if stream == "residual stream" else entering.get("layout"),
                                 dims="batch seq hidden" if stream == "residual stream" else entering.get("dims"))
    nodes["stream.output"] = value_node(layer_output, stream)
    drawn_shapes = []
    for s, shape in enumerate(shapes):
        subs = [sublayers[k] for k in shape]
        mids = []
        # Between two sequential sublayers the stream has a value of its own; a parallel block has no such point.
        for k in range(len(subs) - 1 if spec.get("topology", "sequential") == "sequential" else 0):
            nxt = subs[k + 1]
            after = subs[k]["label"].lower()
            mid = f"stream.mid.{k}" if single else f"stream.mid.{s}.{k}"
            if nxt["pre_norm"]:
                nodes[mid] = node(stream, f"{base}.{nxt['pre_norm']}.input",
                                  f"The stream after the {after} add, as the next pre-norm receives it. No standard value of its own.")
            else:
                nodes[mid] = node(stream, f"{base}.{nxt['host']}.input",
                                  f"The stream after the {after} add, as the next sublayer receives it. No standard value of its own.")
            mids.append(mid)
        terms = " + ".join(f"{sub['host']}.{sub['contribution']}" for sub in subs)
        identity = spec.get("identity", f"{base.removeprefix('model.')}.input + {terms} == layer_output")
        plus = "plus" if single else f"plus.{s}"
        checked = "on this family" if stream == "residual stream" else "on every tower block"
        nodes[plus] = node("the add", identity, spec.get("identity_note", f"The contribution identity nnterp's suite checks {checked}."))
        drawn_shapes.append({"subs": list(shape), "mids": mids, "plus": plus, "identity": identity,
                             "label": " + ".join(sub["label"] for sub in subs)})

    schema = {
        "topology": spec.get("topology", "sequential"),
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
    if info.get("vision"):
        embed = nodes["strip.embed"]
        embed["extra"] = f"{embed['extra']} {IMAGE_SENTENCE}" if embed["extra"] else IMAGE_SENTENCE
    return {"nodes": nodes, "notes": strip}


def tower_path_nodes(v: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The image's path into the text model, as hover nodes: the image, the patch embedding, the tower's
    blocks, its final norm where it has one, the projector, the features and where they enter."""
    tower, wrapper = v["tower"], v["wrapper_fields"]
    root = {row["name"]: row for row in v["values"]["vision"]}
    layer = {row["name"]: row for row in v["values"]["layer"]}
    blocks = value_node(layer["layer_output"], "the tower's blocks")
    blocks["extra"] = f"{v['block']['num_layers']} blocks of {v['layer_class']}; the one drawn below is any of them."
    nodes = {
        "path.image": node("the image", "pixel_values", f"What the processor hands the tower. {tower['rows']}"),
        "path.patch_embed": {**value_node(root["patch_embeddings"], "patch embedding"), "extra": tower["positions"]},
        "path.layers": blocks,
        "path.projector": node(
            "projector", "model.projector",
            f"`{v['projector']['class']}` at `{v['projector']['path']}`: {wrapper['projector']}.",
            extra="`model.projector.input` is what it receives, `model.projector.output` what it returns."),
        "path.features": value_node(root["image_features"], "image features"),
        "path.scatter": {**value_node(root["image_token_mask"], "into the text model"),
                         "extra": "`model.layers[0].input[model.vision.image_token_mask] == model.vision.image_features`: "
                                  "the features replace the image tokens' embeddings before block 0."},
    }
    if v["has_norm"]:
        nodes["path.norm"] = {**value_node(root["tower_output"], "final norm"), "extra": f"`model.vision.norm`. {tower['norm']}"}
    else:
        nodes["path.layers"]["extra"] += f" `model.vision.tower_output` is the last block's `layer_output`: {tower['norm']}"
    return nodes


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
    if info.get("vision"):  # the tower's own values belong to the stream; its blocks' take their hosts' roles, as above
        roles.update({row["name"]: "stream" for row in info["vision"]["values"]["vision"]})
    return roles


LEXER = PythonLexer()
FORMATTER = HtmlFormatter(nowrap=True)
NAME_SPAN = re.compile(r'<span class="n">(\w+)</span>')
FENCE = re.compile(r'<pre><code(?: class="language-(\w+)")?>(.*?)</code></pre>', re.S)
MODULE_LINE = re.compile(r"^\((?P<name>[\w/]+)\): (?P<cls>[\w.]+)(?P<args>\(.*)?$")


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
    if "vision" not in model._aliases:
        return "\n".join(out)
    # The tower likewise: a native container whose every child is mounted on the tower under a standard name
    # (CLIP's and SigLIP's `encoder`, holding only `layers`) is left out; one with other children stays.
    vision = model.vision
    mounted = set(vision._aliases.values())
    containers = {path.split(".")[0] for path in mounted if "." in path}
    full = {name for name in containers
            if all(f"{name}.{child}" in mounted for child, _ in vision._module.get_submodule(name).named_children())}
    start = next(k for k, line in enumerate(out) if line.startswith("  (vision): "))
    lines, out, i = out, out[:start], start
    while i < len(lines):
        name = re.match(r"    \((\w+)\): ", lines[i])
        if name and name[1] in full and lines[i].endswith("("):
            i = lines.index("    )", i)
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
            role = roles.get(module["name"].split("/")[0])
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
    return [quirk(slug, entry.MODEL_TYPE) for slug in entry.QUIRKS]


#: A small eye, marking a vision-language checkpoint in the selector, the checkpoints ledger and the index.
EYE = Markup('<svg class="eye" viewBox="0 0 24 14" aria-hidden="true" focusable="false">'
             '<path d="M1.5 7C5 1.8 19 1.8 22.5 7 19 12.2 5 12.2 1.5 7Z"/><circle cx="12" cy="7" r="2.8"/></svg>')
#: The parts of a family page built once per checkpoint, each a template macro in panes.html.j2; the page holds
#: one copy per distinct rendering and shows the selected checkpoint's.
PANES = ("chips", "block_head", "tower", "values", "printout", "sizes", "vision_notes", "quirk_list", "envoys")


def unavailable_reason(error: BaseException) -> str:
    """Why a checkpoint's config cannot be read, in a few words: the Hub's error class where one is in the chain."""
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError

    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, GatedRepoError):
            return "the repository is gated, and this build has no access to it"
        if isinstance(seen, RepositoryNotFoundError):
            return "no such repository on the Hub"
        if isinstance(seen, LocalEntryNotFoundError):
            return "its config is not in the Hub cache, and the build is offline"
        seen = seen.__cause__ or seen.__context__
    text = str(error).strip()
    return f"{type(error).__name__}: {text.splitlines()[0] if text else 'no message'}"[:240]


def read_checkpoint(entry: ModuleType, checkpoint: str, cache: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    """One checkpoint as the page holds it: ``info`` from `introspect`, or ``unavailable`` saying why its config
    cannot be read (gated, missing, offline), so the page lists it greyed out and the build goes on.

    The task comes from the config: a ``model_type`` transformers maps to image-text-to-text, and the entry
    describes in ``WRAPPERS``, loads as that wrapper with its tower; any other loads for text generation, as a
    page always has. An entry with a ``load`` builds its checkpoints itself (its configs need more than
    ``AutoConfig``). Checkpoints whose configs differ only in their name share one introspection."""
    key, wrapper = checkpoint, None
    if not hasattr(entry, "load"):
        try:
            config = AutoConfig.from_pretrained(checkpoint)
        except Exception as error:  # noqa: BLE001 - whatever makes a config unreadable is a reason the page states
            return {"id": checkpoint, "unavailable": unavailable_reason(error)}
        if config.model_type in IMAGE_TEXT_TO_TEXT and config.model_type in getattr(entry, "WRAPPERS", {}):
            wrapper = config.model_type
        fields = {k: v for k, v in config.to_dict().items() if k != "_name_or_path"}
        key = (wrapper, json.dumps(fields, sort_keys=True, default=str))
    if key not in cache:
        cache[key] = introspect(entry, checkpoint, wrapper=wrapper)
    return {"id": checkpoint, "info": {**cache[key], "reference": checkpoint}}


def ledgers(info: dict[str, Any]) -> list[dict[str, Any]]:
    """The API's ledgers, one per host: the text model's, then on a vision-language checkpoint the tower's."""
    out = []
    for host, rows in info["values"].items():
        label = "model" if host == "root" else "model.layers[i]" if host == "layer" else f"model.layers[i].{host}"
        label = Markup(html.escape(label).replace(".", ".<wbr>"))  # a long host name breaks at its dots
        classes = " / ".join(info["module_classes"][host]) if host in info["module_classes"] else (rows[0]["module"] if rows else "")
        nodes = {"token_embeddings": "strip.embed", "logits": "strip.logits"} if host == "root" else {}
        out.append({"label": label, "role": HOST_ROLES.get(host, "mlp"), "classes": classes, "rows": rows, "nodes": nodes})
    v = info.get("vision")
    if v:
        for host, rows in v["values"].items():
            label = "model.vision" if host == "vision" else "model.vision.layers[i]" if host == "layer" else f"model.vision.layers[i].{host}"
            label = Markup(html.escape(label).replace(".", ".<wbr>"))
            role = "stream" if host in ("vision", "layer") else HOST_ROLES.get(host, "mlp")
            out.append({"label": label, "role": role, "classes": rows[0]["module"] if rows else "", "rows": rows, "nodes": {},
                        "tower": v["tower"]["title"] if host == "vision" else None})
    return out


def checkpoint_model(entry: ModuleType, info: dict[str, Any], family_quirks: list[dict[str, str]]) -> dict[str, Any]:
    """Everything on the page that depends on the checkpoint: what its panes render, and ``data`` for the page's
    script (the block's schema and node texts, the identity, and the tower's on a vision-language checkpoint)."""
    roles = name_roles(entry, info)
    block = block_schema(entry.MODEL_TYPE, entry.BLOCK, info)
    strip = strip_schema(entry, info)
    nodes = {**block.pop("nodes"), **strip["nodes"]}
    for shape in block.get("shapes", []):
        shape["identity_html"] = str(highlight_python(shape["identity"], roles))
    block["roles"] = {s.get("key", s["host"]): HOST_ROLES.get(s["host"], "mlp") for s in block["sublayers"]}
    data: dict[str, Any] = {"schema": block, "nodes": nodes, "identity_html": str(highlight_python(block["identity"], roles)), "tower": None}
    shown = list(family_quirks)
    model: dict[str, Any] = {**info, "block": block, "roles": roles, "ledgers": ledgers(info),
                             "eager_only": [row["name"] for row in info["support"] if row["condition"] and row["condition"]["kind"] == "eager"],
                             "repr_html": highlight_repr(info["repr"], roles), "logits_label": logits_label(family_quirks)}
    v = info.get("vision")
    if v:
        tower, wrapper = v["tower"], v["wrapper_fields"]
        schema = block_schema(f"{entry.MODEL_TYPE} ({tower['slug']} tower)", tower["block"], v["block"],
                              base="model.vision.layers[i]", stream="tower's stream")
        tower_nodes = schema.pop("nodes")
        attention = next((key for key in tower_nodes if key.startswith("sub.") and "self_attn" in key), None)
        if attention:
            tower_nodes[attention]["extra"] += " " + tower["masking"]
        tower_nodes.update(tower_path_nodes(v))
        nodes.update({"v:" + key: value for key, value in tower_nodes.items()})
        schema["roles"] = {s.get("key", s["host"]): HOST_ROLES.get(s["host"], "mlp") for s in schema["sublayers"]}
        data["tower"] = {"schema": schema, "identity_html": str(highlight_python(schema["identity"], roles))}
        seen = {q["slug"] for q in shown}
        for slug in ["vision", *tower["quirks"], *wrapper.get("quirks", [])]:
            if slug not in seen:
                seen.add(slug)
                shown.append(quirk(slug, f"{entry.MODEL_TYPE} {v['wrapper']}"))
        model["tower"] = {
            "title": tower["title"], "wrapper": wrapper["title"], "schema": schema, "identity_html": data["tower"]["identity_html"],
            "rows": tower["rows"], "positions": tower["positions"], "masking": tower["masking"], "norm": tower["norm"],
            "has_norm": v["has_norm"], "num_layers": v["block"]["num_layers"], "layer_class": v["layer_class"],
            "module_class": v["module_class"], "projector": v["projector"], "sizes": v["sizes"], "envoys": v["envoys"],
            # The parts' headings are h2; the notes' own headings sit under them.
            "notes": Markup(str(md(tower["notes"], roles=roles)).replace("<h2", "<h3").replace("</h2>", "</h3>")),
            "wrapper_notes": Markup(str(md(wrapper.get("notes", ""), roles=roles)).replace("<h2", "<h3").replace("</h2>", "</h3>")),
        }
    model["quirks"] = shown
    model["data"] = data
    return model


def logits_label(family_quirks: list[dict[str, str]]) -> str:
    slugs = {q["slug"] for q in family_quirks}
    return "softcapped" if "softcapped-logits" in slugs else "scaled" if "scaled-logits" in slugs else "the output"


def quirk(slug: str, where: str) -> dict[str, str]:
    assert slug in QUIRKS, f"{where}: unknown quirk {slug!r}; known: {sorted(QUIRKS)}"
    label, blurb = QUIRKS[slug]
    return {"slug": slug, "label": label, "blurb": blurb}


def page_model(entry: ModuleType, read: list[dict[str, Any]], default: str) -> dict[str, Any]:
    """The page: the family's shared parts from the default checkpoint, and every available checkpoint's panes."""
    by_id = {c["id"]: c for c in read}
    assert "info" in by_id[default], f"{entry.MODEL_TYPE}: the default checkpoint {default} is unavailable: {by_id[default].get('unavailable')}"
    info = by_id[default]["info"]
    family_quirks = quirks(entry)
    models = {c["id"]: checkpoint_model(entry, c["info"], family_quirks) for c in read if "info" in c}
    module = environment().get_template("panes.html.j2").make_module({"eye": EYE})
    panes: dict[str, list[dict[str, Any]]] = {}
    for name in PANES:
        groups: dict[str, list[str]] = {}
        for cid, model in models.items():
            groups.setdefault(str(getattr(module, name)(model)).strip(), []).append(cid)
        panes[name] = [{"html": Markup(text), "ckpts": ids, "shown": default in ids} for text, ids in groups.items()]
    wrappers = getattr(entry, "WRAPPERS", {})
    options = []
    for c in read:
        v = models[c["id"]]["vision"] if c["id"] in models else None
        options.append({
            "id": c["id"], "name": c["id"].split("/")[-1], "url": f"https://huggingface.co/{c['id']}",
            "reason": c.get("unavailable"), "vision": bool(v),
            "tower": v["tower"]["title"] if v else None, "wrapper": v["wrapper_fields"]["title"] if v else None,
            "group": "unavailable" if "unavailable" in c else "vision" if v else "text",
        })
    groups = [(label, [o for o in options if o["group"] == key])
              for key, label in (("text", "text"), ("vision", "vision-language"), ("unavailable", "not available"))]
    pinned = [entry.PINNED] + [w["pinned"] for w in wrappers.values() if w["pinned"] != entry.PINNED]
    roles = models[default]["roles"]
    available = [models[o["id"]] for o in options if o["id"] in models]
    towers_found = list(dict.fromkeys(m["tower"]["title"] for m in available if m.get("tower")))
    return {
        **info,
        "model_type": entry.MODEL_TYPE,
        "title": entry.TITLE,
        "subtitle": entry.SUBTITLE,
        "palette": palette(entry),
        "default": default,
        "options": options,
        "option_groups": [(label, items) for label, items in groups if items],
        "checkpoints": [{"id": o["id"], "url": o["url"]} for o in options],
        "pinned": pinned,
        "vllm": getattr(entry, "VLLM", False),
        "quirks": family_quirks + ([quirk("vision", entry.MODEL_TYPE)] if towers_found else []),
        "notes": md(entry.NOTES, roles=roles),
        "docstring": md(info["docstring"], rst=True, roles=roles),
        "panes": panes,
        "eye": EYE,
        "identity_html": Markup(models[default]["data"]["identity_html"]),
        "checkpoints_json": embed_json({"default": default, "checkpoints": {cid: m["data"] for cid, m in models.items()}}),
        "blocks": sorted({m["num_layers"] for m in available}),
        "towers": towers_found,
        "wrappers": list(dict.fromkeys(m["tower"]["wrapper"] for m in available if m.get("tower"))),
        "source_url": f"{GITHUB}/{info['family_file']}",
        "test_url": f"{GITHUB}/tests/families/test_{entry.MODEL_TYPE}.py",
        "families_url": f"{GITHUB}/docs/reference/families.md",
        "built": dt.date.today().isoformat(),
    }


def environment() -> Environment:
    env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["md"] = md
    env.filters["code"] = lambda s: Markup(f"<code>{html.escape(str(s))}</code>")
    # a sentence with code names in backticks, as the hover cards show it
    env.filters["inline"] = lambda s: Markup(re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(str(s))))
    return env


def read_entry(entry: ModuleType, reference: str | None = None, checkpoints: list[str] | None = None) -> tuple[list[dict[str, Any]], str]:
    """The entry's checkpoints, each read (`read_checkpoint`), and the default one: ``reference`` or the entry's
    ``REFERENCE``. Given a ``reference`` and no ``checkpoints``, that one alone (the test suite's pinned build)."""
    default = reference or entry.REFERENCE
    ids = list(checkpoints or ([reference] if reference else entry.CHECKPOINTS))
    if default not in ids:
        ids.insert(0, default)
    cache: dict[Any, dict[str, Any]] = {}
    return [read_checkpoint(entry, checkpoint, cache) for checkpoint in ids], default


def build_page(entry: ModuleType, reference: str | None = None, checkpoints: list[str] | None = None) -> str:
    """One family page as HTML: over ``checkpoints`` (the entry's by default), opening on ``reference``."""
    read, default = read_entry(entry, reference, checkpoints)
    return environment().get_template("family.html.j2").render(**page_model(entry, read, default))


def index_model(built: list[dict[str, Any]]) -> dict[str, Any]:
    done = {page["model_type"] for page in built}
    stubs = [name for name in nnterp.families.known() if name not in done]
    found = list(dict.fromkeys(title for page in built for title in page["towers"]))
    return {"pages": built, "stubs": stubs, "total": len(nnterp.families.known()), "built": dt.date.today().isoformat(),
            "palette": site_palette(), "eye": EYE,
            # the vision slugs live on the pages; the first filter row keeps the text quirks and the Vision chip
            "quirks": [{"slug": s, "label": l} for s, (l, _) in QUIRKS.items() if s not in VISION_QUIRKS],
            "towers": [{"slug": tower_slug(title), "label": title} for title in found]}


def tower_slug(title: str) -> str:
    return "tower-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


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
        read, default = read_entry(entry)
        model = page_model(entry, read, default)
        path = out / f"{entry.MODEL_TYPE}.html"
        path.write_text(env.get_template("family.html.j2").render(**model))
        written.append(path)
        blocks = model["blocks"]
        cards.append({**{k: model[k] for k in ("model_type", "title", "subtitle", "palette", "checkpoints", "quirks", "vllm",
                                                "architecture", "family_module", "towers", "wrappers")},
                      "blocks": str(blocks[0]) if len(blocks) == 1 else f"{blocks[0]}–{blocks[-1]}",
                      "tower_slugs": [tower_slug(t) for t in model["towers"]]})
        print(f"wrote {path.relative_to(HERE.parent)}")
    if not only:
        index = out / "index.html"
        index.write_text(env.get_template("index.html.j2").render(**index_model(cards)))
        written.append(index)
        print(f"wrote {index.relative_to(HERE.parent)}")
    return written


if __name__ == "__main__":
    build(sys.argv[1:] or None)
