"""vLLM's best-effort family, for a ``model_type`` no vLLM family module covers.

`nnterp.families.lookup` falls back to it, with a warning, for
``engine="vllm"``, as it falls back to `nnterp.families.default` for
transformers. vLLM's implementations keep transformers' module names, so
``RENAME`` is that default's, and ``ENVOYS`` is keyed on the same standard
names, over `nnterp.components.vllm`'s envoys.

What vLLM changes is the block's convention, which a family states and the
default reads off the block's forward instead: a block that takes a
``residual`` and reads it is fused (`FusedLayer`: the stream is
``hidden_states + residual``, entering and leaving); any other is called with
the stream (`Layer`), whose position in the call is the ``hidden_states``
argument's, and returns it alone or first in a tuple, as its ``return`` says
(Exaone-4 and Cohere take a ``residual`` they overwrite before reading).

`check` is the transformers default's without the scan, which vLLM has no
forward for: the root must have ``embed_tokens``, ``layers`` and ``norm``
(``lm_head`` may be tied into the embedding, which `project_on_vocab` then
unembeds with), and each sublayer's contribution is reported unavailable
where its block's forward does not add its output as is (a residual add fused
into the next norm counts as the add). The attention interior is read on the
module's ``attn`` child, vLLM's attention layer, and is unavailable where the
module has none.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

import torch
from nnsight.intervention.envoy import Envoy

from .. import default
from ...components import DerivedEProperty, HeadOutputs, Keys, Queries, Residual, Values
from ...components.vllm import Attention, Flat, FusedLayer, Layer, Mlp, on_decode_step, project
from ...components.vllm.attention import probabilities, scores

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

MODEL_TYPES = ()

RENAME = default.RENAME


# -- the block's convention, off its forward ---------------------------------------

def _forward(block_type: type) -> ast.FunctionDef | None:
    try:
        function = ast.parse(textwrap.dedent(inspect.getsource(block_type.forward))).body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    return function if isinstance(function, ast.FunctionDef) else None


@functools.lru_cache(maxsize=None)
def fused(block_type: type) -> bool:
    """Whether the block is fused: it takes a ``residual`` and reads it before assigning it."""
    function = _forward(block_type)
    if function is None or "residual" not in [arg.arg for arg in function.args.args]:
        return False
    uses = sorted(
        (node for node in ast.walk(function) if isinstance(node, ast.Name) and node.id == "residual"),
        key=lambda node: (node.lineno, node.col_offset),
    )
    return bool(uses) and isinstance(uses[0].ctx, ast.Load)


@functools.lru_cache(maxsize=None)
def _call(block_type: type) -> tuple[int, bool]:
    """``(index of the stream in the call, whether the block returns a tuple)`` for a block called with the stream."""
    function = _forward(block_type)
    if function is None:
        return 0, False
    arguments = [arg.arg for arg in function.args.args[1:]]
    stream = arguments.index("hidden_states") if "hidden_states" in arguments else 0
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return) and node.value is not None]
    return stream, any(isinstance(node.value, ast.Tuple) for node in returns)


# -- the envoys ----------------------------------------------------------------------

class StreamLayer(Layer):
    """A block called with the residual stream, which it returns alone or first in a tuple, as its forward says."""

    @property
    def STREAM(self) -> int:  # noqa: N802 (the base's class attribute, read per block here)
        return _call(type(self._module))[0]

    @property
    def returns_tuple(self) -> bool:
        return _call(type(self._module))[1]

    def support(self) -> dict[str, str | None]:
        return {**super().support(), **default.missing_modules(self, {"self_attn": Attention, "mlp": Mlp})}


class FusedLayer(FusedLayer):
    """A block with the residual add fused into the next norm: the stream is ``hidden_states + residual``."""

    def support(self) -> dict[str, str | None]:
        return {**super().support(), **default.missing_modules(self, {"self_attn": Attention, "mlp": Mlp})}


class Layers(Envoy):
    """The block container: wraps each block in `FusedLayer` or `StreamLayer`, as its forward says (see `default.Layers`)."""

    def _resolve_envoy_class(self, module: torch.nn.Module, path: str) -> type[Envoy]:
        envoy_class = super()._resolve_envoy_class(module, path)
        if envoy_class is not Envoy:
            return envoy_class
        return FusedLayer if fused(type(module)) else StreamLayer


def _interior(envoy: "Attention") -> str | None:
    """Why the module's interior is not on an ``attn`` child that is vLLM's attention layer, or ``None``."""
    layer = getattr(envoy._module, "attn", None)
    if layer is None or not hasattr(layer, "impl"):
        return (
            f"{type(envoy._module).__name__} has no `attn` child that is vLLM's attention layer, where the default "
            "family reads the interior; a family module maps it"
        )
    return None


def _head_outputs(envoy: "Attention") -> str | None:
    """Why ``attn``'s output is not what the output projection takes (the forward transforms it first), or ``None``."""
    reason = _interior(envoy)
    if reason is not None:
        return reason
    after = default._after_interface(type(envoy._module), "attn")
    return f"{after}, so it is not what the output projection takes; a family module maps it" if after else None


def _pattern(envoy: "Attention") -> str | None:
    return _interior(envoy) or on_decode_step(envoy)


class Attention(Attention):
    """An attention module as the default finds it: the base's values, unavailable where the default cannot vouch for them."""

    attention_scores = DerivedEProperty(scores, description=Attention.attention_scores.description, unavailable=_pattern)
    attention_probabilities = DerivedEProperty(probabilities, description=Attention.attention_probabilities.description, unavailable=_pattern)

    @Flat("attn.inputs", select=0, heads="first", description=Attention.attention_queries.description, unavailable=_interior)
    def attention_queries(self, value: torch.Tensor) -> Queries:
        return value

    @Flat("attn.inputs", select=1, heads="first", description=Attention.attention_keys.description, unavailable=_interior)
    def attention_keys(self, value: torch.Tensor) -> Keys:
        return value

    @Flat("attn.inputs", select=2, heads="first", description=Attention.attention_values.description, unavailable=_interior)
    def attention_values(self, value: torch.Tensor) -> Values:
        return value

    @Flat("attn.output", heads="last", description=Attention.attention_head_outputs.description, unavailable=_head_outputs)
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        return value

    @Flat("output", description=Attention.attention_output.description, unavailable=default._contribution)
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value


class Mlp(Mlp):
    """A feed-forward module as the default finds it: its output, unavailable where its block does not add it as is."""

    @Flat("output", description=Mlp.mlp_output.description, unavailable=default._contribution)
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value


#: Keyed on the standard names (nnsight matches a string key on alias paths too).
ENVOYS = {"layers": Layers, "self_attn": Attention, "mlp": Mlp}


# -- the load-time check ---------------------------------------------------------------

def check(model: "StandardizedVLLM") -> None:
    """Confirm the default's names on a freshly built model; called by `StandardizedVLLM` at load.

    `default.check_names` with the head optional and fused adds counted as
    adds: raises `UnsupportedFamily` when the root lacks ``embed_tokens``,
    ``layers`` or ``norm``, and records why a sublayer's output is not its
    contribution. There is no shape check: vLLM has no forward to scan.
    """
    default.check_names(model, (FusedLayer, StreamLayer), required=("embed_tokens", "layers", "norm"), fused=True)


# -- the logit lens and the sizes ----------------------------------------------------

def project_on_vocab(model: "StandardizedVLLM", hidden: torch.Tensor) -> torch.Tensor:
    """The final norm, then vLLM's logits processor over ``lm_head`` (with its bias, if it has one), or over ``embed_tokens`` where the head is tied into it."""
    head = model.__dict__.get("lm_head")
    if head is None:
        return project(model, hidden, model.embed_tokens)
    bias = getattr(head._module, "bias", None)
    return project(model, hidden, head) if bias is None else project(model, hidden, head, head.bias)


def intermediate_size(model: Any) -> int:
    """The config's ``intermediate_size`` / ``ffn_dim`` / ``ffn_hidden_size`` / ``n_inner`` (``None`` being four times the hidden size): vLLM's modules hold their tensor-parallel share, so the config is what says."""
    config = model.config.get_text_config()
    for key in ("intermediate_size", "ffn_dim", "ffn_hidden_size", "n_inner"):
        if isinstance(getattr(config, key, None), int):
            return getattr(config, key)
    if hasattr(config, "n_inner"):  # ``None``: transformers' convention for four times the hidden size
        return 4 * model.hidden_size
    raise NotImplementedError(
        f"{type(config).__name__} has no intermediate_size, ffn_dim, ffn_hidden_size or n_inner; a family module defines `def intermediate_size(model)`"
    )
