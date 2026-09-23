"""The rows that read an operation of a forward, on every family.

A ``.source`` row names operations the way nnsight names them
(``attention_interface_1``, ``nn_functional_dropout_0``), which is a fact about
the transformers version installed, not about the architecture: an upgrade that
moves a call, renames a local or routes a family through the shared attention
interface silently invalidates a row. nnterp has been broken that way twice.

This is the suite that catches it: every family of ``invariant_models`` is loaded
with the attention probabilities *enabled*, so the op has to resolve, and the
tensor it resolves to is checked to be a pattern rather than merely present. A
row nnterp declares unavailable on a family must say why; it is never skipped
silently. When a row does break, ``LayerAccessor.get_operation`` raises naming
the row, the class, the transformers version and every operation that does exist
at that level of the source, which is what makes the fix a one-row edit.
"""

from pathlib import Path

import nnsight
import pytest
import torch as th
import yaml

from nnterp import StandardizedTransformer
from nnterp.rename_utils import check_attention_probabilities

MODELS = yaml.safe_load((Path(__file__).parent / "test_config.yaml").read_text())[
    "invariant_models"
]
# token ids rather than text: GPT-OSS's tiny checkpoint has a 1000-entry embedding
# under a 200k-entry tokenizer
TOKENS = th.tensor([[3, 4, 5, 6]])
#: What the softmax over the keys is called where a family's forward makes it
#: visible next to the operation the row reads. Its input is the scores.
SOFTMAX_OPS = ("nn_functional_softmax_0", "F_softmax_0")


@pytest.fixture(scope="module", params=list(MODELS))
def with_probs(request):
    """Loading with enable_attention_probs=True is itself the first assertion:
    it resolves every op the family's rows name and runs
    ``check_attention_probabilities``."""
    model = StandardizedTransformer(
        request.param, enable_attention_probs=True, device_map="cpu", dtype=th.float32
    )
    return model, MODELS[request.param]


def operation_rows(model) -> list[str]:
    return [name for name in model.internals if model.internals[name].address.op]


def test_every_operation_row_reads_a_tensor_on_every_layer(with_probs):
    """Resolving the name is not enough: the operation is read on every layer
    that has the place, in forward order, and a layer that does not have it says
    why."""
    model, _ = with_probs
    rows = operation_rows(model)
    assert "attention_probabilities" in rows, rows
    places = []
    for name in rows:
        accessor = model.internals[name]
        for layer in range(model.num_layers):
            reason = accessor.unavailable_on(layer)
            if reason is None:
                places.append((model.internals.rank(name, layer), name, layer))
            else:
                assert reason.strip(), (name, layer)
    assert places, f"{rows} are unavailable on every layer of this model"
    with th.no_grad(), model.trace(TOKENS):
        got = nnsight.save({})
        for _, name, layer in sorted(places):
            got[f"{name}[{layer}]"] = model.internals[name][layer].clone()
    for key, tensor in dict(got).items():
        assert isinstance(tensor, th.Tensor), (key, type(tensor))


def test_the_attention_probabilities_are_a_pattern(with_probs):
    """The row reads the pattern the values are mixed with, not a neighbouring
    tensor of the same shape: non-negative, one row per query over the keys, and
    summing to one — to less than one on a family with an attention sink, whose
    row says so with a tag. ``check_attention_probabilities`` also writes them
    and checks the logits move."""
    model, _ = with_probs
    layer = model.attention_layers[0]
    accessor = model.attention_probabilities
    with th.no_grad(), model.trace(TOKENS):
        probs = accessor[layer].clone().save()
    batch, seq = TOKENS.shape
    assert probs.shape == (batch, model.num_heads, seq, seq)
    assert (probs >= 0).all()
    sums = probs.sum(-1)
    if "sink" in accessor.address.tags:
        # the softmax spans the keys plus a sink, and the sink is dropped
        assert (sums > 0).all() and (sums < 1 - 1e-4).all(), float(sums.max())
    else:
        assert th.allclose(sums, th.ones_like(sums), atol=1e-5)
    check_attention_probabilities(model, layer)


def test_the_attention_probabilities_are_the_softmax_of_the_scores(with_probs):
    """Where the family's forward makes the softmax visible beside the operation
    the row reads, the two are compared: a row that drifted onto the scores, the
    values or the post-mix output would still look like a tensor of the right
    shape, and only this says it is the pattern."""
    model, _ = with_probs
    layer = model.attention_layers[0]
    accessor = model.attention_probabilities
    with th.no_grad(), model.trace(TOKENS):
        source = accessor.get_operation(layer, containing_source=True)
        found = [name for name in SOFTMAX_OPS if getattr(source, name, None) is not None]
        assert found, (
            "no softmax operation beside the one attention_probabilities reads; "
            f"the source has:\n{source}"
        )
        scores = getattr(source, found[0]).input.clone().save()
        probs = accessor[layer].clone().save()
    manual = th.softmax(scores.float(), dim=-1)
    # a sink family softmaxes over the keys plus the sink and drops the sink, so
    # the pattern is the leading part of that distribution
    assert th.allclose(manual[..., : probs.shape[-1]], probs, atol=1e-5)
