"""What nnterp's accessors mean, asserted on every family.

The accessors are defined by what they are *of* the block, so on every model that
has them:

    layers_mid       == layers_input + attentions_output
    layers_output    == layers_mid   + mlps_output
    mlps_norm_output == mlps_input

and on a parallel block, which has no mid-stream,
``layers_output == layers_input + attentions_output + mlps_output``. A family where
a name resolves to a module that does not satisfy these reads a real tensor of the
right shape from the wrong place (Gemma-2's ``post_attention_layernorm`` is the
attention's post-norm, GPT-NeoX's normalizes the block input), so nothing else
would notice.
"""

from pathlib import Path

import nnsight
import pytest
import torch as th
import yaml

from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError

MODELS = yaml.safe_load((Path(__file__).parent / "test_config.yaml").read_text())[
    "invariant_models"
]
# token ids rather than text: GPT-OSS's tiny checkpoint has a 1000-entry embedding
# under a 200k-entry tokenizer
TOKENS = th.tensor([[3, 4, 5, 6, 7]])


@pytest.fixture(scope="module", params=list(MODELS))
def loaded(request):
    # fp32: the Falcon checkpoints are stored in bfloat16, where the additive
    # identity holds only to one unit of the format (0.0156 at magnitude 2)
    model = StandardizedTransformer(
        request.param, enable_attention_probs=False, device_map="cpu", dtype=th.float32
    )
    return model, MODELS[request.param]


def read(model, layer: int, *names: str) -> dict[str, th.Tensor]:
    """Each value cloned *as it is reached*, in forward order. Falcon adds its
    attention output into its MLP output in place, so an MLP output cloned after
    the block has finished already contains the attention's."""
    ordered = sorted(names, key=lambda name: model.internals[name].address.order)
    with th.no_grad(), model.trace(TOKENS):
        got = nnsight.save({})
        for name in ordered:
            got[name] = model.internals[name][layer].clone()
    return dict(got)


def tensor_accessors(model) -> list[str]:
    return [name for name in model.internals if model.internals[name].io_type is not None]


def test_the_block_structure_is_published(loaded):
    model, expected = loaded
    assert model.block_structure == expected["structure"]


def test_what_a_family_lacks_is_known_before_any_trace(loaded):
    model, expected = loaded
    status = model.internals.status()
    status.pop("attention_probabilities")  # disabled by how the model was loaded here
    assert sorted(name for name, reason in status.items() if reason) == sorted(
        expected.get("unavailable", [])
    )
    for name in expected.get("unavailable", []):
        with pytest.raises(RenamingError):
            model.internals[name][0]


def test_every_available_accessor_reads_a_tensor_on_every_layer(loaded):
    model, expected = loaded
    status = model.internals.status()
    names = [name for name in tensor_accessors(model) if status[name] is None]
    dense = expected.get("dense_layers")
    for layer in range(model.num_layers):
        here = [
            name
            for name in names
            if dense is None or layer in dense or name not in ("mlps_activation", "mlps_neurons")
        ]
        got = read(model, layer, *here)
        assert sorted(got) == sorted(here)
        for name, tensor in got.items():
            assert tensor.shape[:2] == TOKENS.shape, (name, tensor.shape)
    if dense is not None:
        sparse = next(layer for layer in range(model.num_layers) if layer not in dense)
        with pytest.raises(RenamingError, match=f"does not exist on layer {sparse}"):
            model.mlps_activation[sparse]


def test_the_residual_identities_hold(loaded):
    model, expected = loaded
    status = model.internals.status()
    if status["mlps_output"] is not None:  # OPT: no MLP module, so no MLP contribution to add
        return
    wanted = ["layers_input", "attentions_output", "mlps_input", "mlps_output", "layers_output"]
    wanted += [name for name in ("layers_mid", "mlps_norm_output") if status[name] is None]
    for layer in range(model.num_layers):
        got = read(model, layer, *wanted)
        if "layers_mid" in got:
            assert th.allclose(got["layers_input"] + got["attentions_output"], got["layers_mid"], atol=1e-5)
            assert th.allclose(got["layers_mid"] + got["mlps_output"], got["layers_output"], atol=1e-5)
        else:
            assert model.block_structure == "parallel"
            total = got["layers_input"] + got["attentions_output"] + got["mlps_output"]
            assert th.allclose(total, got["layers_output"], atol=1e-5)
        if "mlps_norm_output" in got:
            assert th.equal(got["mlps_norm_output"], got["mlps_input"])
        # a contribution is not the stream it is added to
        assert not th.allclose(got["attentions_output"], got["layers_input"] + got["attentions_output"])


def test_the_published_sizes_are_the_tensors(loaded):
    """Against the tensors and the modules, not the config they were read from: a
    GPT-2 config can say intermediate_size 37 beside 128-wide MLPs."""
    model, expected = loaded
    status = model.internals.status()
    layer = expected.get("dense_layers", [0])[0]
    names = [
        name
        for name in ("layers_input", "attentions_premix", "mlps_activation", "mlps_neurons")
        if status[name] is None
    ]
    got = read(model, layer, *names)
    assert got["layers_input"].shape[-1] == model.hidden_size
    assert got["attentions_premix"].shape[-1] == model.num_heads * model.head_dim
    for name in ("mlps_activation", "mlps_neurons"):
        if name in got:
            assert got[name].shape[-1] == model.intermediate_size, name
    # where the family has separate projections, they are as wide as the heads say
    # (multi-head latent attention has none of these: its q/k/v are low-rank pairs)
    attention = model.attentions[layer]._module
    projections = {name: getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj")}
    if projections["k_proj"] is not None:
        assert projections["k_proj"].out_features == model.num_kv_heads * model.qk_head_dim
        assert projections["v_proj"].out_features == model.num_kv_heads * model.head_dim
    if projections["q_proj"] is not None:
        assert projections["q_proj"].out_features == model.num_heads * model.qk_head_dim
    assert model.num_heads % model.num_kv_heads == 0


def test_a_write_at_the_new_accessors_lands(loaded):
    """Each is an address to write at too: zeroing the mid-stream's attention share
    is zeroing attentions_output, by the identity."""
    model, _ = loaded
    if model.internals.status()["layers_mid"] is not None:
        return
    clean = read(model, 0, "layers_input", "layers_mid")
    with th.no_grad(), model.trace(TOKENS):
        model.attentions_output[0] = th.zeros_like(model.attentions_output[0])
        mid = model.layers_mid[0].clone().save()
    assert th.allclose(mid, clean["layers_input"], atol=1e-6)
    assert not th.allclose(mid, clean["layers_mid"])
