"""The best-effort default family: forced onto families that have their own, and on a model_type that has none.

Forced onto a shipped family's checkpoint, the default must run the whole
`FamilySuite` (GPT-2, Llama, GPT-NeoX, OPT, Phi), and every value it reports
available must read exactly what the dedicated family reads, on every shipped
family whose checkpoint runs here: where the dedicated family does arithmetic
of its own, the default reports the value unavailable instead of a wrong number.
"""

import importlib
import re
import warnings

import pytest
import torch
from nnsight import TransformersModel  # nnsight before any transformers submodule
from nnsight.intervention.envoy import Envoy
from suite import PROMPT, FamilySuite, recurrent_mixer
from transformers import AutoConfig

from nnterp import StandardizedTransformer, UnsupportedFamily, families
from nnterp.components import SelectiveScan, StateSpace
from nnterp.families import default


def load_default(repo, **kwargs):
    """``repo`` standardized by the default family, as if its ``model_type`` had none."""
    model_type = AutoConfig.from_pretrained(repo).get_text_config().model_type
    families.REGISTRY[model_type] = default
    try:
        return StandardizedTransformer(repo, **kwargs)
    finally:
        del families.REGISTRY[model_type]


def dedicated_suite(name):
    """The shipped family's `FamilySuite` class: its checkpoint and its quirks."""
    module = importlib.import_module(f"test_{name}")
    family = getattr(families, name)
    return next(
        cls for cls in vars(module).values()
        if isinstance(cls, type) and issubclass(cls, FamilySuite) and cls.FAMILY is family and "REPO" in vars(cls)
    )


class DefaultSuite(FamilySuite):
    """`FamilySuite` with the default family forced onto the checkpoint."""

    FAMILY = default

    @pytest.fixture(scope="class")
    def model(self, request):
        cls = request.cls
        return load_default(cls.REPO, dispatch=True, attn_implementation="eager", **cls.LOAD_KWARGS)

    def expected_values(self, model):
        """A module the default found on no block is listed as unavailable, not left out (OPT's ``mlp``)."""
        return super().expected_values(model) | {"mlp.mlp_output"}


def forced(name, **overrides):
    """A `DefaultSuite` over the shipped family ``name``'s checkpoint and suite settings."""
    source = dedicated_suite(name)
    settings = {key: getattr(source, key) for key in dir(source) if key.isupper() and key != "FAMILY"}
    return type(f"TestDefaultOn{source.__name__.removeprefix('Test')}", (DefaultSuite,), {**settings, **overrides})


TestDefaultOnGPT2 = forced("gpt2")
TestDefaultOnLlama = forced("llama")
TestDefaultOnGPTNeoX = forced("gpt_neox")
TestDefaultOnPhi = forced("phi")
TestDefaultOnOPT = forced("opt", EXPECTED_UNAVAILABLE={"mlp.mlp_output": "no mlp module found"})


NANOCHAT = "hf-tiny-v2/tiny-random-NanoChatForCausalLM"


@pytest.fixture(scope="module")
def nanochat():
    """NanoChat has no family module: the default is what a load resolves to, with a warning."""
    with pytest.warns(UserWarning, match=r"no family for model_type 'nanochat'.*best-effort.*model\.support\(\)"):
        return StandardizedTransformer(NANOCHAT, dispatch=True, attn_implementation="eager")


def test_unknown_model_type_loads_with_the_default(nanochat):
    assert nanochat.family is default
    assert type(nanochat.layers[0]) is default.Layer and type(nanochat.layers[0].self_attn) is default.Attention
    assert all(reason is None for reason in nanochat.support().values())
    assert (nanochat.num_layers, nanochat.num_heads, nanochat.head_dim, nanochat.intermediate_size) == (2, 2, 16, 32)


def test_unknown_model_type_reads_the_raw_model(nanochat):
    """Each standard value is the native module's, and the contributions add up to the stream."""
    raw = TransformersModel(NANOCHAT, task="text-generation", dispatch=True, attn_implementation="eager")
    native, standard = [], []
    with raw.trace(PROMPT):
        for layer in raw.model.layers:
            native.append((layer.self_attn.output[0].save(), layer.mlp.output.save(), layer.output.save()))
        logits = raw.output.logits.save()
    with nanochat.trace(PROMPT):
        for layer in nanochat.layers:
            standard.append((layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save()))
        lens = nanochat.project_on_vocab(nanochat.layers[-1].layer_output).save()
        ours = nanochat.logits.save()
    for (attn, mlp, out), (x, our_attn, our_mlp, our_out) in zip(native, standard):
        assert torch.equal(our_attn, attn) and torch.equal(our_mlp, mlp) and torch.equal(our_out, out)
        torch.testing.assert_close(x + our_attn + our_mlp, our_out)
    assert torch.equal(ours, logits)
    torch.testing.assert_close(lens, logits)


# -- the default against every dedicated family -----------------------------------------

def comparable():
    """Every shipped family but those whose mixer runs a kernel the default does not route (Mamba-1/2 and their hybrids)."""
    names = []
    for name in families.known():
        if name == "gpt_neox_japanese":
            continue  # its container is named after the model_type; see test_unknown_container_is_refused_with_a_rename
        mixer = recurrent_mixer(getattr(families, name))
        if mixer is None or not issubclass(mixer, (SelectiveScan, StateSpace)):
            names.append(name)
    return names


VALUES = (
    ("", "layer_output"), ("self_attn", "attention_output"), ("mlp", "mlp_output"),
    ("self_attn", "attention_queries"), ("self_attn", "attention_keys"), ("self_attn", "attention_values"),
    ("self_attn", "attention_scores"), ("self_attn", "attention_probabilities"), ("self_attn", "attention_head_outputs"),
)


def counterpart(dedicated, model, layer, host):
    """The dedicated family's name for the module the default calls ``host`` on block ``layer``.

    A hybrid whose transformers code keeps both mixers under ``self_attn``
    (Kimi-Linear) has its family name the recurrent one ``linear_attn``; the
    default knows no mixer kinds, so it calls both ``self_attn``.
    """
    theirs = dedicated.layers[layer]
    if not host or theirs.__dict__.get(host) is not None:
        return host
    mine = getattr(model.layers[layer], host).path  # the native path: the two loads share no module
    return next((name for name, child in vars(theirs).items() if isinstance(child, Envoy) and child.path == mine), host)


def read(model, layer, host, value):
    saved = {}
    with model.trace(PROMPT):
        block = model.layers[layer]
        saved["value"] = getattr(getattr(block, host) if host else block, value).save()
    return saved["value"]


@pytest.mark.parametrize("name", comparable())
def test_default_reads_what_the_family_reads(name):
    """Wherever the default reports a value available, it reads exactly what the dedicated family reads."""
    source = dedicated_suite(name)
    kwargs = dict(dispatch=True, attn_implementation="eager", **source.LOAD_KWARGS)
    dedicated = StandardizedTransformer(source.REPO, **kwargs)
    model = load_default(source.REPO, **kwargs)
    compared = 0
    for i in range(len(model.layers)):
        theirs, ours = dedicated.support(layer=i), model.support(layer=i)
        for host, value in VALUES:
            key = f"{host}.{value}" if host else value
            if ours.get(key, "absent") is not None:
                continue
            other = counterpart(dedicated, model, i, host)
            their_key = f"{other}.{value}" if other else value
            assert theirs.get(their_key) is None, f"layer {i}: the default serves {key}, which {name} reports unavailable: {theirs.get(their_key)}"
            expected, actual = read(dedicated, i, other, value), read(model, i, host, value)
            assert torch.equal(actual, expected), f"layer {i}: {key}"
            compared += 1
    with dedicated.trace(PROMPT):
        expected = dedicated.logits.save()
    with model.trace(PROMPT):
        actual = model.logits.save()
    assert torch.equal(actual, expected)
    assert compared >= len(model.layers)  # at least layer_output on every block


# -- what the default reports and refuses -----------------------------------------------

def test_lookup_falls_back_to_the_default_with_a_warning():
    with pytest.warns(UserWarning, match="no family for model_type 'not_a_model_type'"):
        assert families.lookup("not_a_model_type") is default
    assert "default" not in families.known()


def test_registered_and_shipped_families_win_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert families.lookup("gpt2") is families.gpt2


def test_post_norm_contributions_are_unavailable():
    """Gemma-2 norms each sublayer's output before adding it: the module output is not the contribution."""
    model = load_default(dedicated_suite("gemma2").REPO)
    support = model.support(layer=0)
    assert "post_attention_layernorm" in support["self_attn.attention_output"]
    assert "post_feedforward_layernorm" in support["mlp.mlp_output"]
    assert support["layer_output"] is None


def test_residual_inside_the_sublayer_is_unavailable():
    """BLOOM's attention and MLP take the residual and add it themselves."""
    model = load_default(dedicated_suite("bloom").REPO)
    support = model.support(layer=0)
    assert "takes `residual`" in support["self_attn.attention_output"]
    assert "takes `residual`" in support["mlp.mlp_output"]
    assert "attention_interface" in support["self_attn.attention_probabilities"]


def test_a_value_the_default_cannot_trust_raises_at_the_read():
    model = load_default(dedicated_suite("gemma2").REPO)
    with pytest.raises(Exception, match="post_attention_layernorm"):
        with model.trace(PROMPT):
            model.layers[0].self_attn.attention_output.save()


def test_unknown_names_are_refused_with_a_rename():
    """RWKV's tree uses no spelling the default knows: the error names a rename read off the module tree."""
    with pytest.warns(UserWarning, match="no family for model_type 'rwkv'"):
        with pytest.raises(UnsupportedFamily) as refused:
            StandardizedTransformer("hf-internal-testing/tiny-random-RwkvForCausalLM")
    message = str(refused.value)
    assert "found no embed_tokens, layers, norm, lm_head" in message
    assert "'rwkv.blocks': 'layers'" in message and "'head': 'lm_head'" in message
    assert "adding-a-family.md" in message


def test_unknown_container_is_refused_with_a_rename():
    """GPT-NeoX-Japanese keeps its modules under ``gpt_neox_japanese``; the suggested rename completes the default."""
    repo = dedicated_suite("gpt_neox_japanese").REPO
    with pytest.raises(UnsupportedFamily, match=re.escape("'gpt_neox_japanese.layers': 'layers'")):
        load_default(repo)
    rename = {"gpt_neox_japanese.embed_in": "embed_tokens", "gpt_neox_japanese.layers": "layers", "gpt_neox_japanese.final_layer_norm": "norm"}
    model = load_default(repo, rename=rename)
    assert model.layers[0].self_attn is model.gpt_neox_japanese.layers[0].attention


def test_a_stream_of_another_shape_is_refused():
    """Gemma-3n's blocks carry several parallel streams (``[streams, batch, seq, hidden]``), which the default cannot name."""
    with pytest.warns(UserWarning, match="gemma3n_text"):
        with pytest.raises(UnsupportedFamily, match=r"the blocks do not pass a \[batch, seq, hidden\] stream"):
            StandardizedTransformer("hf-tiny-v2/tiny-random-Gemma3nForCausalLM")


def test_tuple_blocks_are_read_off_the_scan():
    """GPT-J's block returns a tuple; `skip_with` packs the stream the way it does."""
    model = load_default(dedicated_suite("gptj").REPO, dispatch=True)
    assert all(layer.returns_tuple for layer in model.layers)
    with model.trace(PROMPT):
        model.skip_layers(0, 0)
        logits = model.logits.save()
    assert logits.shape[-1] == model.vocab_size
