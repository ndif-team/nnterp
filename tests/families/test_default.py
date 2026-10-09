"""The best-effort default family: forced onto families that have their own, and on a model_type that has none.

Forced onto a shipped family's checkpoint, the default must run the
`FamilySuite` (GPT-2, Llama, GPT-NeoX, OPT, Phi) with the contributions and the
logit lens unavailable, and every value it serves must read exactly what the
dedicated family reads, on every shipped family whose checkpoint it loads.
"""

import importlib
import re
import warnings

import pytest
import torch
from nnsight import TransformersModel  # nnsight before any transformers submodule
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import SourceNotAvailable
from suite import INTERIOR, PROMPT, FamilySuite, recurrent_mixer
from transformers import AutoConfig

from nnterp import StandardizedTransformer, Unavailable, UnsupportedFamily, families
from nnterp.components import Moe, SelectiveScan, StateSpace
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


#: What the default reports for the values it does not serve.
NOT_SERVED = {"self_attn.attention_output": "cannot tell what this sublayer adds", "mlp.mlp_output": "cannot tell what this sublayer adds"}


class DefaultSuite(FamilySuite):
    """`FamilySuite` with the default family forced onto the checkpoint.

    The tests that read a contribution or the logit lens check instead that
    the default refuses them.
    """

    FAMILY = default

    @pytest.fixture(scope="class")
    def model(self, request):
        cls = request.cls
        return load_default(cls.REPO, dispatch=True, attn_implementation="eager", **cls.LOAD_KWARGS)

    def expected_values(self, model):
        """A module the default found on no block is listed as unavailable, not left out (OPT's ``mlp``)."""
        return super().expected_values(model) | {"mlp.mlp_output"}

    def test_every_value_reads_a_tensor_on_every_layer(self, model):
        read = {}
        with model.trace(PROMPT):
            for layer in model.layers:
                read[layer.path] = layer.layer_output.save()
        assert all(value.shape[-1] == model.hidden_size for value in read.values()) and len(read) == len(model.layers)

    def test_contribution_identity(self, model):
        with pytest.raises(Unavailable, match="cannot tell what this sublayer adds"):
            model.layers[0].self_attn.attention_output

    def test_interior_writes_are_causal(self, model):
        block = self.attn_block(model)
        with model.trace(PROMPT):
            clean = model.logits.save()
        for name in INTERIOR:
            with model.trace(PROMPT):
                attn = block.self_attn
                setattr(attn, name, getattr(attn, name) * 0)
                logits = model.logits.save()
            assert not torch.equal(clean, logits), name

    def test_boundary_writes_land(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            model.layers[0].layer_output = model.layers[0].layer_output * 0
            edited = model.logits.save()
        assert not torch.equal(clean, edited)

    def test_skip_layers_hands_the_stream_straight_through(self, model):
        last = model.num_layers - 1
        with model.trace(PROMPT):
            first = model.layers[0].layer_output.save()
            model.skip_layers(1, last)
            skipped_last = model.layers[last].layer_output.save()
        assert torch.equal(skipped_last, first)

    def test_project_on_vocab_is_the_logit_lens(self, model):
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
        with pytest.raises(Unavailable, match="cannot tell what follows lm_head"):
            model.project_on_vocab(resid)
        with pytest.raises(Unavailable, match="cannot tell what follows lm_head"):
            model.get_topk_closest_tokens(resid[0, -1], k=3)

    def test_logits_are_the_models_output(self, model):
        with model.trace(PROMPT):
            logits = model.logits.save()
            result = model.output.logits.save()
        assert torch.equal(logits, result) and logits.shape[-1] == model.vocab_size


def forced(name, **overrides):
    """A `DefaultSuite` over the shipped family ``name``'s checkpoint and suite settings."""
    source = dedicated_suite(name)
    settings = {key: getattr(source, key) for key in dir(source) if key.isupper() and key != "FAMILY"}
    settings["EXPECTED_UNAVAILABLE"] = {**NOT_SERVED, **settings.get("EXPECTED_UNAVAILABLE", {}), **overrides.pop("EXPECTED_UNAVAILABLE", {})}
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
    support = nanochat.support()
    assert {name for name, reason in support.items() if reason is not None} == {"self_attn.attention_output", "mlp.mlp_output"}
    assert (nanochat.num_layers, nanochat.num_heads, nanochat.head_dim, nanochat.intermediate_size) == (2, 2, 16, 32)


def test_unknown_model_type_reads_the_raw_model(nanochat):
    """The stream, the logits and the attention pattern are the native model's, bit for bit."""
    raw = TransformersModel(NANOCHAT, task="text-generation", dispatch=True, attn_implementation="eager")
    native, standard = [], []
    with raw.trace(PROMPT):
        for layer in raw.model.layers:
            native.append(layer.output.save())
        logits = raw.output.logits.save()
    with nanochat.trace(PROMPT):
        pattern = nanochat.layers[0].self_attn.attention_probabilities.save()
        for layer in nanochat.layers:
            standard.append(layer.layer_output.save())
        ours = nanochat.logits.save()
    for out, our_out in zip(native, standard):
        assert torch.equal(our_out, out)
    assert torch.equal(ours, logits)
    torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))


def test_contributions_and_the_lens_are_unavailable(nanochat):
    """What a sublayer adds to the stream, and what follows lm_head, are not served, with one reason each."""
    reason = "the default family cannot tell what this sublayer adds to the stream; add a family module (docs/extending/adding-a-family.md)"
    assert nanochat.support(layer=0)["self_attn.attention_output"] == reason
    assert nanochat.support(layer=0)["mlp.mlp_output"] == reason
    with pytest.raises(Unavailable, match=re.escape(reason)):
        with nanochat.trace(PROMPT):
            nanochat.layers[0].mlp.mlp_output.save()
    with nanochat.trace(PROMPT):
        resid = nanochat.layers[-1].layer_output.save()
    with pytest.raises(Unavailable, match="the default family cannot tell what follows lm_head"):
        nanochat.project_on_vocab(resid)
    with pytest.raises(Unavailable, match="the default family cannot tell what follows lm_head"):
        nanochat.get_topk_closest_tokens(resid[0, -1])


def test_steer_and_skip_layers_work_on_the_stream(nanochat):
    vector = torch.ones(nanochat.hidden_size)
    with nanochat.trace(PROMPT):
        clean = nanochat.layers[0].layer_output.save()
    with nanochat.trace(PROMPT):
        nanochat.steer(0, vector, token_positions=-1)
        steered = nanochat.layers[0].layer_output.save()
    torch.testing.assert_close(steered[:, -1], clean[:, -1] + vector.to(clean))
    with nanochat.trace(PROMPT):
        first = nanochat.layers[0].layer_output.save()
        nanochat.skip_layers(1, -1)
        last = nanochat.layers[-1].layer_output.save()
    assert torch.equal(last, first)


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


#: Shipped families the default refuses at load: the scan cannot run their forward, or their stream is not [batch, seq, hidden].
REFUSED = {
    "dbrx": r"its shape check could not run under fake tensors \(GuardOnDataDependentSymNode",
    "deepseek_v4": r"the blocks do not pass a \[batch, seq, hidden\] stream",
}
#: Shipped families whose eager attention forward is their own, without the op a pattern value is read at: those reads fail loudly, naming the op.
NO_OP = {
    "gpt_oss": {"attention_scores"},
    "mimo_v2_flash": {"attention_scores"},
    "granite_swa": {"attention_scores", "attention_probabilities"},
    "granitemoe_swa": {"attention_scores", "attention_probabilities"},
}


@pytest.mark.parametrize("name", comparable())
def test_default_reads_what_the_family_reads(name):
    """Wherever the default serves a value the dedicated family also serves, it reads exactly what that family reads."""
    source = dedicated_suite(name)
    kwargs = dict(dispatch=True, attn_implementation="eager", **source.LOAD_KWARGS)
    if any(issubclass(envoy, Moe) for envoy in getattr(families, name).ENVOYS.values()):
        kwargs.setdefault("experts_implementation", "batched_mm")  # grouped_mm has no float32 fake-tensor kernel for the scan
    if name in REFUSED:
        with pytest.raises(UnsupportedFamily, match=REFUSED[name]):
            load_default(source.REPO, **kwargs)
        return
    dedicated = StandardizedTransformer(source.REPO, **kwargs)
    model = load_default(source.REPO, **kwargs)
    compared = 0
    for i in range(len(model.layers)):
        theirs, ours = dedicated.support(layer=i), model.support(layer=i)
        for host, value in VALUES:
            key = f"{host}.{value}" if host else value
            if ours.get(key, "absent") is not None:
                continue
            if value in NO_OP.get(name, ()):
                with pytest.raises(SourceNotAvailable, match="has no operation 'nn_functional_"):
                    read(model, i, host, value)
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


def test_post_norm_contributions_are_unavailable_not_wrong():
    """Gemma-2 norms each sublayer's output before adding it; forced through the default, the contributions are refused, the stream served."""
    model = load_default(dedicated_suite("gemma2").REPO)
    support = model.support(layer=0)
    assert "cannot tell what this sublayer adds" in support["self_attn.attention_output"]
    assert "cannot tell what this sublayer adds" in support["mlp.mlp_output"]
    assert support["layer_output"] is None
    with pytest.raises(Unavailable, match="cannot tell what this sublayer adds"):
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


def test_family_default_at_load_runs_the_check():
    """``family=default`` skips the lookup (no warning) but not the check on the built tree."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnsupportedFamily, match="found no embed_tokens, layers, norm, lm_head"):
            StandardizedTransformer("hf-internal-testing/tiny-random-RwkvForCausalLM", family=default)
    assert not any("no family for model_type" in str(warning.message) for warning in caught)


def test_a_scan_that_cannot_run_is_refused():
    """A forward the shape scan cannot run (grouped expert matmuls on fake float32 tensors) leaves the guess unchecked: refused, naming the error."""
    repo = "hf-tiny-v2/tiny-random-InklingForCausalLM"
    with pytest.warns(UserWarning, match="no family for model_type"):
        with pytest.raises(UnsupportedFamily, match=r"its shape check could not run under fake tensors \(RuntimeError: .*experts_implementation='batched_mm'"):
            StandardizedTransformer(repo)
    with pytest.warns(UserWarning, match="no family for model_type"):
        assert StandardizedTransformer(repo, experts_implementation="batched_mm").family is default
