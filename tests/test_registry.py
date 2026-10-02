"""The registry and the load path, across families."""

import types

import pytest
import torch
from nnsight.intervention.envoy import Envoy
from transformers import AutoModelForCausalLM

from nnterp import StandardizedTransformer, UnsupportedFamily, families
from nnterp.families import gpt2

GPT2 = "hf-internal-testing/tiny-random-gpt2"


def test_every_family_module_is_named_after_its_model_type():
    for name in families.known():
        family = getattr(families, name)   # lazy: imported here
        assert family.MODEL_TYPES == (name,), name
        assert families.lookup(name) is family
    assert len(families.known()) == len(families.all_families()) >= 31


def test_import_is_lazy():
    """`import nnterp` pulls in no transformers modeling module; a family loads on first use."""
    import subprocess
    import sys

    code = (
        "import sys, nnsight, nnterp\n"
        "before = sorted(m for m in sys.modules if m.startswith('transformers.models.') and 'modeling_' in m)\n"
        "nnterp.families.lookup('gpt2')\n"
        "after = sorted(m for m in sys.modules if m.startswith('transformers.models.') and 'modeling_' in m)\n"
        "print(len(before), len(after), 'nnterp.families.gpt2' in sys.modules, 'nnterp.families.llama' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "0" and int(out[1]) >= 1 and out[2] == "True" and out[3] == "False", out


def test_unknown_family_refused():
    with pytest.raises(UnsupportedFamily, match="zamba"):
        StandardizedTransformer("hf-tiny-v2/tiny-random-ZambaForCausalLM")


def test_register_adds_a_family_and_can_override():
    custom = types.SimpleNamespace(MODEL_TYPES=("gpt2",), RENAME=gpt2.RENAME, ENVOYS=gpt2.ENVOYS)
    try:
        families.register(custom)
        assert families.lookup("gpt2") is custom
        assert StandardizedTransformer(GPT2).family is custom
    finally:
        del families.REGISTRY["gpt2"]


def test_an_engine_has_its_own_families():
    """vLLM's families are a package under the transformers ones, looked up by the same model type."""
    assert "vllm" not in families.known() and {"llama", "gpt2"} <= set(families.known("vllm"))
    with pytest.raises(UnsupportedFamily, match="'stablelm' on vllm.*nnterp/families/vllm/stablelm.py"):
        families.lookup("stablelm", engine="vllm")
    custom = types.SimpleNamespace(MODEL_TYPES=("stablelm",), RENAME={}, ENVOYS={})
    try:
        families.register(custom, engine="vllm")
        assert families.lookup("stablelm", engine="vllm") is custom
        assert families.lookup("stablelm") is not custom
    finally:
        del families.REGISTRY["vllm.stablelm"]


def test_preloaded_module_uses_its_own_config():
    module = AutoModelForCausalLM.from_pretrained(GPT2)
    model = StandardizedTransformer(module)
    assert model.family is gpt2
    assert model.layers[0].self_attn is model.transformer.h[0].attn


def test_user_rename_merges_over_family():
    model = StandardizedTransformer(GPT2, rename={"mlp": "ffn"})
    block = model.layers[0]
    assert block.ffn is block.mlp is model.transformer.h[0].mlp


def test_user_envoys_merge_over_defaults():
    """A user's entry replaces the family's for the same key. nnsight tries type
    keys before path keys, so a family's type key is beaten by a type key."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

    class Marker(Envoy):
        pass

    model = StandardizedTransformer(GPT2, envoys={GPT2MLP: Marker})
    assert type(model.layers[0]) is model.family.Layer
    assert type(model.layers[0].mlp) is Marker


def test_remote_key_names_the_plain_transformers_model():
    """A server deploys a TransformersModel; the standardized wrapper must produce that key."""
    from nnsight.modeling.transformers import TransformersModel

    model = StandardizedTransformer(GPT2)
    assert model._remoteable_class() is TransformersModel


def test_default_load_keeps_the_checkpoints_attention():
    model = StandardizedTransformer("hf-internal-testing/tiny-random-LlamaForCausalLM")
    assert model.config._attn_implementation != "eager"
    assert "eager" in model.support()["self_attn.attention_probabilities"][0]
    assert model.support()["self_attn.attention_output"] is None


def test_register_needs_only_names_and_envoys_for_support():
    """`support()` walks the tree, so a family without `Attention`/`Mlp` classes still reports every block value."""
    custom = types.SimpleNamespace(MODEL_TYPES=("gpt2",), RENAME=gpt2.RENAME, ENVOYS=gpt2.ENVOYS)
    try:
        families.register(custom)
        support = StandardizedTransformer(GPT2).support()
    finally:
        del families.REGISTRY["gpt2"]
    assert "self_attn.attention_output" in support and "mlp.mlp_output" in support


def test_custom_value_through_envoys_is_in_support():
    """A value added on an envoy subclass passed through ``envoys=`` is listed by `support()` like the family's own."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    from nnterp.components import DerivedEProperty

    class Attention(gpt2.Attention):
        heads = DerivedEProperty(lambda self: self._module.num_heads, description="The head count")

    model = StandardizedTransformer(GPT2, envoys={GPT2Attention: Attention})
    assert model.support()["self_attn.heads"] is None
    assert model.support(layer=0)["self_attn.heads"] is None
    assert "self_attn.heads" not in StandardizedTransformer(GPT2).support()


def test_family_defines_a_size_instead_of_the_root():
    """A function named like a `StandardizedProperty` in the family module wins over the root's implementation."""
    custom = types.SimpleNamespace(MODEL_TYPES=("gpt2",), RENAME=gpt2.RENAME, ENVOYS=gpt2.ENVOYS, hidden_size=lambda model: 999)
    try:
        families.register(custom)
        model = StandardizedTransformer(GPT2)
    finally:
        del families.REGISTRY["gpt2"]
    assert model.hidden_size == 999
    assert model.num_heads == model.config.num_attention_heads  # the rest keep the root's
    assert StandardizedTransformer(GPT2).hidden_size == StandardizedTransformer(GPT2).config.hidden_size


def test_family_defines_project_on_vocab_instead_of_the_softcap():
    """A `project_on_vocab(model, hidden)` in the family module is bound in the root's place; the root's applies the softcap."""
    from nnterp.standardized import StandardizedCapability

    custom = types.SimpleNamespace(
        MODEL_TYPES=("gpt2",), RENAME=gpt2.RENAME, ENVOYS=gpt2.ENVOYS,
        project_on_vocab=lambda model, hidden: model.lm_head(model.norm(hidden)) * 2,
    )
    try:
        families.register(custom)
        model = StandardizedTransformer(GPT2, dispatch=True)
    finally:
        del families.REGISTRY["gpt2"]
    plain = StandardizedTransformer(GPT2, dispatch=True)
    hidden = torch.randn(1, 3, plain.hidden_size).to(plain.lm_head.weight)
    assert isinstance(StandardizedTransformer.project_on_vocab, StandardizedCapability)
    torch.testing.assert_close(model.project_on_vocab(hidden), plain.project_on_vocab(hidden) * 2)
    plain.config.final_logit_softcapping = 2.0
    try:
        raw = plain.lm_head(plain.norm(hidden))
        torch.testing.assert_close(plain.project_on_vocab(hidden), 2.0 * torch.tanh(raw / 2.0))
    finally:
        plain.config.final_logit_softcapping = None


def test_sizes_are_read_only():
    model = StandardizedTransformer(GPT2)
    with pytest.raises(AttributeError, match="def hidden_size"):
        model.hidden_size = 5

