"""vLLM's best-effort default family, forced onto checkpoints whose vLLM implementation has a family of its own.

On the meta build (no engine), for every vLLM family: the default finds the
same modules under the standard names, reads the block's convention (fused or
called with the stream, where the stream sits, whether it returns a tuple) as
the family states it, and reports unavailable what the family maps by hand.
On the engine, the whole `VLLMFamilySuite` runs with the default forced on
Llama, GPT-2, GPT-NeoX and Phi, against the transformers engine as the oracle.
"""

import glob
import importlib
import os

import pytest
from transformers import AutoConfig
from vllm_suite import VLLMFamilySuite

from nnterp import StandardizedVLLM, families
from nnterp.components.vllm import FusedLayer
from nnterp.families.vllm import default

HERE = os.path.dirname(__file__)


def load_default(repo, **kwargs):
    """``repo`` on vLLM, standardized by the default family as if vLLM's ``model_type`` had none."""
    model_type = AutoConfig.from_pretrained(repo).get_text_config().model_type
    key = f"vllm.{model_type}"
    families.REGISTRY[key] = default
    try:
        return StandardizedVLLM(repo, **kwargs)
    finally:
        del families.REGISTRY[key]


def dedicated_suites():
    """Every vLLM family's suite class, by module name."""
    suites = {}
    for path in sorted(glob.glob(os.path.join(HERE, "test_vllm_*.py"))):
        name = os.path.basename(path)[len("test_vllm_"):-len(".py")]
        if name == "default":
            continue
        module = importlib.import_module(f"test_vllm_{name}")
        for cls in vars(module).values():
            if isinstance(cls, type) and issubclass(cls, VLLMFamilySuite) and "REPO" in vars(cls):
                suites.setdefault(name, cls)
    return suites


SUITES = dedicated_suites()


class TestVLLMDefaultNames:
    """Meta builds only, no engine: one class, so `run.sh` runs these in a process of their own."""

    @pytest.mark.parametrize("name", sorted(SUITES))
    def test_default_finds_what_the_family_names(self, name):
        """The same modules under the standard names, the same block convention, and honest availability."""
        suite = SUITES[name]
        dedicated = StandardizedVLLM(suite.REPO, **suite.ENGINE)
        model = load_default(suite.REPO, **suite.ENGINE)
        assert model.family is default
        for standard, native in suite.NATIVE.items():
            assert model.get(standard).path == dedicated.get(standard).path == model.get(native).path, standard
        for ours, theirs in zip(model.layers, dedicated.layers):
            assert isinstance(ours, FusedLayer) == isinstance(theirs, FusedLayer), ours.path
            if not isinstance(theirs, FusedLayer):
                assert (ours.STREAM, ours.returns_tuple) == (theirs.STREAM, theirs.returns_tuple), ours.path
        support, theirs = model.support(), dedicated.support()
        for value, reason in support.items():
            if reason is None:
                assert theirs.get(value) is None, f"the default serves {value}, which {name} reports unavailable: {theirs.get(value)}"


class DefaultVLLMSuite(VLLMFamilySuite):
    """`VLLMFamilySuite` with the default forced onto the checkpoint."""

    FAMILY = default

    @pytest.fixture(scope="class")
    def model(self, request, reference):
        cls = request.cls
        model = load_default(
            cls.REPO, dispatch=True, dtype=cls.DTYPE, gpu_memory_utilization=cls.MEMORY, max_model_len=reference["max_len"], **cls.ENGINE
        )
        yield model
        model.vllm_entrypoint.llm_engine.engine_core.shutdown()

    def test_family_resolved(self, model):
        """The default's envoys, with each block's convention the family's."""
        theirs = self.DEDICATED.Layer
        assert model.family is default
        assert all(type(layer) is (default.FusedLayer if issubclass(theirs, FusedLayer) else default.StreamLayer) for layer in model.layers)
        assert all(type(layer.self_attn) is default.Attention and type(layer.mlp) is default.Mlp for layer in model.layers)


def forced(name):
    source = SUITES[name]
    settings = {key: getattr(source, key) for key in dir(source) if key.isupper() and key != "FAMILY"}
    return type(f"TestVLLMDefaultOn{source.__name__.removeprefix('TestVLLM')}", (DefaultVLLMSuite,), {**settings, "DEDICATED": source.FAMILY})


TestVLLMDefaultOnLlama = forced("llama")
TestVLLMDefaultOnGPT2 = forced("gpt2")
TestVLLMDefaultOnGPTNeoX = forced("gpt_neox")
TestVLLMDefaultOnPhi = forced("phi")
