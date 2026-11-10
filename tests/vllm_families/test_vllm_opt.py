"""OPT on vLLM, against the transformers engine: no MLP module on either engine, and pre-scaled queries."""

import gc

import nnsight
import pytest
import torch
import vllm_suite
from vllm_suite import BOUNDARY, INTERIOR, NO_ROTARY, ROOT, VLLMFamilySuite, close

from nnterp import StandardizedTransformer, StandardizedVLLM
from nnterp.components import POST_NORM_ATTENTION, POST_NORM_MLP
from nnterp.families.vllm import opt

suite_boundary = vllm_suite.boundary


def fc2_output(layer):
    """What the block's feed-forward path adds: ``fc2``'s output (vLLM's returns ``(output, bias)``), ``[1, tokens, hidden_size]``."""
    output = layer.fc2.output
    output = output[0] if isinstance(output, tuple) else output
    return output.reshape(1, -1, output.shape[-1]).clone()


def boundary(layer, interior=()):
    """The suite's boundary values, with ``mlp_output`` read where OPT has it on both engines: the block's ``fc2``."""
    values = {"layer_input": layer.layer_input.cpu()}
    for name in sorted(interior, key=(*INTERIOR[:3], *vllm_suite.PATTERN, INTERIOR[3]).index):
        values[name] = getattr(layer.self_attn, name).cpu()
    values["attention_output"] = layer.self_attn.attention_output.cpu()
    values["mlp_output"] = fc2_output(layer).cpu()
    values["layer_output"] = layer.layer_output.cpu()
    return values


class TestVLLMOPTPostNormBoundaries:
    """OPT-350m's post-norm blocks (``do_layer_norm_before`` false), on vLLM against transformers: the norms follow the
    adds, so ``layer_mid`` is ``self_attn_layer_norm``'s output and no norm precedes either sublayer, on both engines."""

    REPO = "facebook/opt-350m"
    BOUNDARIES = ("attention_norm_output", "layer_mid", "mlp_norm_output")

    @pytest.fixture(scope="class")
    def reference(self):
        hf = StandardizedTransformer(self.REPO, device_map="cuda", dtype=torch.float32, attn_implementation="eager")
        ids = hf.tokenizer(vllm_suite.PROMPT)["input_ids"]
        picked = sorted({0, hf.num_layers // 2, hf.num_layers - 1})
        support = {name: hf.support(layer=0)[name] for name in self.BOUNDARIES}
        with torch.no_grad(), hf.trace(torch.tensor([ids])):
            mids = nnsight.save({i: hf.layers[i].layer_mid.cpu() for i in picked})
        del hf
        gc.collect()
        torch.cuda.empty_cache()
        return {"ids": ids, "picked": picked, "support": support, "mids": mids}

    @pytest.fixture(scope="class")
    def model(self, reference):
        model = StandardizedVLLM(self.REPO, dispatch=True, dtype="float32", gpu_memory_utilization=0.2, max_model_len=512)
        yield model
        model.vllm_entrypoint.llm_engine.engine_core.shutdown()

    def test_boundaries_are_transformers(self, model, reference):
        support = model.support(layer=0)
        assert {name: support[name] for name in self.BOUNDARIES} == reference["support"]
        assert support["attention_norm_output"] == POST_NORM_ATTENTION and support["mlp_norm_output"] == POST_NORM_MLP
        with model.trace(reference["ids"], temperature=0.0, max_tokens=1):
            mids = nnsight.save({i: model.layers[i].layer_mid.cpu() for i in reference["picked"]})
        for i, mid in mids.items():
            want = reference["mids"][i]  # transformers' rows are flat
            close(mid, want.reshape(mid.shape), 5e-3, f"layers[{i}].layer_mid")


class TestVLLMOPT(VLLMFamilySuite):
    REPO = "facebook/opt-125m"
    FAMILY = opt
    UNAVAILABLE = (VLLMFamilySuite.UNAVAILABLE | NO_ROTARY | {"mlp_input"}) - {"mlp.mlp_activation"}
    NATIVE = {
        "embed_tokens": "model.decoder.embed_tokens",
        "layers": "model.decoder.layers",
        "norm": "model.decoder.final_layer_norm",
        "lm_head": "lm_head",
        "layers.0.self_attn": "model.decoder.layers.0.self_attn",
        "layers.0.input_layernorm": "model.decoder.layers.0.self_attn_layer_norm",
    }

    @pytest.fixture(scope="class", autouse=True)
    def fc2_boundary(self):
        """The suite reads ``layers[i].mlp.mlp_output``, which OPT has on neither engine: read the block's ``fc2`` instead, for this class."""
        vllm_suite.boundary = boundary
        yield
        vllm_suite.boundary = suite_boundary

    def test_family_resolved(self, model):
        assert model.family is self.FAMILY
        assert all(type(layer) is self.FAMILY.Layer for layer in model.layers)
        assert all(type(layer.self_attn) is self.FAMILY.Attention for layer in model.layers)
        assert not any(hasattr(layer, "mlp") for layer in model.layers)

    def test_support(self, model):
        support = model.support()
        assert ROOT <= set(support)
        assert {name for name, reason in support.items() if reason} == self.UNAVAILABLE
        assert not any(name.startswith("mlp.") for name in support)  # no MLP module, as on transformers
        for name in ("layer_input", "layer_output", "self_attn.attention_output", *(f"self_attn.{name}" for name in self.SERVED)):
            assert support[name] is None, name

    @pytest.mark.parametrize("name", BOUNDARY)
    def test_writes_land(self, model, reference, name):
        """The suite's check, whose hosts include ``layers[i].mlp``, which OPT does not have: its own three values only."""
        if name == "mlp_output":
            pytest.skip("OPT has no MLP module on either engine; its feed-forward path is the block's fc1/fc2")
        clean = self.clean(model, reference)
        layer = model.layers[reference["middle"]]
        host = layer.self_attn if name == "attention_output" else layer
        vector = reference["vector"] * reference["scale"]
        with self.run(model, reference):
            value = getattr(host, name)
            value[:, -1] += vector.to(value)
            in_place = model.logits.cpu().save()
        with self.run(model, reference):
            getattr(host, name)[:, -1] += vector.to(getattr(host, name))
            read_twice = model.logits.cpu().save()
        with self.run(model, reference):
            value = getattr(host, name).clone()
            value[:, -1] += vector.to(value)
            setattr(host, name, value)
            assigned = model.logits.cpu().save()
        assert not torch.allclose(clean, in_place)
        assert torch.equal(in_place, read_twice)
        close(assigned, in_place, self.TOLERANCE, name)

    def test_lm_head_is_the_embedding(self, model):
        """With tied embeddings vLLM unembeds with the embedding module itself: one module, two names."""
        assert model.lm_head._module is model.embed_tokens._module

    def test_queries_are_pre_scaled(self, model, reference):
        attention = model.layers[0].self_attn
        with self.run(model, reference):
            raw = attention.attn.inputs[0][0].clone().cpu().save()
            queries = attention.attention_queries.cpu().save()
        heads, head_dim = model.num_heads, model.head_dim
        unscaled = raw.unflatten(-1, (heads, head_dim)).transpose(0, 1).unsqueeze(0)
        torch.testing.assert_close(queries, unscaled * head_dim ** -0.5)


class TestVLLMOPT350m(TestVLLMOPT):
    """OPT-350m: post-norm blocks, no final norm and ``project_in`` / ``project_out`` (512 <-> 1024) around the stack,
    on both engines. The lens is ``lm_head(project_out(h))``, the decoder's own path."""

    REPO = "facebook/opt-350m"
    UNAVAILABLE = TestVLLMOPT.UNAVAILABLE | {"final_norm_output"}
    NATIVE = {standard: native for standard, native in TestVLLMOPT.NATIVE.items() if standard != "norm"}

    def test_no_final_norm(self, model):
        assert model.model.decoder.final_layer_norm is None and model.model.decoder.project_out is not None
        assert model.support()["final_norm_output"] == "no norm module on this model"

    def test_contribution_identity(self, model, reference):
        pytest.skip("the norms follow the adds: input + attention_output + fc2.output is normed before it is the block's output")

    def test_module_boundaries_match_transformers(self, model, reference):
        pytest.skip("vLLM's layer_mid and mlp_norm_output read the block's final_layer_norm, which follows the MLP's add on this post-norm layout")
