"""Every check a vLLM family must pass, written once, with the transformers engine as the oracle.

A family's test file subclasses `VLLMFamilySuite` and names a checkpoint and
its native paths. The same checkpoint is first run through
`StandardizedTransformer` in float32 and its standard values kept; the vLLM
engine is then built in float32 and every value it serves is held against
them. One engine per class, so run each file in its own process: engines
sharing a card overrun their memory fractions.
"""

import gc

import nnsight
import pytest
import torch
from nnsight.intervention.envoy import Envoy

from nnterp import StandardizedTransformer, StandardizedVLLM, Unavailable
from nnterp.components import EProperty
from nnterp.components.vllm import Attention, Mlp

PROMPT = "The Eiffel Tower is in the city of"
ROOT = {"logits", "token_embeddings", "next_token_probs", "input_ids", "input_size", "attention_mask"}
BOUNDARY = ("layer_input", "attention_output", "mlp_output", "layer_output")
INTERIOR = ("attention_queries", "attention_keys", "attention_values", "attention_head_outputs")
PATTERN = ("attention_scores", "attention_probabilities")
SIZES = ("num_layers", "hidden_size", "vocab_size", "num_heads", "num_kv_heads", "head_dim", "intermediate_size")

LLAMA_ROWS = {
    "embed_tokens": "model.embed_tokens",
    "layers": "model.layers",
    "norm": "model.norm",
    "lm_head": "lm_head",
    "layers.0.self_attn": "model.layers.0.self_attn",
    "layers.0.mlp": "model.layers.0.mlp",
}


def boundary(layer, interior=()):
    """A block's boundary values, and the named values of its attention's interior, read in forward order, on the CPU."""
    values = {"layer_input": layer.layer_input.cpu()}
    for name in sorted(interior, key=(*INTERIOR[:3], *PATTERN, INTERIOR[3]).index):  # the head outputs leave the kernel last
        values[name] = getattr(layer.self_attn, name).cpu()
    values["attention_output"] = layer.self_attn.attention_output.cpu()
    values["mlp_output"] = layer.mlp.mlp_output.cpu()
    values["layer_output"] = layer.layer_output.cpu()
    return values


def shifted(scores, seen):
    """Scores on the keys a query sees, each query's largest subtracted: what the softmax keeps of them.

    Transformers masks with the dtype's minimum where vLLM's are ``-inf``,
    and an ALiBi family writes its bias from another origin, which moves a
    query's scores by one constant.
    """
    scores = scores.masked_fill(~seen, float("-inf"))
    return (scores - scores.amax(-1, keepdim=True)).masked_fill(~seen, 0)


def close(got, want, tolerance, name=""):
    """Equal up to ``tolerance`` of the reference's largest magnitude: two engines' float32 kernels differ in the last bits."""
    assert got.shape == want.shape, (name, got.shape, want.shape)
    scale = want.float().abs().max().item()
    worst = (got.float() - want.float()).abs().max().item()
    assert worst <= tolerance * scale, f"{name}: max |diff| {worst:.3e} against a scale of {scale:.3e}"


@torch.no_grad()
def transformers_values(repo, skip):
    """`PROMPT` through `StandardizedTransformer` in float32: the standard values, and the logits under three edits.

    Without gradients: a value saved with its graph keeps the weights it was
    computed from on the card, and the engine sizes itself from what is free.
    The attention's interior is read one value a trace: each transformers
    family reads its own in its own order.
    """
    hf = StandardizedTransformer(repo, device_map="cuda", dtype=torch.float32, attn_implementation="eager")
    ids = hf.tokenizer(PROMPT)["input_ids"]
    picked = sorted({0, hf.num_layers // 2, hf.num_layers - 1})
    middle = picked[len(picked) // 2]
    vector = torch.randn(hf.hidden_size, generator=torch.Generator().manual_seed(0))
    with hf.trace(torch.tensor([ids])):
        embeddings = hf.token_embeddings.cpu().save()
        layers = nnsight.save({i: boundary(hf.layers[i]) for i in picked})
        logits = hf.logits[:, -1:].cpu().save()
    support = hf.support()
    for name in (*INTERIOR, *PATTERN):
        if support.get(f"self_attn.{name}"):
            continue  # transformers does not serve it on this checkpoint
        with hf.trace(torch.tensor([ids])):
            read = nnsight.save({i: getattr(hf.layers[i].self_attn, name).cpu() for i in picked})
        for i in picked:
            layers[i][name] = read[i]
    scale = layers[middle]["layer_output"].norm(dim=-1).mean().item()
    with hf.trace(torch.tensor([ids])):
        hf.steer(middle, vector, factor=scale)
        steered = hf.logits[:, -1:].cpu().save()
    headless = None
    if "attention_head_outputs" in layers[middle]:
        with hf.trace(torch.tensor([ids])):
            hf.layers[middle].self_attn.attention_head_outputs[:, :, 0] = 0
            headless = hf.logits[:, -1:].cpu().save()
    with hf.trace(torch.tensor([ids])):
        hf.skip_layers(*(skip or (middle, middle)))
        skipped = hf.logits[:, -1:].cpu().save()
    positions = getattr(hf.config, "max_position_embeddings", None) or getattr(hf.config, "n_positions", None) or 512
    return {
        "ids": ids, "picked": picked, "middle": middle, "vector": vector, "scale": scale, "embeddings": embeddings,
        "layers": layers, "logits": logits, "steered": steered, "headless": headless, "skipped": skipped,
        "skip": skip or (middle, middle), "max_len": min(512, positions), "sizes": {name: getattr(hf, name) for name in SIZES},
    }


class VLLMFamilySuite:
    """Subclass per family: set the class attributes, add the family's own tests."""

    #: A checkpoint both engines load.
    REPO: str
    #: The vLLM family module the checkpoint must resolve to.
    FAMILY = None
    #: Standard path -> native path on vLLM's tree.
    NATIVE: dict = LLAMA_ROWS
    #: The engine's share of the card.
    MEMORY = 0.2
    #: Agreement with transformers, relative to each value's largest magnitude.
    TOLERANCE = 5e-3
    #: Agreement of the attention's interior (and of the logits after editing it). The two engines' kernels sum a
    #: sharp softmax differently, which a checkpoint with very large queries and keys shows in its head outputs, and
    #: from there in every later block's queries, keys and values, more than in the residual stream they are small in.
    KERNEL_TOLERANCE = 5e-3
    #: The blocks `test_skip_layers` skips; the middle one when not said.
    SKIP = None
    #: The attention values vLLM's implementation of this family serves (the scores and the pattern included).
    SERVED = (*INTERIOR, *PATTERN)
    #: The values this family lacks on vLLM, by `support()` name.
    UNAVAILABLE = frozenset({"attention_mask"})
    #: Engine arguments this checkpoint needs.
    ENGINE: dict = {}
    #: The engine's dtype. The reference is float32 either way; a kernel that only runs in half precision is
    #: compared at the tolerances its class sets.
    DTYPE = "float32"

    @pytest.fixture(scope="class")
    def reference(self, request):
        """The transformers engine's values for `PROMPT`; its model is off the card before vLLM sizes itself from what is free."""
        values = transformers_values(request.cls.REPO, request.cls.SKIP)
        gc.collect()
        torch.cuda.empty_cache()
        return values

    @pytest.fixture(scope="class")
    def model(self, request, reference):
        cls = request.cls
        model = StandardizedVLLM(
            cls.REPO, dispatch=True, dtype=cls.DTYPE, gpu_memory_utilization=cls.MEMORY, max_model_len=reference["max_len"], **cls.ENGINE
        )
        yield model
        model.vllm_entrypoint.llm_engine.engine_core.shutdown()

    def embedding_scale(self, model):
        """What vLLM's model multiplies the embedding module's output by before the first block (Gemma: sqrt(hidden))."""
        return 1.0

    def run(self, model, reference, **kwargs):
        """A one-step greedy trace of the reference prompt."""
        return model.trace(reference["ids"], temperature=0.0, max_tokens=1, **kwargs)

    def clean(self, model, reference):
        with self.run(model, reference):
            logits = model.logits.cpu().save()
        return logits

    # -- names ------------------------------------------------------------------

    def test_family_resolved(self, model):
        assert model.family is self.FAMILY
        assert all(type(layer) is self.FAMILY.Layer for layer in model.layers)
        assert all(type(layer.self_attn) is self.FAMILY.Attention for layer in model.layers)
        assert all(type(layer.mlp) is self.FAMILY.Mlp for layer in model.layers)
        assert issubclass(self.FAMILY.Attention, Attention) and issubclass(self.FAMILY.Mlp, Mlp)

    def test_standard_names_alias_native_envoys(self, model):
        for standard, native in self.NATIVE.items():
            assert isinstance(model.get(standard), Envoy), standard
            assert model.get(standard) is model.get(native), (standard, native)

    def test_sizes(self, model, reference):
        assert {name: getattr(model, name) for name in SIZES} == reference["sizes"]

    # -- availability -----------------------------------------------------------

    def test_support(self, model):
        support = model.support()
        assert ROOT <= set(support)
        unavailable = {name for name, reason in support.items() if reason}
        assert unavailable == self.UNAVAILABLE
        for name in ("layer_input", "layer_output", "self_attn.attention_output", "mlp.mlp_output", *(f"self_attn.{name}" for name in self.SERVED)):
            assert support[name] is None, name

    def test_support_matches_what_reads(self, model):
        layer = model.layers[0]
        for name, reason in model.support(layer=0).items():
            module, _, value = name.rpartition(".")
            host = getattr(layer, module) if module else layer
            if reason is None:
                assert isinstance(getattr(type(host), value), EProperty)
            else:
                with pytest.raises(Unavailable, match="vLLM"):
                    getattr(host, value)
        with pytest.raises(Unavailable, match="no mask"):
            model.attention_mask

    # -- the values are transformers' ---------------------------------------------

    def test_boundary_values_match_transformers(self, model, reference):
        """Saved as read, with no clone: a value is a private copy, in nnterp's layout."""
        interior = self.SERVED
        with self.run(model, reference):
            got = nnsight.save({i: boundary(model.layers[i], interior) for i in reference["picked"]})
        tokens, hidden = len(reference["ids"]), model.hidden_size
        for i, values in got.items():
            for name in BOUNDARY:
                assert values[name].shape == (1, tokens, hidden), (i, name)
                close(values[name], reference["layers"][i][name], self.TOLERANCE, f"layers[{i}].{name}")

    def test_attention_interior_matches_transformers(self, model, reference):
        """The queries, keys, values and head outputs are transformers' eager ones, in its layouts, heads split out."""
        interior = tuple(name for name in INTERIOR if name in self.SERVED)
        if not interior:
            pytest.skip("vLLM serves no attention interior on this family")
        with self.run(model, reference):
            got = nnsight.save({i: boundary(model.layers[i], interior) for i in reference["picked"]})
        tokens, heads, kv_heads, head_dim = len(reference["ids"]), model.num_heads, model.num_kv_heads, model.head_dim
        shapes = {
            "attention_queries": (1, heads, tokens, head_dim),
            "attention_keys": (1, kv_heads, tokens, head_dim),
            "attention_values": (1, kv_heads, tokens, head_dim),
            "attention_head_outputs": (1, tokens, heads, head_dim),
        }
        compared = 0
        for i, values in got.items():
            for name in interior:
                assert values[name].shape == shapes[name], (i, name)
                if name in reference["layers"][i]:  # transformers serves it too
                    close(values[name], reference["layers"][i][name], self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.{name}")
                    compared += 1
        assert compared, "transformers serves none of this family's interior to compare with"

    def test_pattern_matches_transformers(self, model, reference):
        """The scores and the pattern, recomputed from the queries and keys, are transformers' eager ones."""
        if not set(PATTERN) <= set(self.SERVED):
            pytest.skip("no recomputed pattern on this family")
        with self.run(model, reference):
            got = nnsight.save({i: boundary(model.layers[i], PATTERN) for i in reference["picked"]})
        tokens, heads = len(reference["ids"]), model.num_heads
        seen = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        for i, values in got.items():
            scores, probs = values["attention_scores"], values["attention_probabilities"]
            assert scores.shape == probs.shape == (1, heads, tokens, tokens), i
            assert torch.isinf(scores[..., ~seen]).all() and torch.equal(probs, probs.tril())
            torch.testing.assert_close(probs.sum(-1), torch.ones(1, heads, tokens), atol=1e-5, rtol=0)
            wanted = reference["layers"][i]
            assert "attention_probabilities" in wanted, "transformers serves no pattern on this checkpoint to compare with"
            if "attention_scores" in wanted:
                close(shifted(scores, seen), shifted(wanted["attention_scores"], seen), self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.attention_scores")
            close(probs, wanted["attention_probabilities"], self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.attention_probabilities")

    def test_pattern_is_the_prefills_and_read_only(self, model, reference):
        if not set(PATTERN) <= set(self.SERVED):
            pytest.skip("no recomputed pattern on this family")
        attention = model.layers[reference["middle"]].self_attn
        with model.trace(reference["ids"], temperature=0.0, max_tokens=3, ignore_eos=True):  # no iter: the block runs on the prefill
            prefill = attention.attention_probabilities.cpu().save()
        assert prefill.shape[-1] == len(reference["ids"])
        with pytest.raises(RuntimeError, match="on a decode step"):
            with model.trace(reference["ids"], temperature=0.0, max_tokens=2, ignore_eos=True) as tracer:
                for step in tracer.iter[:2]:
                    pattern = attention.attention_probabilities.cpu().save()
        with pytest.raises(RuntimeError, match="derived and read-only"):
            with self.run(model, reference):
                attention.attention_probabilities = attention.attention_probabilities * 0
        assert torch.equal(self.clean(model, reference), self.clean(model, reference))  # the engine lives

    @pytest.mark.parametrize("name", INTERIOR)
    def test_interior_writes_land(self, model, reference, name):
        if name not in self.SERVED:
            pytest.skip(f"vLLM does not serve {name} on this family")
        clean = self.clean(model, reference)
        attention = model.layers[reference["middle"]].self_attn
        with self.run(model, reference):
            getattr(attention, name)[:] *= 0.5
            in_place = model.logits.cpu().save()
        with self.run(model, reference):
            setattr(attention, name, getattr(attention, name) * 0.5)
            assigned = model.logits.cpu().save()
        assert not torch.allclose(clean, in_place)
        close(assigned, in_place, self.TOLERANCE, name)

    def test_a_zeroed_head_is_transformers_zeroed_head(self, model, reference):
        if "attention_head_outputs" not in self.SERVED or reference["headless"] is None:
            pytest.skip("the head outputs are not served on both engines for this family")
        with self.run(model, reference):
            model.layers[reference["middle"]].self_attn.attention_head_outputs[:, :, 0] = 0
            headless = model.logits.cpu().save()
        close(headless, reference["headless"], self.KERNEL_TOLERANCE, "logits with one head's output zeroed")

    def test_contribution_identity(self, model, reference):
        """``layer_input + attention_output + mlp_output == layer_output`` on every block."""
        with self.run(model, reference):
            parts = nnsight.save([boundary(layer) for layer in model.layers])
        for i, part in enumerate(parts):
            out = part["layer_output"]
            eps = torch.finfo(out.dtype).eps * out.abs().max().item()
            torch.testing.assert_close(part["layer_input"] + part["attention_output"] + part["mlp_output"], out, rtol=0, atol=8 * eps, msg=f"layer {i}")

    def test_root_values(self, model, reference):
        with self.run(model, reference):
            ids = model.input_ids.cpu().save()
            size = model.input_size.save()
            embeddings = model.token_embeddings.cpu().save()
            logits = model.logits.cpu().save()
            probs = model.next_token_probs.cpu().save()
        assert ids.tolist() == [reference["ids"]] and size == ids.shape
        close(embeddings * self.embedding_scale(model), reference["embeddings"], self.TOLERANCE, "token_embeddings")
        assert logits.shape == (1, 1, model.vocab_size)
        close(logits, reference["logits"], self.TOLERANCE, "logits")
        first, second = reference["logits"].flatten().topk(2).values
        if first - second > self.TOLERANCE * reference["logits"].abs().max():  # a clear winner, not a tie two kernels may break either way
            assert logits.argmax(-1).item() == reference["logits"].argmax(-1).item()
        torch.testing.assert_close(probs, logits[:, -1].softmax(-1), rtol=1e-3, atol=1e-5)  # the worker's softmax, on its card

    # -- reads and writes ---------------------------------------------------------

    def test_a_read_changes_nothing(self, model, reference):
        clean, interior = self.clean(model, reference), self.SERVED
        with self.run(model, reference):
            embeddings = model.token_embeddings
            for layer in model.layers:
                boundary(layer, interior)
            logits = model.logits.cpu().save()
        assert torch.equal(clean, logits)

    @pytest.mark.parametrize("name", BOUNDARY)
    def test_writes_land(self, model, reference, name):
        """An in-place edit, a statement that reads twice, and an assignment are the same write."""
        clean = self.clean(model, reference)
        index = reference["middle"]
        layer = model.layers[index]
        host = {"attention_output": layer.self_attn, "mlp_output": layer.mlp}.get(name, layer)
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

    def test_a_value_of_other_rows_is_refused(self, model, reference):
        layer = model.layers[reference["middle"]]
        with pytest.raises(RuntimeError, match="cannot be replaced by a value of shape"):
            with self.run(model, reference):
                layer.layer_output = layer.layer_output[:, :-1]
        assert torch.equal(self.clean(model, reference), self.clean(model, reference))  # the engine lives

    # -- methods ------------------------------------------------------------------

    def test_logit_lens(self, model, reference):
        """`project_on_vocab` of the last block's stream is the model's logits; of any block's, a distribution."""
        middle, last = model.layers[reference["middle"]], model.layers[-1]
        with self.run(model, reference):
            early = model.project_on_vocab(middle.layer_output).cpu().save()
            top = nnsight.save(model.get_topk_closest_tokens(last.layer_output[:, -1], k=3))
            lens = model.project_on_vocab(last.layer_output[:, -1:]).cpu().save()
            logits = model.logits.cpu().save()
        assert early.shape == (1, len(reference["ids"]), model.vocab_size)
        close(lens, logits, 1e-5, "last-layer lens")
        assert len(top) == 1 and len(top[0]) == 3
        assert model.tokenizer.decode(logits.argmax(-1).item()) in top[0]  # among the top three: a near-tie may order them either way

    def test_steer(self, model, reference):
        with self.run(model, reference):
            model.steer(reference["middle"], reference["vector"], factor=reference["scale"])
            steered = model.logits.cpu().save()
        close(steered, reference["steered"], self.TOLERANCE, "steered logits")
        assert not torch.allclose(steered, self.clean(model, reference))

    # -- the engine's own shape: steps and invokes -----------------------------------

    def test_generation_steps(self, model, reference):
        layer = model.layers[reference["middle"]]
        with model.trace(reference["ids"], temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
            shapes = nnsight.save([])
            for step in tracer.iter[:3]:
                shapes.append((tuple(layer.layer_output.shape), tuple(model.logits.shape)))
        tokens, hidden, vocab = len(reference["ids"]), model.hidden_size, model.vocab_size
        assert shapes == [((1, tokens, hidden), (1, 1, vocab))] + [((1, 1, hidden), (1, 1, vocab))] * 2
        with model.trace(reference["ids"], temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
            streams = nnsight.save([])
            for step in tracer.iter[:3]:  # one value a step and nothing between: each step is the model's next call
                streams.append(layer.layer_output.cpu())
        assert [s.shape[1] for s in streams] == [tokens, 1, 1] and not torch.equal(streams[1], streams[2])
        if "attention_keys" not in self.SERVED:
            return
        with model.trace(reference["ids"], temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
            keys = nnsight.save([])
            for step in tracer.iter[:3]:  # a decode step's keys are the new token's; the rest are in the engine's cache
                keys.append(tuple(layer.self_attn.attention_keys.shape))
        assert [k[2] for k in keys] == [tokens, 1, 1]

    def test_one_prompt_per_invoke(self, model, reference):
        layer = model.layers[reference["middle"]]
        prompts = ["The capital of France is", "Two plus two is equal to the number"]
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            for prompt in prompts:
                with tracer.invoke(prompt):
                    stream = layer.layer_output.cpu().save()
        assert [s.shape[0] for s in stream] == [1, 1] and stream[0].shape[1] != stream[1].shape[1]

    # -- last: a skip answers for the whole step, so it runs alone and after everything else --

    def test_skip_layers(self, model, reference):
        with self.run(model, reference):
            model.skip_layers(*reference["skip"])
            skipped = model.logits.cpu().save()
        close(skipped, reference["skipped"], self.TOLERANCE, "logits with blocks skipped")
