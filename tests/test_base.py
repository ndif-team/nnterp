"""The base envoys and descriptors, apart from any family."""

import pytest
import torch
from nnsight import TransformersModel  # nnsight before any transformers submodule
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.llama.modeling_llama import LlamaAttention

from nnterp import Layer, StandardizedTransformer, Unavailable, unavailable
from nnterp.components import Attention


@pytest.fixture(scope="module")
def tuple_model():
    """GPT-J's block returns ``(hidden_states, present)``; the base Layer on a plain TransformersModel."""
    return TransformersModel(
        "hf-internal-testing/tiny-random-GPTJForCausalLM", task="text-generation",
        dispatch=True, envoys={GPTJBlock: Layer},
    )


def test_tuple_block_unwrapped_and_rewrapped(tuple_model):
    m = tuple_model
    block = m.transformer.h[0]
    with m.trace("Hello world"):
        raw = block.output.save()
        std = block.layer_output.save()
        clean = m.lm_head.output.save()
    assert isinstance(raw, tuple) and torch.equal(std, raw[0])
    with m.trace("Hello world"):
        block.layer_output = block.layer_output * 0
        after = block.output.save()
        edited = m.lm_head.output.save()
    assert isinstance(after, tuple) and len(after) == len(raw)
    assert torch.equal(after[0], torch.zeros_like(raw[0]))
    assert not torch.equal(clean, edited)


def test_unavailable_marker_is_listed_and_raises():
    class Linear(Attention):
        attention_probabilities = unavailable("no softmax: the attention is linear")

    model = StandardizedTransformer("hf-internal-testing/tiny-random-LlamaForCausalLM", envoys={LlamaAttention: Linear})
    attn = model.layers[0].self_attn
    assert type(attn) is Linear
    assert "(attention_probabilities): Unavailable: no softmax" in repr(attn)
    assert attn.support()["attention_probabilities"] == "no softmax: the attention is linear"
    assert model.support()["self_attn.attention_probabilities"] == {i: "no softmax: the attention is linear" for i in range(model.num_layers)}
    with pytest.raises(Unavailable, match="the attention is linear"):
        attn.attention_probabilities
    with pytest.raises(Unavailable):  # TODO in nnterp.components.Unavailable: hasattr should be False instead
        hasattr(attn, "attention_probabilities")


@pytest.fixture(scope="module")
def gpt2_paths():
    """A GPT-2 attention with one value per kind of path an `EProperty` key can take."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    from nnterp.components import EProperty, Pattern, Residual
    from nnterp.families import gpt2

    class Paths(gpt2.Attention):
        @EProperty("../ln_2.output", description="A sibling module's output")
        def sibling(self, value) -> Residual:
            return value

        @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.output", description="An op two drills down")
        def softmax(self, value) -> Pattern:
            return value

        @EProperty(lambda self: "source.attention_interface_1.inputs", select="scaling", description="A function key, a keyword argument")
        def scaling(self, value) -> float:
            return value

        @EProperty(lambda self: "source.attention_interface_1.inputs", select=lambda self: "scaling", description="A select function")
        def scaling_selected(self, value) -> float:
            return value

        @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.input", description="A call's first argument")
        def scores(self, value) -> Pattern:
            return value

        @EProperty("source.attention_interface_1.inputs", select=lambda self: 1, description="A selector function: the queries")
        def queries(self, value):
            return value

        @EProperty("/norm.output", description="The final norm's output, by its standard name from the root")
        def final_norm(self, value) -> Residual:
            return value

        @EProperty("/inputs", select="input_ids", description="The model's input ids, from the root")
        def root_ids(self, value):
            return value

    model = StandardizedTransformer("hf-internal-testing/tiny-random-gpt2", dispatch=True, attn_implementation="eager", envoys={GPT2Attention: Paths})
    return model, Paths


def test_eproperty_paths_resolve_and_write(gpt2_paths):
    model, Paths = gpt2_paths
    attn = model.layers[0].self_attn
    assert Paths.softmax.inside_forward() and Paths.scaling.inside_forward() and not Paths.sibling.inside_forward()
    assert model.support()["self_attn.sibling"] is None and model.support()["self_attn.softmax"] is None
    with model.trace("Hello world"):  # forward order: the call's arguments, then the ops inside it, then the sibling norm
        scaling = attn.scaling.save()
        scores = attn.scores.save()
        softmax = attn.softmax.save()
        sibling = attn.sibling.save()
        ln_2 = model.layers[0].ln_2.output.save()
        clean = model.logits.save()
    assert torch.allclose(scores.softmax(-1).to(softmax.dtype), softmax) and torch.equal(sibling, ln_2)
    assert isinstance(scaling, float) and scaling == attn._module.head_dim ** -0.5
    with model.trace("Hello world"):
        attn.scores = attn.scores * 0                       # a first-argument write
        edited = model.logits.save()
    assert not torch.equal(clean, edited)
    with model.trace("Hello world"):
        attn.scaling = 0.0                                   # a keyword-argument write repacks the call
        pattern = attn.softmax.save()
    causal = torch.ones_like(pattern).tril()                 # with no scaling every score is 0: uniform over the causal keys
    assert torch.allclose(pattern, causal / causal.sum(-1, keepdim=True))


def test_root_anchored_keys(gpt2_paths):
    """A key with a leading ``/`` is walked from the model's root, aliases included, whatever the host."""
    model, Paths = gpt2_paths
    attn = model.layers[0].self_attn
    assert attn.root is model and not Paths.final_norm.inside_forward()
    assert model.support()["self_attn.final_norm"] is None
    with model.trace("Hello world"):
        ids = attn.root_ids.save()
        normed = attn.final_norm.save()
    with model.trace("Hello world"):
        input_ids = model.input_ids.save()
        ln_f = model.transformer.ln_f.output.save()  # the native name of `norm`
    assert torch.equal(ids, input_ids) and torch.equal(normed, ln_f)
    with model.trace("Hello world"):
        attn.final_norm = attn.final_norm * 0  # a write lands where the root's path leads
        logits = model.logits.save()
    assert torch.equal(logits, model.lm_head._module(torch.zeros_like(ln_f)))


def test_select_can_be_a_function_of_the_host(gpt2_paths):
    """``select`` given as a function picks the element at read time, for a read and for a write."""
    model, _ = gpt2_paths
    attn = model.layers[0].self_attn
    with model.trace("Hello world"):
        queries = attn.queries.save()
        clean = model.logits.save()
    with model.trace("Hello world"):
        standard = attn.attention_queries.save()
    assert torch.equal(queries, standard)
    with model.trace("Hello world"):
        attn.queries = attn.queries * 0
        edited = model.logits.save()
    assert not torch.equal(clean, edited)


def test_select_function_picks_the_element_per_access(gpt2_paths):
    """A `select` that is a function of the host picks the element at each read and write (`StateSpace`'s two kernels)."""
    model, Paths = gpt2_paths
    attn = model.layers[0].self_attn
    with model.trace("Hello world"):
        by_function = attn.scaling_selected.save()
    assert by_function == attn._module.head_dim ** -0.5
    with model.trace("Hello world"):
        attn.scaling_selected = 0.0
        pattern = attn.softmax.save()
    causal = torch.ones_like(pattern).tril()
    assert torch.allclose(pattern, causal / causal.sum(-1, keepdim=True))


def test_route_kernels_binds_each_state_space_kernel_to_its_own_torch_function():
    """A mixer with no per-token state (`StateSpace`) keeps its prompt kernel: each name gets its own pure-torch function."""
    import sys

    from nnterp import StateSpace, route_kernels
    from nnterp.families import mamba2

    module = sys.modules[mamba2.Mamba2Mixer.__module__]
    names = [StateSpace.CHUNK_KERNEL.rsplit("_", 1)[0], StateSpace.RECURRENT_KERNEL.rsplit("_", 1)[0]]
    before = {name: getattr(module, name) for name in names}
    try:
        route_kernels(mamba2, "torch")
        assert all(getattr(module, name) is before[name].__wrapped__ for name in names)
        route_kernels(mamba2, "default")
        assert {name: getattr(module, name) for name in names} == before
    finally:
        route_kernels(mamba2, "default")


def test_recurrent_mixer_without_a_state_op_reports_the_state_unavailable():
    """A `RecurrentMixer` whose kernels do not materialize the state per token says so, and still serves its call's values."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from nnterp import LinearAttention, RecurrentMixer
    from nnterp.components import EProperty
    from nnterp.components.recurrent import kernel, needs_torch_kernels

    class NoState(RecurrentMixer):
        CHUNK_KERNEL = LinearAttention.CHUNK_KERNEL
        RECURRENT_KERNEL = LinearAttention.RECURRENT_KERNEL

        @EProperty(kernel("output"), select=1, unavailable=needs_torch_kernels)
        def state_output(self, value):
            return value

    assert NoState.STATE_OP is None and NoState.KERNEL is RecurrentMixer.KERNEL
    model = StandardizedTransformer("yujiepan/qwen3.5-tiny-random", dispatch=True, envoys={Qwen3_5GatedDeltaNet: NoState})
    mix = model.layers[0].linear_attn
    assert type(mix) is NoState
    reason = "this mixer's kernels do not materialize the state per token"
    for name in ("state", "states"):
        assert mix.support()[name] == reason
        assert model.support()[f"linear_attn.{name}"][0] == reason
    assert mix.support()["state_output"] is None
    with pytest.raises(Unavailable, match="do not materialize the state per token"):
        mix.state_after(0)
    with pytest.raises(Unavailable, match="do not materialize the state per token"):
        mix.set_state_after(0, torch.zeros(1))
    with model.trace("Hello world"):
        final = mix.state_output.save()
    assert final.dim() == 4


def test_route_kernels_round_trips_the_bindings():
    """``"torch"`` binds both kernel names to the token-by-token loop; ``"default"`` restores what the module bound."""
    import sys

    from nnterp import LinearAttention, route_delta_rule, route_kernels
    from nnterp.families import qwen3_5_text

    module = sys.modules[qwen3_5_text.Qwen3_5GatedDeltaNet.__module__]
    chunk, recurrent = LinearAttention.CHUNK_KERNEL.rsplit("_", 1)[0], LinearAttention.RECURRENT_KERNEL.rsplit("_", 1)[0]
    before = {chunk: getattr(module, chunk), recurrent: getattr(module, recurrent)}
    loop = before[recurrent].__wrapped__  # functools.wraps on the dispatcher: the pure-torch token-by-token rule
    try:
        route_kernels(qwen3_5_text, "torch")
        assert getattr(module, chunk) is loop and getattr(module, recurrent) is loop
        route_kernels(qwen3_5_text, "default")
        assert {name: getattr(module, name) for name in before} == before
        route_delta_rule(module, "recurrent")  # the DeltaNet spelling, given the modeling module itself
        assert getattr(module, chunk) is loop
        route_delta_rule(module, "chunked")
        assert {name: getattr(module, name) for name in before} == before
        with pytest.raises(ValueError, match="'torch' or 'default'"):
            route_kernels(qwen3_5_text, "recurrent")
    finally:
        route_kernels(qwen3_5_text, "default")


def test_route_kernels_binds_a_single_step_decode_kernel_to_its_own():
    """On Mamba-1 (``STEP_STATE_OP`` set) the scan is the token loop: each name is bound to its own pure-torch function."""
    import sys

    from nnterp import SelectiveScan, route_kernels
    from nnterp.families import mamba

    module = sys.modules[mamba.MambaMixer.__module__]
    names = [op.rsplit("_", 1)[0] for op in (SelectiveScan.CHUNK_KERNEL, SelectiveScan.RECURRENT_KERNEL)]
    before = {name: getattr(module, name) for name in names}
    try:
        route_kernels(mamba, "torch")
        assert all(getattr(module, name) is before[name].__wrapped__ for name in names)
        assert SelectiveScan._loop_kernel() == SelectiveScan.CHUNK_KERNEL
        assert SelectiveScan._state_op(SelectiveScan.RECURRENT_KERNEL) == SelectiveScan.STEP_STATE_OP
        route_kernels(mamba, "default")
        assert {name: getattr(module, name) for name in names} == before
    finally:
        route_kernels(mamba, "default")


def test_decode_step_whose_first_read_is_relaxed_takes_its_own_branch():
    """A decode step that first reads ``input`` (a relaxed read) and then a kernel value binds on that
    step's own kernel: `per_call` counts the relaxed read as the step of the last pinned one."""
    import warnings

    from nnterp import route_kernels
    from nnterp.families import qwen3_5_text

    route_kernels(qwen3_5_text, "torch")
    try:
        model = StandardizedTransformer("yujiepan/qwen3.5-tiny-random", dispatch=True)
        mix = model.layers[0].linear_attn
        inputs, keys = [], []  # made outside the block: names bound inside do not survive it
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with model.generate("Hello world", max_new_tokens=3, do_sample=False) as tracer:
                for step in tracer.iter[:3]:
                    inputs.append(mix.input.save())
                    keys.append(mix.attention_keys.save())
        assert not [str(w.message) for w in caught if "cut short" in str(w.message).lower()]
        assert len(inputs) == 3 and len(keys) == 3
        assert all(value is not None for value in inputs + keys)
        assert keys[0].shape[1] == len(model.tokenizer("Hello world").input_ids)
        assert keys[1].shape[1] == 1 and keys[2].shape[1] == 1
    finally:
        route_kernels(qwen3_5_text, "default")


def _mamba2():
    from nnterp import route_kernels
    from nnterp.families import mamba2

    route_kernels(mamba2, "torch")
    return StandardizedTransformer("yujiepan/mamba2-tiny-random", dispatch=True)


@pytest.mark.parametrize("second", ["attention_keys", "state_output"])
def test_two_invokes_read_one_mixers_values(second):
    """Each invoke's worker keeps its own per-call records, so two invokes reading one mixer do not share a kernel choice."""
    import warnings

    deltanet = StandardizedTransformer("yujiepan/qwen3.5-tiny-random", dispatch=True)
    for model in (deltanet, _mamba2()):
        mix = next(block.linear_attn for block in model.layers if hasattr(block._module, "linear_attn") or hasattr(block._module, "mixer"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with model.trace() as tracer:
                with tracer.invoke("Hello world there"):
                    queries_a = mix.attention_queries.save()
                    second_a = getattr(mix, second).save()
                with tracer.invoke("Another prompt here ok"):
                    queries_b = mix.attention_queries.save()
                    second_b = getattr(mix, second).save()
        assert queries_a.shape[0] == queries_b.shape[0] == 1
        assert second_a.shape[0] == second_b.shape[0] == 1
        assert "_per_call" not in mix.__dict__  # the records are the worker's, freed with it


def test_edit_on_a_kernel_value_replays_on_another_prompt():
    """A ``model.edit`` replay runs on a fresh worker, so it reads the new call's arguments, not the first run's."""
    model = _mamba2()
    mix = model.layers[0].linear_attn
    with model.edit() as (tracer, edited):
        gate = edited.layers[0].linear_attn
        gate.betas = gate.betas * 0.5
    prompts = ("Hello world there", "A much longer prompt than the first one was")
    changed = []
    for prompt in prompts:  # back to back: nothing else touches the mixer between the replays
        with edited.trace(prompt):
            changed.append(edited.logits.save())
    for prompt, logits in zip(prompts, changed):
        with model.trace(prompt):
            betas = mix.betas.save()
            clean = model.logits.save()
        assert logits.shape == clean.shape and logits.shape[1] == betas.shape[1]
        assert not torch.equal(logits, clean)


def test_cached_call_with_several_tokens_reads_the_prompts_kernel():
    """Several tokens over a cached state run the prompt's kernel (the forward decodes only one token at a time); `KERNEL` follows."""
    import warnings

    model = StandardizedTransformer("yujiepan/qwen3.5-tiny-random", dispatch=True)
    mix = next(block.linear_attn for block in model.layers if hasattr(block._module, "linear_attn"))
    ids = model.tokenizer("Hello world there", return_tensors="pt").input_ids.to(model._module.device)
    with torch.no_grad():
        cache = model._module(ids, use_cache=True).past_key_values
    more = ids[:, :2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.trace(more, past_key_values=cache, use_cache=True):
            queries = mix.attention_queries.save()
            carried = mix.state_input.save()
    assert queries.shape[1] == 2 and carried is not None


def test_support_with_a_module_another_block_owns():
    """A block holding a module owned by another block (shared weights): its aliases are read off their own bindings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "hf-internal-testing/tiny-random-gpt2"
    raw = AutoModelForCausalLM.from_pretrained(repo)
    raw.transformer.h[1].shared_mlp = raw.transformer.h[0].mlp
    model = StandardizedTransformer(raw, tokenizer=AutoTokenizer.from_pretrained(repo))
    support = model.support(layer=1)
    assert support["mlp.mlp_output"] is None and support["layer_output"] is None
    assert model.support()["mlp.mlp_output"] is None


def test_value_repr_line_names_the_layout():
    """A value's repr line is ``(name) -> Layout [axes]: description``; a value without a layout keeps nnsight's line."""
    from nnterp.components import Attention, Layer
    from nnterp.components.linear_attention import LinearAttention
    from nnterp.standardized import StandardizedTransformer as Root

    assert str(Layer.layer_output) == (
        "(layer_output) -> Residual [batch seq hidden]: The residual stream leaving the block, a tensor even when the block returns a tuple"
    )
    assert str(Attention.attention_probabilities).startswith("(attention_probabilities) -> Pattern [batch heads query key]: ")
    assert str(LinearAttention.state_input).startswith("(state_input) -> State | None [batch heads key_dim value_dim]: ")
    assert str(Root.input_size) == "(input_size): [batch, seq] of the current call; read-only"


# -- top-k tokens ----------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2():
    return StandardizedTransformer("hf-internal-testing/tiny-random-gpt2", dispatch=True)


def test_probs_to_dict_keeps_tokens_that_decode_alike(gpt2):
    """Byte tokens 94-97 all decode to the replacement character; each keeps its own entry, keyed by its raw token."""
    tokenizer = gpt2.tokenizer
    colliding = [94, 95, 96, 97]
    assert {tokenizer.decode(i) for i in colliding} == {"�"}
    probs = torch.zeros(gpt2.vocab_size)
    weights = [0.3, 0.2, 0.15, 0.1, 0.05]
    for index, weight in zip([500, *colliding], weights):
        probs[index] = weight
    got = gpt2.probs_to_dict(probs, k=5)
    raw = [tokenizer.convert_ids_to_tokens(i) for i in colliding]
    assert list(got) == [tokenizer.decode(500), *raw]
    assert list(got.values()) == pytest.approx(weights)


def test_probs_to_dict_without_a_collision_keys_by_text(gpt2):
    probs = torch.zeros(gpt2.vocab_size)
    ids, weights = [500, 600, 700], [0.5, 0.3, 0.2]
    probs[ids] = torch.tensor(weights)
    got = gpt2.probs_to_dict(probs, k=3)
    assert list(got) == [gpt2.tokenizer.decode(i) for i in ids]
    assert list(got.values()) == pytest.approx(weights)


def test_get_topk_closest_tokens_returns_k_per_position(gpt2):
    with gpt2.trace("The quick brown fox jumps over the lazy dog"):
        resid = gpt2.layers[0].layer_output.save()
    k = 50
    top = gpt2.get_topk_closest_tokens(resid[0], k=k)
    assert len(top) == resid.shape[1]
    for row in top:
        assert len(row) == k
        assert list(row.values()) == sorted(row.values(), reverse=True)
