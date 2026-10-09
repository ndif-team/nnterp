"""`model.order` / `model.rank`: the forward order of the values, measured by one probe scan."""

import pytest

from nnterp import StandardizedTransformer, StandardizedVLLM, Unavailable, route_kernels
from nnterp.families import qwen3_5_text

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
QWEN3_5 = "yujiepan/qwen3.5-tiny-random"

#: Tiny Llama's block, in forward order: the queries, keys and values are the attention call's arguments, one location.
LLAMA_BLOCK = {
    "layer_input": 0,
    "self_attn.attention_queries": 1,
    "self_attn.attention_keys": 1,
    "self_attn.attention_values": 1,
    "self_attn.attention_scores": 2,
    "self_attn.attention_probabilities": 3,
    "self_attn.attention_head_outputs": 4,
    "self_attn.attention_output": 5,
    "mlp.mlp_output": 6,
    "layer_output": 7,
}


@pytest.fixture(scope="module")
def llama():
    return StandardizedTransformer(LLAMA, attn_implementation="eager")


def test_llama_order(llama):
    assert llama.num_layers == 2
    assert llama.order() == {
        "input_ids": 0, "attention_mask": 0, "input_size": 0, "token_embeddings": 1, "logits": 2, "next_token_probs": 2,
    }
    assert llama.order(0) == llama.order(1) == llama.order(-1) == LLAMA_BLOCK
    assert [llama.rank(name) for name in ("input_ids", "attention_mask", "input_size", "token_embeddings")] == [
        (-1, 0), (-1, 0), (-1, 0), (-1, 1),
    ]
    assert llama.rank("logits") == llama.rank("next_token_probs") == (2, 2)
    assert {name: llama.rank(name, 1) for name in LLAMA_BLOCK} == {name: (1, r) for name, r in LLAMA_BLOCK.items()}
    assert llama.rank("layer_output", -1) == (1, 7)


def test_ranks_sort_reads_into_forward_order(llama):
    reads = [("logits", None), ("layer_output", 1), ("mlp.mlp_output", 0), ("layer_input", 1), ("token_embeddings", None)]
    reads.sort(key=lambda read: llama.rank(*read))
    assert reads == [("token_embeddings", None), ("mlp.mlp_output", 0), ("layer_input", 1), ("layer_output", 1), ("logits", None)]


def test_a_name_that_is_no_value(llama):
    with pytest.raises(KeyError, match="layer="):
        llama.rank("layer_output")          # a block value without its layer
    with pytest.raises(KeyError):
        llama.rank("self_attn.nothing", 0)


def test_falcon_values_come_before_queries_and_keys():
    model = StandardizedTransformer("Rocketknight1/tiny-random-falcon-7b", attn_implementation="eager")
    order = model.order(0)
    assert order["self_attn.attention_values"] < order["self_attn.attention_queries"] == order["self_attn.attention_keys"]
    assert order["layer_input"] < order["self_attn.attention_values"]
    assert model.rank("self_attn.attention_values", 0) < model.rank("self_attn.attention_queries", 0)


def test_hybrid_orders_per_block_shape():
    model = StandardizedTransformer(QWEN3_5, attn_implementation="eager")
    linear, attention = model.order(0), model.order(3)
    assert model.order(1) == model.order(2) == linear
    assert not any(name.startswith("self_attn.") for name in linear)
    assert not any(name.startswith("linear_attn.") for name in attention)
    assert attention == LLAMA_BLOCK
    kernel_inputs = ["attention_queries", "attention_keys", "attention_values", "betas", "decays", "state_input"]
    assert len({linear[f"linear_attn.{name}"] for name in kernel_inputs}) == 1
    assert linear["layer_input"] < linear["linear_attn.attention_queries"] < linear["linear_attn.attention_head_outputs"]
    assert linear["linear_attn.attention_output"] < linear["mlp.mlp_output"] < linear["layer_output"]
    # Layer-major: block 2's last value sorts before block 3's first, whatever the shapes' own ranks.
    reads = [(name, layer) for layer in (3, 2, 0) for name in model.order(layer)]
    reads.sort(key=lambda read: model.rank(*read))
    assert [layer for _, layer in reads] == sorted(layer for _, layer in reads)


def test_sdpa_ranks_only_what_it_has(llama):
    model = StandardizedTransformer(LLAMA, attn_implementation="sdpa")
    order = model.order(0)
    assert "self_attn.attention_scores" not in order and "self_attn.attention_probabilities" not in order
    eager = [name for name in sorted(LLAMA_BLOCK, key=LLAMA_BLOCK.get) if name in order]
    assert sorted(order, key=order.get) == eager
    assert model.order() == llama.order()
    with pytest.raises(Unavailable, match="eager"):
        model.rank("self_attn.attention_probabilities", 0)


def test_scan_needs_no_dispatch(llama):
    order = {layer: llama.order(layer) for layer in (None, 0, 1)}
    assert all(p.device.type == "meta" for p in llama._module.parameters())   # the probe loaded no weights
    dispatched = StandardizedTransformer(LLAMA, attn_implementation="eager", dispatch=True)
    assert {layer: dispatched.order(layer) for layer in (None, 0, 1)} == order


def test_a_forward_meta_tensors_cannot_run_is_probed_for_real():
    """Granite's forward branches on its data (``torch.equal``), so the scan fails and the probe traces instead."""
    model = StandardizedTransformer("hf-internal-testing/tiny-random-GraniteForCausalLM", attn_implementation="eager", dispatch=True)
    assert model.order(0) == LLAMA_BLOCK
    assert model.rank("logits") == (model.num_layers, 2)


def test_routed_kernels_add_the_state():
    """Kernels are routed before the model runs; the order is then measured under them."""
    chunked = StandardizedTransformer(QWEN3_5, attn_implementation="eager").order(0)
    assert "linear_attn.state" not in chunked and "linear_attn.states" not in chunked
    route_kernels(qwen3_5_text, "torch")
    try:
        routed = StandardizedTransformer(QWEN3_5, attn_implementation="eager").order(0)
    finally:
        route_kernels(qwen3_5_text, "default")
    assert routed["linear_attn.state"] < routed["linear_attn.states"] < routed["linear_attn.attention_head_outputs"]
    assert set(routed) - set(chunked) == {"linear_attn.state", "linear_attn.states"}


def test_vllm_raises():
    model = object.__new__(StandardizedVLLM)  # the methods raise before touching the engine
    with pytest.raises(NotImplementedError, match="vllm engine"):
        model.order()
    with pytest.raises(NotImplementedError, match="vllm engine"):
        model.rank("logits")
