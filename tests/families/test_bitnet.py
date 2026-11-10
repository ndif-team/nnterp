"""BitNet b1.58, end to end."""

import json
import os
import tempfile

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, cached_snapshot

from nnterp import StandardizedTransformer
from nnterp.components.attention import bitnet_weight
from nnterp.families import bitnet

REPO = "hf-tiny-v2/tiny-random-BitNetForCausalLM"


class TestBitnet(FamilySuite):
    REPO = REPO
    FAMILY = bitnet
    NATIVE = LLAMA_ROWS
    PREMIX_IS_HEAD_OUTPUTS = False           # attn_sub_norm between the heads and o_proj
    NEURONS_FROM_ACTIVATION = False          # ffn_sub_norm between act(gate)*up and down_proj


def _quantized_checkpoint(mode: str) -> str:
    """The tiny checkpoint with the released checkpoints' ``quantization_config`` in its config: every projection loads as
    transformers' ``AutoBitLinear``, ``online`` (ternarized on every call, microsoft/bitnet-b1.58-2B-4T-bf16) or
    ``offline`` (the stored weight times ``weight_scale``, microsoft/bitnet-b1.58-2B-4T)."""
    snapshot = cached_snapshot(REPO)
    patched = tempfile.mkdtemp(prefix="bitnet-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["quantization_config"] = {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": mode}
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_head_weights_are_the_weights_the_layers_apply(mode):
    """``value_weight`` and ``output_weight`` slice the weight each ``AutoBitLinear`` multiplies by, not the one it stores:
    with the per-token 8-bit rounding of its input (``ActQuant``), they give the served values and ``o_proj``'s output."""
    from transformers.integrations.bitnet import ActQuant, AutoBitLinear

    model = StandardizedTransformer(_quantized_checkpoint(mode), dispatch=True, attn_implementation="eager")
    attn = model.layers[0].self_attn
    v_proj = attn.v_proj._module
    assert isinstance(v_proj, AutoBitLinear) and v_proj.online_quant == (mode == "online")
    if mode == "offline":  # the tiny checkpoint stores no weight_scale; give every layer one away from 1
        for module in model._module.modules():
            if isinstance(module, AutoBitLinear):
                module.weight_scale.fill_(2.5)
    with torch._dynamo.config.patch(disable=True), torch.no_grad():  # the quantizers are torch.compile'd; run them eagerly
        with model.trace(PROMPT):
            x = model.layers[0].attention_norm_output.save()
            values = attn.attention_values.save()
            premix = attn.attention_premix.save()
            out = attn.o_proj.output.save()
        x = ActQuant.apply(x)
        rounded = ActQuant.apply(premix.flatten(-2)).unflatten(-1, premix.shape[-2:])
    assert not torch.equal(attn.value_weight(0), v_proj.weight.unflatten(0, (attn.num_kv_heads, attn.head_dim))[0])
    for kv in range(attn.num_kv_heads):
        torch.testing.assert_close(x @ attn.value_weight(kv).T, values[:, kv])
    mixed = sum(rounded[:, :, j] @ attn.output_weight(j).T for j in range(attn.num_heads))
    torch.testing.assert_close(mixed, out)


def test_bitlinear_weight_is_the_unpacked_codes_over_the_scale():
    """A packed ``BitLinear`` (``linear_class: bitlinear``) stores 2-bit codes and divides by ``weight_scale``: the
    weight it applies is the unpacked codes over that scale, which ``bitnet_weight`` returns."""
    from transformers.integrations.bitnet import BitLinear, pack_weights

    torch.manual_seed(0)
    layer = BitLinear(16, 8, bias=False, dtype=torch.float32)
    ternary = torch.randint(-1, 2, (8, 16)).to(torch.float32)
    layer.weight.copy_(pack_weights(ternary.to(torch.int8)))
    layer.weight_scale.fill_(0.25)
    x = torch.randn(2, 3, 16)
    with torch._dynamo.config.patch(disable=True), torch.no_grad():  # the unpacking and the rounding are torch.compile'd
        applied = bitnet_weight(layer)
        quantized, scale = layer.activation_quant(x)
        out = layer(x)
    torch.testing.assert_close(applied, ternary / 0.25)
    torch.testing.assert_close(out, (quantized.float() / scale) @ applied.T)
