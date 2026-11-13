"""BitNet b1.58, end to end."""

import pytest
from suite import FamilySuite, LLAMA_ROWS, patched_checkpoint

from nnterp import StandardizedTransformer, Unavailable
from nnterp.families import bitnet

REPO = "hf-tiny-v2/tiny-random-BitNetForCausalLM"


class TestBitnet(FamilySuite):
    REPO = REPO
    FAMILY = bitnet
    NATIVE = LLAMA_ROWS
    PREMIX_IS_HEAD_OUTPUTS = False           # attn_sub_norm between the heads and o_proj
    NEURONS_FROM_ACTIVATION = False          # ffn_sub_norm between act(gate)*up and down_proj


def test_head_weights_refuse_on_bitnet_layers():
    """The released checkpoints load their projections as ``AutoBitLinear``, which applies a function of the weight it
    stores: ``value_weight`` and ``output_weight`` refuse there rather than slice the stored weight."""
    quantization = {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}
    model = StandardizedTransformer(patched_checkpoint(REPO, quantization_config=quantization), dispatch=True)
    attn = model.layers[0].self_attn
    assert type(attn.v_proj._module).__name__ == "AutoBitLinear"
    for read in (attn.value_weight, attn.output_weight):
        with pytest.raises(Unavailable, match="BitNet AutoBitLinear"):
            read(0)
