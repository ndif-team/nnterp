"""AFMoE on vLLM, against the transformers engine: a fused block with sandwich norms, a dense block then mixtures of experts."""

import pytest
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp import Unavailable
from nnterp.families.vllm import afmoe


class TestVLLMAfmoe(VLLMFamilySuite):
    REPO = "hf-tiny-v2/tiny-random-AfmoeForCausalLM"
    FAMILY = afmoe
    NATIVE = LLAMA_ROWS

    def test_dense_then_mixture(self, model):
        """The first ``num_dense_layers`` blocks are dense; the others a mixture whose shared expert serves no contribution."""
        dense = model.config.num_dense_layers
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == ["LlamaMLP"] * dense + ["AfmoeMoE"] * (len(kinds) - dense)  # vLLM's AfmoeMLP is its LlamaMLP
        shared = model.layers[dense].mlp.shared_experts
        assert type(shared) is afmoe.Mlp
        with pytest.raises(Unavailable, match="vLLM"):
            shared.mlp_output
