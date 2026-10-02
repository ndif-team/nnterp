"""OPT, end to end: no MLP module."""

import torch
from suite import FamilySuite, PROMPT, rows

from nnter.families import opt


class TestOPT(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-OPTForCausalLM"
    FAMILY = opt
    NATIVE = rows("model.decoder", "layers", "embed_tokens", "final_layer_norm", mlp=None, ln1="self_attn_layer_norm", ln2=None)
    MLP_NORM = "final_layer_norm"            # the block's own, native name (the decoder's has the same name)

    def test_no_mlp_module(self, model):
        """No block has an MLP module, so `support()` lists no ``mlp`` value at all."""
        assert not hasattr(model.layers[0], "mlp")
        assert not any(name.startswith("mlp.") for name in model.support())
        assert not any(name.startswith("mlp.") for name in model.support(layer=0))

    def test_contribution_identity(self, model):
        """No MLP module, but an MLP path on the block: ``fc2``'s output is what the block adds."""
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                parts[i] = (layer.input.save(), layer.self_attn.attention_output.save(), layer.fc2.output.save(), layer.layer_output.save())
        for i, (x, attn, mlp, out) in parts.items():
            eps = torch.finfo(out.dtype).eps
            torch.testing.assert_close(x.float() + attn.float() + mlp.float(), out.float(), rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")
