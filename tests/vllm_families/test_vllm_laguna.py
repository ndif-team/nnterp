"""Laguna on vLLM, against the transformers engine: gated attention, a dense block then a mixture of experts.

The tiny checkpoint does not run on vLLM as it is: its heads are 8 wide,
which vLLM's attention kernels cannot take (the engine hangs on the first
step), and its config says ``hidden_act: gelu``, which vLLM's Laguna refuses
(SiLU only, as the released checkpoints specify). So the test runs a copy
with ``head_dim`` 32 and ``hidden_act`` ``silu``, written once per snapshot:
the tiny weights where the shapes agree, a seeded initialisation of the
attention projections where they do not; the tokenizer is symlinked.
"""

import os
import tempfile

import pytest
import torch
from huggingface_hub import snapshot_download
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp import Unavailable
from nnterp.families.vllm import laguna


def _wide_heads_checkpoint(repo="hf-tiny-v2/tiny-random-LagunaForCausalLM", head_dim=32):
    """The tiny checkpoint with ``head_dim`` widened and ``hidden_act`` ``silu``; tiny weights where the shapes agree."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-laguna-heads{head_dim}-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    config = AutoConfig.from_pretrained(snapshot)
    config.head_dim, config.hidden_act = head_dim, "silu"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    tiny = load_file(os.path.join(snapshot, "model.safetensors"))
    state = model.state_dict()
    model.load_state_dict({name: tiny[name] if name in tiny and tiny[name].shape == tensor.shape else tensor for name, tensor in state.items()})
    model.save_pretrained(patched)
    for name in os.listdir(snapshot):
        if name.startswith("tokenizer") or name == "chat_template.jinja":
            target = os.path.join(patched, name)
            if not os.path.exists(target):
                os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


class TestVLLMLaguna(VLLMFamilySuite):
    REPO = _wide_heads_checkpoint()
    FAMILY = laguna
    NATIVE = LLAMA_ROWS

    def test_dense_then_mixture(self, model):
        """``mlp_layer_types`` decides each block's MLP; a mixture's shared expert serves no contribution."""
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == [{"dense": "LagunaMLP", "sparse": "LagunaMoE"}[kind] for kind in model.config.mlp_layer_types]
        sparse = kinds.index("LagunaMoE")
        shared = model.layers[sparse].mlp.shared_experts
        assert type(shared) is laguna.Mlp
        with pytest.raises(Unavailable, match="vLLM"):
            shared.mlp_output
