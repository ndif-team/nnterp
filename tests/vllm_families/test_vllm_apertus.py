"""Apertus on vLLM, against the transformers engine: transformers' norm names, aliased.

vLLM's xIELU activation reads its parameters back as Python floats when it
is built (``.cpu().float().item()``). nnsight builds the client's tree on the
meta device, where only ``tolist`` is answered today, so the build fails
(``Cannot copy out of meta tensor``) until nnsight's meta build answers
``item`` and ``cpu`` too; the class is skipped until then.
"""

import pytest
import torch
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import apertus


def _meta_build_reads_scalars():
    """Whether nnsight's meta build of a vLLM model answers ``.cpu().item()`` on a meta tensor, as xIELU's constructor asks."""
    from nnsight.modeling.vllm import VLLM

    with VLLM._meta_values():
        try:
            torch.ones(1, device="meta").detach().cpu().float().item()
        except (NotImplementedError, RuntimeError):
            return False
    return True


@pytest.mark.skipif(not _meta_build_reads_scalars(), reason="nnsight's meta build cannot read back xIELU's scalars (.cpu().item() on meta)")
class TestVLLMApertus(VLLMFamilySuite):
    REPO = "swiss-ai/Apertus-v1.1-0.5B"
    FAMILY = apertus
    NATIVE = {
        **LLAMA_ROWS,
        "layers.0.input_layernorm": "model.layers.0.attention_layernorm",
        "layers.0.post_attention_layernorm": "model.layers.0.feedforward_layernorm",
    }
    # The first block's queries, keys and values agree with transformers' to 3e-7 of their scale; vLLM's attention
    # kernel makes head outputs 8e-4 off theirs, and the squared branch of xIELU in every MLP amplifies that: later
    # blocks' contributions and attention interior differ by up to 1.2e-2 of their scale (block 10's attention output,
    # values and scores), while the residual stream stays within 1e-4 and the logits within 2e-3. Transformers'
    # eager and SDPA kernels agree to 3e-6 everywhere on this checkpoint.
    TOLERANCE = 2e-2
    KERNEL_TOLERANCE = 2e-2
