"""StableLM-2 on vLLM, against the transformers engine: a block returning ``(stream, stale residual)``."""

import nnsight
import torch
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import stablelm


class TestVLLMStableLm(VLLMFamilySuite):
    REPO = "stabilityai/stablelm-2-1_6b"
    FAMILY = stablelm
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
    # StableLM-2-1.6B's first block has keys near 50, so its softmax is sharp: block 0's queries, keys and values
    # agree with transformers' to 3e-7 of their scale and the recomputed pattern to 1.2e-5, but the kernel's head
    # outputs to 5.2e-3. Carried through the stack, block 12's attention_output differs by 6.1e-3 and the last
    # block's layer_output by 6.6e-3 of their scales (logits 2.3e-3).
    TOLERANCE = 1e-2
    KERNEL_TOLERANCE = 1e-2

    def test_the_second_element_is_not_part_of_the_stream(self, model, reference):
        """The block's first element is the whole stream, which the next block takes alone; the second is stale."""
        first, second = model.layers[0], model.layers[1]
        with self.run(model, reference):
            pair = nnsight.save(tuple(value.clone().cpu() for value in first.output))
            stream = second.layer_input.cpu().save()
        torch.testing.assert_close(pair[0].unsqueeze(0), stream)
        assert not torch.allclose(pair[0], pair[1])
