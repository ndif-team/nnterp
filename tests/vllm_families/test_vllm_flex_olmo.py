"""FlexOlmo on vLLM, against the transformers engine: post-norms around a mixture of experts, a block returning ``(stream, None)``.

The tiny checkpoint's config says ``hidden_act: gelu``; vLLM's FlexOlmo
ignores it and runs its experts with SiLU, as the released checkpoints
specify. So the checkpoint is config-patched to ``silu`` for both engines
(the weights and tokenizer are symlinked); unpatched, the experts' outputs
differ by 1-3% on every token.
"""

import json
import os
import tempfile

import torch
from huggingface_hub import snapshot_download
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import flex_olmo


def _silu_checkpoint(repo="hf-tiny-v2/tiny-random-FlexOlmoForCausalLM"):
    """The tiny checkpoint with ``hidden_act`` rewritten to ``silu``; everything else symlinked."""
    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-flex-olmo-silu-{os.path.basename(snapshot)}")
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        target = os.path.join(patched, name)
        if name != "config.json" and not os.path.exists(target):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["hidden_act"] = "silu"
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestVLLMFlexOlmo(VLLMFamilySuite):
    REPO = _silu_checkpoint()
    FAMILY = flex_olmo
    NATIVE = LLAMA_ROWS

    def test_skip_layers_hands_on_no_residual(self, model, reference):
        """A skipped block returns ``(stream, None)`` like the block does, so the final norm adds nothing beside it."""
        last = model.num_layers - 1
        with self.run(model, reference):
            stream = model.layers[last].layer_input.save()
            model.layers[last].skip_with(stream)
            skipped = model.logits.cpu().save()
        with self.run(model, reference):
            lens = model.project_on_vocab(model.layers[last].layer_input[:, -1:]).cpu().save()
        torch.testing.assert_close(skipped, lens)
