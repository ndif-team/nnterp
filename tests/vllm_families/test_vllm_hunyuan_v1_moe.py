"""Hunyuan MoE V1 on vLLM, against the transformers engine: a fused block returning a third element, a mixture of experts.

The tiny checkpoint's config is transformers' alone, and vLLM's Hunyuan
reads the keys Tencent's released configs carry beside it, so it is
config-patched for both engines (the weights and tokenizer are symlinked):
``hidden_act`` becomes ``silu`` (vLLM refuses anything else; the released
checkpoints use it), and ``moe_intermediate_size``, ``use_mixed_mlp_moe``,
``num_shared_expert`` and ``use_qk_norm`` say what transformers' modules
always do (experts and one shared expert ``intermediate_size`` wide, per-head
query and key norms), as Hunyuan-A13B's config says them.
"""

import json
import os
import tempfile

import torch
from huggingface_hub import snapshot_download
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import hunyuan_v1_moe


def _released_style_checkpoint(repo="hf-tiny-v2/tiny-random-HunYuanMoEV1ForCausalLM"):
    """The tiny checkpoint with the config keys vLLM reads written as a released config writes them; everything else symlinked."""
    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-hunyuan-moe-{os.path.basename(snapshot)}")
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        target = os.path.join(patched, name)
        if name != "config.json" and not os.path.exists(target):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config.update(
        hidden_act="silu", moe_intermediate_size=config["intermediate_size"], use_mixed_mlp_moe=True, num_shared_expert=1,
        use_qk_norm=True,
    )
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestVLLMHunyuanV1Moe(VLLMFamilySuite):
    REPO = _released_style_checkpoint()
    FAMILY = hunyuan_v1_moe
    NATIVE = LLAMA_ROWS

    def test_the_block_passes_its_keys_and_values_through(self, model, reference):
        """An edit of the stream leaves the block's third element, its attention's keys and values, as they were."""
        layer = model.layers[reference["middle"]]
        with self.run(model, reference):
            kept = layer.output[2][1].clone().cpu().save()
            layer.layer_output[:, -1] += 1.0
            after = layer.output[2][1].cpu().save()
        torch.testing.assert_close(after, kept)
