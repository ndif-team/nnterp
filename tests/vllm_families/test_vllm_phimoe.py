"""Phi-3.5-MoE on vLLM, against the transformers engine: a pair-returning block that is not fused, a mixture of experts for an MLP."""

import gc
import json

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from vllm_suite import LLAMA_ROWS, VLLMFamilySuite, close, transformers_values

from nnterp.families.vllm import phimoe

#: The reference's logits, which on transformers carry the head's bias.
LOGITS = ("logits", "steered", "headless", "skipped")


def head_bias(repo):
    """The checkpoint's ``lm_head.bias``, read from its safetensors."""
    index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
    with safe_open(hf_hub_download(repo, index["weight_map"]["lm_head.bias"]), "pt") as weights:
        return weights.get_tensor("lm_head.bias").float()


class TestVLLMPhimoe(VLLMFamilySuite):
    REPO = "microsoft/Phi-tiny-MoE-instruct"   # Phi-3.5-MoE distilled to 3.8B: the architecture's smallest real checkpoint
    FAMILY = phimoe
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}
    MEMORY = 0.4

    @pytest.fixture(scope="class")
    def reference(self, request):
        """The transformers values, with the head's bias taken off every logit: vLLM's logits leave it out (see the family)."""
        values = transformers_values(self.REPO, self.SKIP, self.REFERENCE)
        gc.collect()
        torch.cuda.empty_cache()
        values["bias"] = head_bias(self.REPO)
        values["biased"] = values["logits"]
        for name in LOGITS:
            if values[name] is not None:
                values[name] = values[name] - values["bias"]
        return values

    def test_logits_leave_out_the_head_bias(self, model, reference):
        """vLLM loads ``lm_head.bias`` and never adds it: its logits are transformers' less the bias.

        On Phi-tiny-MoE the bias is up to 0.13 against logits up to 68: the
        logits differ from transformers' by 2.2e-3 of their scale, and by
        3.6e-4 once the bias is added back.
        """
        with self.run(model, reference):
            bias = model.lm_head.bias.cpu().save()
            logits = model.logits.cpu().save()
        torch.testing.assert_close(bias, reference["bias"], rtol=0, atol=0)
        close(logits + bias, reference["biased"], 1e-3, "logits + lm_head.bias")
        assert not torch.allclose(logits, reference["biased"], rtol=0, atol=bias.abs().max().item() / 2)
