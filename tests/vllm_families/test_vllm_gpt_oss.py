"""GPT-OSS on vLLM, against the transformers engine: the stream first in the block's call, and attention sinks."""

import nnsight
import torch
from vllm_suite import PATTERN, VLLMFamilySuite, boundary, close, shifted

from nnterp.families.vllm import gpt_oss


class TestVLLMGptOss(VLLMFamilySuite):
    REPO = "tengomucho/tiny-random-gpt-oss"
    FAMILY = gpt_oss
    NATIVE = {
        "embed_tokens": "model.embedding",
        "layers": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
        "layers.0.self_attn": "model.layers.0.attn",
        "layers.0.mlp": "model.layers.0.mlp",
    }

    def test_pattern_matches_transformers(self, model, reference):
        """The suite's check, with the sink: each row sums to less than one, and to transformers' sum."""
        with self.run(model, reference):
            got = nnsight.save({i: boundary(model.layers[i], PATTERN) for i in reference["picked"]})
        tokens, heads = len(reference["ids"]), model.num_heads
        seen = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        for i, values in got.items():
            scores, probs = values["attention_scores"], values["attention_probabilities"]
            wanted = reference["layers"][i]
            assert scores.shape == probs.shape == (1, heads, tokens, tokens), i
            assert torch.isinf(scores[..., ~seen]).all() and torch.equal(probs, probs.tril())
            assert (probs.sum(-1) < 1).all()
            close(probs.sum(-1), wanted["attention_probabilities"].sum(-1), self.KERNEL_TOLERANCE, f"layers[{i}] row sums")
            close(shifted(scores, seen), shifted(wanted["attention_scores"], seen), self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.attention_scores")
            close(probs, wanted["attention_probabilities"], self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.attention_probabilities")
