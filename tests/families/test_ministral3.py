"""Ministral 3, end to end."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, built, patched_checkpoint, rows
from vision_suite import PixtralSuite

from nnterp.families import ministral3


class TestMinistral3(FamilySuite):
    REPO = "yujiepan/ministral-3-tiny-random"
    FAMILY = ministral3
    NATIVE = LLAMA_ROWS

    def rotated_as_served(self, attn, name, rotated, positions):
        """Past the rotary the queries are scaled by position, ``1 + beta * log(1 + floor(position / original))``
        (``llama_4_scaling_beta``, ``original_max_position_embeddings``): exactly one below that length, so the
        released checkpoints (0.1, 16384) scale only prompts longer than 16384 tokens."""
        rope = attn._module.config.rope_parameters
        beta, original = rope.get("llama_4_scaling_beta"), rope.get("original_max_position_embeddings")
        if name != "attention_queries_prerope" or not beta:
            return rotated
        scale = 1 + beta * torch.log(1 + torch.floor(torch.as_tensor(positions)[None] / original))
        return rotated * scale[:, None, :, None].to(rotated.device, rotated.dtype)


class TestMinistral3Scaled(TestMinistral3):
    """The query scale the released checkpoints apply past 16384 tokens (``llama_4_scaling_beta`` 0.1), brought within
    the suite's prompt (``original_max_position_embeddings`` 2): ``attention_queries`` is the rotated value times the
    scale, ``rotate`` the rotary alone. The tiny checkpoint's weights."""

    REPO = built(patched_checkpoint, "yujiepan/ministral-3-tiny-random", rope_parameters={"llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 2})

    def test_the_queries_are_scaled_past_the_original_length(self, model):
        attn = model.layers[0].self_attn
        with model.trace(PROMPT):
            pre = attn.attention_queries_prerope.save()
        with model.trace(PROMPT):
            served = attn.attention_queries.save()
        positions = torch.arange(pre.shape[-2])
        rotated = attn.rotate(pre, positions)
        assert torch.equal(served[..., :2, :], rotated[..., :2, :]) and not torch.equal(served[..., 2:, :], rotated[..., 2:, :])


class TestMistral3Wrapper(TestMinistral3):
    """Mistral 3 loaded as the wrapper with its processor: the text stack at ``model.language_model``."""

    REPO = "yujiepan/mistral-3-tiny-random"
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text", "dtype": torch.float32}   # in bf16 a zeroed query head moves no logit


class TestMistral3Vision(PixtralSuite):
    """Mistral 3's Pixtral tower and patch-merging projector, and the tower's image values."""

    REPO = "yujiepan/mistral-3-tiny-random"
    FAMILY = ministral3
    TEXT_REPO = TestMinistral3.REPO
