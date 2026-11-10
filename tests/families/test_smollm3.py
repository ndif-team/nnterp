"""SmolLM3, end to end."""

from suite import NO_ROTARY, FamilySuite, LLAMA_ROWS, built, patched_checkpoint

from nnterp.families import smollm3


class TestSmolLM3(FamilySuite):
    REPO = "yujiepan/smollm3-tiny-random"
    FAMILY = smollm3
    NATIVE = LLAMA_ROWS
    EXPECTED_UNAVAILABLE = NO_ROTARY  # block 1 is a NoPE block (no_rope_layers' 0)


class TestSmolLM3Full(TestSmolLM3):
    """SmolLM3-3B's layout: ``use_sliding_window`` false, every block full attention (the NoPE blocks as on the tiny
    checkpoint). The tiny checkpoint's weights."""

    REPO = built(patched_checkpoint, TestSmolLM3.REPO, use_sliding_window=False, layer_types=["full_attention", "full_attention"])

    def test_no_block_slides(self, model):
        assert all(layer.self_attn._module.sliding_window is None for layer in model.layers)
