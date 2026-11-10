"""Phi-3.5-MoE, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, draw_biases, patched_checkpoint

from nnterp.families import phimoe


class TestPhimoe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-PhimoeForCausalLM"
    FAMILY = phimoe
    NATIVE = LLAMA_ROWS


class TestPhimoeTiny(TestPhimoe):
    """Phi-tiny-MoE's layout: biased attention projections and head, heads (2 x 32 here, 16 x 128 there) wider than the
    stream, a sliding window (7 tokens here, 2047 there). Weights drawn for it, the biases too."""

    REPO = built(
        patched_checkpoint, TestPhimoe.REPO, init=draw_biases, attention_bias=True, lm_head_bias=True, head_dim=32, sliding_window=7,
    )
