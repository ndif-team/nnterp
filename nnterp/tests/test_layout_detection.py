"""Unit tests for ``LayerAccessor`` layout-detection helpers.

These don't load any model — they exercise the pure-Python helpers that
inspect ``forward`` signatures and source code to decide which I/O convention
a layer follows. Cheap to run, useful as a regression net for the layout
inference logic that the heavier vLLM↔HF equivalence tests depend on.
"""
from __future__ import annotations

import torch as th
from torch import nn

from nnterp.rename_utils import (
    InputLayout,
    OutputLayout,
    RenamingError,
    _infer_input_layout,
    _infer_output_layout,
    _parent_calls_with_kwargs,
)


# --- Fake modules emulating each calling convention ---


class _HFLikeAttention(nn.Module):
    """HF attention: ``forward(hidden_states)``."""

    def forward(self, hidden_states):
        return hidden_states


class _VLLMAttention(nn.Module):
    """vLLM Llama attention: ``forward(positions, hidden_states)``."""

    def forward(self, positions, hidden_states):
        return hidden_states


class _VLLMDecoderLayer(nn.Module):
    """vLLM Llama decoder: ``forward(positions, hidden_states, residual)``."""

    def __init__(self):
        super().__init__()
        self.self_attn = _VLLMAttention()
        self.mlp = _HFLikeAttention()  # mlp(x) — same shape as hf

    def forward(self, positions, hidden_states, residual):
        # vLLM calls self_attn with kwargs; mlp positionally — exactly the pattern
        # _parent_calls_with_kwargs needs to detect.
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class _HFLikeDecoder(nn.Module):
    """HF decoder: ``forward(hidden_states, ...)``, sub-modules called positionally."""

    def __init__(self):
        super().__init__()
        self.self_attn = _HFLikeAttention()
        self.mlp = _HFLikeAttention()

    def forward(self, hidden_states):
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


# --- Input layout from forward signature ---


def test_input_layout_hf_decoder():
    """HF decoder: first positional arg is ``hidden_states`` → HIDDEN."""
    assert _infer_input_layout(_HFLikeDecoder()) is InputLayout.HIDDEN


def test_input_layout_vllm_decoder():
    """vLLM decoder: ``(positions, hidden_states, residual)`` → POSITIONS_HIDDEN_RESIDUAL."""
    assert _infer_input_layout(_VLLMDecoderLayer()) is InputLayout.POSITIONS_HIDDEN_RESIDUAL


def test_input_layout_vllm_attention():
    """vLLM attention: ``(positions, hidden_states)`` → POSITIONS_HIDDEN (no residual)."""
    assert _infer_input_layout(_VLLMAttention()) is InputLayout.POSITIONS_HIDDEN


def test_input_layout_hf_attention():
    """HF attention: ``(hidden_states,)`` → HIDDEN."""
    assert _infer_input_layout(_HFLikeAttention()) is InputLayout.HIDDEN


# --- Calling-convention detection from parent source ---


def test_parent_calls_self_attn_with_kwargs():
    """vLLM decoder calls ``self.self_attn(...=...)`` → kwargs path."""
    assert _parent_calls_with_kwargs(_VLLMDecoderLayer(), "self_attn") is True


def test_parent_calls_mlp_positionally():
    """vLLM decoder calls ``self.mlp(hidden_states)`` → positional path."""
    assert _parent_calls_with_kwargs(_VLLMDecoderLayer(), "mlp") is False


def test_parent_calls_hf_decoder_positionally():
    """HF decoder always calls sub-modules positionally."""
    assert _parent_calls_with_kwargs(_HFLikeDecoder(), "self_attn") is False
    assert _parent_calls_with_kwargs(_HFLikeDecoder(), "mlp") is False


# --- Output layout from runtime values ---


def test_output_layout_single_tensor():
    """A bare tensor is SINGLE."""
    t = th.zeros(2, 4)
    assert _infer_output_layout(t) is OutputLayout.SINGLE


def test_output_layout_dual_stream():
    """Two same-shape float tensors → DUAL_STREAM."""
    a = th.zeros(2, 4)
    b = th.ones(2, 4)
    assert _infer_output_layout((a, b)) is OutputLayout.DUAL_STREAM


def test_output_layout_tuple_first():
    """``(hidden, non-tensor metadata)`` → TUPLE_FIRST (HF's past_kv pattern)."""
    a = th.zeros(2, 4)
    assert _infer_output_layout((a, None)) is OutputLayout.TUPLE_FIRST
    assert _infer_output_layout((a, "metadata")) is OutputLayout.TUPLE_FIRST


def test_output_layout_dual_stream_shape_mismatch_raises():
    """Two float tensors with different shapes look like dual-stream gone wrong → RenamingError."""
    a = th.zeros(2, 4)
    b = th.zeros(2, 8)
    try:
        _infer_output_layout((a, b))
    except RenamingError as e:
        assert "mismatched shapes" in str(e), f"unexpected message: {e}"
    else:
        raise AssertionError("expected RenamingError for mismatched dual-stream shapes")


def test_output_layout_tuple_int_dtype_is_tuple_first():
    """Two-tensor tuple where one is an int (e.g. positions) → TUPLE_FIRST, not DUAL_STREAM."""
    a = th.zeros(2, 4)
    b = th.zeros(2, 4, dtype=th.int64)
    assert _infer_output_layout((a, b)) is OutputLayout.TUPLE_FIRST
