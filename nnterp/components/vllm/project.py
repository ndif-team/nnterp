"""`project`: the logit lens on vLLM, with whichever module a family unembeds with."""

from __future__ import annotations

from typing import Any

import torch


def project(model: Any, hidden: torch.Tensor, head: Any, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Logits for ``hidden`` (``[..., hidden]``): the model's final norm, then vLLM's logits processor over ``head``.

    ``head`` is the module whose weight unembeds: ``lm_head`` where the model
    has one, the embedding where the two are tied into it (Gemma, Cohere).
    ``bias`` is the head's own, for a model whose logits add one (GPT-J, Phi).
    The processor applies the model's logit scale and softcap. The norm's
    kernel takes ``[rows, hidden]``, and a norm that takes the residual
    beside the hidden state hands back a pair.
    """
    rows = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    normed = model.norm(rows)
    if isinstance(normed, tuple):
        normed = normed[0]
    logits = model.logits_processor(head, normed) if bias is None else model.logits_processor(head, normed, bias)
    return logits.reshape(*hidden.shape[:-1], -1)
