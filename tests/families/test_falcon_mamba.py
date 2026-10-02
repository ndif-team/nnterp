"""Falcon-Mamba, end to end: Mamba-1 with RMS norms on the scan's step size, ``B`` and ``C``.

Every Mamba test applies unchanged; this file adds that the scan's ``B`` and
``C`` are the normed ones.
"""

import torch
from suite import PROMPT
from test_mamba import TestMamba as MambaTests

from nnter.families import falcon_mamba


class TestFalconMamba(MambaTests):
    REPO = "trl-internal-testing/tiny-FalconMambaForCausalLM"
    FAMILY = falcon_mamba
    REAL = ("tiiuae/falcon-mamba-7b", 64, 4096, 8192)

    def test_keys_and_queries_are_the_normed_b_and_c(self, model):
        """``B`` and ``C`` enter the scan through weightless RMS norms: unit root-mean-square per token."""
        mix = model.layers[0].linear_attn
        for name in ("attention_keys", "attention_queries"):
            with model.trace(PROMPT):
                value = getattr(mix, name).save()
            rms = value.float().pow(2).mean(-1).sqrt()
            torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-2, rtol=0)
