"""StableLM-2, end to end: a parallel block."""

from suite import FamilySuite, rows

from nnter.families import stablelm


class TestStableLm(FamilySuite):
    REPO = "stabilityai/tiny-random-stablelm-2"
    FAMILY = stablelm
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel

    def test_block_is_parallel(self, model):
        assert model.config.use_parallel_residual
