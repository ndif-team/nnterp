"""A model split over two devices the way ``device_map`` splits one: accelerate's hooks, here over one GPU and the CPU.

Each module runs on its own device and its hooks move its inputs there, so the model's rotary embedding computes its
cos and sin on the first device while a later block's queries and keys live on that block's.
"""

import pytest
import torch
from nnsight import TransformersModel  # noqa: F401  nnsight before any transformers submodule

from nnterp import StandardizedTransformer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the second device is a GPU beside the CPU")

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
#: The rotary embedding with the first block on the GPU, the second block on the CPU.
DEVICE_MAP = {
    "model.embed_tokens": 0, "model.rotary_emb": 0, "model.layers.0": 0, "lm_head": 0,
    "model.layers.1": "cpu", "model.norm": "cpu",
}
PROMPT = "Hello world there"


@pytest.fixture(scope="module")
def model():
    """The tiny Llama dispatched over ``cuda:0`` and the CPU.

    ``from_pretrained(device_map=...)`` with a GPU visible offloads the CPU entries (their weights stay on the CPU and
    run on the GPU), so the model is dispatched with ``main_device="cpu"``, which runs them on the CPU.
    """
    from accelerate import dispatch_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = AutoModelForCausalLM.from_pretrained(REPO, attn_implementation="eager")
    raw = dispatch_model(raw, DEVICE_MAP, main_device="cpu")
    return StandardizedTransformer(raw, tokenizer=AutoTokenizer.from_pretrained(REPO))


@pytest.mark.parametrize("pre, post", [("attention_queries_prerope", "attention_queries"), ("attention_keys_prerope", "attention_keys")])
def test_rotate_on_a_block_off_the_rotary_device(model, pre, post):
    """``rotate(prerope, positions)`` is the served queries and keys bit for bit on a block whose device is not the rotary's."""
    attn = model.layers[1].self_attn
    with model.trace(PROMPT):
        before = getattr(attn, pre).save()
    with model.trace(PROMPT):
        after = getattr(attn, post).save()
    assert before.device.type == "cpu" and model.model.rotary_emb._module.inv_freq.device.type == "cuda"
    rotated = attn.rotate(before, torch.arange(before.shape[-2]))
    assert rotated.device == after.device and torch.equal(rotated, after), (rotated - after.to(rotated.device)).abs().max()
