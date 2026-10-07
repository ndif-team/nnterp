"""MiMo-V2-Flash on vLLM, against the transformers engine: values narrower than queries, sinks on the sliding blocks, MoE.

The tiny checkpoint does not run on vLLM as it is: its query and key heads
are 8 wide, which vLLM's attention kernels cannot take, and its config is
transformers' alone, without the names vLLM reads (Xiaomi's released
``config.json`` carries both). So the test runs a copy with ``head_dim`` 64
and ``v_head_dim`` 32 (vLLM's attention for values narrower than keys) and
Xiaomi's names written from transformers' values, saved once per snapshot:
the tiny weights where the shapes agree, a seeded initialisation of the
attention projections where they do not; the tokenizer is symlinked.
"""

import os
import tempfile

import nnsight
import torch
from huggingface_hub import snapshot_download
from vllm_suite import INTERIOR, LLAMA_ROWS, PATTERN, VLLMFamilySuite, boundary, close

from nnterp.families.vllm import mimo_v2_flash


def _vllm_ready_checkpoint(repo="hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM", head_dim=64, v_head_dim=32):
    """The tiny checkpoint with wider heads and the config names vLLM reads; tiny weights where the shapes agree."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = snapshot_download(repo, local_files_only=True)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-vllm-mimo-{head_dim}-{v_head_dim}-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    config = AutoConfig.from_pretrained(snapshot)
    config.head_dim, config.v_head_dim = head_dim, v_head_dim
    full, sliding = config.rope_parameters["full_attention"], config.rope_parameters["sliding_attention"]
    assert full["partial_rotary_factor"] == sliding["partial_rotary_factor"]  # vLLM takes one factor for both
    xiaomi = {
        "hybrid_layer_pattern": [int(kind == "sliding_attention") for kind in config.layer_types],
        "moe_layer_freq": [int(kind == "sparse") for kind in config.mlp_layer_types],
        "swa_num_attention_heads": config.num_attention_heads,
        "swa_num_key_value_heads": 2 * config.num_key_value_heads,
        "swa_head_dim": head_dim,
        "swa_v_head_dim": v_head_dim,
        "sliding_window_size": config.sliding_window,
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "layernorm_epsilon": config.rms_norm_eps,
        "rope_theta": full["rope_theta"],
        "swa_rope_theta": sliding["rope_theta"],
        "partial_rotary_factor": full["partial_rotary_factor"],
    }
    for name, value in xiaomi.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    tiny = load_file(os.path.join(snapshot, "model.safetensors"))
    state = model.state_dict()
    model.load_state_dict({name: tiny[name] if name in tiny and tiny[name].shape == tensor.shape else tensor for name, tensor in state.items()})
    with torch.no_grad():
        for layer in model.model.layers:
            if layer.self_attn.sinks is not None:
                layer.self_attn.sinks.normal_(generator=torch.Generator().manual_seed(1))  # sinks that move the pattern
    model.save_pretrained(patched)
    for name in os.listdir(snapshot):
        if name.startswith("tokenizer") or name == "chat_template.jinja":
            target = os.path.join(patched, name)
            if not os.path.exists(target):
                os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


class TestVLLMMimoV2Flash(VLLMFamilySuite):
    REPO = _vllm_ready_checkpoint()
    FAMILY = mimo_v2_flash
    NATIVE = LLAMA_ROWS

    def test_pattern_matches_transformers(self, model, reference):
        """The suite's check, but a sliding block's rows sum to one less the sink's share, as on transformers."""
        with self.run(model, reference):
            got = nnsight.save({i: boundary(model.layers[i], PATTERN) for i in reference["picked"]})
        tokens = len(reference["ids"])
        seen = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        sliding = [kind == "sliding_attention" for kind in model.config.layer_types]
        for i, values in got.items():
            scores, probs = values["attention_scores"], values["attention_probabilities"]
            assert torch.isinf(scores[..., ~seen]).all() and torch.equal(probs, probs.tril())
            sums = probs.sum(-1)
            assert (sums < 1 - 1e-4).all() if sliding[i] else torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
            close(probs, reference["layers"][i]["attention_probabilities"], self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.attention_probabilities")
            masked = scores.masked_fill(~seen, 0)
            close(masked, reference["layers"][i]["attention_scores"].masked_fill(~seen, 0), self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.attention_scores")

    def test_attention_interior_matches_transformers(self, model, reference):
        """The suite's check, with each block's own shapes: queries and keys ``qk_head_dim`` wide, values and head outputs
        ``head_dim`` (``v_head_dim``), and a sliding block's key/value heads twice the config's ``num_key_value_heads``."""
        with self.run(model, reference):
            got = nnsight.save({i: boundary(model.layers[i], INTERIOR) for i in reference["picked"]})
        tokens, heads = len(reference["ids"]), model.num_heads
        for i, values in got.items():
            kv_heads = model.num_kv_heads * (2 if model.config.layer_types[i] == "sliding_attention" else 1)
            shapes = {
                "attention_queries": (1, heads, tokens, model.qk_head_dim),
                "attention_keys": (1, kv_heads, tokens, model.qk_head_dim),
                "attention_values": (1, kv_heads, tokens, model.head_dim),
                "attention_head_outputs": (1, tokens, heads, model.head_dim),
            }
            for name in INTERIOR:
                assert values[name].shape == shapes[name], (i, name, values[name].shape)
                close(values[name], reference["layers"][i][name], self.KERNEL_TOLERANCE, f"layers[{i}].self_attn.{name}")
        assert model.qk_head_dim != model.head_dim and "sliding_attention" in [model.config.layer_types[i] for i in got]
