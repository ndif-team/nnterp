"""DeepSeek-V3, end to end: multi-head latent attention, and a dense block before a mixture of experts."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.components import Moe
from nnter.families import deepseek_v3

REPO = "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM"


def _moe_checkpoint(repo=REPO):
    """The tiny checkpoint with a mixture of experts on its second block.

    As published it has none: ``first_k_dense_replace`` is 3 on a 2-block
    model, and its ``topk_group`` (4) exceeds ``n_group`` (2), which the
    router's group-limited top-k cannot take. The config is rewritten to
    ``first_k_dense_replace=1, topk_group=1``, so block 0 stays the dense MLP
    the weights hold and block 1 is a mixture; the mixture's weights, which the
    checkpoint lacks, are drawn once, seeded, from the model's own
    initialization, and written beside the checkpoint's. Written once per
    snapshot; the tokenizer is symlinked.
    """
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnter-deepseek-v3-moe-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    for name in os.listdir(snapshot):
        if name not in ("config.json", "model.safetensors") and not os.path.exists(os.path.join(patched, name)):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config.update(first_k_dense_replace=1, topk_group=1)
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    torch.manual_seed(0)
    fresh = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(patched), dtype=torch.float32).state_dict()
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    weights = {name: weights.get(name, tensor).contiguous() for name, tensor in fresh.items() if name in weights or "mlp" in name}
    partial = os.path.join(patched, f"model.safetensors.{os.getpid()}")
    save_file(weights, partial, metadata={"format": "pt"})
    os.replace(partial, os.path.join(patched, "model.safetensors"))
    return patched


class TestDeepseekV3(FamilySuite):
    REPO = _moe_checkpoint()
    FAMILY = deepseek_v3
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim

    def test_a_dense_block_then_a_mixture(self, model):
        assert type(model.layers[0].mlp) is deepseek_v3.Mlp and not isinstance(model.layers[0].mlp, Moe)
        assert type(model.layers[1].mlp) is deepseek_v3.Moe
        assert model.support()["mlp.router_logits"] == {0: "no router_logits value on this block's mlp"}

    def test_router_logits_write_by_hand(self, model):
        """Sigmoid scoring, group-limited top-k, renormalized and times ``routed_scaling_factor``, from written logits."""
        moe = model.layers[1].mlp
        with model.trace(PROMPT):
            logits = moe.router_logits.save()
        written = torch.randn(logits.shape, generator=torch.Generator().manual_seed(0)).to(logits)
        with model.trace(PROMPT):
            moe.router_logits = written
            idx = moe.expert_indices.save()
            w = moe.expert_weights.save()
            x = moe.experts.inputs[0][0].save()
            routed = moe.routed_output.save()
        gate = moe.router._module
        scores = written.sigmoid()
        choice = scores + gate.e_score_correction_bias  # one group of experts per token: topk_group is 1
        groups = choice.view(*choice.shape[:2], gate.num_group, -1).topk(2, dim=-1)[0].sum(-1)
        group = groups.argmax(-1, keepdim=True)
        mask = torch.nn.functional.one_hot(group, gate.num_group).bool().repeat_interleave(gate.num_experts // gate.num_group, -1).squeeze(-2)
        expected_idx = choice.masked_fill(~mask, float("-inf")).topk(gate.top_k, dim=-1).indices
        assert torch.equal(idx.sort(-1).values, expected_idx.sort(-1).values)
        expected_w = scores.gather(-1, idx)
        expected_w = expected_w / expected_w.sum(-1, keepdim=True) * gate.routed_scaling_factor
        torch.testing.assert_close(w, expected_w)
        experts = moe.experts._module
        by_hand = torch.zeros_like(routed)
        for t in range(routed.shape[1]):
            for j in range(gate.top_k):
                hidden = experts._apply_gate(torch.nn.functional.linear(x[t], experts.gate_up_proj[idx[0, t, j]])[None])[0]
                by_hand[0, t] += torch.nn.functional.linear(hidden, experts.down_proj[idx[0, t, j]]) * w[0, t, j]
        torch.testing.assert_close(routed, by_hand, rtol=1e-5, atol=1e-6)
