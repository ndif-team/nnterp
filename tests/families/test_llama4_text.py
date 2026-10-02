"""Llama 4, text, end to end: interleaved dense and MoE blocks, iRoPE, chunked attention, a wrapper checkpoint.

The pinned checkpoint, ``yujiepan/llama-4-tiny-random``, is a
``Llama4ForConditionalGeneration`` (model_type ``llama4``) whose
``text_config`` is ``llama4_text``: four blocks, dense MLPs on 0 and 2 and a
mixture of experts on 1 and 3, block 3 a NoPE block with full attention and
temperature tuning, the others RoPE with qk-norm and chunked attention
(``attention_chunk_size`` 128). The text-generation task loads it as
``Llama4ForCausalLM``.

It is config-patched: its ``attn_temperature_tuning`` is the int ``4``, which
transformers 5.17's config validation rejects (it wants a bool; the model
only tests it for truth), so ``config.json`` is rewritten with ``True`` and
the weights and tokenizer are symlinked. The one ``Llama4ForCausalLM`` tiny
checkpoint on the Hub, ``trl-internal-testing/tiny-Llama4ForCausalLM``,
parses but its logits are NaN in plain transformers 5.17, and it has no
dense block and no NoPE block.
"""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import MOE, FamilySuite, LLAMA_ROWS, PROMPT

from nnter import StandardizedTransformer, Unavailable
from nnter.families import llama4_text

NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.feed_forward"}


def _patched_checkpoint(repo="yujiepan/llama-4-tiny-random"):
    """The tiny checkpoint with ``attn_temperature_tuning`` as the bool 5.17 validates; weights and tokenizer symlinked."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="llama4-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    text = config["text_config"]
    text["attn_temperature_tuning"] = bool(text["attn_temperature_tuning"])
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestLlama4Text(FamilySuite):
    REPO = _patched_checkpoint()
    FAMILY = llama4_text
    MOE_UNAVAILABLE = {"expert_weights": "dense score", "expert_outputs": "dense score"}
    NATIVE = NATIVE

    def test_the_checkpoint_has_every_kind_of_block(self, model):
        """Dense and MoE blocks, RoPE and NoPE blocks, chunked and full attention: the quirks are all exercised."""
        config = model.config
        assert config.model_type == "llama4_text"
        assert config.moe_layers == [1, 3] and config.no_rope_layers == [1, 1, 1, 0]
        assert config.layer_types == ["chunked_attention"] * 3 + ["full_attention"]
        assert config.attn_temperature_tuning is True and config.use_qk_norm
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == ["Llama4TextMLP", "Llama4TextMoe", "Llama4TextMLP", "Llama4TextMoe"]
        assert [hasattr(layer.self_attn._module, "qk_norm") for layer in model.layers] == [True, True, True, False]

    def test_moe_contribution_is_in_the_residual_shape(self, model):
        """The mixture returns its output flattened over batch and sequence; ``mlp_output`` is the block's view of it."""
        with model.trace(PROMPT):
            raw = model.layers[1].mlp.output[0].save()
            contribution = model.layers[1].mlp.mlp_output.save()
            out = model.layers[1].layer_output.save()
        assert raw.dim() == 2 and contribution.shape == out.shape
        assert torch.equal(contribution.reshape(raw.shape), raw)
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            model.layers[1].mlp.mlp_output[:, -1] = 0  # in place, on the view: lands on the mixture's output
            edited = model.logits.save()
        assert not torch.equal(clean[:, -1], edited[:, -1])
        torch.testing.assert_close(clean[:, :-1], edited[:, :-1])

    def test_shared_expert_has_no_contribution_of_its_own(self, model):
        shared = model.layers[1].mlp.shared_expert
        assert type(shared) is llama4_text.Mlp  # the same module class as a dense block's MLP
        with pytest.raises(Unavailable, match="shared expert"):
            shared.mlp_output

    def test_intermediate_size_is_the_dense_width(self, model):
        assert model.intermediate_size == model.config.intermediate_size_mlp == model.layers[0].mlp._module.gate_proj.out_features
        assert model.layers[1].mlp._module.shared_expert.gate_proj.out_features == model.config.intermediate_size

    def test_nope_block_interior(self, model):
        """On the NoPE block the queries are temperature-scaled and the keys carry no rotary embedding or qk-norm.

        The checkpoint's ``floor_scale`` (8192) leaves the scale at one on a short
        prompt, so the module's is lowered for the trace to make it vary by position.
        """
        attn = model.layers[3].self_attn
        module = attn._module
        assert not module.use_rope and not hasattr(module, "qk_norm")
        floor_scale, module.floor_scale = module.floor_scale, 2
        try:
            with model.trace(PROMPT):
                normed = model.layers[3].input_layernorm.output.save()
                keys = attn.attention_keys.save()
            with model.trace(PROMPT):
                queries = attn.attention_queries.save()
        finally:
            module.floor_scale = floor_scale
        batch, seq, _ = normed.shape
        heads_first = lambda x: x.view(batch, seq, -1, module.head_dim).transpose(1, 2)
        torch.testing.assert_close(keys, heads_first(module.k_proj(normed)))
        positions = torch.arange(seq, dtype=torch.float32, device=normed.device)
        scales = torch.log1p(torch.floor((positions + 1.0) / 2)) * module.attn_scale + 1.0
        assert scales[-1] > 1
        plain = heads_first(module.q_proj(normed))
        torch.testing.assert_close(queries, (plain * scales.view(1, 1, seq, 1)).to(plain.dtype))

    def test_moe_contribution_per_invoke(self, model):
        """Two invokes: each reads its own rows of the mixture's output, in the residual's shape."""
        other = "A completely different and longer prompt"
        with model.trace(PROMPT):
            alone = model.layers[1].mlp.mlp_output.save()
        with model.trace() as tracer:
            with tracer.invoke(PROMPT):
                first = model.layers[1].mlp.mlp_output.save()
            with tracer.invoke(other):
                second = model.layers[1].mlp.mlp_output.save()
        assert first.shape[0] == second.shape[0] == 1 and first.shape[-1] == second.shape[-1] == model.hidden_size
        assert second.shape[1] == len(model.tokenizer(other).input_ids)
        assert first.shape[1] == second.shape[1]  # padded to the longer prompt
        assert alone.shape[1] < second.shape[1]

    def test_first_trace_of_a_meta_load_reads_the_mlp_after_the_attention(self):
        """``mlp_output`` is an operation in the block's forward; it is served on the very first call, after weights are dispatched."""
        model = StandardizedTransformer(self.REPO, attn_implementation="eager")  # meta until the first trace
        read = {}
        with model.trace(PROMPT):
            for layer in model.layers:
                read[layer.path] = (layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save())
        assert len(read) == len(model.layers)
        for attn, mlp, out in read.values():
            assert attn.shape == mlp.shape == out.shape

    def test_chunked_attention_across_a_chunk_boundary(self, model):
        """A prompt longer than ``attention_chunk_size``: a chunked block attends within its chunk, the NoPE block across."""
        chunk = model.config.attention_chunk_size
        prompt = " ".join(["word"] * (chunk + 20))
        assert len(model.tokenizer(prompt).input_ids) > chunk
        with model.trace(prompt):
            chunked = model.layers[0].self_attn.attention_probabilities.save()
        with model.trace(prompt):
            full = model.layers[3].self_attn.attention_probabilities.save()
        after = chunked[:, :, chunk:]  # queries in the second chunk
        assert torch.equal(after[..., :chunk], torch.zeros_like(after[..., :chunk]))
        assert (full[:, :, chunk:, :chunk] > 0).all()
        eps = torch.finfo(chunked.dtype).eps
        for probs in (chunked, full):
            assert torch.equal(probs.tril(), probs)
            torch.testing.assert_close(probs.sum(-1).float(), torch.ones(probs.shape[:-1], device=probs.device), atol=8 * eps, rtol=0)


def test_a_wrapper_module_binds_through_language_model():
    """A ``Llama4ForConditionalGeneration`` passed in already loaded keeps the text model at ``language_model``."""
    from transformers import Llama4ForConditionalGeneration

    repo = TestLlama4Text.REPO
    wrapper = Llama4ForConditionalGeneration.from_pretrained(repo, attn_implementation="eager")
    model = StandardizedTransformer(wrapper, tokenizer=StandardizedTransformer(repo).tokenizer)
    assert model.family is llama4_text
    assert model.layers is model.get("language_model.model.layers")
    assert model.lm_head is model.get("language_model.lm_head")
    assert model.embed_tokens is model.get("language_model.model.embed_tokens")
    assert model.norm is model.get("language_model.model.norm")
    assert all(type(layer) is llama4_text.Layer and type(layer.mlp) in (llama4_text.Mlp, llama4_text.Moe) for layer in model.layers)
    # the mixture values are per block (dense blocks have none) and two are unavailable on Llama 4: TestLlama4Text checks them
    assert all(reason is None for name, reason in model.support().items() if name.removeprefix("mlp.") not in MOE)
    causal = StandardizedTransformer(repo, attn_implementation="eager")
    parts = []  # filled inside the block: a name bound there does not survive the trace
    with model.trace(PROMPT):
        for layer in model.layers:
            parts.append((layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save()))
        logits = model.logits.save()
    with causal.trace(PROMPT):
        expected = causal.logits.save()
    torch.testing.assert_close(logits, expected)
    for x, attn, mlp, out in parts:
        torch.testing.assert_close(x + attn + mlp, out)
