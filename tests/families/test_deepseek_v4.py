"""DeepSeek-V4, end to end: a hyper-connection residual, compressed attention with a sink, all-MoE."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.components import Residual, StreamMixing, Streams, StreamWeights
from nnterp.families import deepseek_v4

#: Long enough that the compressed-sparse blocks (compress rate 4) carry compressed entries.
LONG = "The quick brown fox jumps over the lazy dog while the cat sleeps in the warm sun"

WEIGHTS = ("attention_post", "attention_comb", "mlp_post", "mlp_comb")


def read_blocks(model, prompt=PROMPT):
    """Per block: its input, the four stream weights, both contributions and its output."""
    parts = []  # bound outside: a name bound inside the block does not survive it
    with model.trace(prompt):
        for layer in model.layers:
            x = layer.input.save()
            post_a, comb_a = layer.attention_post.save(), layer.attention_comb.save()
            attn = layer.self_attn.attention_output.save()
            post_f, comb_f = layer.mlp_post.save(), layer.mlp_comb.save()
            mlp = layer.mlp.mlp_output.save()
            parts.append((x, post_a, comb_a, attn, post_f, comb_f, mlp, layer.layer_output.save()))
    return parts


def mixed(comb, streams):
    """What the block does with ``comb``: stream ``k`` becomes ``sum_j comb[j, k] * streams[j]``."""
    return torch.matmul(comb.transpose(-1, -2), streams)


def written(post, output):
    """A sublayer's output written into every stream, weighted by ``post``."""
    return post.unsqueeze(-1) * output.unsqueeze(-2)


class TestDeepseekV4(FamilySuite):
    REPO = "yujiepan/deepseek-v4-bf16-tiny-random"
    FAMILY = deepseek_v4
    NATIVE = LLAMA_ROWS
    LOAD_KWARGS = {"dtype": torch.float32}  # the norms are kept in float32; a bf16 forward fails on CPU inside transformers
    ATTENTION_SINK = True
    MLP_WIDTH_KEY = "moe_intermediate_size"

    def expected_values(self, model):
        return super().expected_values(model) | set(WEIGHTS)

    def pattern_from_scores(self, model, scores):
        """The sink joins the softmax as one extra key column and is dropped afterwards (GPT-OSS's arithmetic)."""
        sinks = model.layers[0].self_attn._module.sinks.to(scores.dtype)
        batch, heads, q, _ = scores.shape
        combined = torch.cat([scores, sinks.view(1, heads, 1, 1).expand(batch, heads, q, 1)], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        return combined.softmax(-1)[..., :-1]

    def test_contribution_identity(self, model):
        """The block's own formula, exact on every block:
        ``mlp_combᵀ(attention_combᵀ·input + attention_post⊗attention_output) + mlp_post⊗mlp_output == layer_output``."""
        for i, (x, post_a, comb_a, attn, post_f, comb_f, mlp, out) in enumerate(read_blocks(model)):
            middle = written(post_a, attn) + mixed(comb_a, x)
            total = written(post_f, mlp) + mixed(comb_f, middle)
            eps = torch.finfo(out.dtype).eps
            torch.testing.assert_close(total, out, rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")

    def test_mean_identity_is_additive(self, model):
        """``comb`` is doubly stochastic, so the stream mean is additive, up to the Sinkhorn projection's residual."""
        for i, (x, post_a, _, attn, post_f, _, mlp, out) in enumerate(read_blocks(model)):
            total = x.mean(2) + post_a.mean(-1, keepdim=True) * attn + post_f.mean(-1, keepdim=True) * mlp
            torch.testing.assert_close(total, out.mean(2), rtol=1e-5, atol=1e-5, msg=f"layer {i}")

    def test_plain_identity_does_not_hold(self, model):
        x, _, _, attn, _, _, mlp, out = read_blocks(model)[0]
        assert not torch.allclose(x + attn.unsqueeze(2) + mlp.unsqueeze(2), out, atol=1e-3)

    def test_layer_output_is_the_streams(self, model):
        with model.trace(PROMPT):
            embedding = model.token_embeddings.save()
            first = model.layers[0].input.save()
            attn = model.layers[0].self_attn.attention_output.save()
            out = model.layers[0].layer_output.save()
        streams = model.config.hc_mult
        assert out.shape == (1, embedding.shape[1], streams, model.hidden_size) and isinstance(out, Streams)
        assert isinstance(attn, Residual) and not isinstance(out, Residual)
        assert torch.equal(first, embedding.unsqueeze(2).expand_as(first))  # block 0 reads the embedding in every stream
        assert deepseek_v4.Layer.layer_output.layout is Streams

    def test_stream_weights_are_the_hyper_connections(self, model):
        layer = model.layers[1]
        with model.trace(PROMPT):
            attn_hc = layer.attn_hc.output.save()
            post_a, comb_a = layer.attention_post.save(), layer.attention_comb.save()
            ffn_hc = layer.ffn_hc.output.save()
            post_f, comb_f = layer.mlp_post.save(), layer.mlp_comb.save()
        assert torch.equal(post_a, attn_hc[0]) and torch.equal(comb_a, attn_hc[1])
        assert torch.equal(post_f, ffn_hc[0]) and torch.equal(comb_f, ffn_hc[1])
        assert isinstance(post_a, StreamWeights) and isinstance(comb_a, StreamMixing)
        assert ((post_a > 0) & (post_a < 2)).all()
        for comb in (comb_a, comb_f):
            ones = torch.ones(comb.shape[:-1], device=comb.device)
            torch.testing.assert_close(comb.sum(-2), ones, atol=1e-5, rtol=0)  # unit column sums (the last Sinkhorn step)
            torch.testing.assert_close(comb.sum(-1), ones, atol=1e-3, rtol=0)  # unit row sums, to the projection's residual

    def test_stream_weight_writes_land(self, model):
        """A zero ``attention_post`` removes the attention's write: the mid-block streams are the input, mixed."""
        layer = model.layers[0]
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            x = layer.input.save()
            comb = layer.attention_comb.save()
            layer.attention_post = layer.attention_post * 0
            middle = layer.ffn_hc.input.save()
            assigned = model.logits.save()
        with model.trace(PROMPT):
            layer.mlp_comb[:] = torch.eye(model.config.hc_mult, device=clean.device)
            inplace = model.logits.save()
        torch.testing.assert_close(middle, mixed(comb, x))
        assert not torch.equal(clean, assigned) and not torch.equal(clean, inplace)

    def test_project_on_vocab_goes_through_hc_head(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            resid = model.layers[-1].layer_output.save()
            head = model.model.hc_head.output.save()
            logits = model.logits.save()
        assert torch.equal(model.project_on_vocab(resid), logits)
        torch.testing.assert_close(model.project_on_vocab(resid[0, -1]), logits[0, -1])  # one position, [streams, hidden]
        torch.testing.assert_close(model.project_on_vocab(head), logits)  # plain [batch, seq, hidden]: norm and lm_head only
        torch.testing.assert_close(model.project_on_vocab(attn), model.lm_head(model.norm(attn)))
        per_stream = model.lm_head(model.norm(resid[:, :, 0]))
        assert per_stream.shape == logits.shape and not torch.equal(per_stream, logits)

    def test_keys_and_values_are_one_tensor(self, model):
        attn = model.layers[0].self_attn
        with model.trace(PROMPT):
            keys = attn.attention_keys.save()
        with model.trace(PROMPT):
            values = attn.attention_values.save()
        assert torch.equal(keys, values)
        with model.trace(PROMPT):
            attn.attention_keys[:, :, -1] = 0
            edited = attn.attention_values.save()
        assert torch.equal(edited[:, :, -1], torch.zeros_like(edited[:, :, -1]))
        with model.trace(PROMPT):
            attn.attention_keys = attn.attention_keys * 0
            separate = attn.attention_values.save()
        torch.testing.assert_close(separate, values)  # an assignment gives the keys a tensor of their own

    def test_compressed_blocks_have_more_keys_than_tokens(self, model):
        """On a compressed-sparse block the compressor's entries follow the token keys: one per ``compress_rate`` tokens."""
        config = model.config
        seq = len(model.tokenizer(LONG).input_ids)
        rates = config.compress_rates
        for i, kind in enumerate(config.layer_types):
            attn = model.layers[i].self_attn
            with model.trace(LONG):
                keys = attn.attention_keys.save()
            with model.trace(LONG):
                pattern = attn.attention_probabilities.save()
            extra = 0 if kind == "sliding_attention" else seq // rates[kind]
            assert keys.shape[2] == pattern.shape[-1] == seq + extra, (i, kind, tuple(keys.shape))
            assert pattern.shape[-2] == seq
        assert any(seq // rates[kind] for kind in config.layer_types if kind != "sliding_attention")

    def test_head_outputs_are_rotated_back(self, model):
        """``attention_head_outputs`` is what the grouped output projection reads, not the interface's output."""
        attn = model.layers[0].self_attn
        with model.trace(PROMPT):
            interface = attn.source.attention_interface_1.output[0].save()
            heads = attn.attention_head_outputs.save()
            grouped = attn.o_a_proj.input.save()
        assert not torch.allclose(heads, interface)
        batch, seq = heads.shape[:2]
        assert torch.equal(grouped, heads.reshape(batch, seq, model.config.o_groups, -1))
