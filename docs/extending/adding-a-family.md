---
title: Adding a Family
one_liner: Write one module named after `config.model_type` with `RENAME`, three envoy subclasses and `ENVOYS`, a `def <size>(model)` for any root size the config spells its own way, plus one test file subclassing `FamilySuite`.
tags: [extending, families, rename, envoys, tests]
related: [docs/extending/overriding-values.md, docs/extending/custom-values.md, docs/extending/finding-source-ops.md, docs/extending/registering.md]
sources: [nnterp/families/__init__.py, nnterp/families/llama.py, nnterp/families/gpt2.py, nnterp/families/falcon.py, nnterp/families/cohere.py, nnterp/components/__init__.py, nnterp/components/layer.py, nnterp/standardized.py, tests/families/suite.py, tests/families/test_llama.py, tests/families/test_gpt2.py]
---

# Adding a Family

## What this is for

A family is one module under `nnterp/families/` that tells `StandardizedTransformer`
how a checkpoint's native module names map onto the standard vocabulary and which
envoy classes wrap its blocks. The module's file name is the registry: `lookup`
imports `nnterp.families.<model_type>` the first time a checkpoint of that type is
loaded, so adding a family is writing one module and one test file, and nothing
in `nnterp/` core changes. A family that lives outside the package goes through
`register()` instead ([registering.md](registering.md)); this page's template is the
same either way.

## Canonical pattern

`llama.py` is the family the vocabulary is taken from, so it changes nothing but
the container names:

```python
"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from."""

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Llama's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Llama's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
```

Mistral, Qwen2/3, Gemma 1, OLMo 1, Phi-3, SmolLM3 and StableLM are this file with
the class names swapped. `gpt2.py` shows every other part of the template in use:
block-level aliases, a config flag that takes the attention off the shared
interface, and container keys anchored at the root:

```python
"""GPT-2 (``GPT2LMHeadModel``)."""

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GPT-2's attention; the shared eager forward and the residual added in the block, so the base holds.

    A checkpoint with ``reorder_and_upcast_attn`` takes GPT-2's own upcast
    path instead, where nothing on the interface is reachable.
    """

    def off_interface(self):
        if self._module.config.reorder_and_upcast_attn:
            return "this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"
        return super().off_interface()


class Mlp(Mlp):
    """GPT-2's MLP; the residual is added in the block, so the base holds."""


ENVOYS = {GPT2Block: Layer, GPT2Attention: Attention, GPT2MLP: Mlp}
```

## The recipe

1. Read `config.model_type` off the checkpoint (`AutoConfig.from_pretrained(repo).model_type`;
   a multimodal config nests the text model's under `text_config`, and that is the type
   used). Create `nnterp/families/<model_type>.py`. The file name is what `lookup` imports,
   so `gemma3_text.py` covers `gemma3_text` and nothing else; the module declares no
   type of its own.
2. Write `RENAME`. Print the raw model (`TransformersModel(repo)`) or its `named_modules()`
   to see the native tree, then map the containers and any block names that differ.
3. Subclass `Layer`, `Attention`, `Mlp` (and `LinearAttention` for a hybrid), overriding
   only what the family's forward spells differently
   ([overriding-values.md](overriding-values.md)).
4. Key them in `ENVOYS` on the transformers module classes.
5. Define `def <size>(model)` for any root size the config spells its own way
   (`intermediate_size` on a config with `n_inner` or `ffn_dim`); leave the rest to the
   root ([Sizes](#sizes) below). Define `def project_on_vocab(model, hidden)` if the model
   scales the head's output ([The logits](#the-logits)).
6. Add `tests/families/test_<model_type>.py` and run it.

### `RENAME`

Keys are native paths, values the standard names. nnsight resolves every key
relative to every envoy in the tree (nnsight docs/usage/rename-modules.md):

- A key with several components (`"transformer.h"`, `"model.layers"`,
  `"norm_attn_norm.attn"`) binds where it resolves from: the root. That is what lifts
  the containers out of the inner model so the alias is `model.layers`, not
  `model.transformer.h`. DBRX's `"norm_attn_norm.attn": "self_attn"` resolves from each
  block, so the block reads like any other.
- A single-component key (`"attn"`, `"ln_1"`, `"ffn"`) binds on every envoy that has a
  child of that name: every block gets `self_attn`.
- A key that resolves nowhere is skipped, not an error. GPT-NeoX's `RENAME` carries both
  `"embed_out": "lm_head"` and the containers; on current transformers the head is
  already `lm_head`, the `embed_out` key binds nothing, and the family covers both
  spellings.
- An alias that would shadow an existing name (a sibling module, an `Envoy` attribute
  such as `output`, an `nn.Module` attribute such as `config`) raises at construction.
  OPT keeps its block-level `final_layer_norm` native for a related reason: a
  single-component alias for it would also bind on the decoder's final norm.
- A name that depends on the block goes in `RENAME` keyed on the child's *class*
  (nnsight binds a class key on every envoy with exactly one direct child of that
  class): Nemotron-H's one `mixer` per block is `linear_attn`, `self_attn` or `mlp` by
  its class, `{NemotronHMamba2Mixer: "linear_attn", NemotronHAttention: "self_attn", ...}`.
  A class that two children of one envoy share is an error there; key those by name.

The standard names are `embed_tokens`, `layers`, `norm`, `lm_head`, and on blocks
`self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`, `linear_attn`. Bind
the norms only where the family has a module in that position; their meaning varies
and the suite checks the sublayer inputs against the family's own norm
(`ATTENTION_NORM`, `MLP_NORM` below), not the alias. Anything else (`wpe`, `drop`,
BLOOM's `word_embeddings_layernorm`) stays under its native name.

### The three envoy subclasses

Every family defines all three classes even when nothing changes: `test_envoy_classes`
asserts the block envoys are exactly the family's classes. `support()` does not read them:
it walks each block's children that are `Standard` envoys, under their standard names, so
what it lists is what the tree has. OPT has no MLP module and still defines `Mlp`,
unkeyed, for the class convention; no block has one, so `support()` lists no `mlp.*` key.
A docstring says what holds and why; the phrase for the common case is "so the base
holds":

```python
class Layer(Layer):
    """<Family>'s decoder block; returns a bare tensor, so the base holds."""
```

- **`Layer`**: set `returns_tuple = True` when the block returns `(hidden_states, ...)`
  rather than the tensor (GPT-J, BLOOM, MPT, Falcon). `layer_output` unwraps either way;
  the flag is what `skip_with` needs to hand back the right shape.
- **`Attention`**: the base holds when the attention runs transformers' shared
  `attention_interface` call and the block adds the residual. Override `off_interface()`
  when a config flag routes around the interface (GPT-2). Redefine the values when the
  attention does its own arithmetic (GPT-J, BLOOM, MPT, Falcon) or the residual is added
  inside the module (BLOOM).
- **`Mlp`**: the base holds when the module returns the contribution, as a tensor or as
  the first element of a tuple (GPT-OSS's `(hidden_states, router_scores)`). Redefine
  `mlp_output` when the residual is added inside (BLOOM, MPT) or when a post-norm's
  output is what reaches the stream (Gemma-2/3, OLMo-2/3).
- **`Moe`** (a mixture of experts): key a subclass of `nnterp.components.Moe` on the
  MoE module class. Where every MLP is a mixture, that subclass is the family's `Mlp`
  (`class Mlp(Moe)`, Mixtral); where dense blocks come first or alternate, it is a
  `Moe` beside the `Mlp`, inheriting the family's `mlp_output` (`class Moe(Moe, Mlp)`,
  DeepSeek-V3), and `test_envoy_classes` accepts either class on a block. Alias the
  router `router` (`"gate": "router"` in `RENAME`) and the shared expert
  `shared_experts` (`"shared_expert"` / `"shared_mlp"`). The base holds when the router
  computes its logits with `F.linear` (`LOGITS`), the experts module takes
  `(hidden, top_k_index, top_k_weights)` under transformers' `@use_experts_implementation`,
  and the shared expert's output is what the mixture adds. Otherwise redefine the value
  at the right place, a `TokenEProperty` with the base's layout: the router's
  projection module (`"router.wg.output"`, Hunyuan), an op of the mixture's own forward
  (`"source.hidden_states_2.output"`, Laguna, with `sourced = True` since the read
  follows a child's), `unavailable(...)` where no tensor holds the value (DBRX's
  `expert_outputs`), and set `SCORING`. A mixture with no module of its own is hosted on
  `layers[i].mlp`: the family's `Layer.__init__` hands the block's `router` and
  `experts` envoys down (`self.mlp.router = self.router`, Gemma-4) and the `Mlp`
  overrides `no_mixture()` for the checkpoints without one. A family never looks its
  parent up through the interleaver's envoys
  ([../usage/mixture-of-experts.md](../usage/mixture-of-experts.md)).
- **`LinearAttention`** (hybrids only): the base holds for transformers' pure-torch gated
  delta rule; Qwen3-Next, Qwen3.5 and OLMo-Hybrid subclass it with a docstring and nothing
  else. A delta-rule mixer with its own kernels and the same values subclasses it and names
  them: Kimi-Linear's sets `CHUNK_KERNEL`, `RECURRENT_KERNEL` and `STATE_OP` to transformers'
  KDA kernels and redeclares `decays` with its per-channel layout. It is a `RecurrentMixer`: a mixer with other kernels (a state-space layer) is a new
  `RecurrentMixer` subclass that sets `CHUNK_KERNEL`, `RECURRENT_KERNEL` and `STATE_OP` and
  declares its own values
  ([../developing/recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md)).
  Key it in `ENVOYS` like the others, so `route_kernels(model.family, ...)` finds it.

### `ENVOYS`

A dict from transformers module class to envoy class. Import the modeling module at
the top of the family module and nowhere else in nnterp: `import nnterp` loads no
transformers modeling code because families are imported on first use. Several
module types may share one envoy class (Llama's `LlamaMLP` and a shared expert of the
same class); a mixture is keyed to its own (`DeepseekV2MLP: Mlp, DeepseekV2Moe: Moe`). `envoys=`
matches by type or by native path suffix, never by alias, and nnsight tries type keys
before path keys.

### Sizes

The root's sizes (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`,
`head_dim`, `qk_head_dim`, `intermediate_size`) are each a `StandardizedProperty` on
`StandardizedTransformer` that reads the text config (a multimodal checkpoint's `text_config`,
else the config itself) by the plain Llama-style rule:
`num_kv_heads` is `config.num_key_value_heads` or `num_heads`, `head_dim` is
`config.head_dim` or `hidden_size // num_heads`, `qk_head_dim` is `head_dim`,
`intermediate_size` is `config.intermediate_size`. Where the family's config spells one
of them differently, define a module-level function of the same name taking the model;
the descriptor calls it instead of the rule, and every size the family does not define
keeps the root's. Falcon's:

```python
# nnterp/families/falcon.py

def num_kv_heads(model: "StandardizedTransformer") -> int:
    """``num_kv_heads`` on the 40B layout (``new_decoder_architecture``); 1 under ``multi_query``; else every head."""
    config = model.config
    if config.new_decoder_architecture:
        return config.num_kv_heads
    return 1 if config.multi_query else model.num_heads


def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``ffn_hidden_size``."""
    return model.config.ffn_hidden_size
```

The function may read other sizes off the model (`model.num_heads`, `model.hidden_size`),
which resolve the same way. Twenty-five shipped families define one or more, for example
Falcon (`num_kv_heads`, `intermediate_size`), DeepSeek-V2 (`head_dim` = `v_head_dim`,
`qk_head_dim` = `qk_nope_head_dim + qk_rope_head_dim`; DeepSeek-V3, -V3.2, GLM-5,
GLM-4.7-Flash and Youtu import both from `deepseek_v2`), GPT-2 (`intermediate_size` =
`n_inner`, `4 * hidden_size` when `None`) and Mamba-2 (`num_heads`, the SSD heads); the
full list with what each reads is in
[families.md](../reference/families.md#logits-scales-and-sizes). The suite's
`test_sizes_match_the_model` checks each size against the weights, so a wrong spelling
fails there; `MLP_WIDTH_KEY` is for a family whose `intermediate_size` is right but
unused by the model (an all-MoE family's experts).

Each block's own sizes (`self_attn.num_heads`, `.num_kv_heads`, `.head_dim`,
`.qk_head_dim`, `mlp.intermediate_size`) are read off the module by the base `Attention`
and `Mlp`. A module that keeps a size under another name overrides the property on the
family's subclass (JetMoE's `Mlp` returns its module's `hidden_size`);
`test_per_module_sizes_match_each_block` fails when one is wrong.

### The logits

`project_on_vocab(hidden)` is `lm_head(norm(hidden))`, then what the model does after the
head: the root's applies the text config's `final_logit_softcapping` when it is set. A family
whose model does something else to the head's output defines
`def project_on_vocab(model, hidden)` in its module, which is bound in the root's place the
way a size function is (`project_on_vocab` is a `StandardizedCapability`,
`StandardizedProperty` for a method), so the logit lens on the last block still equals
`logits`:

```python
# nnterp/families/cohere.py

def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``logit_scale``."""
    return model.lm_head(model.norm(hidden)) * model.config.logit_scale
```

Granite's divides by `logits_scaling`; Cohere-2 imports Cohere's. The suite's
`test_logits_are_the_models_output` and `test_project_on_vocab_is_the_logit_lens` both check
`project_on_vocab(layers[-1].layer_output) == logits`, so a missing or wrong one fails there.

## A complete template

```python
"""<Family> (``<Class>ForCausalLM``).

<One paragraph: the native tree, where the residual is added, whether the
attention runs the shared interface, what returns a tuple, what has no standard name.>
"""

from transformers.models.<module>.modeling_<module> import <X>Attention, <X>DecoderLayer, <X>MLP

from ..components import Attention, Layer, Mlp  # add LinearAttention for a hybrid

RENAME = {
    "<container>.<embedding>": "embed_tokens",
    "<container>.<blocks>": "layers",
    "<container>.<final norm>": "norm",
    # block-level names that differ from Llama's, single-component:
    # "<attention>": "self_attn", "<feed-forward>": "mlp",
    # "<pre-attention norm>": "input_layernorm", "<pre-mlp norm>": "post_attention_layernorm",
    # "<old head name>": "lm_head",   # binds only where it resolves
}


class Layer(Layer):
    """<Family>'s decoder block; returns a bare tensor, so the base holds."""

    # returns_tuple = True   # when the block returns (hidden_states, ...)


class Attention(Attention):
    """<Family>'s attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """<Family>'s MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {<X>DecoderLayer: Layer, <X>Attention: Attention, <X>MLP: Mlp}


# -- sizes: only where the config spells one its own way -------------------------
# def intermediate_size(model: "StandardizedTransformer") -> int:
#     """The MLP width is ``<key>``."""
#     return model.config.<key>
```

## The test file

One file per family under `tests/families/`, subclassing `FamilySuite`
(`tests/families/suite.py`). Llama's is the minimum:

```python
"""Llama, end to end: the family the vocabulary is taken from."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import llama


class TestLlama(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    FAMILY = llama
    NATIVE = LLAMA_ROWS
```

GPT-2's states one quirk and adds one family-specific test:

```python
class TestGPT2(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-gpt2"
    FAMILY = gpt2
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v are split views of one c_attn tensor

    def test_reorder_and_upcast_makes_the_interface_unavailable(self, model):
        config = model.layers[0].self_attn._module.config
        config.reorder_and_upcast_attn = True
        try:
            support = model.support()
            assert all("reorder_and_upcast_attn" in support[f"self_attn.{name}"][0] for name in ("attention_probabilities", "attention_queries"))
        finally:
            config.reorder_and_upcast_attn = False
```

`rows(container, layers, embed, norm, attn="self_attn", mlp="mlp", ln1="input_layernorm", ln2="post_attention_layernorm")`
builds the standard-path to native-path dict; pass `None` for a module the family
does not have (`ln2=None` on a parallel block, `mlp=None` on OPT). `LLAMA_ROWS` is
`rows("model", "layers", "embed_tokens", "norm")`.

Every class attribute of `FamilySuite`:

| attribute | meaning |
| --- | --- |
| `REPO` | The pinned tiny checkpoint, offline-cached. |
| `FAMILY` | The family module the checkpoint must resolve to (`model.family is FAMILY`). |
| `NATIVE` | Standard path to native path; each pair must be the same envoy, and every `layers.0.*` name must exist on every block. |
| `EXPECTED_UNAVAILABLE` | `support()` key to a substring of the reason, for values this checkpoint lacks; every other value must report `None`. Default `{}`. |
| `REFUSES_IN_PLACE_QKV` | torch refuses in-place edits on q/k/v that are views out of a `split`/`chunk` (GPT-2); the suite expects a `RuntimeError` and skips the in-place query edit. |
| `ATTENTION_SINK` | The pattern's rows sum to less than one (GPT-OSS). |
| `KV_HEADS_EXPANDED` | Keys and values are read already expanded to `num_heads` (latent attention; Falcon's 40B layout). |
| `MLP_WIDTH_KEY` | A config key naming the first block's MLP width when it is not `intermediate_size` (an all-MoE family's experts). |
| `LOAD_KWARGS` | Extra load arguments the checkpoint needs (`{"dtype": torch.float32}` on DBRX's degenerate tiny checkpoint). |
| `QUERY_GATED` | `q_proj` produces the query and a gate side by side, twice the width. |
| `ATTENTION_NORM` | The block's own module whose output enters the attention; `None` when the block input enters directly (OLMo-2/3). Default `"input_layernorm"`. |
| `MLP_NORM` | The block's own module whose output enters the MLP, whatever the family calls it. Default `"post_attention_layernorm"`; `"input_layernorm"` on a parallel block. |
| `MLP_NORM_BEFORE_ATTENTION` | The MLP's norm runs before the attention does (Falcon's 40B layout takes both norms from the block input up front). |
| `MOE_UNAVAILABLE` | Mixture value to a substring of the reason, for the mixture values this checkpoint lacks on its mixture; a mixture without a shared expert needs no entry for `shared_expert_output`. Default `{}`. |
| `ROUTER_EXTRA_CLASSES` | Router classes beyond the experts (ZAYA's skip class is one more column of `router_logits`). Default `0`. |

`pattern_from_scores(self, model, scores)` is the one method a subclass may override:
what the softmax makes of `attention_scores`; a sink family adds its column.

The suite is every end-to-end statement a family must satisfy: aliases reach the native
modules and nothing is bound at `model.model`; the envoys are the family's classes;
`support()` lists every standard value and matches what reads and raises; `layer_output`
is the block's tensor; the contribution identity `input + attention_output + mlp_output
== layer_output` holds on every block; `self_attn.input` and `mlp.input` equal the
family's own norms' outputs; the standardized model equals a raw `TransformersModel`;
boundary writes move the logits; the pattern has the right shape, dtype, row sums and
causal mask; a written pattern moves the logits; every value inside a forward resolves on every
attention block; the interior's shapes, causal writes and in-place edits; `skip_layers`,
`steer`, `project_on_vocab`; every value's tensor matches its layout annotation (the
name the base carries, imported from `..components`: `Residual`, `Pattern`, ...); the
input accessors; the root values; the sizes; the repr.

Run it:

```
HF_HUB_OFFLINE=1 pytest tests/families/test_<model_type>.py
```

A failure in `test_every_source_value_resolves_on_every_layer` names the operation the
forward does not have; [finding-source-ops.md](finding-source-ops.md) is how to find the
one it does.

## Verified outside the package

The template above, written as a standalone module for `gpt2` and passed to
`nnterp.families.register(module, "gpt2")` at the top of a test file that subclasses `FamilySuite`
with GPT-2's `NATIVE` and `REFUSES_IN_PLACE_QKV`, passes the whole suite (35 tests) on
`hf-internal-testing/tiny-random-gpt2`; `model.family` is the standalone module. Nothing
under `nnterp/families/` was touched. A family that will ship is the same module moved
into the package and the `register()` line removed.

## Gotchas

- **The file name is the registry.** The module covers exactly the type it is named
  after; `test_every_family_module_is_named_after_its_model_type` checks it.
- **Import the modeling module only inside the family module.** An import at
  `nnterp/__init__.py` or in `components/` would load transformers modeling code on
  `import nnterp`; `test_import_is_lazy` fails.
- **Import nnterp (or nnsight) before any `transformers.models...` import** in a script or
  test; the reverse order segfaults at import on this stack. The suite and `conftest.py`
  do this on their first line.
- **Define `Layer`, `Attention` and `Mlp` even when a module type has no such module.**
  The convention is one family, three classes; OPT keys no `Mlp` in `ENVOYS` and still
  defines one. `support()` walks the tree, so a module no block has is not listed.
- **Every family test loads with `attn_implementation="eager"`** so the interior values
  are available. Values that are still unavailable on that checkpoint go in
  `EXPECTED_UNAVAILABLE` with a substring of the reason.
- **Interior values bind at different points of the forward.** The suite reads one per
  trace; a family-specific test that reads two in one trace must read them in forward
  order (Falcon without alibi: `attention_values` before `attention_queries`; with alibi
  queries, keys, values).
- **`known()` and `all_families()` list the shipped modules only.** A registered family is
  reached through `lookup` and the model's `family` attribute, and appears in the
  `UnsupportedFamily` message's list.

## Related

- [overriding-values.md](overriding-values.md): what to write in the subclasses when the base does not hold.
- [custom-values.md](custom-values.md): adding a value the base classes do not have.
- [finding-source-ops.md](finding-source-ops.md): discovering the operation names a `source.` path needs.
- [registering.md](registering.md): a family outside the package, or overriding a shipped one.
- nnsight docs/usage/rename-modules.md: the alias rules `RENAME` relies on.
