# The nnterp encyclopedia

One web page per family: the block drawn and annotated with nnterp's names, every standard value
and where it is read, the model's printout, the sizes, and notes on what to know before tracing
that family. The pages are static HTML, built by `build.py` into `site/` (not committed).

This file is the brief for adding a family. `entries/gemma2.py` is the reference entry; a new
family is one new file beside it.

## Build

```
pip install -e ".[encyclopedia]"
PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py            # every entry, plus the index
PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py gemma2     # one entry
```

Every checkpoint in an entry's `CHECKPOINTS` is built on the `meta` device (config, tokenizer and,
for a vision-language wrapper, processor; no weights). The `REFERENCE` checkpoint's config has to be
in the Hub cache or reachable; any other checkpoint whose config cannot be read (gated without
access, no such repository, offline and not cached) is listed on the page, greyed out, with the
reason, and the build goes on. Checkpoints whose configs differ only in their name share one
build. A single entry's build writes its page and not the index, and stops at the first error. The
full build goes on past an entry that fails to import or render: it prints `FAILED <model_type>: <error>`,
writes the index from the pages that built, and exits non-zero naming the failed entries. Open
`encyclopedia/site/index.html`.

What the Hub says about an entry beyond its configs (its org's name and avatar, each checkpoint's
parameter count and creation date) is kept in `hub_cache.json` and `static/orgs/`, both committed, so
a build offline and the tests show the same page. A build without `HF_HUB_OFFLINE` asks the Hub only
for what the cache does not hold and writes it back; commit the cache and any new avatar with the entry.

## The page, top to bottom

Every family page has the same sections in the same order. An entry fills them; it does not
rearrange them.

1. **Hero.** `family <model_type>`, the title, a one-sentence subtitle, the
   quirk chips (each links to the notes; the quirks that hold on the shown checkpoint), five overlapping circles in the family's five colours, and
   under them the **checkpoint selector**: a button naming the shown checkpoint that opens a list of
   the entry's `CHECKPOINTS`, text checkpoints first, then the vision-language ones, each marked with
   an eye, then any the build could not read, greyed out and not selectable, their hover saying
   "not available in the encyclopedia: <reason>". The list is a keyboard listbox (arrows, Home, End,
   Enter, Escape). Under it, the shown checkpoint's parameter count (`8.03B parameters`, the Hub's
   safetensors total; where the Hub has none, the meta model's count, marked `· from the config`) and
   `released Jul 2024` (the repository's creation month on the Hub), each blank for a repository the cache does not hold or the Hub does not let the build
   read; then the Hugging Face logo (`static/hf-logo.svg`, the official file with a
   `viewBox` added so it scales), linking to the shown checkpoint's Hub page. Once the hero has scrolled
   off, a slim bar at the top holds the family's title and a second selector over the same choice:
   picking in either moves both and the hash.
   The page opens on `REFERENCE`; the choice is the URL hash, `#ckpt=<repo id>`, so
   a link opens the page on a checkpoint. Switching does not reload: every part that depends on the
   checkpoint is in the page and the script swaps it. On a vision-language checkpoint the chips add
   `Vision-language` and the vision encoder's and the wrapper's slugs; the subtitle stays the family's.
   Nothing else: no layer count, no vLLM or eager stamps.
   The selector is the page's list of checkpoints; the suite's pinned tiny checkpoints are not on the
   page.
2. **The block** (`01`). The model-level strip (`embed_tokens → layers → norm → lm_head → logits`;
   a checkpoint with no final norm, OPT-350m, has no `norm` node and its `STRIP["norm"]` is not shown),
   a slider over the blocks with one tick per block coloured by `config.layer_types` (GPT-Neo's
   `config.attention_layers` where there is none; by the block's
   shape on a family whose blocks come in several; by the pair of shape and layer type where the
   types also differ within a shape, as on MiMo-V2-Flash, Laguna and Gemma 4; clicking a tick, or Enter or Space on it, selects
   that block as the slider does), and the block diagram: the residual stream as a
   vertical line, each sublayer as a row (pre-norm, the module with its interior values as chips,
   post-norm) whose contribution returns to an `⊕` on the stream. A mixture of experts draws a panel
   each for the router, the routed experts and the shared expert inside its box. On a family with
   several block shapes the diagram and the identity under it redraw for the slider's block. Boxes
   are outlines in their role's colour, clear inside. Hovering any part fills it lightly and shows, in
   the card beside the diagram, the nnterp expression that reads it, its layout and where it is read;
   clicking pins it. Under the diagram, the contribution identity as highlighted code. The slider's
   range, its ticks and the diagram's sizes are the selected checkpoint's.

   On a vision-language checkpoint the strip's `embed` note adds that the projected image features
   replace the image tokens' embeddings before block 0, and a sub-section **The vision encoder** follows: the
   image's path (`image → patch_embed → layers × N → norm`, where the vision encoder has one, `→ projector →
   image_features → layers[0].input`), each node's hover naming the nnterp value
   (`vision.patch_embeddings`, `vision.layers[i].layer_output`, `vision.tower_output`,
   `projector.input`/`.output`, `vision.image_features`, `vision.image_token_mask`). The `projector` node's
   caption is the wrapper's `projector_input`; where the vision encoder has `vision.norm` and the projector
   reads something else (a block before it), the norm hangs off the path under `layers`, dimmed, its hover
   saying the projector does not read it. Then four facts of the
   vision encoder (what a row is, positions, masking, the final norm); and the vision encoder's block, drawn by the same
   code as the text block from the vision encoder's `BLOCK`, with its own card and its identity. It is a fold,
   closed at first, whose summary is a small title, *The vision encoder* (the display face a little
   larger than the body), tagged `· CLIP · CLIPVisionModel`; absent on a text checkpoint.
3. **The API** (`02`). In this order:
   - *Values, by host*: one ledger per host (`model`, `model.layers[i]`, `model.layers[i].self_attn`,
     ...) with every standard value's name, layout and dims, description and where it is read. A value
     the checkpoint has only under some load carries an amber `⚠` before its name; hovering it shows
     the condition in a popup.
   - *Printout*: `print(model)`, highlighted, without the native container (`model`, `transformer`,
     ...): the standard names on the root, the blocks collapsed to one (`(0-25): 26 x ...`), and the
     root's values. On a wrapper the vision encoder follows under `vision`, its native `encoder` left out the
     same way.
   - *Config*: the root sizes and, under a rule, the config keys they come
     from (`architectures` is the checkpoint's own, a wrapper's on a vision-language checkpoint; the
     rest are the text config's). On a vision-language checkpoint a second card, *Vision Config*, whose body folds (closed at
     first, only its band shows, labelled; open, the body is titled as the Config card is): `model.vision.num_layers`, `hidden_size`, `num_heads`,
     `head_dim`, `intermediate_size`, `patch_size`, `image_size` (`varies` where nnterp raises
     `Unavailable`), then `spatial_merge_size` and `window_size` where the vision encoder's envoy defines
     them (the Qwen ViT's `QwenVision`).

   On a vision-language checkpoint the ledgers go on with the vision encoder's hosts (`model.vision`,
   `model.vision.layers[i]`, `.self_attn`, `.mlp`), with the same `⚠` convention: the vision encoder's
   attention interior needs eager. Everything in this section is the selected checkpoint's.
4. **The Notes** (`03`). The entry's notes on the left, the quirks that hold on the shown checkpoint with
   their one-line explanations on the right. On a vision-language checkpoint *Vision notes* follow, a fold closed at first (tagged `· CLIP ·
   Llava 1.5`): the vision encoder module's `NOTES` under "The <title> vision encoder", then the
   wrapper's `notes` under its title, beside the vision quirks; the family's quirk list stays the
   family's.
5. **From the family module** (`04`). The family module's docstring, links to the module, its test
   file and the families table, and which envoy class wraps which module; on a vision-language
   checkpoint also the vision encoder's (`CLIPVisionModel → Vision`, ..., `LlavaModel → ImageScatter`). The
   family's `RENAME` lines for the vision encoder are not singled out.

The index lists every family in `nnterp.families.known()`: a card with the family's circles, title,
subtitle and quirks for each entry (the quirks that hold on any of its checkpoints; blocks as a range when its checkpoints differ; an eye and the
vision encoders found, `CLIP · SigLIP`, and the `Vision-language` chip when any checkpoint has a vision encoder), a
stub for each family not written yet, a search box (title, `model_type`, org, architecture, Hub ids,
quirks, vision encoders, wrappers) and three filter rows, each a fold closed at first whose label opens it and
counts the filters on in it: **Org** (each org's Hub avatar and
name; a card shows when it is any org chosen), **Quirks** (the text quirks and `Vision-language`) and
**Vision encoders** (the encoders found); a card shows when it has every quirk and encoder chosen, a quirk
counting when it holds on any of the family's checkpoints. A
card carries its org's logo, the name on hover. The org is the author of the entry's `REFERENCE`, or the entry's `ORG`.

## Where each part comes from

- **From the code**, by `build.py`, with nothing to write: the architecture class, `support()` under
  an eager load and under the default one (the difference is what gets a `⚠`), every value's key,
  layout, dims and description, the printout, the root sizes and config keys, the native path behind
  each standard name, the family module's docstring, the envoy classes, the palette (generated from a
  hue), and all syntax highlighting.
- **Per checkpoint**, from a meta build of each: the sizes and config card, the slider's
  `num_layers` and `layer_types` (and the block's shapes), the `support()` rows and their `⚠`, the
  printout, the colophon's reference, the quirks that hold on it, and on a vision-language wrapper which vision
  encoder, its sizes, its `vision.*` rows and the wrapper's printout. **Shared**, the family's: the block schema,
  the vocabulary, the notes and the family module section.
- **Derived from the config**: a checkpoint loads with `task="image-text-to-text"` when its
  `config.model_type` is a key of transformers' `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` (the
  mapping is keyed by `model_type`) and of the entry's `WRAPPERS`; any other loads for text
  generation, as before. The family must resolve to the entry's `MODEL_TYPE`. The vision encoder is found by
  `config.vision_config.model_type`, and the build asserts `type(model.vision._module).__name__` is
  one of the vision encoder's `MODULE_CLASSES`.
- **From the Hub**, by `hub.py`, cached in `hub_cache.json`: the org's display name and avatar, and
  each checkpoint's parameter count and creation date.
- **From the entry**, `entries/<model_type>.py`: the title and subtitle, the checkpoints, the block
  schema the diagram draws, the strip's notes, the quirk tags, the hue, the notes, and `WRAPPERS`.
- **From a vision encoder module**, `vision/<tower>.py`: what holds on every host of a vision encoder.

## Writing an entry

### Read first

The entry states facts about a family, and each one has a source. Before writing:

- `nnterp/families/<model_type>.py`: the names, the envoys, where each value is read, the family's
  docstring (the page prints it, so the notes add to it, they do not repeat it).
- `tests/families/test_<model_type>.py`: the pinned tiny checkpoint and what the suite asserts on this
  family (identities, shapes, read orders).
- `docs/reference/families.md` (the family's row and quirks) and the usage pages it points to.
- The transformers modeling file for the architecture: the block's forward, in order.
- `entries/gemma2.py`, for the shape of a finished entry.

### The fields

1. **`MODEL_TYPE`**: the family's `model_type`, which is the entry's file name.
2. **`TITLE`**, **`SUBTITLE`**: the family's public name (several model lines separated by ` / `,
   `"Qwen2 / Qwen2.5"`, are shown stacked, one per line), and one sentence saying how this family
   differs from a plain Llama block: what a person must know first. Present tense, no adjectives.
3. **`REFERENCE`**: the public checkpoint the page opens on. Choose the one most used for
   interpretability (usually the smallest base model).
   **`CHECKPOINTS`**: the family's public checkpoints as Hub ids, the reference among them; each is a
   choice in the page's selector, linked, and searchable on the index. A vision-language wrapper
   whose text model is this family goes here too, with its `model_type` in `WRAPPERS`. **`PINNED`**: the tiny checkpoint in
   `tests/families/test_<model_type>.py` (its `REPO`; where the suite rewrites the tiny checkpoint's
   config into a local copy, `PINNED` is the Hub id and the test builds the page from the copy).
   **`load`** (optional): `def load(checkpoint, **kwargs)` returning the `StandardizedTransformer`
   the page reads, for a checkpoint a repo id alone does not build on the meta device (a tokenizer
   that is not cached, a config `AutoConfig` maps only through remote code). It receives the
   reference, the pinned checkpoint and `attn_implementation`; say in its docstring why it exists.
   **`GREYED`** (optional): `{"<repo id>": "<reason>"}`, checkpoints of `CHECKPOINTS` the page lists greyed
   out with that reason, without reading or building them: a checkpoint whose config builds but whose
   weights do not load (Doge's repositories in their remote code's layout). A checkpoint whose config
   cannot be read needs no entry here; the build greys it by itself.
   **`ORG`** (optional): the Hub org the index files the family under, where the reference's author is
   not it (an ungated copy: `cohere.py`'s reference is unsloth's copy of a CohereLabs model).
4. **`VLLM`**: whether the family has a module under `nnterp/families/vllm/` on the `0.8-refactor-vllm`
   branch (`git show origin/0.8-refactor-vllm:nnterp/families/vllm/<model_type>.py`); this branch has no
   `StandardizedVLLM`, so the flag cannot be run here.
5. **`BLOCK`**: what the diagram draws.
   - `topology`: `"sequential"` (each sublayer reads the stream after the one before it) or
     `"parallel"` (one read feeds every sublayer and the block sums them).
   - `sublayers`: in forward order, each a dict with
     - `host`: the standard name the sublayer has on the block (`self_attn`, `linear_attn`, `mlp`), or,
       where the block has no module for it, a native dotted path on the block (OPT's and XGLM's MLP
       path is `final_layer_norm`, `fc1`, `activation_fn` and `fc2` on the block itself, so its host is
       `"fc2"`). A sublayer on a native path is an `"attention"` or an `"mlp"` with no `interior`; its
       `contribution` is a module output named from the block (`"fc2.output"`), which the identity uses
       as written; `reads` names the module whose input is the pre-norm's output (`"fc1"`), and
       `host_note` is shown on its box's and its contribution's hovers (the shape it runs on). The
       ledgers stay as nnterp serves them: a native path has no host there;
     - `kind`: `"attention"`, `"mixer"` (a recurrent mixer, `linear_attn`: DeltaNet, KDA, a
       selective scan), `"mlp"` or `"moe"` (a mixture of experts, `mlp` where it is a `Moe`);
       `label`: the box's title (`"Attention"`, `"Linear attention"`, `"MLP"`, `"MoE"`);
     - `contribution`: the standard value this sublayer adds to the stream (`attention_output`,
       `mlp_output`);
     - `pre_norm`, `post_norm`: the native names of the norms before and after it on the block, each
       only if the block has one. In a parallel block whose one norm feeds both sublayers (GPT-J,
       CodeGen), give both sublayers the same `pre_norm`: the diagram draws it in both rows under one
       hover node; add a `pre_norm_note` saying both read the same tensor; `pre_norm_note` / `post_norm_note`: a sentence shown when that norm
       is hovered, for a trap in its name or place;
     - `interior`: the standard values read inside the module, drawn as chips, in forward order
       (three to a row; a box with more than six grows a row at a time). On a `"moe"` sublayer the
       chips are the mixture's values and each is drawn in its part's panel: `router_logits`,
       `expert_weights`, `expert_indices` in the router's, `expert_outputs`, `routed_output` in the
       routed experts', `shared_expert_output` in the shared expert's (list it where some
       checkpoint's mixture has one; a checkpoint whose mixture has none, ERNIE-4.5's 300B-A47B, draws
       no shared chip or panel). The panels stack router, routed experts, shared expert; where the
       mixture runs them in another order, `part_order` gives it (`["shared", "router", "experts"]` on
       Qwen2-MoE, HunYuan, ERNIE-4.5, Laguna; `["router", "shared", "experts"]` on AFMoE). The panels show the scoring, `top_k of num_experts` and
       the parts' classes, all read off the family's first `Moe`; the scoring is read per block, and
       where it differs (DeepSeek-V4's hash-routed first blocks) the router panel shows the slider's block's. On a `"mixer"` sublayer the
       chips are the mixer's values (`attention_queries`/`keys`/`values` as the family maps them,
       `decays`, `betas`, `state_input`, `attention_head_outputs`, `state_output`, `states`), and
       its hover card names the two kernels the values are read at and which values need
       `route_kernels`, from the family's mixer class and `support()`;
     - `detail`: one line under the label (head counts, widths, activation), a format string over the
       sizes (`{num_heads}`, `{hidden_size}`, ...) and the top-level config keys in
       `build.CONFIG_KEYS` (append a key there if the family needs one; only top-level keys resolve,
       so a value that lives only under `rope_parameters` goes in the notes), so the
       numbers come from the reference config and are not typed; on a mixture `{num_experts}` and
       `{top_k}` are the `Moe`'s own. Where a config key and a root size share a name the config's
       value wins, so name the key you mean (`{v_head_dim}` on latent attention, whose config
       `head_dim` need not be the values' width). A key the checkpoint leaves unset or `null`
       (`sliding_window` on StarCoder2's tiny checkpoint) shows as `—`; a name that is no size and
       no key in `CONFIG_KEYS` fails the build. The box shows about 30 characters
       beside the host's name when the sublayer has interior chips (the rest is ellipsised; the hover
       card has the whole line), so keep it short: `"12 heads × 64, fused c_attn"`;
     - `variants`: `{layer_type: detail}` keyed on the values of `config.layer_types` (or of
       `config.attention_layers`, `"global"` / `"local"` on GPT-Neo), shown instead
       of `detail` as the slider moves, for blocks that differ only in a setting; the same length
       limit applies.
     - `stream_norm` (and `stream_norm_note`): optional, the native name of a norm on the stream after
       this sublayer's add, in a post-LN block (OPT-350m: `self_attn_layer_norm` after the attention's
       add, `final_layer_norm` after the MLP path's, whose output is `layer_output`). It is drawn on the
       stream line under the `⊕`; the next sublayer reads its output. Such a block's identity is not a
       sum, so give `identity`;
     - `parallel_with_next`: optional, `True` on a sublayer of a `"sequential"` block that reads the
       same stream point as the sublayer after it, the two joining the stream at one add (Falcon-H1's
       Mamba-2 mixer and attention, then the MLP). The pair is drawn side by side with one `⊕`, the
       stream has no point between them, and the default identity sums the pair in parentheses;
       each keeps its own hover texts. A shared pre-norm is listed on both, as in a parallel block;
     - `block`: optional, the native class name of the blocks this sublayer is drawn on
       (`"OlmoHybridAttentionDecoderLayer"`); a sublayer without it is drawn on every class. Give it
       when one host is drawn differently on two block classes: OLMo-Hybrid's `mlp` is pre-normed on
       its DeltaNet blocks and post-normed on its attention blocks, and `post_attention_layernorm` is
       the MLP's pre-norm on one class and the attention's post-norm on the other, so `mlp` is listed
       once per class, each with its own `block` and norms. A sublayer with a `block` key has its own
       norm hover nodes, so the same norm name can sit in two roles.
   - `identity` (and `identity_note`): the contribution identity, when it is not the plain sum
     `layers[i].input + <contributions> == layer_output` (Gemma-4, DeepSeek-V4; Granite's scaled
     contributions still sum exactly). `identity_note` may be given on its own, for a sum that is
     exact but worth a word. A family with several block shapes takes no `identity`: each
     shape's is the plain sum of what it draws.

   **Blocks that differ.** `sublayers` lists every sublayer any block has, once, in forward order,
   and each block draws the ones that match its own children: a sublayer is drawn where its `host`
   exists, a `"moe"` only where that host runs a mixture (a `Moe` whose `no_mixture()` gives no
   reason) and an `"mlp"` only where it does not. A hybrid
   lists both mixers (`linear_attn` as `"mixer"`, `self_attn` as `"attention"`); a family with
   dense first blocks lists `mlp` twice, as `"mlp"` and as `"moe"`, and so does a family whose
   checkpoints differ (Gemma-4: dense on E2B, a mixture on 26B-A4B), each checkpoint drawing the
   one its blocks have. Nothing is keyed on a config
   key: the matching reads the built model, so `layer_types`, `mlp_layer_types`,
   `first_k_dense_replace` and their kin all come out the same way. The distinct combinations are
   the block's shapes; the slider's ticks are coloured by shape, and the diagram, the identity and
   the variant label redraw when the slider crosses into another. The build fails when a block has
   a host the listed sublayers do not draw, when two sublayers match the same host on one block,
   when a `block` key names no block class of the checkpoint, or when a
   sublayer matches no block of the checkpoint while its host is on some block, and it is not the
   other half of such an `"mlp"` / `"moe"` pair. A sublayer whose host is on no block of a
   checkpoint is left off that checkpoint's drawing and ledgers (Granite 4.0's attention-only
   checkpoints have no `linear_attn`). When the reference and the pinned checkpoint have different block kinds (Nemotron 3 Nano is dense where its tiny checkpoint is a mixture), list both layouts (`"mlp"` and `"moe"` for the same host); each build draws the ones its checkpoint has.

   **Layouts that differ by checkpoint.** Where a family's checkpoints build different blocks from
   one class (Falcon-7B's one norm, Falcon-40B's `ln_attn` and `ln_mlp`, Falcon-RW's sequential block;
   OPT's post-norm `opt-350m`; StableLM-2-12B's parallel block under `use_parallel_residual`), `BLOCK` is a list of `(predicate, block)` pairs, each `block` a dict as
   above and each predicate a function of the checkpoint's text config; a checkpoint draws the first
   whose predicate holds, so end the list with `(lambda config: True, ...)` for the default. A callable
   of the config returning the dict does the same. Each checkpoint is drawn, checked and given its
   identity from its own block; the notes colour the norms of every block listed.

   The build checks every host, contribution and interior value against the family and fails on a
   name the family does not have, and every `pre_norm` and `post_norm` against the children of the
   blocks that draw it (native names and aliases).
6. **`STRIP`**: sentences for the model-level strip, keyed `embed`, `layers`, `norm`, `head`,
   `logits`; each is shown when that node is hovered. Give one wherever the family does something
   there: a scaled embedding, tied weights, a softcap, a logit scale, a position embedding added after
   `embed_tokens`. The `logits` node's own label (`softcapped`, `scaled`, `the output`) follows the
   quirk slugs, not the note.
7. **`QUIRKS`**: slugs from `build.QUIRKS`, in the order a reader should meet them. If the family has
   a property none covers, add a slug there with a label and one sentence, worded so it holds for
   every family that will carry it (check the list first: another entry may have added it). A family
   with nothing that departs from the plain block has `QUIRKS = []`. A slug holds on every checkpoint.
   Where only some checkpoints have the property, the item is a dict instead:
   `{"slug": "parallel-blocks", "when": lambda config: config.use_parallel_residual}`, a predicate on the
   checkpoint's text config (as `BLOCK`'s), so it holds on the pinned tiny checkpoint as on the public
   ones. Each checkpoint's chips, quirk list and `logits` label show the quirks that hold on it; the
   index card shows those that hold on any checkpoint on the page, as plain chips. StableLM (`qkv-bias`
   on 1.6B, `parallel-blocks` and `qk-norm` on 12B), OPT (`post-ln` on 350m), Llama 4 (`interleaved-moe` on Maverick) and VaultGemma
   (`softcapped-logits` on a config that sets `final_logit_softcapping`) carry such items. The notes still
   say which checkpoints have the property.

   The vision slugs are not an entry's: a page adds them on a vision-language checkpoint, from the
   vision encoder module's `QUIRKS` and the wrapper's `quirks`, after `vision` (`Vision-language`: some
   checkpoints carry a vision encoder; the index's filter). The others, each one line in `build.QUIRKS` taken
   from `docs/usage/vision.md`: `cls-token` (a class token beside the patches), `packed-tower` (every
   image's patches in one row), `variable-resolution` (`image_size` raises `Unavailable`),
   `padded-patches` (padded rows run through every block), `tiled-images` (crops or tiles as rows),
   `deepstack` (`layers[k].deepstack_output`), `unpadded-features` (`projector.output` is not
   `image_features`), `pooled-projector` (fewer image tokens than patches), `encoder-free` (no vision encoder
   blocks). The index keeps them off its first filter row.
8. **`PALETTE`**: `{"hue": degrees}` is the base hue the five colours are generated from. The table
   below is the layout: every entry's hue, computed by `hues.py` from one ordered list of lineages
   (each a band, its members a step apart, about 3°) and the families with no kin (each in a gap
   between two bands, a step and a half from both). Run `hues.py` to re-space after adding a family:
   put the family in `hues.LAYOUT`, in its lineage's band or in a gap, and run
   `python encyclopedia/hues.py --write`, which rewrites every entry's `PALETTE` hue and this table.
   Do not pick a hue by hand. `"paper"` tints the page and is rarely needed. `"colors"` (five hex
   fills in role order: attention, MLP, norms, stream, mark) and `"deeps"` bypass generation; a
   palette that fails the contrast or distinctness checks fails the build.

<!-- hues.py: table -->
| lineage | hues |
|---|---|
| DeepSeek (and the Kimi and Youtu lines on its latent attention) | `deepseek_v2` 322, `deepseek_v3` 325, `deepseek_v32` 328, `deepseek_v4` 331, `youtu` 334, `kimi_k2` 337, `kimi_linear` 340 |
| MiniMax / MiMo | `minimax_m2` 346, `mimo_v2_flash` 349 |
| Mamba (and Jamba's Mamba blocks) | `mamba` 359, `mamba2` 2, `falcon_mamba` 5, `jamba` 8 |
| Falcon | `falcon_h1` 14, `falcon` 17 |
| HunYuan | `hunyuan_v1_dense` 26, `hunyuan_v1_moe` 29 |
| GPT-2 | `gpt2` 35, `gpt_bigcode` 38, `starcoder2` 41 |
| GPT-J | `gpt_neo` 50, `gptj` 54, `codegen` 57 |
| GPT-NeoX | `gpt_neox` 63, `gpt_neox_japanese` 66, `stablelm` 69 |
| OPT | `opt` 78, `xglm` 81 |
| OLMo | `olmo` 95, `olmo2` 98, `olmo3` 101, `olmoe` 104, `flex_olmo` 107, `olmo_hybrid` 110 |
| GLM | `glm` 116, `glm4` 119, `glm4_moe` 122, `glm4_moe_lite` 125, `glm_moe_dsa` 128 |
| Phi | `phi` 137, `phi3` 140, `phimoe` 144 |
| Gemma | `gemma` 153, `gemma2` 156, `gemma3_text` 159, `gemma4_text` 162, `gemma4_unified_text` 165, `vaultgemma` 168 |
| Cohere | `cohere` 174, `cohere2` 177 |
| ERNIE | `ernie4_5` 186, `ernie4_5_moe` 189 |
| Granite (and IBM's Bamba) | `bamba` 198, `granitemoehybrid` 201, `granite` 205, `granite_swa` 208, `granitemoe` 211, `granitemoe_swa` 214, `granitemoeshared` 217 |
| Nemotron | `nemotron` 223, `nemotron_h` 226 |
| Llama and its kin | `arcee` 235, `apertus` 238, `llama` 241, `llama4_text` 244, `smollm3` 247, `helium` 250, `bitnet` 253, `seed_oss` 256, `mistral` 259, `mixtral` 263, `ministral` 266, `ministral3` 269 |
| Qwen | `qwen2` 282, `qwen2_moe` 285, `qwen2_vl_text` 288, `qwen2_5_vl_text` 291, `qwen3` 295, `qwen3_moe` 298, `qwen3_vl_text` 301, `qwen3_vl_moe_text` 304, `qwen3_next` 307, `qwen3_5_text` 310, `qwen3_5_moe_text` 313 |
| no kin (in the gaps) | `zaya` 354, `doge` 21, `bloom` 46, `persimmon` 73, `mpt` 86, `dbrx` 90, `jetmoe` 133, `gpt_oss` 148, `exaone4` 182, `laguna` 194, `afmoe` 230, `hyperclovax` 273, `solar_open` 278, `dots1` 317 |
<!-- hues.py: end -->

9. **`NOTES`**: markdown, the part only a person can write. See below.
10. **`WRAPPERS`** (optional): the family's vision-language wrappers, keyed by the wrapper's
    `config.model_type` (`"llava"`, `"idefics3"`, ...), each a dict of what a config does not say:
    - `title`: the wrapper's public name (`"Llava 1.5"`);
    - `pinned`: the tiny wrapper checkpoint the family's `VisionSuite` subclass runs on (its `REPO`);
      the tests build the vision encoder from it, and every `VisionSuite` `REPO` in the family's test file
      must be some wrapper's `pinned`;
    - `projector`: one line, what feeds the projector and what it does;
    - `projector_input`: a short phrase naming what `model.projector.input` is, shown as the `projector` node's
      caption and in its hover (`"vision.layers[-2].layer_output[:, 1:]"`, `"the last block's layer_output,
      before vision.norm"`, `"vision.tower_output"`, `"the merger's input: the last block's output"`), checked
      on the pinned tiny wrapper like the notes. Exactly `` `vision.tower_output` `` puts the vision encoder's
      norm on the image's path; anything else, on a vision encoder with `vision.norm`, draws the norm off it;
    - `quirks` (optional): the wrapper's slugs (`tiled-images`, `unpadded-features`, ...);
    - `notes`: markdown, the wrapper's own facts: what the projector reads, what `tower_output`
      edits reach, image tokens per image, the scatter; checked on the pinned tiny wrapper, real
      numbers from a run or from `docs/patterns/image-pathway.md`;
    - `tower` (optional): a vision encoder's fields inline (below), for a vision encoder only this family hosts;
    - `per_checkpoint` (optional): `{"<repo id>": {...}}`, fields merged over the record for that checkpoint alone,
      for one wrapper class whose checkpoints differ in what a config does not say (`mistral.py`'s `llava`:
      Pixtral-12B reads `vision.tower_output`, BakLLaVA block -2). It may override `title`, `projector`,
      `projector_input`, `quirks` and `notes`; each key is one of the entry's `CHECKPOINTS` or the record's
      `pinned`, and the record's own fields stay the default for its other checkpoints.

### Vision encoders

A vision encoder's facts that hold on every host live in `vision/<tower>.py` (`clip.py`, `siglip.py`), shared by
every family that hosts it; nothing wrapper- or checkpoint-specific goes there. A module declares:

- `TITLE` (`"CLIP"`); `VISION_CONFIG_TYPES`, the `vision_config.model_type` values it covers (SigLIP's
  covers `siglip_vision_model` and the `idefics3`, `idefics3_vision`, `smolvlm_vision` Idefics 3 and
  SmolVLM report); `MODULE_CLASSES`, the class names `model.vision` may have;
- `BLOCK`, the vision encoder block in an entry's `BLOCK` format (`self_attn` and `mlp`, pre-norms
  `input_layernorm`, `post_attention_layernorm`), its `detail` formatted with the vision encoder's sizes and
  its `vision_config` keys;
- `ROWS` (what a row of `Patches` and its patch axis are), `MASKING`, `POSITIONS`, `NORM` (whether the
  final norm is `vision.norm`; the page draws a norm node where the built vision encoder has one): a sentence
  or two each, code names in backticks;
- `QUIRKS`, slugs from `build.QUIRKS`; `NOTES`, markdown, true on every host. Where one module covers
  several `vision_config.model_type` values whose encoders differ in a detail (the Qwen ViT's lines: their
  norms, MLP and windows), `NOTES` may carry a per-variant table, one row per line; it is the one place a
  Markdown table is allowed, so keep it to four short columns.

A vision encoder hosted by two or more families gets a module; a vision encoder one family hosts stays inline, as
`WRAPPERS[<wrapper>]["tower"]`, a dict with the same field names (`VISION_CONFIG_TYPES` may be left
out). `vision.resolve` turns either form into one dict, so the build reads both the same way.

### The notes

What someone needs to know before running an experiment on this family, that the tables above do
not already say. Cover, where the family has something to say:

- **The block, in order**: the forward as two or three lines of pseudo-code in a fenced block, and
  any trap in the native names.
- **The contributions**: what `attention_output` and `mlp_output` are on this family (the module's
  output, a post-norm's, a pre-residual tensor, a scaled copy), with the identity as a snippet, and
  what that means for ablation, steering and attribution.
- **Loading**: what needs `attn_implementation="eager"` and why; whether the default load runs a
  different computation than the eager one; kernels to route on a recurrent mixer.
- **Attention**: grouped-query heads and what an edit to a key/value head reaches, the query scale,
  masks and windows, softcaps, sinks, anything read at an unusual place.
- **The readout**: what `logits` is relative to `lm_head.output`, what `project_on_vocab` applies,
  the final norm's gain, tied weights.
- **The embeddings**: scaling, position embeddings, whether `token_embeddings` equals
  `layers[0].input`.
- **The family's own machinery**: the router and experts of a mixture, a recurrent state, parallel
  streams, borrowed keys and values, read-order traps.
- **Resources**: sparse autoencoders or other published interpretability artefacts trained on the
  family's checkpoints, and which nnterp value their hook point is.

Rules:

- Present tense, factual, this family's own facts. No history, no comparisons of quality, no advice
  that holds for every family (that is in `docs/`).
- Every claim is checked against the family module, its test file, the transformers source or a run.
  Shapes, identities and read orders can be checked on the pinned tiny checkpoint; anything about
  real values (norm scales, sinks, what a scaling does, SAE reconstruction) is checked on the
  reference checkpoint or another real one whose weights are cached, because a tiny random
  checkpoint can contradict a true claim (its activations sit below a norm's `eps`, its random norms
  coincide). A number is one the reference config or a run gives; say which checkpoint it is for
  when sizes differ (`48.0 on 2B`).
- A behaviour that holds on every family is not a family fact: generic nnterp advice is in `docs/`,
  and an nnsight bug is reported, not written into the notes as a trap (a read that fails only on the
  first trace of a fresh model is one).
- `##` headings, one per topic, each a statement where possible ("The contributions are the
  post-norms' outputs"). Paragraphs of two to five sentences.
- Snippets are short and use the standard names; they are highlighted at build time and each nnterp
  name takes its role's colour, so write `model.layers[i].self_attn.attention_output`, not an alias
  of your own. They are illustrative and are not executed, so they must be right as written: run
  each once against the reference checkpoint, and against the pinned one where it can (a pinned
  checkpoint has two or a few blocks, so index `layers[1]`, not `layers[8]`). Inside one trace,
  reads come in forward order: `mlp.output` before `mlp_output` on a post-norm family,
  `lm_head.output` before `logits`; one edit per trace when the text contrasts two edits.
- Inline code for every name, path and config key. Lists rather than Markdown tables: a table
  overflows the page at phone width.

### Verify

```
PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py <model_type>
PYTHONPATH=. HF_HUB_OFFLINE=1 pytest tests/test_encyclopedia.py
```

The test builds every entry's page from its pinned checkpoint and checks the schema against the
family, builds each wrapper's vision encoder from its pinned tiny wrapper, and checks that a checkpoint whose
config cannot be read is listed and skipped. Then look at the page: a headless Chromium renders it to
an image
(`chrome --headless=new --no-sandbox --hide-scrollbars --window-size=1360,9000 --screenshot=page.png file://.../site/<model_type>.html`;
a 500px-wide, 14000px-tall window for the phone layout; a page with long notes needs the height).
Add `#ckpt=<repo id>` to the URL to render it on another checkpoint, and render each wrapper's. Check that

- the diagram reads in the forward's order, every norm the block has is drawn, and each contribution
  edge leaves from the right place (after the post-norm when there is one);
- hovering the sublayers, norms, chips, edges and `⊕` shows the right expression (the node texts are
  in the page's `nodes-json`);
- the slider's variants match `config.layer_types`;
- the values that carry a `⚠` are the ones that need a non-default load, and no value is missing
  from the ledgers;
- the notes' code blocks are highlighted and nothing overflows the page at either width;
- on a vision-language checkpoint, the vision encoder's block reads in forward order, the image's path has a
  norm node only where the vision encoder has `vision.norm`, and the vision encoder's ledgers, sizes and Vision notes
  are there.

### Scope of one entry

An entry is `entries/<model_type>.py`, plus, when needed, a new slug in `build.QUIRKS` or a config
key appended to `build.CONFIG_KEYS`. It does not change the templates, the stylesheet, `block.js`
or the page's sections. The diagram draws sublayers in sequence or in parallel, with optional norms
around each; attention, a recurrent mixer, an MLP or a mixture of experts; and blocks whose
sublayers differ from one index to the next (see *Blocks that differ*). When a family's block does
not fit that (several parallel streams, a sublayer with no module, a block whose sublayers change
order),
do not approximate it: write the rest of the entry, say what the schema lacks, and extend the
generator as its own change, so every family with that shape gains it.

Done means: the page builds from both the reference and the pinned checkpoint, the tests pass, the
page has been looked at in a browser at both widths, and every sentence in the entry has a source.

The page's sentences are the same on every checkpoint: the only checkpoint-specific text a
template composes is the repo id (the printout's caption); counts, sizes and classes appear as
data (the strip's `× N`, the Config cards, the ledgers), never inside a sentence.

## Design

The Sakura Chroma system (`static/encyclopedia.css`): paper and ink, Big Shoulders for display,
Albert Sans for body, JetBrains Mono for data; no rounded corners, no soft shadows. Section titles
are one or two words with one word in the family's colour (`The block`, `The API`, `Notes`).

Each family brings five muted colours, generated in OKLCH from its hue (`palette.py`), one per role,
and the same role takes the same colour everywhere on its page:

| role | where it shows |
|---|---|
| attention | the Attention box's outline and chips, the `attention_output` edge, the `self_attn` (or `linear_attn`) ledger, attention names in code and in the printout |
| MLP | the MLP box, the `mlp_output` edge, the `mlp` ledger, its names |
| norms | the norm boxes, `norm` on the strip, norm names |
| stream | the residual line and `⊕`, `layers[i].input`, `layer_output`, the root's values and names, strings in code |
| mark | the quirk chips, the coloured word in titles, topstrips, hover fills on the strip, keywords in code |

Every colour has a fill tier, which takes ink text, and a deep tier for lines and coloured text on
the paper; the page sets `--c1` … `--c5`, `--c1-deep` … `--c5-deep` and `--paper` on `<html>`, and
the stylesheet names the roles from them. The hero's circles are the five fills. Code is highlighted
at build time (Pygments for the snippets, a line lexer for the printout), with no JavaScript. The
warning mark is amber on every family: it is a status, not a role.
