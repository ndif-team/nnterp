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
build. A single entry's build writes its page and not the index. Open `encyclopedia/site/index.html`.

## The page, top to bottom

Every family page has the same sections in the same order. An entry fills them; it does not
rearrange them.

1. **Hero.** `family <model_type> · <Architecture class>`, the title, a one-sentence subtitle, the
   quirk chips (each links to the notes), five overlapping circles in the family's five colours, and
   under them the **checkpoint selector**: a button naming the shown checkpoint that opens a list of
   the entry's `CHECKPOINTS`, text checkpoints first, then the vision-language ones, each marked with
   an eye, then any the build could not read, greyed out and not selectable, their hover saying
   "not available in the encyclopedia: <reason>". The list is a keyboard listbox (arrows, Home, End,
   Enter, Escape). Under it, the Hugging Face logo (`static/hf-logo.svg`, the official file with a
   `viewBox` added so it scales) links to the shown checkpoint's Hub page. The
   eyebrow's class is the shown checkpoint's (`LlavaForConditionalGeneration` on a Llava checkpoint).
   The page opens on `REFERENCE`; the choice is the URL hash, `#ckpt=<repo id>`, so
   a link opens the page on a checkpoint. Switching does not reload: every part that depends on the
   checkpoint is in the page and the script swaps it. On a vision-language checkpoint the chips add
   `Vision-language` and the vision encoder's and the wrapper's slugs; the subtitle stays the family's.
   Nothing else: no layer count, no vLLM or eager stamps.
   The selector is the page's list of checkpoints; the suite's pinned tiny checkpoints are not on the
   page.
2. **The block** (`01`). The model-level strip (`embed_tokens → layers → norm → lm_head → logits`),
   a slider over the blocks with one tick per block coloured by `config.layer_types` (by the block's
   shape on a family whose blocks come in several), and the block diagram: the residual stream as a
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
4. **The Notes** (`03`). The entry's notes on the left, the quirks with their one-line explanations on
   the right. On a vision-language checkpoint *Vision notes* follow, a fold closed at first (tagged `· CLIP ·
   Llava 1.5`): the vision encoder module's `NOTES` under "The <title> vision encoder", then the
   wrapper's `notes` under its title, beside the vision quirks; the family's quirk list stays the
   family's.
5. **From the family module** (`04`). The family module's docstring, links to the module, its test
   file and the families table, and which envoy class wraps which module; on a vision-language
   checkpoint also the vision encoder's (`CLIPVisionModel → Vision`, ..., `LlavaModel → ImageScatter`). The
   family's `RENAME` lines for the vision encoder are not singled out.

The index lists every family in `nnterp.families.known()`: a card with the family's circles, title,
subtitle and quirks for each entry (blocks as a range when its checkpoints differ; an eye and the
vision encoders found, `CLIP · SigLIP`, and the `Vision-language` chip when any checkpoint has a vision encoder), a
stub for each family not written yet, a search box (title, `model_type`, architecture, Hub ids,
quirks, vision encoders, wrappers) and quirk filters: the text quirks and `Vision-language` on the first row,
the vision encoders on a second.

## Where each part comes from

- **From the code**, by `build.py`, with nothing to write: the architecture class, `support()` under
  an eager load and under the default one (the difference is what gets a `⚠`), every value's key,
  layout, dims and description, the printout, the root sizes and config keys, the native path behind
  each standard name, the family module's docstring, the envoy classes, the palette (generated from a
  hue), and all syntax highlighting.
- **Per checkpoint**, from a meta build of each: the sizes and config card, the slider's
  `num_layers` and `layer_types` (and the block's shapes), the `support()` rows and their `⚠`, the
  printout, the colophon's reference, and on a vision-language wrapper which vision encoder, its sizes, its
  `vision.*` rows and the wrapper's printout. **Shared**, the family's: the block schema, the
  vocabulary, the quirks, the notes and the family module section.
- **Derived from the config**: a checkpoint loads with `task="image-text-to-text"` when its
  `config.model_type` is a key of transformers' `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` (the
  mapping is keyed by `model_type`) and of the entry's `WRAPPERS`; any other loads for text
  generation, as before. The family must resolve to the entry's `MODEL_TYPE`. The vision encoder is found by
  `config.vision_config.model_type`, and the build asserts `type(model.vision._module).__name__` is
  one of the vision encoder's `MODULE_CLASSES`.
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
4. **`VLLM`**: whether the family has a module under `nnterp/families/vllm/` on the `0.8-refactor-vllm`
   branch (`git show origin/0.8-refactor-vllm:nnterp/families/vllm/<model_type>.py`); this branch has no
   `StandardizedVLLM`, so the flag cannot be run here.
5. **`BLOCK`**: what the diagram draws.
   - `topology`: `"sequential"` (each sublayer reads the stream after the one before it) or
     `"parallel"` (one read feeds every sublayer and the block sums them).
   - `sublayers`: in forward order, each a dict with
     - `host`: the standard name the sublayer has on the block (`self_attn`, `linear_attn`, `mlp`);
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
       routed experts', `shared_expert_output` in the shared expert's (list it only where the
       mixture has one: the build checks). The panels show the scoring, `top_k of num_experts` and
       the parts' classes, all read off the family's first `Moe`. On a `"mixer"` sublayer the
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
       `head_dim` need not be the values' width). The box shows about 30 characters
       beside the host's name when the sublayer has interior chips (the rest is ellipsised; the hover
       card has the whole line), so keep it short: `"12 heads × 64, fused c_attn"`;
     - `variants`: `{layer_type: detail}` keyed on the values of `config.layer_types`, shown instead
       of `detail` as the slider moves, for blocks that differ only in a setting; the same length
       limit applies.
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
   a host the listed sublayers do not draw, when two sublayers match the same host, or when a
   sublayer matches no block of the checkpoint and is not the other half of such an `"mlp"` /
   `"moe"` pair. When the reference and the pinned checkpoint have different block kinds (Nemotron 3 Nano is dense where its tiny checkpoint is a mixture), list both layouts (`"mlp"` and `"moe"` for the same host); each build draws the ones its checkpoint has.

   The build checks every host, contribution and interior value against the family and fails on a
   name the family does not have.
6. **`STRIP`**: sentences for the model-level strip, keyed `embed`, `layers`, `norm`, `head`,
   `logits`; each is shown when that node is hovered. Give one wherever the family does something
   there: a scaled embedding, tied weights, a softcap, a logit scale, a position embedding added after
   `embed_tokens`. The `logits` node's own label (`softcapped`, `scaled`, `the output`) follows the
   quirk slugs, not the note.
7. **`QUIRKS`**: slugs from `build.QUIRKS`, in the order a reader should meet them. If the family has
   a property none covers, add a slug there with a label and one sentence, worded so it holds for
   every family that will carry it (check the list first: another entry may have added it). A family
   with nothing that departs from the plain block has `QUIRKS = []`.

   The vision slugs are not an entry's: a page adds them on a vision-language checkpoint, from the
   vision encoder module's `QUIRKS` and the wrapper's `quirks`, after `vision` (`Vision-language`: some
   checkpoints carry a vision encoder; the index's filter). The others, each one line in `build.QUIRKS` taken
   from `docs/usage/vision.md`: `cls-token` (a class token beside the patches), `packed-tower` (every
   image's patches in one row), `variable-resolution` (`image_size` raises `Unavailable`),
   `padded-patches` (padded rows run through every block), `tiled-images` (crops or tiles as rows),
   `deepstack` (`layers[k].deepstack_output`), `unpadded-features` (`projector.output` is not
   `image_features`), `pooled-projector` (fewer image tokens than patches), `encoder-free` (no vision encoder
   blocks). The index keeps them off its first filter row.
8. **`PALETTE`**: `{"hue": degrees}` is the base hue the five colours are generated from. Related
   families sit together: the first entry of a lineage sets a hue and adds it to the table below, and
   its kin take one within about 15° of it (not equal) and add themselves to the row. A family with no kin leaves `PALETTE` out and
   gets a hash of its `model_type` (also listed below once an entry exists, so later kin can find it).
   `"paper"` tints the page and is rarely needed. `"colors"` (five hex fills in role order: attention,
   MLP, norms, stream, mark) and `"deeps"` bypass generation; a palette that fails the contrast or
   distinctness checks fails the build.

   | lineage | hue | set by |
   |---|---|---|
   | Gemma | 145 | gemma2 (gemma3_text 157, gemma4_text 133) |
   | Qwen | 285 | qwen2 (qwen3 297, qwen3_5_text 273; qwen2_moe hashes to 243, so set it near 285) |
   | Llama and its relatives (Mistral, SmolLM, ...) | hash of `llama` | llama |
   | OLMo (olmo3, olmo_hybrid, flex_olmo, olmoe) | hash of `olmo2` (58) | olmo2 (olmoe 70) |
   | GPT-2 | hash of `gpt2` | gpt2 |
   | GPT-NeoX / Pythia | hash of `gpt_neox` | gpt_neox |
   | EXAONE | hash of `exaone4` (46) | exaone4 |
   | GPT-J and CodeGen | hash of `gptj` (222) | gptj (codegen 234) |
   | Granite (granitemoe, granitemoeshared, granitemoehybrid, granite_swa, granitemoe_swa) | 205 | granite (granitemoe 215; `granite` hashes to 156, beside Gemma) |
   | Nemotron (`nemotron_h` is the hybrid line) | 262, the hash of `nemotron` | nemotron |
   | Granite (granitemoe, granitemoeshared, granitemoehybrid, granite_swa, granitemoe_swa) | 205 | granite (`granite` hashes to 156, beside Gemma) |
   | Nemotron (`nemotron_h` is the hybrid line) | 262, the hash of `nemotron` | nemotron (nemotron_h 250) |
   | Kimi (kimi_k2, kimi_linear) | 330 | kimi_k2 (kimi_linear 342) |
   | Mamba (mamba2, falcon_mamba, jamba's Mamba blocks) | hash of `mamba` (17) | mamba |

   `python -c "import sys; sys.path.insert(0, 'encyclopedia'); import palette; print(palette.hue_of('olmo2'))"`
   prints a hashed hue.
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
    - `tower` (optional): a vision encoder's fields inline (below), for a vision encoder only this family hosts.

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
