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

Each entry's `REFERENCE` checkpoint is built on the `meta` device (config and tokenizer only, no
weights), so its config has to be in the Hub cache or reachable. Open `encyclopedia/site/index.html`.

## The page, top to bottom

Every family page has the same sections in the same order. An entry fills them; it does not
rearrange them.

1. **Hero.** `family <model_type> · <Architecture class>`, the title, a one-sentence subtitle, the
   quirk chips (each links to the notes), and five overlapping circles in the family's five colours.
   Nothing else: no layer count, no vLLM or eager stamps.
2. **Checkpoints.** A collapsed fold. One row per public checkpoint, the name and the Hub id both
   linking to the Hub; the reference checkpoint is tagged "this page's sizes", and the last row is
   the suite's pinned tiny checkpoint, tagged "the test suite's".
3. **The block** (`01`). The model-level strip (`embed_tokens → layers → norm → lm_head → logits`),
   a slider over the blocks with one tick per block coloured by `config.layer_types`, and the block
   diagram: the residual stream as a vertical line, each sublayer as a row (pre-norm, the module with
   its interior values as chips, post-norm) whose contribution returns to an `⊕` on the stream. Boxes
   are outlines in their role's colour, clear inside. Hovering any part fills it lightly and shows, in
   the card beside the diagram, the nnterp expression that reads it, its layout and where it is read;
   clicking pins it. Under the diagram, the contribution identity as highlighted code.
4. **The API** (`02`). In this order:
   - *Values, by host*: one ledger per host (`model`, `model.layers[i]`, `model.layers[i].self_attn`,
     ...) with every standard value's name, layout and dims, description and where it is read. A value
     the checkpoint has only under some load carries an amber `⚠` before its name; hovering it shows
     the condition in a popup.
   - *Printout*: `print(model)`, highlighted, without the native container (`model`, `transformer`,
     ...): the standard names on the root, the blocks collapsed to one (`(0-25): 26 x ...`), and the
     root's values.
   - *Sizes*: the root sizes and, under a rule, the config keys they come from.
5. **Notes** (`03`). The entry's notes on the left, the quirks with their one-line explanations on
   the right.
6. **From the family module** (`04`). The family module's docstring, links to the module, its test
   file and the families table, and which envoy class wraps which module.

The index lists every family in `nnterp.families.known()`: a card with the family's circles, title,
subtitle and quirks for each entry, a stub for each family not written yet, a search box (title,
`model_type`, architecture, Hub ids, quirks) and quirk filters.

## Where each part comes from

- **From the code**, by `build.py`, with nothing to write: the architecture class, `support()` under
  an eager load and under the default one (the difference is what gets a `⚠`), every value's key,
  layout, dims and description, the printout, the root sizes and config keys, the native path behind
  each standard name, the family module's docstring, the envoy classes, the palette (generated from a
  hue), and all syntax highlighting.
- **From the entry**, `entries/<model_type>.py`: the title and subtitle, the checkpoints, the block
  schema the diagram draws, the strip's notes, the quirk tags, the hue, and the notes.

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
3. **`REFERENCE`**: one public checkpoint whose config the page's sizes, config keys and `support()`
   are read from. Choose the one most used for interpretability (usually the smallest base model).
   **`CHECKPOINTS`**: the family's public checkpoints as Hub ids, the reference among them; they are
   linked from the page and searchable on the index. **`PINNED`**: the tiny checkpoint in
   `tests/families/test_<model_type>.py` (its `REPO`).
4. **`VLLM`**: whether the family also runs on `StandardizedVLLM`.
5. **`BLOCK`**: what the diagram draws.
   - `topology`: `"sequential"` (each sublayer reads the stream after the one before it) or
     `"parallel"` (one read feeds every sublayer and the block sums them).
   - `sublayers`: in forward order, each a dict with
     - `host`: the standard name the sublayer has on the block (`self_attn`, `linear_attn`, `mlp`);
     - `kind`: `"attention"` or `"mlp"`; `label`: the box's title (`"Attention"`, `"MLP"`, `"MoE"`);
     - `contribution`: the standard value this sublayer adds to the stream (`attention_output`,
       `mlp_output`);
     - `pre_norm`, `post_norm`: the native names of the norms before and after it on the block, each
       only if the block has one; `pre_norm_note` / `post_norm_note`: a sentence shown when that norm
       is hovered, for a trap in its name or place;
     - `interior`: the standard values read inside the module, drawn as chips, in forward order;
     - `detail`: one line under the label (head counts, widths, activation), a format string over the
       sizes (`{num_heads}`, `{hidden_size}`, ...) and the top-level config keys in
       `build.CONFIG_KEYS` (append a key there if the family needs one; a nested value such as
       `rope_parameters.partial_rotary_factor` cannot be formatted, so it goes in the notes), so the
       numbers come from the reference config and are not typed. The box shows about 30 characters
       beside the host's name when the sublayer has interior chips (the rest is ellipsised; the hover
       card has the whole line), so keep it short: `"12 heads × 64, fused c_attn"`;
     - `variants`: `{layer_type: detail}` keyed on the values of `config.layer_types`, shown instead
       of `detail` as the slider moves, for blocks that differ only in a setting; the same length
       limit applies.
   - `identity` (and `identity_note`): the contribution identity, when it is not the plain sum
     `layers[i].input + <contributions> == layer_output` (Gemma-4, DeepSeek-V4; Granite's scaled
     contributions still sum exactly). `identity_note` may be given on its own, for a sum that is
     exact but worth a word.

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
8. **`PALETTE`**: `{"hue": degrees}` is the base hue the five colours are generated from. Related
   families sit together: the first entry of a lineage sets a hue and adds it to the table below, and
   its kin take one within about 15° of it (not equal). A family with no kin leaves `PALETTE` out and
   gets a hash of its `model_type` (also listed below once an entry exists, so later kin can find it).
   `"paper"` tints the page and is rarely needed. `"colors"` (five hex fills in role order: attention,
   MLP, norms, stream, mark) and `"deeps"` bypass generation; a palette that fails the contrast or
   distinctness checks fails the build.

   | lineage | hue | set by |
   |---|---|---|
   | Gemma | 145 | gemma2 (gemma3_text 157) |
   | Qwen | 285 | qwen2 (qwen3 hashes to 288; qwen2_moe hashes to 243, so set it near 285) |
   | Llama and its relatives (Mistral, SmolLM, ...) | hash of `llama` | llama |
   | OLMo (olmo3, olmo_hybrid, flex_olmo) | hash of `olmo2` | olmo2 |
   | GPT-2 | hash of `gpt2` | gpt2 |
   | GPT-NeoX / Pythia | hash of `gpt_neox` | gpt_neox |
   | EXAONE | hash of `exaone4` (46) | exaone4 |
   | GPT-J (CodeGen) | hash of `gptj` (222) | gptj |
   | Granite (granitemoe, granitemoeshared, granitemoehybrid, granite_swa, granitemoe_swa) | 205 | granite (`granite` hashes to 156, beside Gemma) |

   `python -c "import sys; sys.path.insert(0, 'encyclopedia'); import palette; print(palette.hue_of('olmo2'))"`
   prints a hashed hue.
9. **`NOTES`**: markdown, the part only a person can write. See below.

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
HF_HUB_OFFLINE=1 pytest tests/test_encyclopedia.py
```

The test builds every entry's page from its pinned checkpoint and checks the schema against the
family. Then look at the page: a headless Chromium renders it to an image
(`chrome --headless=new --no-sandbox --hide-scrollbars --window-size=1360,5600 --screenshot=page.png file://.../site/<model_type>.html`;
a 500px-wide window for the phone layout). Check that

- the diagram reads in the forward's order, every norm the block has is drawn, and each contribution
  edge leaves from the right place (after the post-norm when there is one);
- hovering the sublayers, norms, chips, edges and `⊕` shows the right expression (the node texts are
  in the page's `nodes-json`);
- the slider's variants match `config.layer_types`;
- the values that carry a `⚠` are the ones that need a non-default load, and no value is missing
  from the ledgers;
- the notes' code blocks are highlighted and nothing overflows the page at either width.

### Scope of one entry

An entry is `entries/<model_type>.py`, plus, when needed, a new slug in `build.QUIRKS` or a config
key appended to `build.CONFIG_KEYS`. It does not change the templates, the stylesheet, `block.js`
or the page's sections. The diagram draws one block shape per family: sublayers in sequence or in
parallel, with optional norms around each. When a family's block does not fit that (blocks whose
sublayers differ from one index to the next, several parallel streams, a sublayer with no module),
do not approximate it: write the rest of the entry, say what the schema lacks, and extend the
generator as its own change, so every family with that shape gains it.

Done means: the page builds from both the reference and the pinned checkpoint, the tests pass, the
page has been looked at in a browser at both widths, and every sentence in the entry has a source.

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
