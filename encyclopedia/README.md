# The nnterp encyclopedia

One page per family: the block drawn and annotated with nnterp's names, `support()` explained,
the sizes and native paths, and notes on what to know before tracing that family. The pages are
static HTML, built by `build.py` into `site/` (not committed).

## Build

```
pip install -e ".[encyclopedia]"
PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py            # every entry, plus the index
PYTHONPATH=. HF_HUB_OFFLINE=1 python encyclopedia/build.py gemma2     # one entry
```

Each entry's `REFERENCE` checkpoint is built on the `meta` device (config and tokenizer only, no
weights), so the config has to be in the Hub cache or reachable. Open `encyclopedia/site/index.html`.

## What a page is made of

Two sources, merged by `build.py`:

- **From the code.** The family module (`nnterp/families/<model_type>.py`) and a meta build of the
  reference checkpoint: `support()` under an eager load and under the default one (the diff marks
  the values that need `attn_implementation="eager"`), every standard value's key, layout, dims and
  description, the root sizes, `RENAME`, the native path behind each standard name, the block's
  repr and the module's docstring.
- **From a person.** `entries/<model_type>.py`: title, subtitle, the block schema the visualization
  draws, quirk tags, palette, notes. `entries/__init__.py` lists the fields; `entries/gemma2.py` is
  the reference entry.

## Writing an entry

1. Copy `entries/gemma2.py` to `entries/<model_type>.py` and set `MODEL_TYPE`, `TITLE`, `SUBTITLE`,
   `REFERENCE` (a public checkpoint whose config is cached), `PINNED` (the suite's tiny checkpoint,
   from `tests/families/test_<model_type>.py`), `CHECKPOINTS`.
2. `BLOCK`: the sublayers in forward order. Each names its `host` (`self_attn`, `linear_attn`, `mlp`),
   its `contribution` (the standard value the block adds), its norms if any, the `interior` values
   to draw as chips, and a `detail` line; `variants` keyed on `config.layer_types` change the detail
   with the layer slider. `topology` is `sequential` or `parallel`. Give `identity` when the block is
   not a plain sum (Gemma-4, Granite, DeepSeek-V4).
3. `QUIRKS`: slugs from `build.QUIRKS`, which follow the themes of `docs/reference/families.md`.
4. `PALETTE`: `{"hue": degrees}` is the base hue the page's five colours are generated from
   (`palette.py`); pick it by lineage so related families read alike (Gemma 2 sits at 145, so a
   Gemma 3 entry near it reads as kin), or leave it out for a hash of the `model_type`. To bypass
   generation give `"colors"`, five hex fills in role order (attention, MLP, norms, stream, mark),
   and optionally `"deeps"`, their five darker partners; `"paper"` tints the page. A palette that
   fails the contrast or distinctness checks fails the build.
5. `NOTES`: markdown. Present tense, factual, the family's own facts: what the contributions are,
   what needs eager, what the logits are, read-order traps, which SAEs exist. Snippets are
   illustrative; the executed ones live in `docs/`.
6. `HF_HUB_OFFLINE=1 pytest tests/test_encyclopedia.py` builds every entry's page from its pinned
   checkpoint and checks the schema against the family.

The design is the Sakura Chroma system (`static/encyclopedia.css`): paper and ink, Big Shoulders for
display, Albert Sans for body, JetBrains Mono for data; no rounded corners, no soft shadows. Each
family brings five colours, one per role, and the same role takes the same colour everywhere on its
page: attention (the box, its chips, `attention_output`, the `self_attn` host, attention names in
ledgers and code), the MLP, the norms, the residual stream (the line, `layers[i].input`,
`layer_output`, the root's values, `model.` and strings in code) and the family's mark (chips, the
`em` in titles, topstrips, hover fills, keywords in code). Every colour comes in a fill tier, which
takes ink text, and a deep tier for lines and coloured text on the paper; the page sets `--c1` …
`--c5`, `--c1-deep` … `--c5-deep` and `--paper` on `<html>`, and the stylesheet names the roles from
them. Code is highlighted at build time (Pygments for the snippets, a line lexer for the printout),
with no JavaScript. The warning mark stays amber on every family: it is a status, not a role.
