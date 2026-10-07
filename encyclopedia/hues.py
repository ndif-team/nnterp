"""The hue layout of every entry: one table, spaced around the circle by rule.

``LAYOUT`` lists the lineages in their order around the hue circle, each a band of its members in
order, and the families with no kin, each in the gap between the two bands it sits between. The
spacing is computed, not chosen: neighbours in a band are ``STEP`` apart, two bands with nothing
between them are two steps apart, and a family in a gap sits a step and a half from each
neighbour. ``STEP`` is what fills the circle exactly. ``START`` is the first band's first hue.

    python encyclopedia/hues.py            # print the layout
    python encyclopedia/hues.py --write    # and rewrite every entry's PALETTE hue and README's table

After adding a family, put it in ``LAYOUT`` (in its lineage's band, or in a gap) and run with
``--write``: every hue moves a little and the table stays one layout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import palette  # noqa: E402

#: The circle, in order. A tuple is a lineage: (name, [members]); a string is a family with no kin.
LAYOUT: list[tuple[str, list[str]] | str] = [
    ("DeepSeek (and the Kimi and Youtu lines on its latent attention)",
     ["deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4", "youtu", "kimi_k2", "kimi_linear"]),
    ("MiniMax / MiMo", ["minimax_m2", "mimo_v2_flash"]),
    "zaya",
    ("Mamba (and Jamba's Mamba blocks)", ["mamba", "mamba2", "falcon_mamba", "jamba"]),
    ("Falcon", ["falcon_h1", "falcon"]),
    "doge",
    ("HunYuan", ["hunyuan_v1_dense", "hunyuan_v1_moe"]),
    ("GPT-2", ["gpt2", "gpt_bigcode", "starcoder2"]),
    "bloom",
    ("GPT-J", ["gpt_neo", "gptj", "codegen"]),
    ("GPT-NeoX", ["gpt_neox", "gpt_neox_japanese", "stablelm"]),
    "persimmon",
    ("OPT", ["opt", "xglm"]),
    "mpt",
    "dbrx",
    ("OLMo", ["olmo", "olmo2", "olmo3", "olmoe", "flex_olmo", "olmo_hybrid"]),
    ("GLM", ["glm", "glm4", "glm4_moe", "glm4_moe_lite", "glm_moe_dsa"]),
    "jetmoe",
    ("Phi", ["phi", "phi3", "phimoe"]),
    "gpt_oss",
    ("Gemma", ["gemma", "gemma2", "gemma3_text", "gemma4_text", "gemma4_unified_text", "vaultgemma"]),
    ("Cohere", ["cohere", "cohere2"]),
    "exaone4",
    ("ERNIE", ["ernie4_5", "ernie4_5_moe"]),
    "laguna",
    ("Granite (and IBM's Bamba)",
     ["bamba", "granitemoehybrid", "granite", "granite_swa", "granitemoe", "granitemoe_swa", "granitemoeshared"]),
    ("Nemotron", ["nemotron", "nemotron_h"]),
    "afmoe",
    ("Llama and its kin",
     ["arcee", "apertus", "llama", "llama4_text", "smollm3", "helium", "bitnet", "seed_oss",
      "mistral", "mixtral", "ministral", "ministral3"]),
    "hyperclovax",
    "solar_open",
    ("Qwen",
     ["qwen2", "qwen2_moe", "qwen2_vl_text", "qwen2_5_vl_text", "qwen3", "qwen3_moe", "qwen3_vl_text",
      "qwen3_vl_moe_text", "qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"]),
    "dots1",
]
#: The first band's first hue. DeepSeek starts the circle in magenta, so Gemma lands in green and Llama in
#: blue, far from it, beside Qwen's purple.
START = 322
#: Steps, in units of STEP: between neighbours in a band, between two bands, between a band and a family in a gap.
IN_BAND, BETWEEN_BANDS, BESIDE_GAP = 1.0, 2.0, 1.5
#: What the layout must keep, in degrees after rounding.
MIN_BETWEEN_BANDS, MIN_BESIDE_GAP = 6, 4


def flatten() -> list[tuple[str, str | None]]:
    """(model_type, lineage or None) around the circle, in order."""
    out: list[tuple[str, str | None]] = []
    for item in LAYOUT:
        if isinstance(item, str):
            out.append((item, None))
        else:
            out.extend((member, item[0]) for member in item[1])
    return out


def steps(items: list[tuple[str, str | None]]) -> list[float]:
    """The weight of the step from each item to the next, the last wrapping to the first."""
    out = []
    for (_, here), (_, there) in zip(items, items[1:] + items[:1]):
        if here is not None and here == there:
            out.append(IN_BAND)
        elif here is None or there is None:
            out.append(BESIDE_GAP)
        else:
            out.append(BETWEEN_BANDS)
    return out


def layout() -> dict[str, int]:
    """model_type -> hue, every entry once."""
    items = flatten()
    weights = steps(items)
    step = 360 / sum(weights)
    hues, at = {}, float(START)
    for (name, _), weight in zip(items, weights):
        hues[name] = round(at) % 360
        at += weight * step
    return hues


def check(hues: dict[str, int]) -> list[str]:
    import build
    import entries

    problems = []
    names = [n for n, _ in flatten()]
    if len(names) != len(set(names)):
        problems.append(f"listed twice: {sorted({n for n in names if names.count(n) > 1})}")
    missing = set(entries.names()) - set(names)
    extra = set(names) - set(entries.names())
    if missing or extra:
        problems.append(f"LAYOUT misses {sorted(missing)}, names no entry {sorted(extra)}")
    items = flatten()
    for ((a, la), (b, lb)), weight in zip(zip(items, items[1:] + items[:1]), steps(items)):
        apart = abs((hues[a] - hues[b] + 180) % 360 - 180)
        need = {IN_BAND: 1, BETWEEN_BANDS: MIN_BETWEEN_BANDS, BESIDE_GAP: MIN_BESIDE_GAP}[weight]
        if apart < need:
            problems.append(f"{a} {hues[a]} and {b} {hues[b]}: {apart} apart < {need}")
    for name, hue in hues.items():
        generated = palette.generate(hue)
        if bad := palette.check(generated["fills"], generated["deeps"], build.PAPER, build.INK):
            problems.append(f"{name} {hue}: {bad}")
    return problems


def table(hues: dict[str, int]) -> str:
    rows = ["| lineage | hues |", "|---|---|"]
    for item in LAYOUT:
        if isinstance(item, str):
            continue
        name, members = item
        rows.append(f"| {name} | {', '.join(f'`{m}` {hues[m]}' for m in members)} |")
    singles = [item for item in LAYOUT if isinstance(item, str)]
    rows.append(f"| no kin (in the gaps) | {', '.join(f'`{m}` {hues[m]}' for m in singles)} |")
    return "\n".join(rows)


PALETTE_LINE = re.compile(r'^PALETTE = \{"hue": \d+(?P<rest>[^}\n]*)\}$', re.M)
COMMENT_ABOVE = re.compile(r'(?:^#:[^\n]*\n)+(?=PALETTE = )', re.M)
TABLE = re.compile(r"(<!-- hues\.py: table -->\n).*?(<!-- hues\.py: end -->)", re.S)


def write(hues: dict[str, int]) -> None:
    lineage = dict(flatten())
    for name, hue in hues.items():
        path = HERE / "entries" / f"{name}.py"
        text = path.read_text()
        where = f"lineage: {lineage[name].split(' (')[0]}" if lineage[name] else "no kin, in a gap between lineages"
        line = f"#: Set by hues.py ({where}).\n"
        if PALETTE_LINE.search(text):
            text = COMMENT_ABOVE.sub("", text, count=1)
            text = PALETTE_LINE.sub(lambda m: f'{line}PALETTE = {{"hue": {hue}{m["rest"]}}}', text, count=1)
        else:
            # an entry that hashed: PALETTE goes before VLLM, where the others have it
            assert re.search(r"^VLLM = ", text, re.M), f"{name}: no VLLM line to put PALETTE before"
            text = re.sub(r"^VLLM = ", f'{line}PALETTE = {{"hue": {hue}}}\nVLLM = ', text, count=1, flags=re.M)
        path.write_text(text)
    readme = HERE / "README.md"
    text, n = TABLE.subn(lambda m: m[1] + table(hues) + "\n" + m[2], readme.read_text())
    assert n == 1, "README.md has no <!-- hues.py: table --> ... <!-- hues.py: end --> block"
    readme.write_text(text)


def main() -> None:
    sys.path.insert(0, str(HERE.parent))
    hues = layout()
    items = flatten()
    weights = steps(items)
    print(f"step {360 / sum(weights):.2f} deg, {len(items)} families")
    for item in LAYOUT:
        if isinstance(item, str):
            print(f"  {hues[item]:>3}  {item}")
        else:
            print(f"  {hues[item[1][0]]:>3}-{hues[item[1][-1]]:<3} {item[0]}: "
                  + ", ".join(f"{m} {hues[m]}" for m in item[1]))
    problems = check(hues)
    for p in problems:
        print("PROBLEM", p)
    if problems:
        sys.exit(1)
    if "--write" in sys.argv:
        write(hues)
        print("wrote every entry's PALETTE and README.md's table")


if __name__ == "__main__":
    main()
