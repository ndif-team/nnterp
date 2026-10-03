"""Every encyclopedia entry names a shipped family and builds a page from its pinned tiny checkpoint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encyclopedia"))

import build  # noqa: E402
import entries  # noqa: E402
import nnterp.families  # noqa: E402


@pytest.mark.parametrize("name", entries.names())
def test_entry_builds_a_page(name):
    entry = entries.load(name)
    assert name in nnterp.families.known()
    for slug in entry.QUIRKS:
        assert slug in build.QUIRKS, slug
    page = build.build_page(entry, reference=entry.PINNED)
    assert entry.TITLE in page
    for sub in entry.BLOCK["sublayers"]:
        assert f"model.layers[i].{sub['host']}.{sub['contribution']}" in page
        for value in sub.get("interior", []):
            assert f"interior.{sub['host']}.{value}" in page
    assert '"identity"' in page and "stream.output" in page


def test_every_quirk_has_a_label():
    for slug, (label, blurb) in build.QUIRKS.items():
        assert label and blurb, slug
