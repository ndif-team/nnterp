"""What the Hugging Face Hub says about an entry: its organisation (name and avatar) and, per checkpoint, the
parameter count and the date the repository was created.

Every answer is kept in ``hub_cache.json`` beside this file, and every avatar in ``static/orgs/``, both committed,
so a build offline (and the tests) reads the same page a build online writes. A build online asks the Hub only for
what the cache does not hold (`refresh`), and `save` writes the cache back.

``hub_cache.json``::

    {"orgs":  {"<hub id>": {"name": "<display name>", "avatar": "orgs/<hub id>.<ext>" or null}},
     "repos": {"<repo id>": {"params": <int> or null, "created": "<YYYY-MM-DD>" or null}}}

An org is the author of the entry's ``REFERENCE`` (``ORG`` in the entry overrides it), looked up as an organisation
and then as a user; an id neither resolves keeps the id as its name and no avatar. A repo's ``params`` is null when
the Hub has no safetensors metadata for it (the page then counts the meta model's parameters); a repo the Hub does
not let this build read (gated, private, gone) holds nulls for both and the page leaves them blank.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
CACHE = HERE / "hub_cache.json"
AVATARS = HERE / "static" / "orgs"
HUB = "https://huggingface.co"
AVATAR_PX = 64  # avatars are stored this many pixels on a side
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#: Whether a lookup the cache does not hold goes to the Hub. `build.build` turns it on for a build that is online;
#: the tests leave it off, so a page they build reads the committed cache alone.
REFRESH = False
_cache: dict[str, dict[str, Any]] | None = None
_dirty = False


def cache() -> dict[str, dict[str, Any]]:
    global _cache
    if _cache is None:
        data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        _cache = {"orgs": data.get("orgs", {}), "repos": data.get("repos", {})}
    return _cache


def online() -> bool:
    from huggingface_hub import constants

    return not (constants.HF_HUB_OFFLINE or os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"))


def save() -> None:
    """Write the cache back when a lookup added to it, sorted so a diff shows only what changed."""
    global _dirty
    if not _dirty:
        return
    data = {kind: dict(sorted(cache()[kind].items(), key=lambda kv: kv[0].lower())) for kind in ("orgs", "repos")}
    CACHE.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    _dirty = False


class Unreachable(Exception):
    """The Hub did not answer (no network, a timeout, a server error): nothing is cached, a later build asks again."""


def _get_json(url: str) -> dict[str, Any] | None:
    """The JSON at ``url``, or None when the Hub answers that there is nothing there for this build."""
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            return None
        raise Unreachable(url) from error
    except OSError as error:
        raise Unreachable(url) from error


def _download_avatar(org: str, url: str) -> str | None:
    """The avatar, AVATAR_PX square as a PNG in static/orgs/, or None when it cannot be read as an image."""
    import io

    from PIL import Image

    url = url if url.startswith("http") else HUB + url
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:  # noqa: BLE001 - an SVG or a broken link: the chip shows the name alone
        return None
    image.thumbnail((AVATAR_PX, AVATAR_PX), Image.LANCZOS)
    AVATARS.mkdir(parents=True, exist_ok=True)
    path = AVATARS / f"{org}.png"
    image.save(path, optimize=True)
    return f"orgs/{path.name}"


def org_id(entry: ModuleType) -> str:
    return getattr(entry, "ORG", None) or entry.REFERENCE.split("/")[0]


def org(entry: ModuleType) -> dict[str, Any]:
    """The entry's org as the index shows it: ``{"id", "name", "avatar"}`` (``avatar`` a path under static/, or None)."""
    global _dirty
    oid = org_id(entry)
    known = cache()["orgs"].get(oid)
    if known is None and REFRESH:
        try:
            overview = _get_json(f"{HUB}/api/organizations/{oid}/overview") or _get_json(f"{HUB}/api/users/{oid}/overview")
        except Unreachable:
            overview = False
        if overview is not False:
            known = {"name": oid, "avatar": None}
            if overview:
                known["name"] = overview.get("fullname") or oid
                if overview.get("avatarUrl"):
                    known["avatar"] = _download_avatar(oid, overview["avatarUrl"])
            cache()["orgs"][oid] = known
            _dirty = True
    known = known or {"name": oid, "avatar": None}
    return {"id": oid, "name": known["name"], "avatar": known["avatar"]}


def repo(repo_id: str) -> dict[str, Any] | None:
    """``{"params", "created"}`` for a Hub repo, or None when the cache does not hold it and the build cannot ask.
    A local path (a test's rewritten copy of a tiny checkpoint) is not a repo: None."""
    global _dirty
    known = cache()["repos"].get(repo_id)
    if known is None and REFRESH and not os.path.isdir(repo_id):
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        try:
            info = HfApi().model_info(repo_id)
            known = {"params": info.safetensors.total if info.safetensors else None,
                     "created": info.created_at.date().isoformat() if info.created_at else None}
        except (GatedRepoError, RepositoryNotFoundError):
            known = {"params": None, "created": None}
        except Exception:  # noqa: BLE001 - the Hub did not answer: nothing is cached, a later build asks again
            return None
        cache()["repos"][repo_id] = known
        _dirty = True
    return known


def format_params(count: int) -> str:
    """A parameter count in three significant figures: 8.03B, 124M, 1.04T."""
    for scale, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= scale:
            value = count / scale
            text = f"{value:.3g}" if value < 1000 else f"{value:.0f}"
            return text + unit
    return str(count)


def format_month(date: str) -> str:
    """``2024-07-14`` as ``Jul 2024``."""
    year, month = date.split("-")[:2]
    return f"{MONTHS[int(month) - 1]} {year}"


def facts(repo_id: str, meta_params: int | None) -> dict[str, Any]:
    """The checkpoint's line under the selector: ``params`` (formatted) and whether it is ``from_config`` (the
    Hub has no safetensors metadata, so the meta model's parameters are counted), and ``since`` (``Jul 2024``).
    A repo the cache does not hold, or the Hub does not let this build read, gives blanks."""
    known = repo(repo_id)
    if not known or (known["params"] is None and known["created"] is None):
        return {"params": None, "from_config": False, "since": None}
    params, from_config = known["params"], False
    if params is None and meta_params:
        params, from_config = meta_params, True
    return {"params": format_params(params) if params else None, "from_config": from_config,
            "since": format_month(known["created"]) if known["created"] else None}
