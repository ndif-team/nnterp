---
title: Releasing
one_liner: A GitHub release publishes nnterp to PyPI through `.github/workflows/publish.yml`; the tag is the version, Trusted Publishing signs the upload, and nothing is bumped by hand.
tags: [developing, release, pypi, versioning, setuptools-scm, trusted-publishing]
related: [docs/developing/contributing.md, docs/developing/testing.md, docs/developing/transformers-compat.md]
sources: [.github/workflows/publish.yml, pyproject.toml, nnterp/__init__.py]
---

# Releasing

## What a release does

Publishing a GitHub release runs `.github/workflows/publish.yml`. The `build` job checks the
tagged commit out with its history, builds the sdist and the wheel with `python -m build`,
runs `twine check`, installs the wheel without dependencies to read its version back, and
confirms the wheel carries `nnterp/families`, `nnterp/components` and `nnterp/_version.py`.
The `publish` job then uploads both to PyPI with `pypa/gh-action-pypi-publish`, pinned to a
commit. Running the workflow by hand (`workflow_dispatch`) builds and checks without
publishing, which is how to confirm a branch packages cleanly before tagging it.

## The version is the tag

`pyproject.toml` declares the version dynamic and lets setuptools-scm derive it from the git
tag: tag `v2.0.0` builds `nnterp 2.0.0`, tag `v2.0.0rc1` builds the pre-release `2.0.0rc1`,
and a commit with no tag builds `1.3.1.devN+g<sha>`, which PyPI refuses. setuptools-scm
writes `nnterp/_version.py` at build time (it is not tracked), and `nnterp.__version__`
reads it, falling back to the installed distribution's metadata. Nothing in the repository
states the version, so a release is: tag the commit, publish the release for the tag, and
let the workflow run.

## Trusted Publishing

The upload carries no API token. PyPI mints a short-lived token from the publish job's OIDC
identity and checks it against the publisher registered for the `nnterp` project: owner
`ndif-team`, repository `nnterp`, workflow `publish.yml`, environment `pypi`. All three must
match the workflow, so the file is not renamed and the job keeps `environment: pypi`. The
environment exists in the repository's settings (Settings, Environments) and can require a
reviewer, which makes every publish a click by a maintainer.

## Before the first 2.0 release

`pip install nnterp` resolves `nnsight>=0.8`, which no nnsight release satisfies yet: PyPI has
`0.8.0rc1`, a pre-release that `>=0.8` does not accept. A published nnterp 2.0 is installable
only once nnsight 0.8.0 is on PyPI, and that release has to carry the `dev` commits nnterp
reads (`Envoy.parent` and `root`, the dotted `rename` alias; [transformers-compat](transformers-compat.md)).
Until then, tag pre-releases (`v2.0.0rc1`) so a `pip install nnterp` without `--pre` keeps
giving 1.3.0, and say in the release notes which nnsight the pre-release needs.

## Checklist

1. The suite is green on the commit (`tests/`, [testing](testing.md)); the docs and the
   encyclopedia are rebuilt if a value or family changed.
2. Run the workflow by hand on the branch once; the `build` job must pass.
3. Tag the commit (`git tag v2.0.0 && git push origin v2.0.0`) and publish a GitHub release
   for it, with notes.
4. Watch the `publish` job; `https://pypi.org/p/nnterp` shows the new version when it finishes.
