# Getting Started

nnterp is a layer on [nnsight](https://nnsight.net). These pages assume you have written an
nnsight trace before: `with model.trace(...)`, `.save()`, module `.input` and `.output`. If not,
nnsight's own [Quick Start](https://nnsight.net/getting-started/quickstart/) is ten minutes and
everything there carries over unchanged.

What nnterp adds is one set of module names and one set of values that mean the same thing on
every transformer family, so the trace you write for GPT-2 runs on Llama, Gemma or Mamba.

## :material-download: [Installation](installation.md)

`pip install nnterp` and the extras.

## :material-rocket-launch: [Quick Start](quickstart.md)

Load a model, read the standard values, check what a checkpoint supports. GPT-2 on CPU.
