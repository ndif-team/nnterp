# Installation

nnterp is a PyPI package for Python 3.10 or later. It depends on nnsight 0.8 and transformers 5.17 or later, which pip installs with it.

```bash
pip install nnterp
```

## From source

```bash
pip install git+https://github.com/ndif-team/nnterp.git
```

For a checkout you intend to edit, run `pip install -e .` from the clone.

## Extras

```bash
pip install "nnterp[models]"   # sentencepiece, tiktoken, protobuf, pillow: tokenizers and processors some checkpoints need
pip install "nnterp[vllm]"     # vLLM, for StandardizedVLLM
pip install "nnterp[test]"     # pytest and pytest-xdist, for the test suite
```

The [Quick Start](quickstart.md) loads GPT-2 and is the check that the install works.
