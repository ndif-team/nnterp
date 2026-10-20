# Installation

## Requirements

- Python 3.10 or higher
- nnsight 0.8 or later and transformers 5.x, both installed automatically

## Install from PyPI

```bash
pip install nnterp
```

## Install from source

```bash
pip install git+https://github.com/ndif-team/nnterp.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/ndif-team/nnterp.git
cd nnterp
pip install -e .
```

## Extras

```bash
pip install "nnterp[models]"   # tokenizers and processors some checkpoints need: sentencepiece, tiktoken, protobuf, pillow
pip install "nnterp[vllm]"     # vLLM, for StandardizedVLLM
pip install "nnterp[test]"     # pytest and pytest-xdist, for the test suite
```

## Verify

```python
import nnterp
print(nnterp.__version__)
```

Next: the [Quick Start](quickstart.md).
