"""The vLLM suite needs vLLM and a GPU; without them nothing here is collected."""

import os

import torch

# pyarrow's jemalloc background thread segfaults at start when pyarrow is first loaded after vLLM inside pytest
# (transformers' generation utils pull it in through sklearn and pandas, which a transformers-backend family's
# module import does at collection); without the thread it loads cleanly.
os.environ.setdefault("JE_ARROW_MALLOC_CONF", "background_thread:false")

try:
    import vllm.model_executor  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
else:
    if not torch.cuda.is_available():
        collect_ignore_glob = ["test_*.py"]
