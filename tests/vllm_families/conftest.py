"""The vLLM suite needs vLLM and a GPU; without them nothing here is collected."""

import torch

try:
    import vllm.model_executor  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
else:
    if not torch.cuda.is_available():
        collect_ignore_glob = ["test_*.py"]
