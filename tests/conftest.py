import nnsight  # noqa: F401  nnsight before any transformers submodule, in every test process

import os  # noqa: E402

import torch  # noqa: E402

if "PYTEST_XDIST_WORKER" in os.environ:
    # One thread per worker: eight workers each spawning a thread per core thrash the machine instead of running.
    torch.set_num_threads(1)
