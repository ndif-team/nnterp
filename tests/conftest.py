import nnsight  # noqa: F401  nnsight before any transformers submodule, in every test process

import os  # noqa: E402

# pyarrow's bundled jemalloc starts a background thread while libarrow is still being loaded, which on glibc
# 2.31 (Ubuntu 20.04) faults the process silently (glibc BZ #19329); transformers' import chain reaches pyarrow
# through sklearn and pandas. Inherited by the subprocesses the tests spawn.
os.environ.setdefault("JE_ARROW_MALLOC_CONF", "background_thread:false")

import torch  # noqa: E402

if "PYTEST_XDIST_WORKER" in os.environ:
    # One thread per worker: eight workers each spawning a thread per core thrash the machine instead of running.
    torch.set_num_threads(1)
