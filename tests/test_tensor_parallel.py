"""A model split tensor-parallel: the values split by head or neuron read and edit as they do on one device.

transformers splits the heads from the query projection to the output projection, and an MLP's neurons from the
gate to the down projection. nnsight reassembles a sharded module's own input and output, not what lies between,
so nnterp gathers those values on read and splits an edit back (`EProperty.piece_axis`). `tp_worker.py` runs the
same reads and edits in one process and across two CPU ranks over gloo; every value has to match, and every rank
has to agree. CPU checks the layout, not real kernels or dtypes.
"""

import os
import socket
import subprocess
import sys

import pytest
import torch
import transformers
from packaging.version import Version

pytestmark = [
    pytest.mark.skipif(Version(transformers.__version__).release < (5, 19), reason="nnsight shards a model on transformers 5.19 and later"),
    pytest.mark.xdist_group("tensor_parallel"),  # one worker runs the two launches once
]

#: Grouped-query attention, so the keys' head axis (2) is not the queries' (4).
REPO = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TP_SIZE = 2
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp_worker.py")

READS = [
    "attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_probabilities",
    "attention_head_outputs", "attention_premix", "attention_output", "mlp_activation", "mlp_neurons", "mlp_output",
    "logits",
]
EDITS = [
    "head_outputs_zeroed_in_place", "keys_scaled_in_place", "queries_scaled_by_assignment", "activation_assigned",
    "head_outputs_zeroed_with_grad",
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run(tp: int, out: str) -> None:
    """The worker at ``tp`` CPU ranks, importing the nnterp and nnsight this session imported."""
    import nnsight
    import nnterp

    paths = [os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))) for module in (nnterp, nnsight)]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [*paths, os.environ.get("PYTHONPATH")])),
        "CUDA_VISIBLE_DEVICES": "",  # transformers builds the mesh on CPU only when no GPU is visible
        "OMP_NUM_THREADS": "1",
    }
    command = [sys.executable]
    if tp > 1:
        command += ["-m", "torch.distributed.run", f"--nproc_per_node={tp}", f"--master_port={_free_port()}"]
    command += [WORKER, "--tp", str(tp), "--repo", REPO, "--out", out]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(f"tp={tp} worker failed\n--- stdout ---\n{completed.stdout[-4000:]}\n--- stderr ---\n{completed.stderr[-4000:]}")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[dict, list[dict]]:
    """``(one process's values, [each rank's values])``."""
    whole, split = tmp_path_factory.mktemp("tp1"), tmp_path_factory.mktemp(f"tp{TP_SIZE}")
    _run(1, str(whole))
    _run(TP_SIZE, str(split))
    reference = torch.load(whole / "rank0.pt")
    return reference, [torch.load(split / f"rank{rank}.pt") for rank in range(TP_SIZE)]


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``max|a - b| / max|b|`` over the entries the causal mask leaves (the scores carry its dtype minimum)."""
    kept = expected > torch.finfo(expected.dtype).min / 2
    return ((actual - expected)[kept].abs().max() / expected[kept].abs().max()).item()


def test_every_rank_agrees(runs):
    _, ranks = runs
    first, *rest = ranks
    for rank, values in enumerate(rest, start=1):
        for name, value in first.items():
            assert torch.equal(value, values[name]), f"rank {rank} disagrees on {name}"


@pytest.mark.parametrize("name", READS + EDITS)
def test_matches_one_process(runs, name):
    reference, (rank0, *_) = runs
    assert rank0[name].shape == reference[name].shape
    assert _rel(rank0[name], reference[name]) < 1e-4


@pytest.mark.parametrize("name", EDITS)
def test_the_edit_moves_the_logits(runs, name):
    """Each edit changes the prediction, so matching it across ranks says the edit landed where it was aimed."""
    reference, _ = runs
    assert _rel(reference[name], reference["logits"]) > 1e-3
