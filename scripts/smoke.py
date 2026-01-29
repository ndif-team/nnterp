import json
import logging
import subprocess
import transformers
from collections import defaultdict

from tests.utils import get_arch

logger = logging.getLogger("nnterp")

with open("data/test_loading_status.json", "r") as f:
    data = json.load(f)
all_models = data[transformers.__version__]["0.5.0.dev8"]["available_nn_models"]
from tqdm import trange

num_failures = 0
num_pass = 0
pbar = trange(
    100,
    desc="Running tests",
    postfix=f"num_failures: {num_failures}, num_pass: {num_pass}",
)
for i in pbar:
    result = subprocess.run(
        # pytest -k test_standardized_transformer_num_layers_property[yujiepan/opt-tiny-random
        ["pytest", "-vx", "-k", "[yujiepan/opt-tiny-random]"],
        check=False,
        capture_output=True,
        text=False,
    )
    if result.returncode != 0:
        logger.error(f"Error running pytest for {i}:\n{result.stderr}")
        num_failures += 1
    else:
        num_pass += 1
    pbar.set_postfix(num_failures=num_failures, num_pass=num_pass)


print(f"num_failures: {num_failures}, num_pass: {num_pass}")
