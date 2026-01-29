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

passing_models = []
failing_models = []
for model in data[transformers.__version__]["available_hf_models"]:
    logger.info(f"running {model}")
    result = subprocess.run(
        ["pytest", "-vx", "-k", model], check=False, capture_output=False, text=False
    )
    logger.info(f"Results for {model}:\n{result.stdout}")
    if result.returncode != 0:
        logger.error(f"Error running pytest for {model}:\n{result.stderr}")
        failing_models.append(model)
    else:
        passing_models.append(model)

res = {"passing": defaultdict(list), "failing": defaultdict(list)}
passing_classes = set()
failing_classes = set()
for model in passing_models:
    arch = get_arch(model)
    passing_classes.add(arch)
    res["passing"][arch].append(model)

for model in failing_models:
    arch = get_arch(model)
    failing_classes.add(arch)
    res["failing"][arch].append(model)

with open("data/passing_models.json", "w") as f:
    json.dump(
        {
            "passing_classes": list(passing_classes - failing_classes),
            "failing_classes": list(failing_classes - passing_classes),
            "models": dict(res),
        },
        f,
        indent=2,
    )
