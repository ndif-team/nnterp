from functools import lru_cache
from collections import defaultdict
import copy
import json
import importlib.resources
from pathlib import Path

import yaml
from filelock import FileLock
from tqdm.autonotebook import tqdm
import torch as th
import transformers
from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from huggingface_hub import get_collection
import nnsight
from nnsight import LanguageModel
from nnterp import StandardizedTransformer
from nnterp.logging import logger
from nnterp.utils import dummy_inputs


TRANSFORMERS_VERSION = transformers.__version__
NNSIGHT_VERSION = nnsight.__version__
test_loading_status_path = importlib.resources.files("nnterp.data").joinpath(
    "test_loading_status.json"
)
test_loading_status_lock = str(test_loading_status_path) + ".lock"

# Load test configuration from YAML
_test_config_path = Path(__file__).parent / "test_config.yaml"
with open(_test_config_path, "r") as f:
    _TEST_CONFIG = yaml.safe_load(f)

# Models with llama-like naming conventions
LLAMA_LIKE_MODELS = _TEST_CONFIG["llama_like_models"]

# Core test models
TEST_MODELS = _TEST_CONFIG["core_test_models"]

# Skip patterns for models known to fail
SKIP_PATTERNS = _TEST_CONFIG["skip_patterns"]

# Test file categorization for failure tracking
TEST_FILE_CATEGORIES = _TEST_CONFIG["test_file_categories"]


def should_skip_model(model_name):
    """Check if a model should be skipped based on skip patterns."""
    return any(pattern in model_name for pattern in SKIP_PATTERNS)


def filter_skipped_models(model_list):
    """Filter out models matching skip patterns."""
    if not SKIP_PATTERNS:
        return model_list
    filtered = [m for m in model_list if not should_skip_model(m)]
    skipped = [m for m in model_list if should_skip_model(m)]
    if skipped:
        logger.info(f"Skipping {len(skipped)} models matching skip patterns: {skipped}")
    return filtered


def get_all_toy_models():
    """Get all toy models from HuggingFace collection.

    Uses a cached file to avoid rate limiting when running tests in parallel.
    Cache expires after 24 hours.

    Returns sorted list to ensure deterministic test collection for pytest-xdist.
    """
    from datetime import datetime, timedelta
    import json

    cache_file = Path(__file__).parent.parent / "data" / "toy_models_cache.json"
    cache_lock = str(cache_file) + ".lock"
    cache_ttl = timedelta(hours=24)

    lock = FileLock(cache_lock)
    with lock:
        # Check if valid cache exists
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                cached_time = datetime.fromisoformat(cache_data["timestamp"])
                if datetime.now() - cached_time < cache_ttl:
                    logger.debug(f"Using cached toy models from {cache_file}")
                    return cache_data["models"]
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Invalid cache file, will re-fetch: {e}")

        # Fetch from HuggingFace
        logger.info("Fetching toy models from HuggingFace (cache miss or expired)")
        models = [
            item.item_id
            for item in get_collection(
                "yujiepan/tiny-dummy-models-65acf7ddb68db4f26eb1dec9"
            ).items
        ]
        models = sorted(models)

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "models": models}, f, indent=2
            )

        return models


def sort_json_recursively(obj, preserve_level=None, current_level=0):
    """
    Recursively sort JSON object:
    - Sort dictionary keys (unless current_level == preserve_level)
    - Sort lists (if elements are sortable)
    - Recurse into nested structures

    Args:
        obj: The JSON object to sort
        preserve_level: The depth level to preserve order (None means sort all levels)
        current_level: Current depth level (0 is top-level)
    """
    if isinstance(obj, dict):
        if preserve_level is not None and current_level == preserve_level:
            iterator = obj.items()
        else:
            iterator = sorted(obj.items())
        return {
            key: sort_json_recursively(value, preserve_level, current_level + 1)
            for key, value in iterator
        }
    elif isinstance(obj, list):
        els = [
            sort_json_recursively(el, preserve_level, current_level + 1) for el in obj
        ]
        list_els = sorted([e for e in els if isinstance(e, list)], key=lambda x: str(x))
        dict_els = sorted([e for e in els if isinstance(e, dict)], key=lambda x: str(x))
        other_els = [
            e for e in els if not isinstance(e, list) and not isinstance(e, dict)
        ]
        return other_els + list_els + dict_els
    else:
        return obj


def get_all_test_models(class_names=None) -> list[str]:
    """Get all models used in tests: both collection models and hardcoded ones.

    Returns sorted list to ensure deterministic test collection for pytest-xdist.
    """
    collection_models = get_all_toy_models()
    hardcoded_models = TEST_MODELS + LLAMA_LIKE_MODELS
    all_models = list(dict.fromkeys(collection_models + hardcoded_models))
    if class_names is not None:
        all_models = [m for m in all_models if get_arch(m) in class_names]
    # Filter out models matching skip patterns
    all_models = filter_skipped_models(all_models)
    return sorted(all_models)  # Sort for deterministic ordering


def load_test_loading_status():
    """Load the test loading status for current transformers/nnsight versions.

    Thread-safe: Uses file lock to prevent race conditions with pytest-xdist.
    """
    lock = FileLock(test_loading_status_lock)
    with lock:
        try:
            with test_loading_status_path.open("r") as f:
                full_status = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            full_status = {}

        if (
            TRANSFORMERS_VERSION not in full_status
            or NNSIGHT_VERSION not in full_status[TRANSFORMERS_VERSION]
        ):
            if TRANSFORMERS_VERSION not in full_status:
                logger.info(
                    f"Transformers version {TRANSFORMERS_VERSION} not in status, updating..."
                )
            else:
                logger.info(
                    f"NNsight version {NNSIGHT_VERSION} not in status for transformers {TRANSFORMERS_VERSION}, updating..."
                )
            # Release lock temporarily for update (which will re-acquire it)
            lock.release()
            try:
                full_status = update_test_loading_status()
            finally:
                lock.acquire()

        return full_status[TRANSFORMERS_VERSION]


def save_test_loading_status(transformers_status):
    """Save the test loading status dict for current transformers version.

    Thread-safe: Uses file lock to prevent race conditions with pytest-xdist.
    """
    lock = FileLock(test_loading_status_lock)
    with lock:
        try:
            with test_loading_status_path.open("r") as f:
                full_status = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            full_status = {}

        full_status.setdefault(TRANSFORMERS_VERSION, {}).update(transformers_status)

        with open(str(test_loading_status_path), "w") as f:
            json.dump(full_status, f, indent=4)
        return full_status


def test_model_availability(model_name):
    """Test a single model's HF and nnsight availability. Returns (hf_status, nn_status)."""
    # Test HF availability
    if requires_trust_remote_code(model_name):
        return ("need_trust_remote_code", None)

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=False
        )
    except Exception as e:
        if "trust_remote_code" in str(e):
            return ("need_trust_remote_code", None)
        else:
            return ("cant_load", None)

    try:
        hf_model(th.tensor([[1]]))
    except Exception:
        return ("cant_forward", None)

    # Test nnsight availability (only if HF works)
    try:
        nn_model = LanguageModel(model_name)
    except Exception:
        try:
            nn_model = LanguageModel(hf_model)
        except Exception:
            return ("available_hf", "cant_load_with_hf_in_LanguageModel")
        return ("available_hf", "cant_load_with_LanguageModel")

    try:
        with nn_model.trace(dummy_inputs()):
            pass
    except Exception:
        return ("available_hf", "cant_trace_with_LanguageModel")

    return ("available_hf", "available_nn")


def update_test_loading_status():
    """Run full model testing and update the JSON."""
    cant_load = []
    need_trust_remote_code = []
    cant_forward = []
    cant_load_with_LanguageModel = []
    cant_load_with_hf_in_LanguageModel = []
    cant_trace_with_LanguageModel = []
    available_hf_models = []
    available_nn_models = []

    for model_name in tqdm(get_all_test_models()):
        hf_status, nn_status = test_model_availability(model_name)

        if hf_status == "cant_load":
            cant_load.append(model_name)
        elif hf_status == "need_trust_remote_code":
            need_trust_remote_code.append(model_name)
        elif hf_status == "cant_forward":
            cant_forward.append(model_name)
        elif hf_status == "available_hf":
            available_hf_models.append(model_name)

            if nn_status == "cant_load_with_LanguageModel":
                cant_load_with_LanguageModel.append(model_name)
            elif nn_status == "cant_load_with_hf_in_LanguageModel":
                cant_load_with_hf_in_LanguageModel.append(model_name)
            elif nn_status == "cant_trace_with_LanguageModel":
                cant_trace_with_LanguageModel.append(model_name)
            elif nn_status == "available_nn":
                available_nn_models.append(model_name)

    unavailable_hf_models = cant_load + need_trust_remote_code + cant_forward
    unavailable_nn_models = list(
        dict.fromkeys(
            cant_load_with_LanguageModel
            + cant_load_with_hf_in_LanguageModel
            + cant_trace_with_LanguageModel
        )
    )

    transformers_status = {
        "available_hf_models": available_hf_models,
        "unavailable_hf_models": unavailable_hf_models,
        "cant_load_with_AutoModelForCausalLM": cant_load,
        "need_trust_remote_code": need_trust_remote_code,
        "cant_forward": cant_forward,
        NNSIGHT_VERSION: {
            "available_nn_models": available_nn_models,
            "unavailable_nn_models": unavailable_nn_models,
            "cant_load_with_LanguageModel": cant_load_with_LanguageModel,
            "cant_load_with_hf_in_LanguageModel": cant_load_with_hf_in_LanguageModel,
            "cant_trace_with_LanguageModel": cant_trace_with_LanguageModel,
        },
    }

    return save_test_loading_status(transformers_status)


def is_available(model_name, test_status):
    """Check if model is available, using cached status when possible."""
    nnsight_status = test_status.get(NNSIGHT_VERSION, {})
    available_models = nnsight_status.get("available_nn_models", [])
    if model_name in available_models:
        return True, test_status

    unavailable_models = nnsight_status.get(
        "unavailable_nn_models", []
    ) + test_status.get("unavailable_hf_models", [])
    if model_name in unavailable_models:
        return False, test_status

    # Test model if not in cache
    hf_status, nn_status = test_model_availability(model_name)

    # Update HF status in cache
    if hf_status == "cant_load":
        test_status.setdefault("unavailable_hf_models", []).append(model_name)
        test_status.setdefault("cant_load_with_AutoModelForCausalLM", []).append(
            model_name
        )
        return False, test_status
    elif hf_status == "need_trust_remote_code":
        test_status.setdefault("unavailable_hf_models", []).append(model_name)
        test_status.setdefault("need_trust_remote_code", []).append(model_name)
        return False, test_status
    elif hf_status == "cant_forward":
        test_status.setdefault("unavailable_hf_models", []).append(model_name)
        test_status.setdefault("cant_forward", []).append(model_name)
        return False, test_status
    else:
        test_status.setdefault("available_hf_models", []).append(model_name)

    # Update nnsight status in cache
    if nn_status == "cant_load_with_LanguageModel":
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "unavailable_nn_models", []
        ).append(model_name)
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "cant_load_with_LanguageModel", []
        ).append(model_name)
        return False, test_status
    elif nn_status == "cant_load_with_hf_in_LanguageModel":
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "unavailable_nn_models", []
        ).append(model_name)
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "cant_load_with_hf_in_LanguageModel", []
        ).append(model_name)
        return False, test_status
    elif nn_status == "cant_trace_with_LanguageModel":
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "unavailable_nn_models", []
        ).append(model_name)
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "cant_trace_with_LanguageModel", []
        ).append(model_name)
        return False, test_status

    try:
        StandardizedTransformer(model_name)
    except Exception as e:
        logger.warning(
            f"Model {model_name} can't load with StandardizedTransformer: {e}"
        )
        return False, test_status

    test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
        "available_nn_models", []
    ).append(model_name)
    return True, test_status


def get_available_models(model_names):
    """Check model availability and return sorted list for deterministic test collection."""
    if not model_names:
        return []
    test_status = load_test_loading_status()
    available_models = []

    for model in model_names:
        available, test_status = is_available(model, test_status)
        if available:
            available_models.append(model)

    save_test_loading_status(test_status)
    return sorted(available_models)  # Sort for deterministic ordering


def get_all_available_models():
    """Get all available models using fast status-based filtering.

    Returns sorted list to ensure deterministic test collection for pytest-xdist.
    """
    # Use sorted instead of list(set(...)) for deterministic ordering
    all_models = sorted(set(TEST_MODELS + get_all_toy_models() + LLAMA_LIKE_MODELS))
    all_models = filter_skipped_models(all_models)
    return get_available_models(all_models)


def get_available_llama_models():
    """Get available models with llama-like naming conventions."""
    filtered_models = filter_skipped_models(LLAMA_LIKE_MODELS)
    return get_available_models(filtered_models)


def get_failed_models_from_status(test_status):
    """Extract failed models grouped by architecture from test_loading_status.json."""
    failed_models = defaultdict(list)
    nnsight_status = test_status.get(NNSIGHT_VERSION, {})
    unavailable_models = nnsight_status.get("unavailable_nn_models", [])

    for model_name in unavailable_models:
        arch = get_arch(model_name)
        failed_models[arch].append(model_name)

    return failed_models


@lru_cache(maxsize=1000)
def get_model_config_dict(model_name):
    return PretrainedConfig.get_config_dict(model_name)


@lru_cache(maxsize=1000)
def get_arch(model_name):
    archs = get_model_config_dict(model_name)
    if len(archs) == 0:
        return "Unknown"
    archs = archs[0]
    if "architectures" not in archs:
        return "Unknown"
    archs = archs["architectures"]
    if len(archs) == 0:
        return "Unknown"
    if len(archs) == 1:
        return archs[0]
    for _arch in archs:
        if "ForCausalLM" in _arch:
            return _arch
    try:
        AutoModelForCausalLM.from_pretrained(model_name).__class__.__name__
    except Exception:
        return "Unknown"


def requires_trust_remote_code(model_name):
    config = get_model_config_dict(model_name)
    if "_auto_map" in config or "auto_map" in config:
        return True
    return False


def merge_partial_status(
    prev_status: dict, new_status: dict, tested_models: dict[str, list[str]]
) -> dict:
    """Merge partial test results into existing status by updating only the tested models."""
    merged = copy.deepcopy(prev_status)

    for category in new_status:
        for arch in tested_models:
            if category == "nnsight_unavailable_models":
                continue
            if category in merged and arch in merged[category]:
                # Remove tested models from previous categories
                merged[category][arch] = [
                    m for m in merged[category][arch] if m not in tested_models[arch]
                ]
                # Add new models
                merged[category][arch] = list(
                    dict.fromkeys(
                        merged[category][arch] + new_status[category].get(arch, [])
                    )
                )

                if not merged[category][arch]:
                    del merged[category][arch]
            else:

                if category not in merged:
                    merged[category] = {}
                if arch in new_status[category]:
                    merged[category][arch] = new_status[category][arch]

    # Overwrite nnsight_unavailable_models (as it's external)
    if "nnsight_unavailable_models" in new_status:
        merged["nnsight_unavailable_models"] = new_status["nnsight_unavailable_models"]

    return merged


def rm_empty_list(dict_):
    return {k: v for k, v in dict_.items() if v}


if __name__ == "__main__":
    test_status = load_test_loading_status()
    save_test_loading_status(test_status)
