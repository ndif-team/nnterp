"""Shared pytest fixtures for all tests."""

import datetime
import json
from collections import defaultdict
import importlib.resources
from pathlib import Path
import threading
import time

import pytest
from _pytest.outcomes import Skipped
from nnterp import StandardizedTransformer
from nnterp.logging import logger
from nnsight import LanguageModel

from .utils import (
    get_all_available_models,
    get_available_models,
    get_arch,
    get_available_llama_models,
    get_all_test_models,
    get_failed_models_from_status,
    load_test_loading_status,
    NNSIGHT_VERSION,
    TRANSFORMERS_VERSION,
    LLAMA_LIKE_MODELS,
    TEST_FILE_CATEGORIES,
    merge_partial_status,
    rm_empty_list,
    sort_json_recursively,
)

PROJECT_ROOT = Path(str(importlib.resources.files("nnterp")))

# Define stash keys for custom test session data
has_deselected_key = pytest.StashKey[bool]()
errors_key = pytest.StashKey[defaultdict]()
skips_key = pytest.StashKey[defaultdict]()
failure_categories_key = pytest.StashKey[dict]()  # Dict of category_name -> defaultdict
tested_models_key = pytest.StashKey[defaultdict]()
is_model_specific_key = pytest.StashKey[bool]()
is_full_run_key = pytest.StashKey[bool]()
save_test_logs_key = pytest.StashKey[bool]()

# Lock for thread-safe operations on shared data structures
stash_lock_key = pytest.StashKey[threading.Lock]()


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--model-names",
        action="store",
        nargs="+",
        help="If provided, only run tests for the given model names.",
    )
    parser.addoption(
        "--class-names",
        action="store",
        nargs="+",
        help="If provided, only run tests for the given class names.",
    )
    parser.addoption(
        "--save-test-logs",
        action="store_true",
        help="Save test logs to the data/test_logs directory.",
    )


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on command line options."""
    if (
        "model_name" in metafunc.fixturenames
        or "llama_like_model_name" in metafunc.fixturenames
    ):
        model_names = metafunc.config.getoption("model_names", default=None)
        class_names = metafunc.config.getoption("class_names", default=None)
        if model_names is not None or class_names is not None:
            all_models = []
            if model_names is not None:
                all_models = get_available_models(model_names)
            if class_names is not None:
                all_models += get_all_test_models(class_names)

            # Deduplicate and sort for deterministic test collection with pytest-xdist
            all_models = sorted(set(all_models))

            if not all_models:
                raise ValueError(f"No models available in NNsight from {model_names}")
            llama_like_models = get_available_models(
                sorted([m for m in LLAMA_LIKE_MODELS if m in all_models])
            )
        else:
            all_models = get_all_available_models()
            llama_like_models = get_available_llama_models()

    if "model_name" in metafunc.fixturenames:
        metafunc.parametrize("model_name", all_models)

    if "llama_like_model_name" in metafunc.fixturenames:
        params = (
            llama_like_models
            if llama_like_models
            else [
                pytest.param(
                    None,
                    marks=pytest.mark.skip(reason="No llama-like models available"),
                )
            ]
        )
        metafunc.parametrize("llama_like_model_name", params)


def pytest_configure(config):
    """Initialise per-session flags on the config object."""
    # Warn if pytest-xdist is being used (not fully supported)

    config.stash[has_deselected_key] = False
    config.stash[errors_key] = defaultdict(dict)
    config.stash[skips_key] = defaultdict(dict)
    config.stash[tested_models_key] = defaultdict(list)

    # Initialize lock for thread-safe operations
    config.stash[stash_lock_key] = threading.Lock()

    # Initialize failure categories dynamically from YAML config
    # Always include "failed_test_models" as the default category
    failure_categories = {"failed_test_models": defaultdict(list)}
    for middle_part in TEST_FILE_CATEGORIES.keys():
        category_name = f"failed_{middle_part}_models"
        failure_categories[category_name] = defaultdict(list)
    config.stash[failure_categories_key] = failure_categories

    config.stash[is_model_specific_key] = (
        config.getoption("model_names", default=None) is not None
        or config.getoption("class_names", default=None) is not None
    )
    config.stash[is_full_run_key] = (
        any(
            Path(p).resolve() == PROJECT_ROOT / "tests"
            for p in config.option.file_or_dir
        )
        or len(config.option.file_or_dir) == 0
    )
    config.stash[save_test_logs_key] = config.getoption("save_test_logs", default=False)


def _extract_model_name(item):
    """Extract model name from a parametrized test item."""
    # Try to get from callspec params first (most reliable)
    if hasattr(item, "callspec") and hasattr(item.callspec, "params"):
        model_name = item.callspec.params.get("model_name")
        if model_name:
            return model_name

    # Fallback: parse from item name (e.g., "test_foo[model-name]")
    if "[" in item.name and "]" in item.name:
        try:
            # Extract content between first [ and last ]
            start = item.name.index("[")
            end = item.name.rindex("]")
            return item.name[start + 1 : end]
        except (ValueError, IndexError):
            pass

    return None


def _categorize_test_failure(test_file_name, model_name, arch, config):
    """Categorize test failure based on test file and update appropriate config list.

    Thread-safe: Uses lock to prevent race conditions in parallel execution.
    Falls back to "failed_test_models" if test file not in any specific category.
    """
    lock = config.stash[stash_lock_key]
    failure_categories = config.stash[failure_categories_key]

    # Check if test file is in any specific category
    for middle_part, test_files in TEST_FILE_CATEGORIES.items():
        if test_file_name in test_files:
            category_name = f"failed_{middle_part}_models"
            with lock:
                if model_name not in failure_categories[category_name][arch]:
                    logger.warning(
                        f"Model {model_name} failed test in {test_file_name} (category: {category_name})"
                    )
                    failure_categories[category_name][arch].append(model_name)
            return

    # Default category for general test failures
    with lock:
        if model_name not in failure_categories["failed_test_models"][arch]:
            logger.warning(f"Model {model_name} failed a test")
            failure_categories["failed_test_models"][arch].append(model_name)


def pytest_runtest_makereport(item, call):
    if call.when != "call":
        return

    model_name = _extract_model_name(item)
    if not model_name:
        return

    config = item.session.config
    arch = get_arch(model_name)
    lock = config.stash[stash_lock_key]

    # Thread-safe: Add to tested_models
    tested_models = config.stash[tested_models_key]
    with lock:
        if model_name not in tested_models[arch]:
            tested_models[arch].append(model_name)

    if call.excinfo is None:
        return

    is_skip = call.excinfo.errisinstance(Skipped)
    formatted_tb = str(call.excinfo.getrepr(style="long", chain=True))
    error_message = call.excinfo.exconly()

    # Thread-safe: Add to errors or skips
    with lock:
        if is_skip:
            lst = config.stash[skips_key][arch].setdefault(model_name, [])
        else:
            lst = config.stash[errors_key][arch].setdefault(model_name, [])

        lst.append(
            {
                "test_name": item.name,
                "test_file": item.path.name,
                "error": error_message,
                "error_traceback": formatted_tb,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    # Categorize failures (skip skipped tests)
    if not is_skip:
        _categorize_test_failure(item.path.name, model_name, arch, config)


def pytest_deselected(items):
    if items:
        items[0].session.config.stash[has_deselected_key] = True


@pytest.hookimpl(tryfirst=True)
def pytest_testnodedown(node, error):
    """Hook called when a pytest-xdist worker node finishes.

    Aggregates test results from worker nodes into the master process.
    This runs on the master node when a worker finishes.
    """
    if not hasattr(node, "workeroutput"):
        # If worker crashed (error != None), workeroutput won't exist - that's expected
        # pytest-xdist already handles the crash, so we just skip merging
        worker_id = (
            getattr(node, "gateway", {}).id if hasattr(node, "gateway") else "unknown"
        )
        if error is not None:
            logger.error(
                f"Worker {worker_id} crashed (error should be reported by pytest-xdist):\n{error}"
            )
            return
        # If no error but workeroutput missing, that's unexpected - fail loud
        raise RuntimeError(
            f"Worker {worker_id} finished without error but workeroutput is missing"
        )

    config = node.config
    # Get the lock to safely merge worker data
    lock = config.stash[stash_lock_key]

    with lock:
        # Merge tested_models
        worker_tested = node.workeroutput.get("tested_models", {})
        master_tested = config.stash[tested_models_key]
        for arch, models in worker_tested.items():
            if arch not in master_tested:
                master_tested[arch] = []
            for model in models:
                if model not in master_tested[arch]:
                    master_tested[arch].append(model)

        # Merge failure_categories
        worker_failures = node.workeroutput.get("failure_categories", {})
        master_failures = config.stash[failure_categories_key]
        for cat_name, cat_data in worker_failures.items():
            if cat_name not in master_failures:
                master_failures[cat_name] = defaultdict(list)
            for arch, models in cat_data.items():
                for model in models:
                    if model not in master_failures[cat_name][arch]:
                        master_failures[cat_name][arch].append(model)

        # Merge errors
        worker_errors = node.workeroutput.get("errors", {})
        master_errors = config.stash[errors_key]
        for arch, errors in worker_errors.items():
            if arch not in master_errors:
                master_errors[arch] = []
            master_errors[arch].extend(errors)

        # Merge skips
        worker_skips = node.workeroutput.get("skips", {})
        master_skips = config.stash[skips_key]
        for arch, skips in worker_skips.items():
            if arch not in master_skips:
                master_skips[arch] = []
            master_skips[arch].extend(skips)


def pytest_sessionfinish(session, exitstatus):
    """Hook called after whole test session finishes."""
    success = exitstatus <= 1
    config = session.config

    # Check if pytest_configure was called
    if is_full_run_key not in config.stash:
        logger.warning(
            "pytest_sessionfinish called without pytest_configure. Skipping status update."
            "You probably called that with --lf"
        )
        return

    # pytest-xdist: If this is a worker node, serialize data and send to master
    if hasattr(config, "workerinput"):
        # This is a worker - send our collected data to the master
        config.workeroutput["tested_models"] = dict(config.stash[tested_models_key])
        config.workeroutput["failure_categories"] = {
            cat_name: dict(cat_data)
            for cat_name, cat_data in config.stash[failure_categories_key].items()
        }
        config.workeroutput["errors"] = dict(config.stash[errors_key])
        config.workeroutput["skips"] = dict(config.stash[skips_key])
        return  # Workers don't write files - only the master does

    # If we get here, we're the master process (or running without xdist)
    is_full_run = config.stash[is_full_run_key]
    has_deselected = config.stash[has_deselected_key]
    is_model_specific = config.stash[is_model_specific_key]
    is_partial = not is_full_run or has_deselected or not success

    existing_data = {}
    if not is_partial:
        log_folder = PROJECT_ROOT / "data" / "test_logs"
        log_folder.mkdir(exist_ok=True, parents=True)

        status_file = log_folder / "latest_status.json"
        versioned_status_file = (
            log_folder / f"{TRANSFORMERS_VERSION}_{NNSIGHT_VERSION}_status.json"
        )

        if status_file.exists():
            try:
                with open(status_file, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Error loading {status_file}: {e}")

    try:
        new_status = _update_status(existing_data, config)
    except Exception as e:
        if existing_data == {} or is_model_specific:
            raise e
        logger.warning(f"Error updating status: {e}")
        new_status = _update_status({}, config)

    if not is_partial:
        tested_models = config.stash[tested_models_key]
        with open(status_file, "w") as f:
            json.dump(new_status, f, indent=2)
        with open(versioned_status_file, "w") as f:
            json.dump(new_status, f, indent=2)
        print(
            f"\nModels tested during this session: {tested_models}\n"
            f"Test status saved to: {status_file}\n"
            f"Versioned copy: {versioned_status_file}"
        )

    if config.stash[save_test_logs_key]:
        log_folder = PROJECT_ROOT / "data" / "test_logs"
        log_folder.mkdir(exist_ok=True)

        file_name = str(int(time.time()))
        if has_deselected:
            file_name += "_partial"
        log_entry_file = log_folder / f"{file_name}.json"
        print(f"Saving log entry to {log_entry_file}")

        log_entry = {
            "is_full_run": is_full_run,
            "has_deselected": has_deselected,
            "transformers_version": TRANSFORMERS_VERSION,
            "nnsight_version": NNSIGHT_VERSION,
            "errors": dict(config.stash[errors_key]),
            "status": new_status.get(TRANSFORMERS_VERSION, {}).get(NNSIGHT_VERSION, {}),
        }

        with open(log_entry_file, "w") as f:
            json.dump(log_entry, f, indent=2)


def _update_status(prev_status: dict, config):
    nnsight_unavailable_models = get_failed_models_from_status(
        load_test_loading_status()
    )
    transformers_section = prev_status.setdefault(TRANSFORMERS_VERSION, {})
    prev_nnsight_status = transformers_section.get(NNSIGHT_VERSION, {})

    # Get data from stash
    tested_models = rm_empty_list(config.stash[tested_models_key])
    failure_categories = config.stash[failure_categories_key]

    # Clean up failure categories
    cleaned_categories = {}
    for category_name, category_data in failure_categories.items():
        cleaned_categories[category_name] = rm_empty_list(category_data)

    nnsight_unavailable_models = rm_empty_list(nnsight_unavailable_models)
    is_model_specific = config.stash[is_model_specific_key]

    # Build new status structure with all failure categories
    new_status = {
        "fully_available_models": {},
        "nnsight_unavailable_models": nnsight_unavailable_models,
        "ran_tests_on": tested_models,
    }

    # Initialize availability keys for all specific categories dynamically
    for middle_part in TEST_FILE_CATEGORIES.keys():
        availability_key = f"no_{middle_part}_available_models"
        new_status[availability_key] = {}

    # Add all failure categories to status
    for category_name, category_data in cleaned_categories.items():
        new_status[category_name] = category_data

    # Compute all failures
    all_failed_tests_models = sum(
        cleaned_categories.get("failed_test_models", {}).values(), []
    )
    all_nnsight_unavailable_models = sum(nnsight_unavailable_models.values(), [])
    all_general_fails = set(all_failed_tests_models + all_nnsight_unavailable_models)

    # Collect all specific category failures (excluding default/general test failures)
    all_specific_fails = set()
    for category_name, category_data in cleaned_categories.items():
        if category_name != "failed_test_models":
            all_specific_fails |= set(sum(category_data.values(), []))

    all_fails = all_general_fails | all_specific_fails

    # Compute available models for each class
    for model_class in tested_models:
        fully_available = set(tested_models[model_class]) - all_fails
        if len(fully_available) > 0:
            new_status["fully_available_models"][model_class] = sorted(fully_available)

        # Compute partially available models (failed specific tests but not general ones)
        # Dynamically handle all specific categories from YAML
        for middle_part in TEST_FILE_CATEGORIES.keys():
            category_name = f"failed_{middle_part}_models"
            availability_key = f"no_{middle_part}_available_models"

            available_models = (
                set(cleaned_categories[category_name].get(model_class, []))
                - all_general_fails
            )
            if available_models:
                new_status[availability_key][model_class] = sorted(available_models)

    if is_model_specific:
        new_status = merge_partial_status(
            prev_nnsight_status, new_status, tested_models
        )

    new_status["last_updated"] = datetime.datetime.now().isoformat()
    new_status = sort_json_recursively(new_status, preserve_level=0)
    transformers_section[NNSIGHT_VERSION] = new_status
    return prev_status


@pytest.fixture(scope="session")
def failed_model_cache():
    """Cache of models that failed to load during the test session."""
    return set()


@pytest.fixture
def model_name(request):
    """Parametrized fixture providing test model names."""
    return request.param


@pytest.fixture
def llama_like_model_name(request):
    """Parametrized fixture for models with llama-like naming conventions."""
    return request.param


@pytest.fixture
def model(model_name, failed_model_cache, request):
    """Fixture providing StandardizedTransformer instances.

    If a model fails to load, it is cached and subsequent tests
    for that model are skipped.
    """
    # Check cache under lock to avoid races in parallel contexts
    lock = request.session.config.stash[stash_lock_key]
    with lock:
        if model_name in failed_model_cache:
            pytest.skip(f"Model {model_name} previously failed to load")

    try:
        return StandardizedTransformer(model_name)
    except Exception as e:
        # Update cache under lock to ensure thread safety
        with lock:
            failed_model_cache.add(model_name)

        # Thread-safe: Add to failed_test_models immediately
        config = request.session.config
        arch = get_arch(model_name)
        failure_categories = config.stash[failure_categories_key]

        with lock:
            if model_name not in failure_categories["failed_test_models"][arch]:
                failure_categories["failed_test_models"][arch].append(model_name)

        pytest.skip(f"Failed to load model {model_name}: {e}")


@pytest.fixture
def raw_model(model_name, failed_model_cache, request):
    """Fixture providing raw LanguageModel instances.

    If a model fails to load, it is cached and subsequent tests
    for that model are skipped.
    """
    # Check cache under lock to avoid races in parallel contexts
    lock = request.session.config.stash[stash_lock_key]
    with lock:
        if model_name in failed_model_cache:
            pytest.skip(f"Model {model_name} previously failed to load")

    try:
        return LanguageModel(model_name, device_map="auto")
    except Exception as e:
        # Update cache under lock to ensure thread safety
        with lock:
            failed_model_cache.add(model_name)

        # Thread-safe: Add to failed_test_models immediately
        config = request.session.config
        arch = get_arch(model_name)
        failure_categories = config.stash[failure_categories_key]

        with lock:
            if model_name not in failure_categories["failed_test_models"][arch]:
                failure_categories["failed_test_models"][arch].append(model_name)

        pytest.skip(f"Failed to load model {model_name}: {e}")
