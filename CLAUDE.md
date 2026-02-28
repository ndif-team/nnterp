## Sphinx Documentation Guidelines
- **IMPORTANT: Respect nnsight execution order**: ALL code examples must access components in forward pass order (layers_output[1] before layers_output[2], attention before layer output of same layer, etc.)
- **Use demo.py tone**: Keep explanations factual and concise, avoid verbose language like "core philosophy" or "research-first design"
- **nnterp is for transformers**: Describe it as a nnsight wrapper specifically for transformer models, not general mechanistic interpretability

## Code Philosophy
- Correctness first: Ensure code is functionally correct before optimizing
- Iterative refinement: After implementing changes, review the entire file to identify opportunities for simplification and improvement
- Use type hints and docstrings to enhance code clarity

## Research Context
You assist me - a researcher - with a research oriented library, not production systems. This context allows for specific approaches:
- Make reasonable assumptions based on common research practices and my instructions. Avoid writting fallbacks in case something is missing. THIS IS VERY IMPORTANT as you shouldn't create bloated code!
- Fail fast philosophy: Design code to crash immediately when assumptions are violated rather than silently handling errors. This means that you should only use try/catch blocks if it explicitely benefits the code logic. No need to state this in comments. DON'T WRITE FALLBACKS FOR NON-COMMON INPUTS! Instead write asserts for you assumptions. This is very important!
        - Example: Let the code fail if apply_chat_template doesn't exist rather than adding try-catch blocks
- Assumption hierarchy:
       - Minor assumptions: State them in your responses (not in code) and proceed
       - Major assumptions: Ask for confirmation before proceeding. Depending on the severity state them in code using comments.
- If you are working with tensors, INCLUDE SHAPE ASSERTIONS in your code. For example, you could write "assert x.shape = (batch_size, self.dictionary_size)".
- It is crucial that you only implement what I asked for. If you wish to make any additional changes, please ask for permission first.
- It is fine if you fail to implement something. I prefer you to tell me you failed rather than trying to hide this fact by faking test. Don't reward hack, Claude :<.

## Test Philosophy
- Tests should FAIL! When writing tests, you should NEVER use try except blocks. Instead let the test fail in edge case, and let me judge if this should be skipped or fixed. NEVER EVER AGAIN REWARD HACKING WITH TRY CATCH IN TEST CLAUDE, OK???
- Never try to fix a test by considering it an edge case and skipping it. I consider that reward hacking. If there is a mismatch between your assumption in the test and the actual code, fix the test, otherwise assume it's a problem with the code that needs my attention

## Development Commands

### Package Management
- `uv install` - Install dependencies
- `uv run python -m pytest` - Run all tests
- `uv run python -m pytest tests/test_interventions.py` - Run specific test file
- `uv run python -m pytest tests/test_interventions.py::test_logit_lens` - Run specific test

### Code Quality
- `uv run black .` - Format code with Black (line length 88)
- `uv run python -m build` - Build package for distribution

### Documentation
- `cd docs && make html` - Build Sphinx documentation
- `cd docs && make clean` - Clean documentation build files

## Architecture Overview

nnterp is a mechanistic interpretability library built on top of nnsight, providing a unified interface for transformer analysis through several key components:

### Core Components

**nnsight** `nnterp` is built on top of `nnsight`. A very important thing about `nnsight` is that interventions in a trace **MUST BE WRITTEN IN ORDER**. This means e.g. you can't access the output of a layer and then access its input / its mlp output.

**StandardizedTransformer** (`standardized_transformer.py`)
- Unified interface for different transformer architectures (extends `nnsight.LanguageModel`)
- Standardizes module naming across models (layers, attention, MLP components)
- **Primary model loading method**: Use `StandardizedTransformer("model_name")` instead of deprecated `load_model()`
- **Attention probabilities**: Opt-in with `enable_attention_probs=True` (automatically sets `attn_implementation="eager"`)

**Key Accessors**:
- `layers_input[i]` / `layers_output[i]` - Layer I/O
- `attentions[i]` / `attentions_input[i]` / `attentions_output[i]` - Attention modules
- `mlps[i]` / `mlps_input[i]` / `mlps_output[i]` - MLP modules
- `token_embeddings` - Token embedding layer (read/write)
- `logits` - Final model logits
- `next_token_probs` - Softmax of last token logits
- `input_ids`, `attention_mask`, `input_size` - Input tensor accessors

**Key Methods**:
- `skip_layer(layer_idx)` / `skip_layers(layer_indices)` - Skip layer computation
- `project_on_vocab(hidden_state)` - Project to vocabulary space
- `get_topk_closest_tokens(hidden_state, k)` - Get k closest tokens

**Intervention Framework** (`interventions.py`)

**Key Functions**:
- `logit_lens(model, prompts, token_idx=-1)` - Get next token probabilities at each layer
  - Returns shape: `(num_prompts, num_layers, vocab_size)`
- `patchscope_lens(model, target_prompts, source_prompts, layer_to_patch)` - Replace hidden states and observe output
  - Uses `TargetPrompt(prompt, index_to_patch)` to specify patching location
- `patchscope_generate(model, target_prompt, source_prompt, layer_to_patch, max_new_tokens)` - Generate with patched states
- `patch_object_attn_lens(model, prompts, object_token_idx, attention_layer)` - Attention-based patching

**Helper Functions**:
- `repeat_prompt(prompt, n_times)` - Create repeated prompts for patchscope
- `it_repeat_prompt(prompt, n_times)` - Iterator version

**Utilities** (`nnsight_utils.py`)
- **Activation Collection**:
  - `get_token_activations(model, prompts, token_idx, get_activations)` - Single batch, most efficient for small data, uses `tracer.stop()` for memory optimization
  - `collect_token_activations_batched(model, prompts, token_idx, get_activations, batch_size)` - Multiple batches with session management for large datasets
  - Both support custom extraction via `get_activations` callable (e.g., `lambda m: m.layers_output[5]`)
- **Layer Access**: `get_layers()`, `get_num_layers()`, `get_layer()`, `get_layer_input/output()`
- **Component Access**: `get_attention()`, `get_attention_output()`, `get_mlp()`, `get_mlp_output()`
- **Projection**: `project_on_vocab()`, `get_next_token_probs()`, `compute_next_token_probs()`
- **Note**: Most functions work with both `StandardizedTransformer` and raw `LanguageModel`

**Prompt Management** (`prompt_utils.py`)
- `Prompt` class for tracking target tokens
  - `from_strings(prompt, targets_dict, tokenizer)` - Create from string descriptions
  - `get_target_probs(probs)` - Extract probabilities for specific targets
  - `has_no_collisions()` - Check for token collisions
- `get_first_tokens(tokenizer, word)` - Handles both "word" and " word" tokenization variants
- `run_prompts(model, prompts)` - Batch process with target tracking

**Module Renaming** (`rename_utils.py`)
- `RenameConfig` - Dataclass for model-specific module renaming configuration
- `get_rename_dict()` - Generate renaming dictionary for a model
- `check_model_renaming()` - Validate module standardization after renaming
- **Supported Architectures**: OPT, Mixtral, Bloom, GPT-2, Qwen2Moe, Dbrx, GPT-J, LLaMA, Qwen3, Qwen2
- Includes attention probability accessors for different architectures

**Display** (`display.py`, optional `[display]` dependency)
- `plot_topk_tokens()` - Plotly heatmap visualization of top-k tokens across layers
- `prompts_to_df()` - Convert Prompt objects to pandas DataFrame

### Module Relationships

```
StandardizedTransformer
  ├── rename_utils (module renaming config)
  ├── nnsight_utils (activation collection)
  └── utils (TraceTensor type, DummyCache)

interventions.py
  └── nnsight_utils

prompt_utils.py
  ├── nnsight_utils
  └── standardized_transformer

display.py (optional)
  └── prompt_utils
```

**Key Pattern**: Most utility functions accept either `StandardizedTransformer` OR raw `LanguageModel` through type unions.

### Critical Conventions

**Type Aliases** (`utils.py`):
- `TraceTensor = Union[torch.Tensor, nnsight.envoy.Envoy]` - Used throughout for tensors that may be traced
- `GetModuleOutput = Callable[[LanguageModel], TraceTensor]` - Function type for extracting activations

**Device Management**:
- Results moved to CPU by default for memory efficiency
- Use `.to(device)` for explicit device placement when needed

**Execution Order** (nnsight constraint):
- Interventions MUST be written in forward-pass order within `model.trace()` context
- Cannot access layer output then layer input of the same layer
- Example valid order: `layers_input[0]` → `attentions_output[0]` → `mlps_output[0]` → `layers_output[0]` → `layers_input[1]`

**Token Handling** (`prompt_utils.py`):
- `get_first_tokens(tokenizer, word)` automatically checks both "word" and " word" variants
- Returns the first token ID found, prioritizing the variant without leading space
- Critical for handling different tokenizer behaviors across models

### Important Implementation Details

**StandardizedTransformer Initialization**:
- Automatically calls `get_rename_dict()` and renames modules on initialization
- Runs validation via `check_model_renaming()` to ensure standardization succeeded
- Extends `nnsight.LanguageModel`, so inherits `.trace()`, `.generate()`, `.scan()` contexts

**Activation Collection Strategy**:
- `get_token_activations()` is most efficient for single batch (uses `tracer.stop()` to halt forward pass early)
- `collect_token_activations_batched()` uses nnsight sessions for batching across multiple prompts
- Both return tensors on CPU by default to save GPU memory

**Module Renaming Configuration**:
- Each architecture has architecture-specific constants in `rename_utils.py` (e.g., `MODEL_NAMES`, `LAYER_NAMES`)
- Some architectures have custom attention probability accessors (e.g., `bloom_attention_prob_source()`)
- Renaming failures are logged but don't necessarily crash (allows partial functionality)

**Test Infrastructure Design**:
- `failed_model_cache` in `conftest.py` prevents re-attempting known failed models during test runs
- Thread locks ensure pytest-xdist compatibility for parallel testing
- Custom pytest options allow targeted testing of specific models/classes

### Key Design Patterns

1. **Batched Processing**: Most functions support both single inputs and batched operations
2. **Fail-Fast Philosophy**: Use assertions for shape validation, no silent error handling
3. **Standardized Interfaces**: Consistent API across different model architectures
4. **Context Management**: Heavy use of `model.trace()` context for interventions
5. **Memory Optimization**: `get_token_activations()` uses `tracer.stop()` to avoid full forward pass

### Test Structure

**Location**: `nnterp/tests/`

**Test Files**:
- `test_interventions.py` - Core intervention methods (logit lens, patchscope, steering)
- `test_model_renaming.py` - Module standardization functionality
- `test_nnsight_utils.py` - Core utility functions
- `test_probabilities.py` - Probability calculations
- `test_prompt_utils.py` - Prompt and target token handling

**Available Fixtures** (`conftest.py`):
- `model_name` - Parametrized fixture with test model names (e.g., "gpt2", "Maykeye/TinyLLama-v0")
- `llama_like_model_name` - Parametrized fixture for LLaMA-like models
- `model` - `StandardizedTransformer` instance with error handling and caching
- `raw_model` - Raw `LanguageModel` instance without standardization
- `failed_model_cache` - Session-scoped cache of failed models (avoids retrying known failures)

**Pytest Options**:
- `--model-names MODEL1,MODEL2` - Test specific model names
- `--class-names CLASS1,CLASS2` - Test specific model classes
- `--save-test-logs` - Save test results to data directory

**Test Infrastructure**:
- Pytest-xdist compatible with thread-safe state management
- Results saved to `data/test_loading_status.json` for tracking compatibility across versions
- Uses thread locks for parallel test execution safety

### Data Directory

**Location**: `nnterp/data/`

Contains JSON files for tracking model compatibility and test results:
- `status.json` - Model compatibility status across transformers/nnsight versions
- `test_loading_status.json` - Which models successfully load with nnsight
- `toy_models_cache.json` - Cached information about small test models

These files are used for test result tracking and cross-version compatibility monitoring.

### Public API

**Exported from `nnterp` package** (`__init__.py`): `StandardizedTransformer`

All other functions/classes must be imported from their respective modules (e.g., `from nnterp.interventions import logit_lens`).

### Dependencies

- **Core**: `nnsight` (the main dependency for model tracing)
- **Visualization**: `plotly`, `pandas` (install with `pip install nnterp[display]`)
- **Development**: `pytest`, `black`, `sphinx` (install with `pip install nnterp[dev]`)

### Common Patterns

**Model Loading and Usage**:
```python
from nnterp import StandardizedTransformer
model = StandardizedTransformer("gpt2")
```

**Intervention Structure**:
```python
with model.trace(prompts) as tracer:
    # Access activations via standardized names
    activations = model.layers_output[layer_idx].save()
    # Apply interventions
    model.steer(layers=layer_idx, steering_vector=vector)
```

**Target Token Tracking**:
```python
from nnterp.prompt_utils import Prompt, run_prompts
prompts = [Prompt.from_strings("input", {"target": "expected"}, tokenizer)]
results = run_prompts(model, prompts)
```
