# Documentation Structure Overview

## Overview

The `nnterp` documentation is built using Sphinx with the PyData theme and hosted on GitHub Pages at https://ndif-team.github.io/nnterp/. The docs present `nnterp` as a nnsight wrapper specifically for transformer models, providing a standardized interface across architectures.

## Configuration (conf.py)

- **Theme**: pydata_sphinx_theme with dark mode default
- **Extensions**:
  - `sphinx.ext.autodoc` - Auto-generate API docs from docstrings
  - `sphinx.ext.napoleon` - NumPy/Google style docstring support
  - `sphinx_copybutton` - Copy buttons on code blocks
  - `sphinx_design` - Bootstrap components
  - `nbsphinx` - Jupyter notebook support
  - `sphinx.ext.viewcode` - Source code links
- **Build**: Standard Sphinx Makefile setup (`make html` to build)

## Documentation Structure

### Entry Point (index.rst)

Main landing page that:
- Positions nnterp as a nnsight wrapper for transformers (NOT as a replacement)
- Emphasizes that users **need to know NNsight** to use nnterp
- Highlights key differences from transformer_lens (HuggingFace vs custom implementation)
- Shows quick example of standardized API
- Organizes docs into 3 main sections:
  1. Getting Started
  2. User Guide (6 pages)
  3. API Reference

### Getting Started Section

**quickstart.rst**
- 5-minute introduction
- Installation (basic + visualization extras)
- Loading models with StandardizedTransformer
- Basic activation access patterns
- Direct assignment examples
- Links to deeper guides

### User Guide Section (6 pages)

**1. basic-usage.rst**
- Standardized interface explanation (llama naming convention)
- Module tree visualization
- Loading models across architectures
- Accessing module I/O (layers, attention, MLP)
- Skip layers functionality
- Built-in methods (project_on_vocab, steer)

**2. interventions.rst**
- Analysis methods:
  - Logit lens
  - Patchscope with TargetPrompt
- Brief code examples for each intervention type

**3. model-validation.rst**
- Automatic testing system that runs on model load
- What nnterp guarantees vs cannot guarantee
- Trade-offs:
  - Model dispatch (device_map="auto")
  - Attention implementation (eager vs sdpa/flash)
- Manual testing commands
- Version compatibility checks

**4. adding-model-support.rst**
- Target standardized structure diagram
- RenameConfig usage for custom architectures
- Path-based renaming
- Multiple alternative names support
- Real GPT-J example walkthrough
- Attention probabilities implementation (optional)
- Troubleshooting common errors
- Testing custom configurations

**5. advanced-features.rst**
- Attention probabilities access (requires enable_attention_probs=True)
- Prompt utilities (Prompt class, target token tracking)
- Visualization functions:
  - plot_topk_tokens
  - prompts_to_df
  - Target evolution plots
- Combining interventions with target tracking

**6. nnsight-best-practices.rst**
- **Critical**: Execution order requirements (must access in forward pass order)
- Gradient computation patterns:
  - Basic backward passes
  - Multiple backward passes with retain_graph
- Performance optimization:
  - tracer.stop() to skip unnecessary computation
  - Caching activations (NNsight 0.5+)

**7. nnsight-utils.rst**
- Standalone utility functions for regular nnsight models
- Requirements (llama/gemma naming conventions)
- Layer access helpers
- Projection utilities
- Activation collection functions
- Recommendation: Use StandardizedTransformer instead for robustness

### API Reference Section

**api.rst**
Auto-generated API documentation from docstrings for:
- **Main modules**:
  - standardized_transformer
  - interventions
  - prompt_utils
  - display
  - nnsight_utils
- **Internal modules**:
  - utils
  - rename_utils

Uses Sphinx autodoc to extract:
- Members
- Undocumented members
- Inheritance info
- Source links

### Supporting Files

**changelog.rst**
- Currently minimal
- Tracks notable changes
- Unreleased section with initial documentation setup

**glossary.md**
- Brief glossary of NNsight-specific terms
- Currently contains: "Dispatch" definition

**Makefile**
- Standard Sphinx build system
- Key commands: `make html`, `make clean`

**llms_header.txt**
- Header content for auto-generated llms.txt
- Contains title and overview section
- Editable without touching Python code

**llms.txt** (auto-generated)
- LLM-readable documentation index for AI agents
- Auto-generated during Sphinx build from RST metadata
- Links to `/_sources/*.rst.txt` files for direct RST access
- Follows llms.txt standard for AI agent navigation

**_static/** and **_build/**
- Static assets and build output directories

## Key Documentation Patterns

### Tone and Style
- Factual and concise (demo.py tone)
- Avoids verbose language ("core philosophy", etc.)
- Clear about limitations (nnterp is NOT a nnsight replacement)

### Code Examples
- **CRITICAL**: All examples respect nnsight execution order (access layers in forward pass order)
- Show practical patterns, not just syntax
- Include shape comments where helpful
- Demonstrate both correct and incorrect patterns (in best practices)

### Content Organization
- Progressive complexity: quickstart → basic → advanced → reference
- Heavy cross-linking between related topics
- Real examples (GPT-J) in addition to generic patterns
- Clear separation of "what works" vs "what's guaranteed"

## LLM-Readable Documentation (llms.txt)

The documentation includes an auto-generated `llms.txt` file for LLM agents to navigate the docs efficiently.

### How It Works

1. **Metadata in RST files**: Each RST file includes a `.. meta::` directive with `:llm-description:`
   ```rst
   Title
   =====
   
   .. meta::
      :llm-description: Brief description for LLM agents
   
   Content...
   ```

2. **Auto-generation**: `conf.py` contains a `setup()` hook that:
   - Parses `index.rst` toctree structure to discover all pages
   - Extracts titles and `:llm-description:` from each RST file
   - Reads header from `llms_header.txt`
   - Generates `llms.txt` with links to `/_sources/*.rst.txt`

3. **Build integration**: 
   - Runs automatically on every Sphinx build
   - `llms.txt` added to `html_extra_path` so it's copied to build output

### Adding New Pages

When adding a new documentation page:

1. Add it to the appropriate toctree in `index.rst`
2. Include `.. meta::` directive with `:llm-description:` after the title
3. The page will automatically appear in `llms.txt` on next build

### Editing the Overview

To change the header/overview section, edit `docs/llms_header.txt` directly.

## Writing New Documentation

When adding/editing docs:

1. **Add metadata**: Include `.. meta::` directive with `:llm-description:` after title for llms.txt generation

2. **Follow execution order**: All code examples MUST access components in forward pass order
   - ✅ `layers_output[1]` before `layers_output[2]`
   - ✅ `attentions_output[i]` before `layers_output[i]`
   - ❌ Never access later layers before earlier ones

3. **Use demo.py tone**: Keep explanations factual and concise

4. **Position correctly**: nnterp is a nnsight wrapper for transformers, not a general MI tool

5. **Show shapes**: Include tensor shape comments where helpful for understanding

6. **Link liberally**: Cross-reference related sections with `:doc:` directives

7. **Test examples**: All code should be runnable (or clearly marked as pseudocode)

8. **Emphasize requirements**: Make clear when features require specific flags (enable_attention_probs, etc.)

