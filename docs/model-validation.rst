Model Validation & Testing
===========================

.. meta::
   :llm-description: Automatic testing system that runs on model load. Learn what nnterp guarantees, trade-offs with model dispatch and attention implementation, and manual testing commands.

``nnterp`` includes automatic validation to prevent silent failures and ensure model correctness. When you load a model, a series of fast tests run automatically to verify the model works as expected.

Automatic Testing System
-------------------------

When loading a ``StandardizedTransformer``, ``nnterp`` automatically runs tests to ensure:

- **Model renaming correctness**: All modules are properly renamed to the standardized interface
- **Module output shapes**: Layer outputs have expected shapes (batch_size, seq_len, hidden_size)
- **Attention probabilities**: If enabled, attention probabilities have correct shape (batch_size, num_heads, seq_len, seq_len), sum to 1 for each token, and modifying them changes model output

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   # Automatic tests run during model loading
   model = StandardizedTransformer("gpt2")
   # Tests passed: model is ready to use
   
   # If tests fail, you'll see detailed error messages
   model = StandardizedTransformer("unsupported-model")
   # Error: Could not find layers module...

What ``nnterp`` Guarantees
~~~~~~~~~~~~~~~~~~~~~~

``nnterp`` guarantees that:

- All models follow the standardized naming convention
- ``model.layers_output[i]`` returns tensors with expected shapes
- ``model.attention_probabilities[i]`` (if enabled) returns properly normalized attention matrices
- ``model.attentions_output[i]`` / ``model.mlps_output[i]`` never include the residual stream. Architectures that add it inside the attention/MLP module (BLOOM, MPT, DBRX) are remapped to the pre-residual submodule, unknown ones are rejected at load (see :doc:`adding-model-support`)

What ``nnterp`` Cannot Guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nnterp`` cannot guarantee:

- **Attention probabilities remain unmodified**: The model might apply additional transformations after the attention probabilities are computed but before they're used. Check ``model.attention_probabilities.print_source()`` to understand the exact hook location in the HuggingFace implementation.
- **Sublayer outputs are the exact residual contribution**: on architectures with post-sublayer layernorms outside the attention/MLP module (e.g. Gemma-2/3), the tensor added to the residual stream is the post-layernorm output, not ``attentions_output[i]`` / ``mlps_output[i]``. See `issue #53 <https://github.com/ndif-team/nnterp/issues/53>`_.
- **Perfect HuggingFace compatibility**: While ``nnterp`` uses original HuggingFace implementations, some edge cases might behave differently due to the renaming process.

.. code-block:: python

   # Check where attention probabilities are hooked
   model.attention_probabilities.print_source()

Trade-offs and Configuration
-----------------------------

The automatic testing system comes with some trade-offs:

Model Dispatch
~~~~~~~~~~~~~~

``nnterp`` automatically dispatches your model to available devices (``device_map="auto"``) during loading. This can be inconvenient if you don't want to load model weights immediately. However you can set ``allow_dispatch=False`` to disable this (but some tests won't be run).

Attention Implementation
~~~~~~~~~~~~~~~~~~~~~~~~

``nnterp`` uses the default HuggingFace attention implementation by default. To access attention probabilities, you must explicitly set ``enable_attention_probs=True``, which automatically configures the model to use ``attn_implementation="eager"`` since other implementations (like sdpa or flash attention) don't support attention pattern tracing.

.. code-block:: python

   # To access attention probabilities (slower but traceable)
   model = StandardizedTransformer(
       "gpt2",
       enable_attention_probs=True
   )

   # Use default HuggingFace implementation (faster, no attention tracing)
   model = StandardizedTransformer("gpt2")  # Uses default HF implementation

If you try to use both ``enable_attention_probs=True`` and a non-eager ``attn_implementation``, ``nnterp`` will raise an error:

.. code-block:: python

   # This will raise an error
   model = StandardizedTransformer(
       "gpt2",
       enable_attention_probs=True,
       attn_implementation="sdpa"  # Conflicts with enable_attention_probs
   )

Manual Testing
--------------

You can run tests manually for specific models or architectures:

.. code-block:: bash

   # Test specific models
   python -m nnterp run_tests --model-names "gpt2" "meta-llama/Llama-2-7b-hf"
   
   # Test using toy models of specific architectures (faster/cheaper)
   python -m nnterp run_tests --class-names "LlamaForCausalLM" "GPT2LMHeadModel"

This is useful when:

- You're using a different version of ``transformers`` or ``nnsight`` than officially tested
- You want to test a new model architecture before using it in research

Version Compatibility
----------------------

``nnterp`` checks if tests were run for your current ``nnsight`` and ``transformers`` versions. If not, it will warn you and suggest running manual tests.

The automatic testing system ensures that even if an architecture hasn't been officially tested, if it loads successfully, it's probably working correctly.