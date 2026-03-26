vLLM Backend (Experimental)
===========================

.. meta::
   :llm-description: How to use nnterp with the vLLM inference backend for faster interventions on transformer models via StandardizedVLLM.

``nnterp`` provides a ``StandardizedVLLM`` class that wraps nnsight's ``VLLM`` backend with the same standardized interface as ``StandardizedTransformer``. This allows you to run interventions using vLLM's optimized inference engine while keeping the same nnterp API.

.. warning::

   The vLLM+nnsight backend is **experimental** and may produce incorrect results.
   Always verify outputs against the HuggingFace backend before relying on them.

Setup
-----

You need vLLM installed at the version required by your nnsight installation. Check the required version with:

.. code-block:: python

   from nnsight import NNS_VLLM_VERSION
   print(NNS_VLLM_VERSION)  # e.g. "0.15.1"

Then install it:

.. code-block:: bash

   pip install vllm==<version>

Loading a Model
---------------

Using ``load_model`` (recommended):

.. code-block:: python

   from nnterp import load_model

   model = load_model(
       "meta-llama/Llama-2-7b-hf",
       use_vllm=True,
       allow_experimental_vllm=True,
   )

Or directly:

.. code-block:: python

   from nnterp.standardized_vllm import StandardizedVLLM

   model = StandardizedVLLM(
       "meta-llama/Llama-2-7b-hf",
       allow_experimental_vllm=True,
   )

The ``allow_experimental_vllm=True`` flag is required to acknowledge the experimental status. A ``UserWarning`` is emitted on every initialization.

Standardized Interface
----------------------

``StandardizedVLLM`` provides the same accessors as ``StandardizedTransformer``:

.. code-block:: python

   # Model properties
   model.num_layers
   model.hidden_size
   model.vocab_size
   model.is_vllm  # True

   # Layer accessors (inside model.trace())
   model.layers_output[i]
   model.layers_input[i]
   model.attentions_output[i]
   model.mlps_output[i]

   # Module accessors
   model.embed_tokens
   model.ln_final
   model.lm_head

Interventions
-------------

Interventions work the same way as with ``StandardizedTransformer``:

.. code-block:: python

   import torch

   # Read activations
   with model.trace("The Eiffel Tower is in the city of"):
       layer_5_out = model.layers_output[5].save()
       logits = model.logits.output.save()  # Note: .output required for vLLM

   # Skip layers
   with model.trace("The Eiffel Tower is in the city of"):
       model.skip_layer(3)
       skipped_logits = model.logits.output.save()

   # Steering
   steering_vector = torch.randn(model.hidden_size)
   with model.trace("The Eiffel Tower is in the city of"):
       model.steer(layers=[2, 3, 4], steering_vector=steering_vector, factor=1.5)
       steered_logits = model.logits.output.save()

   # Sampled tokens
   with model.trace("The Eiffel Tower is in the city of"):
       sampled = model.samples.output.save()  # shape: (1, 1), dtype: int32

Generation
----------

Use ``.trace()`` for single forward pass (like ``StandardizedTransformer.trace()``) and
``.generate()`` for multi-token generation (like ``StandardizedTransformer.generate()``):

.. code-block:: python

   # Single forward pass (default)
   with model.trace("The Eiffel Tower is in the city of"):
       logits = model.logits.output.save()

   # Multi-token generation
   with model.generate("The Eiffel Tower is in the city of", max_new_tokens=10):
       logits = model.logits.output.save()
       tokens = model.samples.output.save()

``.generate()`` accepts the same sampling params as vLLM (``temperature``, ``top_p``,
``stop``, etc.) as keyword arguments. ``max_new_tokens`` is translated to vLLM's
``max_tokens`` automatically.

Differences from StandardizedTransformer
-----------------------------------------

**Logits access**: In ``StandardizedTransformer``, logits are at ``model.output.logits``
(accessible via the ``model.logits`` property). In ``StandardizedVLLM``, ``model.logits``
is a ``WrapperModule`` envoy inherited from nnsight's ``VLLM`` — use ``model.logits.output``
to get the logits tensor. ``model.logits.input`` is the same tensor (pass-through).

Additionally, ``model.samples.output`` provides the sampled token IDs (shape ``[1, 1]``,
dtype ``int32``).

**Steering implementation**: vLLM inference tensors don't support in-place operations.
``StandardizedVLLM.steer()`` uses ``clone()`` + assignment instead of ``+=``. This is handled
automatically — the API is the same.

**Input handling**: ``.trace()`` defaults to ``max_tokens=1`` (single forward pass).
Use ``.generate(max_new_tokens=N)`` for multi-token generation. HuggingFace's
``max_new_tokens`` is automatically translated to vLLM's ``max_tokens``.

**Tensor parallelism**: ``tensor_parallel_size`` defaults to ``torch.cuda.device_count()``.

**Prefix caching**: Disabled by default. Enabling it (``enable_prefix_caching=True``) is dangerous
as interventions from previous requests may leak through the KV cache. Use
``force_dangerous_prefix_caching=True`` to override.

Unsupported Features
--------------------

The following features are not available with the vLLM backend:

- ``model.input_ids`` — vLLM uses flattened inputs without padding
- ``model.input_size`` — same reason
- ``model.attention_mask`` — not exposed in vLLM inputs
- ``model.add_prefix_false_tokenizer`` — tokenizer behavior may differ
- ``enable_attention_probs=True`` — attention probability tracing not supported

These raise ``NotImplementedError`` or ``ValueError`` with descriptive messages.

Known Limitations
-----------------

- The nnsight vLLM backend is under active development. Results may differ from the
  HuggingFace backend for the same model and prompt.
- Each ``tracer.invoke()`` handles one prompt. For multiple prompts, use separate invokes.
- The ``max_tokens`` parameter controls generation length. Setting ``max_tokens != 1`` triggers
  multi-step generation (similar to ``LanguageModel.generate()``).
- vLLM requires a specific version pinned by nnsight. Check ``nnsight.NNS_VLLM_VERSION``.
