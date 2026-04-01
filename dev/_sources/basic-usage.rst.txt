Basic Usage
===========

.. meta::
   :llm-description: Learn the standardized interface for transformer models. Covers model loading, accessing layer inputs/outputs, skip layers functionality, and built-in methods like project_on_vocab and steer.

Standardized Interface
----------------------

Different transformer models use different naming conventions. ``nnterp`` standardizes all models to use the llama naming convention:

.. code-block:: text

   StandardizedTransformer
   ├── embed_tokens
   ├── layers
   │   ├── self_attn
   │   └── mlp
   ├── ln_final
   └── lm_head

Loading Models
~~~~~~~~~~~~~~

.. code-block:: python

   from nnterp import StandardizedTransformer

   # These all work the same way
   model = StandardizedTransformer("gpt2")
   model = StandardizedTransformer("meta-llama/Llama-2-7b-hf")

   # Uses device_map="auto" by default
   print(model.device)
   # Access model configuration attributes
   print(f"number of layers: {model.num_layers}")
   print(f"hidden size: {model.hidden_size}")
   print(f"number of attention heads: {model.num_heads}")
   print(f"vocabulary size: {model.vocab_size}")

Or use ``load_model()`` which auto-detects the model type:

.. code-block:: python

   from nnterp import load_model

   model = load_model("gpt2")  # returns StandardizedTransformer
   vlm = load_model("Qwen/Qwen2-VL-2B-Instruct")  # returns StandardizedVLM

Vision-Language Models
~~~~~~~~~~~~~~~~~~~~~~

``StandardizedVLM`` wraps vision-language models with the same standardized interface.
It extends nnsight's ``VisionLanguageModel`` and supports image inputs via ``model.trace()``:

.. code-block:: python

   from nnterp import StandardizedVLM

   vlm = StandardizedVLM("Qwen/Qwen2-VL-2B-Instruct")

   # Text-only tracing works the same as StandardizedTransformer
   with vlm.trace("Hello world"):
       layer_out = vlm.layers_output[0].save()

   # All standardized accessors are available
   print(vlm.num_layers, vlm.hidden_size, vlm.vocab_size)

.. note::

   Some VLM architectures are not supported:

   - **Heterogeneous layers** (e.g. Mllama/Llama-3.2-Vision): cross-attention layers only activate with image inputs, causing errors during text-only tracing. Pass ``allow_multimodal=True`` to opt in if you know what you're doing.
   - **AltUp models** (e.g. Gemma 3n): use 4D hidden states instead of the standard 3D ``(batch, seq, hidden)`` shape. See `issue #35 <https://github.com/ndif-team/nnterp/issues/35>`_.

Accessing Module I/O
--------------------

Access layer inputs and outputs directly:

.. code-block:: python

   with model.trace("hello"):
       # Access layer outputs
       layer_5_output = model.layers_output[5]
       
   # Access attention and MLP outputs:
   with model.trace("hello"):
       attn_output = model.attentions_output[3]
       mlp_output = model.mlps_output[3]

Skip Layers
~~~~~~~~~~~

.. code-block:: python

   with model.trace("Hello world"):
       # Skip layer 1
       model.skip_layer(1)
       # Skip layers 2 through 3
       model.skip_layers(2, 3)

Use saved activations:

.. code-block:: python

   import torch

   with model.trace("Hello world") as tracer:
       layer_6_out = model.layers_output[6].save()
       tracer.stop()
   
   with model.trace("Hello world"):
       model.skip_layers(0, 6, skip_with=layer_6_out)
       result = model.logits.save()
    
    with model.trace("Hello world"):
        results_vanilla = model.logits.save()
    
    assert torch.allclose(results_vanilla, results_skipped)

Built-in Methods
----------------

Project to vocabulary (apply unembed ln_final and lm_head to an activation):

.. code-block:: python

   with model.trace("The capital of France is"):
       hidden = model.layers_output[5]
       logits = model.project_on_vocab(hidden)

Steering:

.. code-block:: python

   import torch

   steering_vector = torch.randn(768)  # gpt2 hidden size
   with model.trace("The weather today is"):
       model.steer(layers=[1, 3], steering_vector=steering_vector, factor=0.5)

You can target specific token positions or batch elements:

.. code-block:: python

   with model.trace(["The weather today is", "I feel very"]):
       model.steer(layers=1, steering_vector=steering_vector, token_positions=0)  # first token only
       model.steer(layers=1, steering_vector=steering_vector, batch_index=0)  # first prompt only
       model.steer(layers=1, steering_vector=steering_vector, batch_index=1, token_positions=[0, 1])