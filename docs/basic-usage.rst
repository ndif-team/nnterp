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
   │       └── down_proj
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

Accessing Module I/O
--------------------

Access layer inputs and outputs directly:

.. code-block:: python

   with model.trace("hello"):
       # Access layer outputs
       layer_5_output = model.layers_output[5]
       
   # Access attention, MLP, and down_proj outputs:
   with model.trace("hello"):
       attn_output = model.attentions_output[3]
       down_proj_output = model.down_projs_output[3]
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