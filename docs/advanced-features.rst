Advanced Features
=================

.. meta::
   :llm-description: Advanced features including attention probabilities access, using nnterp's renaming with LanguageModel or NNsight classes (without using StandardizedTransformer) or to access a model's PyTorch modules directly, prompt utilities with target token tracking, and visualization functions for plotting logits and token evolution.

Attention Probabilities
-----------------------

For supported models, access attention probabilities. You must enable them when loading the model by setting `enable_attention_probs=True`:

.. code-block:: python

   from nnterp import StandardizedTransformer

   # Load model with attention probabilities enabled
   model = StandardizedTransformer("gpt2", enable_attention_probs=True)

   with model.trace("The cat sat on the mat"):
       # Access attention probabilities for layer 5
       attn_probs = model.attention_probabilities[5].save()
       # Shape: (batch, heads, seq_len, seq_len)

       # Modify attention patterns
       attn_probs[:, :, :, 0] = 0  # Remove attention to first token
       attn_probs /= attn_probs.sum(dim=-1, keepdim=True)  # Renormalize

       modified_logits = model.logits.save()

Check what's happening:

.. code-block:: python

   model.attention_probabilities.print_source()

On hybrid models (Qwen3-Next, Qwen3.5, Qwen3.6), ``attention_probabilities[i]`` is
only defined for ``i`` in ``model.attention_layers`` and raises a ``RenamingError``
on a linear-attention (Gated DeltaNet) layer. The load-time validation and
``print_source()`` use the first softmax-attention layer.

Using nnterp's renaming with ``LanguageModel`` or ``NNsight`` classes
---------------------------------------------------------------------

Even if it is recommended to use ``StandardizedTransformer``, as it abstract some of the quirks of HuggingFace implementations, ``nnterp`` still provides a low level renaming functionality that allows you to rename ``LanguageModel`` or ``NNsight`` classes **and even to have unified access to the model's PyTorch modules.**.

Renaming a ``LanguageModel`` or ``NNsight`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`get_rename_dict` combined with the `rename` feature of ``NNsight`` will allow you to access the modules of the model with the same names as the ``StandardizedTransformer`` class:

.. code-block::

    your_model
    ├── embed_tokens
    ├── layers
    │   ├── self_attn
    │   └── mlp
    ├── ln_final
    └── lm_head

For that, you just need to pass the rename dictionary to the ``LanguageModel`` or ``NNsight`` class:

.. code-block:: python

   from nnterp import get_rename_dict
   from nnsight import LanguageModel, NNsight
   from transformers import AutoModelForCausalLM

   rename_dict = get_rename_dict()
   renamed_model = LanguageModel("gpt2", rename=rename_dict)
   # or
   hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
   renamed_model = LanguageModel(hf_model, rename=rename_dict)
   # or
   renamed_model = NNsight(hf_model, rename=rename_dict) # NNsight works with any PyTorch model
   print(renamed_model.layers, renamed_model.ln_final, renamed_model.lm_head)


Accessing the PyTorch modules using renaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For some reason, you might want a universal way to access the PyTorch modules of your model, without using ``NNsight`` interventions. For example if you want to attach them a hook, or just want to access the weights.
If you embed your model using the ``NNsight`` class with ``nnterp``'s renaming, you get universal access to the PyTorch modules with no overhead:

.. code-block:: python

   from nnsight import NNsight
   from nnterp import get_rename_dict
   from transformers import AutoModelForCausalLM
   import torch.nn as nn

   hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
   renamed_model = NNsight(hf_model, rename=get_rename_dict())
   layers_modules = renamed_model.layers._module
   assert isinstance(layers_modules, nn.ModuleList)

However, the ``._module`` syntax is not very convenient, so you can use the ``ModuleAccessor`` class to access the modules directly:

.. code-block:: python

   from nnterp import ModuleAccessor
   module_accessor = ModuleAccessor(hf_model)
   layers_modules = module_accessor.get_layers()
   # or
   layers_modules = module_accessor.layers
   assert isinstance(layers_modules, nn.ModuleList)



Prompt Utilities
----------------

Track probabilities of specific tokens:

.. code-block:: python

   from nnterp.prompt_utils import Prompt, run_prompts
   
   # Create prompt with target tokens
   targets = {
           "correct_answer": "Paris",
           "traps": ["London", "Madrid"],
           "longstring": "the country of France",
    }
   prompt = Prompt.from_strings(
       "The capital of France (not England or Spain) is",
       targets,
       model.tokenizer,
   )
   
   # Check what tokens are tracked for each target category
   for name, tokens in prompt.target_tokens.items():
       print(f"{name}: {model.tokenizer.convert_ids_to_tokens(tokens)}")
   
   # Get probabilities
   results = run_prompts(model, [prompt])
   for target, probs in results.items():
       print(f"{target}: {probs.shape}")  # Shape: (batch_size,)

``results`` is a dictionary that, for each target category, returns the sum of the probabilities of the first tokens of the target strings **with and without** a space at the beginning. For example for "traps", depending on the tokenizer, it could return the sum of the probabilities of "_London", "Lon", "_Mad" and "Ma".

Combined with Interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nnterp.interventions import logit_lens
   
   # Use interventions with target tracking
   results = run_prompts(model, prompts, get_probs_func=logit_lens)
   # Returns probabilities for each target category across all layers

Visualization
-------------

Plot top tokens at each layer:

.. code-block:: python

   from nnterp.display import plot_topk_tokens, prompts_to_df
   
   probs = logit_lens(model, "The capital of France is")
   
   # Interactive plot
   plot_topk_tokens(
       probs[0],  # First prompt
       model.tokenizer,
       k=5,
       title="Top 5 tokens at each layer"
   )

Prompt Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert prompts to DataFrame
   df = prompts_to_df(prompts, model.tokenizer)
   display(df)

Plot Target Evolution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import plotly.graph_objects as go
   
   # From logit lens results with target tracking
   results = run_prompts(model, prompts, get_probs_func=logit_lens)
   
   fig = go.Figure()
   for category, probs in results.items():
       fig.add_trace(go.Scatter(
           x=list(range(len(probs[0]))),
           y=probs[0].tolist(),
           mode="lines+markers",
           name=category
       ))
   
   fig.update_layout(
       title="Target Token Probabilities Across Layers",
       xaxis_title="Layer", 
       yaxis_title="Probability"
   )
   fig.show()