``nnterp`` - NNsight wrapper for Large Language Models
======================================================

``nnterp`` is a tool for researchers learning or using `NNsight <https://github.com/ndif-team/nnsight>`_. 

Similar to `transformer_lens <https://github.com/TransformerLensOrg/TransformerLens>`_, ``nnterp`` provides a standardized interface for all transformer models. The main difference is that ``nnterp`` uses the HuggingFace implementation through nnsight, while transformer_lens uses its own implementation. This means ``nnterp`` preserves the original model behavior and supports more architectures.

You can install ``nnterp`` using pip:

.. code-block:: bash

   pip install nnterp
   # or for latest improvements
   pip install git+https://github.com/Butanium/nnterp/

**You need to know NNsight to use nnterp.** ``nnterp`` provides a standardized interface for transformer models and common interventions, making it easier to work with different architectures. But for anything complex, you'll still need NNsight directly.

Note that ``nnterp`` doesn't support all models either, since NNsight itself doesn't support all architectures. Additionally, because different models use different naming conventions, ``nnterp`` doesn't support all HuggingFace models, but it does support a good portion of them. When a model is loaded in nnterp, automatic tests are performed to verify that the model has been correctly renamed and that nnterp's hooks return the expected shapes. This means that even if an architecture hasn't been officially tested, the simple fact that it loads successfully indicates it's probably working correctly.

``nnterp`` is not a replacement for NNsight - it's an additional tool that researchers can use alongside NNsight for transformer analysis.

What ``nnterp`` provides:

- **Unified naming**: ``model.layers_output[5]`` works for GPT-2, LLaMA, Gemma, etc.
- **Vision-language models**: ``load_model()`` auto-detects VLMs and returns the right wrapper
- **Common interventions**: logit lens, patchscope, steering built-in
- **HuggingFace compatibility**: Uses original model implementations via nnsight

Quick example:

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   model = StandardizedTransformer("gpt2")  # or any transformer
   with model.trace("Hello"):
       layer_5_out = model.layers_output[5]
       model.layers_output[10] = layer_5_out  # same API for all models

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   getting-started/quickstart

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   basic-usage
   interventions
   model-validation
   adding-model-support
   advanced-features
   nnsight-best-practices
   nnsight-utils

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api

