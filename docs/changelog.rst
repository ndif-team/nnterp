Changelog
=========

All notable changes to this project will be documented in this file.

[Unreleased]
------------

Added
~~~~~
* Hybrid linear/softmax attention models (Qwen3-Next, Qwen3.5, Qwen3.6): the Gated DeltaNet mixer keeps its ``linear_attn`` name, ``model.attention_layers`` / ``model.linear_attention_layers`` list the two kinds of blocks, and the attention accessors raise a ``RenamingError`` on linear-attention layers
* ``StandardizedVLM`` wrapper for vision-language models (Qwen2-VL, Gemma-3, GLM-4v, etc.)
* ``load_model()`` entrypoint that auto-detects VLMs via ``detect_automodel()``
* ``detect_automodel()`` utility to determine the right AutoModel class from config
* Llama-4 support (``feed_forward`` MLP naming)
* Heterogeneous submodule type warnings (e.g. dense MLP vs MoE across layers)
* VLM test suite with auto-discovery from HuggingFace toy model collection
* Skip pattern comments in test config for better maintainability

Changed
~~~~~~~
* Accessors detect tuple outputs per layer (``LayerAccessor.returns_tuple(layer)``), so layers can be accessed in any order; the renaming checks and attention-probability validation run on the first softmax-attention layer
* ``remote=True`` keeps the checkpoint off the client: it sets ``allow_dispatch=False``, validates with ``scan()`` and sends no request to NDIF during construction; ``check_attn_probs_with_trace`` defaults to ``None`` (``True`` for a local model, ``False`` for a remote one)
* ``attn_implementation="eager"`` is accepted next to ``enable_attention_probs=True``; a non-eager value raises a ``ValueError``
* ``StandardizedVLM.allow_multimodal`` defaults to ``False`` (fail loud on heterogeneous layers)
* bfloat16 tolerance for attention probability checks (``1e-2`` instead of ``1e-5``)

Previous
~~~~~~~~
* Initial Sphinx documentation setup
* API documentation for all main modules
* Integration with Read the Docs theme