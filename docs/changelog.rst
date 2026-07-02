Changelog
=========

All notable changes to this project will be documented in this file.

[Unreleased]
------------

Added
~~~~~
* ``StandardizedVLM`` wrapper for vision-language models (Qwen2-VL, Gemma-3, GLM-4v, etc.)
* ``load_model()`` entrypoint that auto-detects VLMs via ``detect_automodel()``
* ``detect_automodel()`` utility to determine the right AutoModel class from config
* Llama-4 support (``feed_forward`` MLP naming)
* Heterogeneous submodule type warnings (e.g. dense MLP vs MoE across layers)
* VLM test suite with auto-discovery from HuggingFace toy model collection
* Skip pattern comments in test config for better maintainability

Changed
~~~~~~~
* ``StandardizedVLM.allow_multimodal`` defaults to ``False`` (fail loud on heterogeneous layers)
* bfloat16 tolerance for attention probability checks (``1e-2`` instead of ``1e-5``)

Previous
~~~~~~~~
* Initial Sphinx documentation setup
* API documentation for all main modules
* Integration with Read the Docs theme