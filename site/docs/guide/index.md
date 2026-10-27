# Guide

- [Loading](loading.md): how a checkpoint becomes a `StandardizedTransformer`.
- [Names](names.md): the module names every family answers to.
- [Residual stream](values/residual-stream.md): the stream leaving a block and what each sublayer adds to it.
- [Attention](values/attention.md): the tensors inside the attention module, from the queries to the head outputs.
- [The whole model](values/whole-model.md): the values on the root, from the input ids to the logits.
- [Layouts](values/layouts.md): the axis layout of every value.
- [Availability](values/availability.md): which values a checkpoint has.
- [Mixture of experts](architectures/mixture-of-experts.md): the router and the experts on every MoE family.
- [Gated DeltaNet](architectures/delta-net.md): the linear-attention mixer on Qwen3-Next, Qwen3.5, OLMo-Hybrid and Kimi-Linear.
- [Mamba](architectures/mamba.md): the selective-scan mixer on Mamba, Falcon-Mamba and Jamba.
- [Mamba-2](architectures/mamba-2.md): the state-space mixer on Mamba-2, Nemotron-H, Bamba and Falcon-H1.
- [Vision-language](architectures/vision-language.md): the image side of an image-text-to-text checkpoint.
- [Methods](methods.md): `steer`, `skip_layers`, `project_on_vocab` and `get_topk_closest_tokens`.
- [Generation](generation.md): the values under `model.generate`.
- [Helpers](helpers.md): activations and next-token distributions over many prompts.
- [Remote](remote.md): a `StandardizedTransformer` trace on NDIF.
