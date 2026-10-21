# Guide

- [Loading](loading.md): How StandardizedTransformer loads a checkpoint and picks its family.
- [Names](names.md): The one set of module names every family answers to.
- [Residual stream](values/residual-stream.md): layer_output, attention_output and mlp_output, and the identity that defines them.
- [Attention](values/attention.md): The queries, keys, values, scores, probabilities and head outputs inside attention.
- [The whole model](values/whole-model.md): logits, token_embeddings, next_token_probs, the inputs and the sizes.
- [Layouts](values/layouts.md): The one axis layout every value has, and how to read it.
- [Availability](values/availability.md): Which values a checkpoint has, and what happens when it lacks one.
- [Mixture of experts](architectures/mixture-of-experts.md): The router and the experts on every MoE family.
- [Gated DeltaNet](architectures/delta-net.md): The linear-attention mixer on Qwen3-Next, Qwen3.5, OLMo-Hybrid and Kimi-Linear.
- [Mamba](architectures/mamba.md): The selective-scan mixer on Mamba, Falcon-Mamba and Jamba.
- [Mamba-2](architectures/mamba-2.md): The state-space mixer on Mamba-2, Nemotron-H, Bamba and Falcon-H1.
- [Vision-language](architectures/vision-language.md): The vision tower, the projector and the image values on image-text-to-text checkpoints.
- [Methods](methods.md): steer, skip_layers, project_on_vocab and get_topk_closest_tokens.
- [Generation](generation.md): What the values mean under model.generate, step by step.
- [Helpers](helpers.md): Activations and next-token distributions over many prompts.
- [Remote](remote.md): Running a StandardizedTransformer trace on NDIF.
