# %% [markdown]
# # Demo: nnterp Features Showcase
#
# This notebook demonstrates the key features of `nnterp`, which aims to offer a unified interface for all transformer models and give best `NNsight` practices for LLMs in everyone's hands.

# %% [markdown]
# ## 1. Standardized Interface
#
# Similar to [`transformer_lens`](https://github.com/TransformerLensOrg/TransformerLens), `nnterp` provides a standardized interface for all transformer models.
# The main difference is that `nnterp` still uses the huggingface implementation under the hood through `NNsight`, while transformer_lens uses its own implementation of the transformer architecture. However, each transformer implementation has its own quirks, such that `transformer_lens` is not able to support all models, and can sometimes have significant difference with the huggingface implementation.
#
# Note that `nnterp` doesn't support all models either, since `NNsight` itself doesn't support all architectures. Additionally, because different models use different naming conventions, `nnterp` doesn't support all HuggingFace models, but it does support a good portion of them. When a model is loaded in `nnterp`, automatic tests are performed to verify that the model has been correctly renamed and that `nnterp`'s hooks return the expected shapes. This means that even if an architecture hasn't been officially tested, the simple fact that it loads successfully indicates it's probably working correctly.
#
# The way it's implemented is based on the `NNsight` built-in renaming feature, to make all models look like the llama naming convention, without having to write `model.model`, namely:
# ```ocaml
# StandardizedTransformer
# ├── embed_tokens
# ├── layers
# │   ├── self_attn
# │   └── mlp
# ├── ln_final
# └── lm_head
# ```

# %%
from transformers import AutoModelForCausalLM

print(AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0"))
print(AutoModelForCausalLM.from_pretrained("gpt2"))

# %% [markdown]
# As you can see, the naming scheme of gpt2 is different from the llama naming convention.
# A simple way to fix this is to use the `rename` feature of `NNsight` to rename the gpt2 modules to the llama naming convention.

# %%
from nnsight import LanguageModel

model = LanguageModel(
    "gpt2",
    rename=dict(transformer="model", h="layers", ln_f="ln_final", attn="self_attn"),
)
print(model)
# Access the attn module as if it was a llama model
print(model.model.layers[0].self_attn)

# %% [markdown]
# You can see the that renamed modules are displayed like `(new_name)/old_name`. However, many models family have their own naming convention, `nnterp` has a global renaming scheme that should transform any model to the llama naming convention. The easiest way to use it is to load your model using the `StandardizedTransformer` class that inherits from `nnsight.LanguageModel`.

# %%
from nnterp import StandardizedTransformer

# You will see the `layers` module printed two times, it'll be explained later.
# Note: We use enable_attention_probs=True to enable attention pattern access
nnterp_gpt2 = StandardizedTransformer("gpt2", enable_attention_probs=True)
print(nnterp_gpt2)
# StandardizedTransformer also use `device_map="auto"` by default:
nnterp_gpt2.dispatch()
print(nnterp_gpt2.model.device)

# %% [markdown]
# Great! But I can see you at the back of the classroom, asking yourself:
# > "Why would you create a package that just pass the right dict to the `NNsight` `rename` feature?"
#
# And actually, I'm glad you asked! `StandardizedTransformer` and `nnterp` have a lot of other features, so bear with me!

# %% [markdown]
# ## 2. Accessing Modules I/O
# With `NNsight`, the most robust way to set the residual stream after layer 1 to be the residual stream after layer 0 for a LLama-like model would be:

# %%
from transformers import __version__ as TRANSFORMERS_VERSION
from packaging.version import parse

llama = LanguageModel("Maykeye/TinyLLama-v0")
# code for transformer "<4.54"
is_old_transformers = parse(TRANSFORMERS_VERSION) < parse("4.54")
if is_old_transformers:
    with llama.trace("hello"):
        llama.model.layers[1].output = (
            llama.model.layers[0].output[0],
            *llama.model.layers[1].output[1:],
        )
else:
    with llama.trace("hello"):
        llama.model.layers[1].output = llama.model.layers[0].output


# %% [markdown]
# Note that the following can cause issues:

# %%
with llama.trace("hello"):
    # can't do this because .output is a tuple

    # Can cause errors with gradient computation
    if is_old_transformers:
        # llama.model.layers[1].output[0] = llama.model.layers[0].output[0]
        llama.model.layers[1].output[0][:] = llama.model.layers[0].output[0]
    else:
        llama.model.layers[1].output[:] = llama.model.layers[0].output

if is_old_transformers:
    with llama.trace("hello"):
        # Can cause errors with opt if you do this at its last layer (thanks pytest)
        llama.model.layers[1].output = (llama.model.layers[0].output[0],)

# %% [markdown]
# `nnterp` makes this much cleaner:

# %%
# the version of transformers does not matter, the tuple vs not tuple stuff is handled internally
# First, you can access layer inputs and outputs directly:
with nnterp_gpt2.trace("hello"):
    # Access layer 5's output
    layer_5_output = nnterp_gpt2.layers_output[5]
    # Set layer 10's output to be layer 5's output
    nnterp_gpt2.layers_output[10] = layer_5_output

# You can also access attention and MLP outputs:
with nnterp_gpt2.trace("hello"):
    attn_output = nnterp_gpt2.attentions_output[3]
    mlp_output = nnterp_gpt2.mlps_output[3]

# %% [markdown]
# ## 3. `nnterp` Guarantees
#

# %% [markdown]
# When designing, `nnterp` I was very worried about silent failures, where you load a model, and then get an unexpected failure in your code downstream, or worst, it doesn't fail but give you fake results. To avoid this, when you load an `nnterp` model, a series of fast tests are run to ensure that:
# - The model has been correctly renamed
# - The model module output are of the expected shape
# - Attention probabilities have the right shape, sum to 1, and changing them changes the output
#
# This comes with the trade-off that `nnterp` will dispatch your model when you load it, which can be annoying if you don't want to load the model's weights. To access attention probabilities, you need to explicitly load your model with `enable_attention_probs=True`, which automatically sets `attn_implementation="eager"` since other implementations (like sdpa or flash attention) don't support attention pattern tracing.
#
# What `nnterp` can NOT guarantee:
# - The attention probabilities won't be modified by the model before being multiplied by the values. To ensure this, you can check `model.attention_probabilities.print_source()` (preferably in a notebook for markdown display) to understand where the attention probabilities are computed.
# - Huggingface's transformers sheringan w
#
# If youe model is not properly renamed, you can pass a `RenameConfig` to the `nnterp` constructor to rename the model. See more in the advanced usage section of this demo.
#
# On top of that, before releasing a new version of `nnterp`, a series of tests covering most architectures are performed. When you load a model, `nnterp` will check if tests were run for your `nnsight` and `transformers` versions, and will check the tests results for the class of your model. I chose to include the tests in the `nnterp` package, so that if your model architecture has not been tested / you use a different version of `nnsight` or `transformers`, you can run `python -m nnterp run_tests --model-names foo bar --class-names LlamaForCausalLM` to run the tests for your model. `--class-names` allow you to run the tests on a toy model of the same class as your model to make it cheaper and faster.

# %% [markdown]
# ## 4. Attention Probabilities
#
# For models that support it, you can access attention probabilities directly. You can check if a model supports it by calling `model.supports_attention_probabilities`.

# %%
import torch as th

nnterp_gpt2.tokenizer.padding_side = (
    "left"  # ensure left padding for easy access to the first token
)

with th.no_grad():
    with nnterp_gpt2.trace("The cat sat on the mat"):
        # Access attention probabilities for layer 5
        attn_probs_l2 = nnterp_gpt2.attention_probabilities[2].save()
        attn_probs = nnterp_gpt2.attention_probabilities[5].save()
        print(
            f"Attention probs shape will be: (batch, heads, seq_len, seq_len): {attn_probs.shape}"
        )
        # knock out the attention to the first token
        attn_probs[:, :, :, 0] = 0
        attn_probs /= attn_probs.sum(dim=-1, keepdim=True)
        corr_logits = nnterp_gpt2.logits.save()
    with nnterp_gpt2.trace("The cat sat on the mat"):
        baseline_logits = nnterp_gpt2.logits.save()

assert not th.allclose(corr_logits, baseline_logits)

sums = attn_probs_l2.sum(dim=-1)
# last dimension is the attention of token i to all other tokens, so should sum to 1
assert th.allclose(sums, th.ones_like(sums))

# %% [markdown]
# Under the hood this uses the new tracing system implemented in `NNsight v0.5` which allow to access most model intermediate variables during the forward pass. This means that if the `transformers` implementation were to change, this could break or give unexpected results, so it is recommended to use one of the tested versions of `transformers` and to check that the attention probabilities hook makes sense by calling `model.attention_probabilities.print_source()` if you want to use a different version of `transformers` / a architecture that has not been tested.

# %%
nnterp_gpt2.attention_probabilities.print_source()  # pretty markdown display in a notebook

# %% [markdown]
# ## 5. Builtin interventions
#
# `StandardizedTransformer` also provides convenient methods for common operations:

# %%
import torch as th

# Project hidden states to vocabulary using the unembed norm and lm_head
with nnterp_gpt2.trace("The capital of France is"):
    hidden = nnterp_gpt2.layers_output[5]
    logits = nnterp_gpt2.project_on_vocab(hidden)

# Skip layers entirely
with nnterp_gpt2.trace("Hello world"):
    # Skip layer 1
    nnterp_gpt2.skip_layer(1)
    # Skip layers 2 through 3 (inclusive)
    nnterp_gpt2.skip_layers(2, 3)

# This is useful if you want to start at a later layer than the first one
with nnterp_gpt2.trace("Hello world") as tracer:
    layer_6_out = nnterp_gpt2.layers_output[6].save()
    tracer.stop()  # avoid computations after layer 6

with nnterp_gpt2.trace("Hello world"):
    nnterp_gpt2.skip_layers(0, 6, skip_with=layer_6_out)
    half_half_logits = nnterp_gpt2.logits.save()

with nnterp_gpt2.trace("Hello world"):
    vanilla_logits = nnterp_gpt2.logits.save()

assert th.allclose(vanilla_logits, half_half_logits)  # they should be the same

# Direct steering
steering_vector = th.randn(768)  # gpt2 hidden size
with nnterp_gpt2.trace("The weather today is"):
    nnterp_gpt2.steer(layers=[1, 3], steering_vector=steering_vector, factor=0.5)

# Steer specific token positions or batch elements
with nnterp_gpt2.trace(["The weather today is", "I feel very"]):
    nnterp_gpt2.steer(layers=1, steering_vector=steering_vector, token_positions=0)
    nnterp_gpt2.steer(layers=1, steering_vector=steering_vector, batch_index=0)
    nnterp_gpt2.steer(
        layers=1, steering_vector=steering_vector, batch_index=1, token_positions=[0, 1]
    )

# %% [markdown]
# ## 6. Specific Token Activation Collection
#
# `nnterp` provides utilities for collecting activations efficiently:

# %%
from nnterp.nnsight_utils import (
    get_token_activations,
    collect_token_activations_batched,
)

# Collect activations for specific tokens
prompts = ["The capital of France is", "The weather today is"]
with nnterp_gpt2.trace(prompts) as tracer:
    # Get last token activations for all layers
    activations = get_token_activations(nnterp_gpt2, prompts, idx=-1, tracer=tracer)
    # activations shape: (num_layers, batch_size, hidden_size)

# For large datasets, use batched collection
large_prompts = ["Sample text " + str(i) for i in range(100)]
batch_activations = collect_token_activations_batched(
    nnterp_gpt2,
    large_prompts,
    batch_size=16,
    layers=[3, 9, 11],  # Only collect specific layers, default is all layers
    idx=-1,  # Last token (default)
)
print(f"Batched activations shape: {batch_activations.shape}")

# %% [markdown]
# ## 7. Prompt Utilities
#
# `nnterp` provides utilities for working with prompts and tracking probabilities of first tokens of certain strings. It tracks both the first token of "string" and " string".
#
# You can provide multiple string per category, the probabilities returned will be the sum of the probabilities of all the first tokens of the strings.

# %%
from nnterp.prompt_utils import Prompt, run_prompts

# Create prompts with target tokens to track
prompt1 = Prompt.from_strings(
    "The capital of France (not England or Spain) is",
    {
        "target": "Paris",
        "traps": ["London", "Madrid"],
        "longstring": "the country of France",
    },
    nnterp_gpt2.tokenizer,
)
for name, tokens in prompt1.target_tokens.items():
    print(f"{name}: {nnterp_gpt2.tokenizer.convert_ids_to_tokens(tokens)}")

prompt2 = Prompt.from_strings(
    "The largest planet (not Earth or Neptune) is",
    {"target": "Jupiter", "traps": ["Earth", "Neptune"], "longstring": "Palace planet"},
    nnterp_gpt2.tokenizer,
)
for name, tokens in prompt2.target_tokens.items():
    print(f"{name}: {nnterp_gpt2.tokenizer.convert_ids_to_tokens(tokens)}")

# Run prompts and get probabilities for target tokens
results = run_prompts(nnterp_gpt2, [prompt1, prompt2], batch_size=2)
print("Target token probabilities:")
for target, probs in results.items():
    print(f"  {target}: shape {probs.shape}")

# %% [markdown]
# ## 8. Interventions
#
# `nnterp` provides several intervention methods inspired by mechanistic interpretability research:

# %%
from nnterp.interventions import (
    logit_lens,
    patchscope_lens,
    TargetPrompt,
    repeat_prompt,
)

# Logit Lens: See predictions at each layer
prompts = ["The capital of France is", "The sun rises in the"]
probs = logit_lens(nnterp_gpt2, prompts)
print(f"Logit lens output shape: {probs.shape}")  # (batch, layers, vocab)

# Patchscope: Replace activations from one context into another
source_prompts = ["Paris is beautiful", "London is foggy"]
custom_target_prompt = TargetPrompt("city: Paris\nfood: croissant\n?", -1)
target_prompt = repeat_prompt()  # Creates a repetition task
custom_repeat_prompt = repeat_prompt(
    words=["car", "cross", "azdrfa"],
    rel=":",
    sep="\n\n",
    placeholder="*",
)
print(f"repeat_prompt: {custom_repeat_prompt}")
print(f"custom_repeat_prompt: {custom_repeat_prompt}")
patchscope_probs = patchscope_lens(
    nnterp_gpt2, source_prompts=source_prompts, target_patch_prompts=target_prompt
)
print(f"patchscope_probs: {patchscope_probs.shape}")
# %% [markdown]
# You can use a combination of run_prompts and interventions to get the probabilities of certain tokens according to your custom intervention.
# %%
demo_model = nnterp_gpt2
# uncomment if you have a GPU for cooler results
# demo_model = StandardizedTransformer("google/gemma-2-2b")

prompts_str = [
    "The translation of 'car' in French is",
    "The translation of 'cat' in Spanish is",
]
tokens = [
    {"target": ["voiture", "bagnole"], "english": "car", "format": "'"},
    {"target": ["gato", "minino"], "english": "cat", "format": "'"},
]
prompts = [
    Prompt.from_strings(prompt, tokens, demo_model.tokenizer)
    for prompt, tokens in zip(prompts_str, tokens)
]
results = run_prompts(demo_model, prompts, batch_size=2, get_probs_func=logit_lens)
for category, probs in results.items():
    print(f"{category}: {probs.shape}")  # (batch, layers)

# Create a plotly plot showing mean probabilities for each category across layers
import plotly.graph_objects as go

# Calculate mean probabilities across batches for each category and layer
mean_probs = {category: probs.mean(dim=0) for category, probs in results.items()}

fig = go.Figure()

# Add a line for each category
for category, probs in mean_probs.items():
    fig.add_trace(
        go.Scatter(
            x=list(range(len(probs))),
            y=probs.tolist(),
            mode="lines+markers",
            name=category,
            line=dict(width=2),
            marker=dict(size=6),
        )
    )

fig.update_layout(
    title="Mean Token Probabilities Across Layers",
    xaxis_title="Layer",
    yaxis_title="Mean Probability",
    hovermode="x unified",
    template="plotly_white",
)

fig.show()


# %% [markdown]
# ## 9. Visualization
#
# Finally, `nnterp` provides visualization utilities for analyzing model probabilities and prompts:

# %%
from nnterp.display import plot_topk_tokens, prompts_to_df

probs = logit_lens(demo_model, prompts_str[0])
# Visualize top tokens from logit lens
plot_topk_tokens(
    probs[0],
    demo_model.tokenizer,
    k=5,
    width=1000,
    height=1000,
    title="Top 5 tokens at each layer for 'The translation of 'car' in French is",
)

# Convert prompts to DataFrame for analysis
df = prompts_to_df(prompts, demo_model.tokenizer)
print("\nPrompts DataFrame:")
display(df)

# %% [markdown]
# # Advanced usage

# %% [markdown]
# Sometime, your model might not be supported yet by nnterp. In this case, you'll be able to use a `RenameConfig` to properly initialize your model.
#
# In this section, I'll show you the steps I took to add support for the `gpt2` to `nnterp`.

# %% [markdown]
# ###  Renaming a module not automatically renamed
#
# Let's say that you load a `gpt2` model that is a bit special: every module is called "super_module" instead of "module".
#
# First, let's build such a model:

# %%
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
for layer in model.transformer.h:
    layer.super_mlp = layer.mlp
    delattr(layer, "mlp")
    layer.super_attn = layer.attn
    delattr(layer, "attn")
model.transformer.super_h = model.transformer.h
delattr(model.transformer, "h")
# Let's keep the final layer norm as is
# model.transformer.super_ln_f = model.transformer.ln_f
# delattr(model.transformer, "ln_f")
model.super_transformer = model.transformer
delattr(model, "transformer")
print(model)

# %% [markdown]
# now if we try to use nnterp, the renaming check automatically performed will fail:

# %%
from nnterp import StandardizedTransformer
from traceback import print_exc

try:
    StandardizedTransformer(model)
except Exception as e:
    print_exc()

# %% [markdown]
# `nnterp` can't find the layers because they're located under `super_transformer`, that nnterp doesn't know about. We have 2 choices in this case:
# 1. Rename `super_transformer` to `model` and `super_h` to `layers` such that it matches the `model.model.layers` Llama architecture and let `nnterp` do the rest.
# 2. Rename `super_transformer.super_h` directly to `layers`, matching the StandardizedTransformer architecture.
#
# Let's try the second option first. And let's not forget that we still need to rename
#
# In order to do that we can instantiate a `StandardizedTransformer` with a `RenameConfig` with the correct aliases provided.

# %%
from nnterp.rename_utils import RenameConfig

rename_cfg = RenameConfig(
    layers_name="super_transformer.super_h",
    attn_name="super_attn",
    mlp_name="super_mlp",
)
try:
    StandardizedTransformer(model, rename_config=rename_cfg)
except Exception as e:
    print_exc()

# %% [markdown]
# We're still getting an error because `nnterp` doesn't find the `ln_f`. This is because `nnterp` will automatically rename the `ln_f` to `ln_final`, but fails to rename `model.ln_final` to `ln_final`. Again, we can either rename `super_transformer` to `model` or directly rename `super_transformer.ln_f` to `ln_final`.
#
# ⚠️ The code will still fail, because our "super_gpt2" model can't run its forward pass as we deleted its modules.

# %%
rename_cfg = RenameConfig(
    model_name="super_transformer",
    layers_name="super_h",
    attn_name="super_attn",
    mlp_name="super_mlp",
    ln_final_name="super_transformer.ln_f",
)
from transformers import AutoConfig

try:
    StandardizedTransformer(
        model, rename_config=rename_cfg, config=AutoConfig.from_pretrained("gpt2")
    )
except Exception as e:
    print_exc()

# %% [markdown]
# ## Adding attention probabilities support

# %% [markdown]
# To access the attention probabilities, `nnterp` uses the `NNsight` ability to hook on most intermediate variables of the forward pass. This is very architecture dependent, as even 2 equivalent models, if they use different names for the intermediate variables, will need different hooks.
#
# As I'm writing this tutorial, I'm adding support for attention probabilities for `GPTJ` models.

# %%
from nnterp import StandardizedTransformer

gptj = StandardizedTransformer(
    "yujiepan/gptj-tiny-random", enable_attention_probs=True
)  # In the current version of nnterp, this will work out of the box

# %% [markdown]
# As you can see, when you load a model,`nnterp` will automatically test if the attention probabilities hook is working and returns a tensor of shape `(batch_size, num_heads, seq_len, seq_len)` where the last dimension sums to 1. In this case, the test failed and `nnterp` logs the error.
#
# Now let's look at the `yujiepan/gptj-tiny-random` forward pass and try to understand where are the attention probabilities computed

# %%
from nnterp.utils import display_source

display_source(gptj.attentions[0].source)

# %% [markdown]
# Lines 60-61:
# ```py
#                                 60     # compute self-attention: V x Softmax(QK^T)
#  self__attn_0                -> 61     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
#  ```
# ⚠️ Be careful! if you set the hook here, you'll be able to successfully access the attention probabilities, but not to edit them! ⚠️
#
# We need to check the source of `self__attn_0` to see where `attn_weights` is used. In order to access a deeper variable like this, we have to actually run the model with `trace` or `scan`. I'd advise to start with `scan` first, but switch to `trace` if you encounter an error.

# %%
with gptj.scan("a"):
    display_source(gptj.attentions[0].source.self__attn_0.source)

# %% [markdown]
# Here, line 20-24:
# ```py
#  self_attn_dropout_0     -> 20     attn_weights = self.attn_dropout(attn_weights)
#                             21
#                             22     # Mask heads if we want to
#                             23     if head_mask is not None:
#                             24         attn_weights = attn_weights * head_mask
# ```
#
# In the current `NNsight` version, the results of operators like `*` are not hooked. But even if they were, I'd be careful to use line 24 here, as it's inside a `if` statement. Therefore, we'll use `self_attn_dropout_0` instead.
#
# Note that we could also look at `torch_matmul_1` input and edit the value here. However, this looks less robust to me as it assumes this is the only place where `attn_weights` is used.

# %%
import torch as th

with gptj.scan(th.tensor([[1, 2, 3]])):
    print(
        gptj.attentions[0].source.self__attn_0.source.self_attn_dropout_0.output.shape
    )

# %% [markdown]
# Nice! The shape looks good. Now we can initialize our model with the right RenameConfig, and let `nnterp` run the tests for us.
#
# To do this, we'll need to create a `AttnProbFunction` and implement the `get_attention_prob_source` method.

# %%
from nnterp.rename_utils import AttnProbFunction, RenameConfig


class GPTJAttnProbFunction(AttnProbFunction):

    def get_attention_prob_source(
        self, attention_module, return_module_source: bool = False
    ):
        if return_module_source:
            # in this case, return source of the module from where the attention probabilities are computed
            return attention_module.source.self__attn_0.source
        else:
            # in this case, return the attention probabilities hook
            return attention_module.source.self__attn_0.source.self_attn_dropout_0


gptj = StandardizedTransformer(
    "yujiepan/gptj-tiny-random",
    enable_attention_probs=True,
    rename_config=RenameConfig(attn_prob_source=GPTJAttnProbFunction()),
)

with gptj.trace("Hello world!"):
    batch_size, seq_len = gptj.input_size
    attn_probs = gptj.attention_probabilities[0].save()
    print(f"attn_probs.shape: {attn_probs.shape}")
    assert attn_probs.shape == (batch_size, gptj.num_heads, seq_len, seq_len)
    gptj.attention_probabilities[0] = attn_probs / 2
    corrupt_logits = gptj.logits.save()

with gptj.trace("Hello world!"):
    clean_logits = gptj.logits.save()

assert gptj.attention_probabilities.enabled
assert not th.allclose(clean_logits, corrupt_logits)
summed_attn_probs = attn_probs.sum(dim=-1)
assert th.allclose(summed_attn_probs, th.ones_like(summed_attn_probs))

# %% [markdown]
# ## Summary
#
# `nnterp` provides a unified, standardized interface for working with transformer models, built on top of `nnsight`. Key features include:
#
# 1. **Standardized naming** across all transformer architectures
# 2. **Easy access** to layer/attention/MLP inputs and outputs
# 3. **Built-in methods** for common operations (steering, skipping layers, projecting to vocab)
# 4. **Efficient activation collection** with batching support
# 5. **Prompt utilities** for tracking target tokens
# 6. **Intervention methods** from mechanistic interpretability research
# 7. **Visualization tools** for analyzing model behavior
#
# All of this while maintaining the full power and flexibility of `nnsight` under the hood!

# %% [markdown]
# # Appendix: `NNsight` cheatsheet

# %% [markdown]
# ## 1) You must execute your interventions in order
# In the new `NNsight` versions, it is enforced that you must access to model internals *in the same order* as the model execute them.

# %%
from nnterp import StandardizedTransformer
from traceback import print_exc

nnterp_gpt2 = StandardizedTransformer("gpt2")
try:
    with nnterp_gpt2.trace("My tailor is rich"):
        l2 = nnterp_gpt2.layers_output[2]
        l1 = nnterp_gpt2.layers_output[1]  # will fail! You need to collect l1 before l2
except Exception as e:
    print_exc()

# %% [markdown]
# ## 2) Gradient computation
# To compute gradients, you need to open a `.backward()` context, and save the gradients *inside it*.

# %%
with nnterp_gpt2.trace("My tailor is rich"):
    l1_out = nnterp_gpt2.layers_output[1]  # get l1 before accessing logits
    logits = nnterp_gpt2.output.logits
    with logits.sum().backward(
        retain_graph=True
    ):  # use retain_graph if you want to do multiple backprops
        if False:
            l1_grad = nnterp_gpt2.layers_output[1].grad.save()
            # this would fail as we'd access nnterp_gpt2.layers_output[1] after nnterp_gpt2.output
        l1_grad = l1_out.grad.save()
    with (logits.sum() ** 2).backward():
        l1_grad_2 = l1_out.grad.save()

assert not th.allclose(l1_grad, l1_grad_2)

# %% [markdown]
# ## 3) Use tracer.stop() to save useless computations
# If you're just computing activations, don't forget to call `tracer.stop()` at the end of your trace. This will stop the model from executing the rest of its computations, and save you some time, as demonstrated below (with the contribution of Claude 4 Sonnet).

# %%
import time
import pandas as pd

print(
    "🎭 Welcome to the Theatrical Performance Comparison! 🎭\n"
    + "=" * 60
    + "\n\n🐌 ACT I: 'The Tragedy of the Unstoppable Tracer' 🐌\nIn which our hero forgets to call tracer.stop()..."
)

start_time = time.time()
for _ in range(30):
    with nnterp_gpt2.trace(["Neel Samba", "Chris Aloha"]):
        out5 = nnterp_gpt2.layers_output[5].save()
end_time = time.time()
nostop_time = end_time - start_time

print(
    f"⏰ Duration of suffering: {nostop_time:.4f} seconds\n\n⚡ ACT II: 'The Redemption of the Stopped Tracer' ⚡\nOur hero learns the ancient art of tracer.stop()..."
)


start_time = time.time()
for _ in range(30):
    with nnterp_gpt2.trace(["Neel Samba", "Chris Aloha"]) as tracer:
        out5 = nnterp_gpt2.layers_output[5].save()
        tracer.stop()
end_time = time.time()
stop_time = end_time - start_time

print(f"⏰ Duration of enlightenment: {stop_time:.4f} seconds")

speedup = nostop_time / stop_time
time_saved = nostop_time - stop_time

# fun display
print("\n" + "=" * 60 + "\n🎉 THE GRAND RESULTS SPECTACULAR! 🎉\n" + "=" * 60)
results_df = pd.DataFrame(
    {
        "🎭 Performance Type": [
            "Without tracer.stop() 🐌",
            "With tracer.stop() ⚡",
            "Time Saved 💰",
        ],
        "⏱️ Time (seconds)": [
            f"{nostop_time:.4f}",
            f"{stop_time:.4f}",
            f"{time_saved:.4f}",
        ],
        "🎯 Rating": ["Tragic 😭", "Magnificent! 🌟", "PROFIT! 📈"],
    }
)
display(results_df)
speedup_bars = int(speedup * 10)
meter = "█" * min(speedup_bars, 48) + "░" * (50 - min(speedup_bars, 48))
print(
    f"\n🏎️ SPEEDUP METER 🏎️\n┌{'─' * 50}┐\n│{meter}│\n└{'─' * 50}┘\n   💫 COSMIC SPEEDUP: {speedup:.2f}x FASTER! 💫"
)


# %% [markdown]
# ## 4) Using NNsight builtin cache to collect activations
#
# `NNsight 0.5` introduces a builtin way to cache activations during the forward pass. Be careful not to call `tracer.stop()` before all the module of the cache have been accessed.
#
# The cache supports both renamed and original module names. You can access cached activations using attribute notation or dictionary keys.
#
# **Note:** There is a bug in the current version of `NNsight` ()

# %%
with nnterp_gpt2.trace("Hello") as tracer:
    cache = tracer.cache(modules=[layer for layer in nnterp_gpt2.layers[::2]]).save()

# Access with renamed names using attribute notation
print(cache.model.layers[10].output)
# Or using dictionary syntax with renamed path
print(cache["model.layers.10"].output)
# Original names still work
print(cache["model.transformer.h.10"].output)
