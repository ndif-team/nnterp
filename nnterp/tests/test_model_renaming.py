import torch as th
import pytest
from nnsight import LanguageModel
from nnterp import StandardizedTransformer
from nnterp.nnsight_utils import (
    get_layer_input,
    get_layer_output,
    get_attention_output,
    get_logits,
    get_next_token_probs,
    project_on_vocab,
    get_mlp_output,
    get_num_layers,
    ModuleAccessor,
)
from nnterp.rename_utils import get_ignores, RenameConfig
from transformers import OPTForCausalLM
import torch.nn as nn


def get_layer_test(model_name, model, renamed, i):
    if renamed:
        return model.layers[i]
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return model.transformer.h[i]
    elif model_name == "Maykeye/TinyLLama-v0":
        return model.model.layers[i]
    elif isinstance(model._model, OPTForCausalLM):
        return model.model.decoder.layers[i]
    else:
        try:
            layer = model.model.layers[i]
            return layer
        except AttributeError:
            return None


def get_norm_test(model_name, model, renamed):
    if renamed:
        return model.ln_final
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return model.transformer.ln_f
    elif model_name == "Maykeye/TinyLLama-v0":
        return model.model.norm
    elif isinstance(model._model, OPTForCausalLM):
        return model.model.decoder.final_layer_norm
    else:
        try:
            norm = model.model.norm
            return norm
        except AttributeError:
            return None


def get_num_layers_test(model_name, model, renamed):
    if renamed:
        return len(model.layers)
    elif model_name == "Maykeye/TinyLLama-v0":
        return len(model.model.layers)
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return len(model.transformer.h)
    elif isinstance(model._model, OPTForCausalLM):
        return len(model.model.decoder.layers)
    else:
        try:
            num_layers = len(model.model.layers)
        except AttributeError:
            return None
        return num_layers


def test_model_renaming_activations_diff(model_name):
    """
    Test that the activations of the model are the same with and without renaming
    """
    with th.no_grad():
        # Load models with and without patching
        model_renamed = StandardizedTransformer(model_name)
        model = LanguageModel(model_name, device_map="auto")
        device = "cuda" if th.cuda.is_available() else "cpu"

        # Set up test input
        prompt = "Hello, world!"

        # Function to collect activations
        def get_token_activations(model, renamed=True):
            activations = []
            num_layers = get_num_layers_test(model_name, model, renamed)
            if (
                num_layers is None
                or get_layer_test(model_name, model, renamed, 0) is None
                or get_norm_test(model_name, model, renamed) is None
            ):
                with open("bump.log", "a") as f:
                    f.write(f"Model {model_name} manual test not implemented\n")
                return None
            with model.trace(prompt):
                # Collect layer outputs
                for i in range(num_layers):
                    layer = get_layer_test(model_name, model, renamed, i)
                    if layer is None:
                        return None
                    activations.append(layer.output[0].save())

                # Collect final layer norm output
                norm = get_norm_test(model_name, model, renamed)
                if norm is None:
                    return None
                activations.append(norm.output[0].save())

                # Collect logits
                activations.append(model.lm_head.output[0].save())

            return activations

        # Collect activations for both models
        activations = get_token_activations(model, renamed=False)
        if activations is None:
            pytest.skip(f"Model {model_name} manual test not implemented")
        activations_renamed = get_token_activations(model_renamed, renamed=True)

        # Compare activations
        assert len(activations_renamed) == len(
            activations
        ), f"Number of activation layers mismatch: {len(activations_renamed)} != {len(activations)}"

        for i, (act_renamed, act) in enumerate(zip(activations_renamed, activations)):
            assert th.allclose(
                act_renamed.to(device), act.to(device), atol=1e-3
            ), f"Mismatch in activation layer {i}. Max diff: {th.max(th.abs(act_renamed.to(device) - act.to(device)))}"


def test_renaming_forward(model_name):
    """
    Test that the renaming is correct for StandardizedTransformer
    """
    with th.no_grad():
        renamed_model = StandardizedTransformer(model_name)
        normal_model = LanguageModel(model_name, device_map="auto")
        device = "cuda" if th.cuda.is_available() else "cpu"

        # Test that key attributes exist and are accessible
        assert renamed_model.model.layers[0].self_attn is not None
        assert renamed_model.attentions[0] is not None
        if not isinstance(renamed_model._model, OPTForCausalLM):
            assert renamed_model.model.layers[0].mlp is not None
            assert renamed_model.mlps[0] is not None
        assert renamed_model.model.ln_final is not None
        assert renamed_model.lm_head is not None

        prompt = "Hello, world!"

        # Compare outputs
        with normal_model.trace(prompt):
            normal_out = normal_model.output.logits.save()

        with renamed_model.trace(prompt):
            renamed_out = renamed_model.output.logits.save()

        assert th.allclose(normal_out.to(device), renamed_out.to(device), atol=1e-5)


def test_standardized_transformer_methods(model_name):
    """
    Test both accessor methods and direct module access of StandardizedTransformer
    """
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        num_layers = model.num_layers
        assert num_layers > 0
        ignores = get_ignores(model._model)
        with model.trace(prompt):
            assert model.layers[0] is not None
            # Test accessor and direct module access
            layer_input_accessor = model.layers_input[0].save()
            layer_input_direct = model.model.layers[0].input.save()
            assert model.attentions[0] is not None
            assert model.layers[0].self_attn is not None
            if "mlp" not in ignores:
                assert model.mlps[0] is not None
                assert model.layers[0].mlp is not None
            attn_output_accessor = model.attentions_output[0].save()
            attn_output_direct = model.model.layers[0].self_attn.output[0].save()
            if "mlp" not in ignores:
                if "down_proj" not in ignores:
                    down_proj_output_accessor = model.down_projs_output[0].save()
                    down_proj_output_direct = model.model.layers[0].mlp.down_proj.output
                    if isinstance(down_proj_output_direct, tuple):
                        down_proj_output_direct = down_proj_output_direct[0]
                    down_proj_output_direct = down_proj_output_direct.save()
                mlps_output_accessor = model.mlps_output[0].save()
                mlps_output_direct = model.model.layers[0].mlp.output
                if isinstance(mlps_output_direct, tuple):
                    mlps_output_direct = mlps_output_direct[0]
                mlps_output_direct = mlps_output_direct.save()
            layer_output_accessor = model.layers_output[0].save()
            layer_output_direct = model.model.layers[0].output[0].save()

            # Test token probability methods
            next_probs = model.next_token_probs.save()

            # Test project_on_vocab with layer output
            projected = model.project_on_vocab(layer_output_accessor).save()

            # Test model-level methods
            logits = model.logits.save()
            logits_direct = model.output.logits.save()

        # Verify accessor and direct access give same results
        assert th.allclose(
            layer_input_accessor, layer_input_direct
        ), "Layer input mismatch between accessor and direct access"
        assert th.allclose(
            attn_output_accessor, attn_output_direct
        ), "Attention output mismatch between accessor and direct access"
        if "mlp" not in ignores:
            assert th.allclose(
                mlps_output_accessor, mlps_output_direct
            ), "MLP output mismatch between accessor and direct access"
            if "down_proj" not in ignores:
                assert th.allclose(
                    down_proj_output_accessor, down_proj_output_direct
                ), "Down proj output mismatch between accessor and direct access"
        assert th.allclose(
            layer_output_accessor, layer_output_direct
        ), "Layer output mismatch between accessor and direct access"
        assert th.allclose(
            logits, logits_direct
        ), "Logits mismatch between logits and direct access"

        # Test shape consistency
        assert next_probs.shape[-1] == model.vocab_size
        assert projected.shape[-1] == model.vocab_size
        assert layer_input_accessor.shape == layer_output_accessor.shape
        assert logits.shape[-1] == model.vocab_size
        assert next_probs.shape == (logits.shape[0], logits.shape[2])
        if "mlp" not in ignores:
            assert mlps_output_accessor.shape == layer_output_accessor.shape
        assert attn_output_accessor.shape == layer_input_accessor.shape


def test_renamed_model_methods(model_name):
    """
    Test all methods with a renamed model loaded via StandardizedTransformer
    """
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        num_layers = get_num_layers(model)
        assert num_layers > 0
        ignores = get_ignores(model._model)
        with model.trace(prompt):
            batch_size = model.input_size[0].save()
            seq_len = model.input_size[1].save()
            embed_tokens = model.token_embeddings.save()
            embed_tokens_out = model.embed_tokens.output.save()

            # Test layer 0 methods
            layer_input = get_layer_input(model, 0).save()
            attn_output = get_attention_output(model, 0).save()
            if "mlp" not in ignores:
                mlps_output = get_mlp_output(model, 0)
                if isinstance(mlps_output, tuple):
                    mlps_output = mlps_output[0]
                mlps_output = mlps_output.save()
            layer_output = get_layer_output(model, 0).save()

            # Test token probability methods
            next_probs = get_next_token_probs(model).save()

            # Test project_on_vocab with layer output
            projected = project_on_vocab(model, layer_output).save()

            # Test model-level methods
            logits = get_logits(model).save()
            logits_output = model.output.logits.save()

        assert next_probs.shape == (batch_size, model.vocab_size)
        assert projected.shape == (batch_size, seq_len, model.vocab_size)
        assert embed_tokens.shape == (batch_size, seq_len, model.hidden_size)
        assert embed_tokens_out.shape == (batch_size, seq_len, model.hidden_size)
        assert th.allclose(embed_tokens, embed_tokens_out)
        assert layer_input.shape == (batch_size, seq_len, model.hidden_size)
        assert logits.shape == (batch_size, seq_len, model.vocab_size)
        assert logits_output.shape == (batch_size, seq_len, model.vocab_size)
        if "mlp" not in ignores:
            assert mlps_output.shape == (batch_size, seq_len, model.hidden_size)
        assert attn_output.shape == (batch_size, seq_len, model.hidden_size)


def test_standardized_transformer_input_accessors(model_name):
    """Test input accessor methods of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        ignores = get_ignores(model._model)
        with model.trace(prompt):
            # Test input accessors
            layer_input_accessor = model.layers_input[0].save()
            layer_input_direct = model.model.layers[0].input.save()

            attn_input_accessor = model.attentions_input[0].save()
            attn_input_direct = model.model.layers[0].self_attn.input.save()

            if "mlp" not in ignores:
                mlp_input_accessor = model.mlps_input[0].save()
                mlp_input_direct = model.model.layers[0].mlp.input.save()
                if "down_proj" not in ignores:
                    down_proj_input_accessor = model.down_projs_input[0].save()
                    down_proj_input_direct = model.model.layers[0].mlp.down_proj.input.save()

        # Verify input accessors work correctly
        assert th.allclose(
            layer_input_accessor, layer_input_direct
        ), "Layer input accessor mismatch"
        assert th.allclose(
            attn_input_accessor, attn_input_direct
        ), "Attention input accessor mismatch"
        if "mlp" not in ignores:
            assert th.allclose(
                mlp_input_accessor, mlp_input_direct
            ), "MLP input accessor mismatch"
            if "down_proj" not in ignores:
                assert th.allclose(
                    down_proj_input_accessor, down_proj_input_direct
                ), "Down proj input accessor mismatch"


def test_standardized_transformer_steer_method(model_name):
    """Test steer method of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        # Create a random steering vector
        hidden_size = model.hidden_size
        if hidden_size is None:
            pytest.fail(f"Model {model_name} has no hidden size")

        steering_vector = th.randn(hidden_size) * 0.1  # Small perturbation

        # Test steering single layer
        with model.trace(prompt):
            baseline_output = model.logits.save()

        with model.trace(prompt):
            model.steer(layers=0, steering_vector=steering_vector, factor=1.0)
            steered_output = model.logits.save()

        # Steered output should be different from baseline
        assert not th.allclose(
            baseline_output, steered_output, atol=1e-4
        ), "Steering should change model output"

        # Test steering multiple layers
        with model.trace(prompt):
            model.steer(
                layers=list(range(min(model.num_layers, 2))),
                steering_vector=steering_vector,
                factor=0.5,
            )
            multi_steered_output = model.logits.save()

        assert not th.allclose(
            baseline_output, multi_steered_output, atol=1e-4
        ), "Multi-layer steering should change output"

        # Test position-specific steering
        num_tokens = len(model.tokenizer.encode(prompt))
        assert num_tokens > 1, "Prompt should have multiple tokens for position testing"

        with model.trace(prompt):
            model.steer(layers=0, steering_vector=steering_vector, positions=0)
            pos_steered_output = model.logits.save()

        assert not th.allclose(
            baseline_output, pos_steered_output, atol=1e-4
        ), "Position-specific steering should change output"


def test_standardized_transformer_skip_methods(model_name):
    """Test skip methods of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        if model.num_layers < 2:
            pytest.skip("Model needs at least 2 layers for skip testing")

        # Test skip_layer method
        with model.trace(prompt):
            baseline_output = model.logits.save()

        with model.trace(prompt):
            model.skip_layer(0)
            skip_output = model.logits.save()

        assert not th.allclose(
            baseline_output, skip_output
        ), "skip_layer should change output"

        # Test skip_layers method
        with model.trace(prompt):
            model.skip_layers(0, 1)
            skip_layers_output = model.logits.save()

        assert not th.allclose(
            baseline_output, skip_layers_output
        ), "skip_layers should change output"

        # Test skip with custom tensor
        with model.trace(prompt):
            custom_tensor = model.layers_input[0]
            model.skip_layer(1, skip_with=custom_tensor)
            custom_skip_output = model.logits.save()

        assert not th.allclose(
            baseline_output, custom_skip_output
        ), "skip with custom tensor should change output"


def test_standardized_transformer_constructor_options(model_name):
    """Test StandardizedTransformer constructor options"""
    with th.no_grad():
        # Test with check_renaming=False
        model_no_check = StandardizedTransformer(model_name, check_renaming=False)
        assert model_no_check.num_layers > 0

        # Test with custom renaming parameters - this should still work even with wrong names
        model_custom = StandardizedTransformer(
            model_name, attn_rename="nonexistent_attn", check_renaming=False
        )
        assert model_custom.num_layers > 0


def test_standardized_transformer_properties(model_name):
    """Test properties of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = ["a", "b b b b b"]
        if model.hidden_size is None:
            pytest.fail(f"Model {model_name} has no hidden size")
        if model.num_heads is None:
            pytest.fail(f"Model {model_name} has no num_heads")

        with model.trace(prompt):
            # Test properties
            unembed_norm = model.ln_final
            assert unembed_norm is not None
            next_token_probs = model.next_token_probs.save()
            logits = model.logits.save()

        # Verify shapes and properties
        assert (
            next_token_probs.shape
            == (logits.shape[0], logits.shape[-1])
            == (
                len(prompt),
                model.vocab_size,
            )
        )
        assert th.allclose(
            next_token_probs.sum(dim=-1),
            th.ones(
                logits.shape[0],
                device=next_token_probs.device,
                dtype=next_token_probs.dtype,
            ),
            atol=1e-5,
        )


def test_standardized_transformer_cache(model_name):
    """Test that StandardizedTransformer works with nnsight cache using renamed names"""
    pytest.skip(
        "Cache is not supported yet due to a nnsight renaming issue."
    )  # TODO: Update once nnsight is fixed
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        # Cache layers using renamed module references
        with model.trace(prompt) as tracer:
            cache = tracer.cache(
                modules=[model.layers[i] for i in range(0, model.num_layers, 2)]
            ).save()

        # Access cached modules using renamed names (attribute notation)
        layer_0_output_attr = cache.model.layers[0].output
        batch_size, seq_len, hidden_size = layer_0_output_attr[0].shape
        assert batch_size == 1
        assert hidden_size == model.hidden_size

        # Access using dictionary notation with renamed path
        layer_0_output_dict = cache["model.layers.0"].output
        assert th.allclose(layer_0_output_attr[0], layer_0_output_dict[0])


def test_module_accessor(model_name, raw_model):
    """Test ModuleAccessor class - all methods and functionality"""
    with th.no_grad():
        # Get the underlying PreTrainedModel
        pretrained_model = raw_model._model

        # Test ModuleAccessor with default config
        accessor = ModuleAccessor(pretrained_model)

        # Test that nn_model is created
        assert accessor.nn_model is not None

        # Test get_embed_tokens returns a module
        embed_tokens = accessor.get_embed_tokens()
        assert isinstance(embed_tokens, nn.Module)

        # Test get_layers returns a ModuleList
        layers = accessor.get_layers()
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) > 0

        # Test get_attention returns a module
        attention = accessor.get_attention(0)
        assert isinstance(attention, nn.Module)

        # Test get_unembed_norm returns a module
        # Note: This may fail for models where get_unembed_norm doesn't handle renamed paths
        # (e.g., models with ln_final instead of model.norm). These failures reveal
        # that get_unembed_norm needs to be updated to handle renamed models.
        unembed_norm = accessor.get_unembed_norm()
        assert isinstance(unembed_norm, nn.Module)

        # Test get_unembed returns a module
        unembed = accessor.get_unembed()
        assert isinstance(unembed, nn.Module)

        # Test get_mlp if not ignored
        ignores = get_ignores(pretrained_model)
        if "mlp" not in ignores:
            mlp = accessor.get_mlp(0)
            assert isinstance(mlp, nn.Module)

        # Test __getattr__ for module attributes
        model_module = accessor.model
        assert isinstance(model_module, nn.Module)

        lm_head_module = accessor.lm_head
        assert isinstance(lm_head_module, nn.Module)

        # Test __getattr__ raises AttributeError for non-existent attributes
        # This will fail at getattr(self.nn_model, name) before our custom error
        with pytest.raises(AttributeError):
            _ = accessor.nonexistent_attr

        # Test __getattr__ raises AttributeError for attributes without _module
        # Find an attribute that exists on nn_model but doesn't have _module
        # Common examples: methods, properties, or non-module attributes
        attrs_to_test = ["tokenizer", "config", "device", "dtype"]
        for attr_name in attrs_to_test:
            if hasattr(accessor.nn_model, attr_name):
                attr = getattr(accessor.nn_model, attr_name)
                if not hasattr(attr, "_module"):
                    # This attribute exists but doesn't have _module, should raise our custom error
                    with pytest.raises(
                        AttributeError, match=f"Attribute {attr_name} is not a module"
                    ):
                        _ = getattr(accessor, attr_name)
                    break

        # Test ModuleAccessor with custom RenameConfig
        custom_config = RenameConfig()
        accessor_custom = ModuleAccessor(pretrained_model, rename_config=custom_config)

        # Verify it still works with custom config
        assert accessor_custom.nn_model is not None
        embed_tokens_custom = accessor_custom.get_embed_tokens()
        assert isinstance(embed_tokens_custom, nn.Module)

        # Test ModuleAccessor with None rename_config
        accessor_none = ModuleAccessor(pretrained_model, rename_config=None)
        assert accessor_none.nn_model is not None

        # Verify all getter methods work across different configs
        layers_custom = accessor_custom.get_layers()
        assert isinstance(layers_custom, nn.ModuleList)
        assert len(layers_custom) == len(layers)

        attention_custom = accessor_custom.get_attention(0)
        assert isinstance(attention_custom, nn.Module)

        unembed_norm_custom = accessor_custom.get_unembed_norm()
        assert isinstance(unembed_norm_custom, nn.Module)

        unembed_custom = accessor_custom.get_unembed()
        assert isinstance(unembed_custom, nn.Module)

        if "mlp" not in ignores:
            mlp_custom = accessor_custom.get_mlp(0)
            assert isinstance(mlp_custom, nn.Module)

        # Test accessing multiple layers
        num_layers = len(layers)
        if num_layers > 1:
            attention_1 = accessor.get_attention(1)
            assert isinstance(attention_1, nn.Module)
            if "mlp" not in ignores:
                mlp_1 = accessor.get_mlp(1)
                assert isinstance(mlp_1, nn.Module)

        # Test __getattr__ with module names that should exist after renaming
        # After renaming via get_rename_dict, model should have certain attributes
        # Test that we can access layers if they exist as an attribute on nn_model
        if hasattr(accessor.nn_model, "layers"):
            layers_attr = accessor.layers
            assert isinstance(layers_attr, nn.ModuleList)
            assert len(layers_attr) == len(layers)
