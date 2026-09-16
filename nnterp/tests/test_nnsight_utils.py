import pytest
import torch as th
from nnsight import TransformersModel
from nnterp.nnsight_utils import (
    get_num_layers,
    get_layer,
    get_layer_input,
    get_layer_output,
    get_attention,
    get_attention_output,
    get_logits,
    project_on_vocab,
    get_next_token_probs,
    get_token_activations,
    collect_last_token_activations_session,
    collect_token_activations_batched,
    compute_next_token_probs,
    get_mlp_output,
    set_layer_output,
)
from nnterp.rename_utils import get_vocab_size


def test_load_model(llama_like_model_name):
    """Test loading model with different configurations"""
    with th.no_grad():
        TransformersModel(llama_like_model_name, task="text-generation", device_map="auto")


def test_basic_utils(llama_like_model_name):
    """Test basic utility functions"""
    with th.no_grad():
        model = TransformersModel(llama_like_model_name, task="text-generation", device_map="auto")
        prompt = "Hello, world!"

        num_layers = get_num_layers(model)
        assert num_layers > 0
        with model.trace(prompt):
            # Test layer count and access

            # Test layer 0 methods
            _layer = get_layer(model, 0)

            layer_input = get_layer_input(model, 0).save()

            _attention = get_attention(model, 0)

            attn_output = get_attention_output(model, 0).save()
            mlps_output = get_mlp_output(model, 0).save()
            layer_output = get_layer_output(model, 0).save()
            if num_layers > 1:
                set_layer_output(model, 1, layer_output)
            else:
                set_layer_output(model, 0, layer_output + 2)
            # Test model-level methods
            logits = get_logits(model).save()

            # Test token probability methods
            next_probs = get_next_token_probs(model).save()

            # Test project_on_vocab with layer output
            projected = project_on_vocab(model, layer_output).save()
            logits_output = model.output.logits.save()

        vocab_size = get_vocab_size(model)

        assert next_probs.shape[-1] == vocab_size
        assert projected.shape[-1] == vocab_size
        assert layer_input.shape == layer_output.shape
        assert logits.shape[-1] == vocab_size
        assert next_probs.shape == (logits.shape[0], vocab_size)
        assert logits_output.shape == projected.shape
        assert mlps_output.shape == layer_output.shape
        assert attn_output.shape == layer_input.shape


def test_activation_collection(llama_like_model_name):
    """Test activation collection functions"""
    with th.no_grad():
        model = TransformersModel(llama_like_model_name, task="text-generation", device_map="auto")
        prompts = ["Hello, world!", "Testing, 1, 2, 3"] * 2

        # Test activation collection with session
        acts_session = collect_last_token_activations_session(
            model, prompts, batch_size=1
        )
        assert acts_session.shape[:2] == (get_num_layers(model), len(prompts))
        # Test basic activation collection
        acts = get_token_activations(model, prompts)
        assert acts.shape[:2] == (
            get_num_layers(model),
            len(prompts),
        )  # Batch dimension

        # Test batched activation collection
        acts_batched = collect_token_activations_batched(model, prompts, batch_size=1)
        assert acts_batched.shape[:2] == (get_num_layers(model), len(prompts))

        acts_batched_no_batch = collect_token_activations_batched(
            model, prompts, batch_size=len(prompts)
        )
        assert th.allclose(acts, acts_batched_no_batch)

        # Test next token probabilities
        probs = compute_next_token_probs(model, prompts)
        assert probs.shape == (len(prompts), get_vocab_size(model))


def test_project_on_vocab_layer_output_backward(llama_like_model_name):
    """Test that project_on_vocab output supports backward and gradients have correct shape."""
    model = TransformersModel(llama_like_model_name, task="text-generation", device_map="auto")
    prompt = "Hello, world!"
    with model.trace(prompt):
        layer_input = get_layer_input(model, 0).save()
        attn_output = get_attention_output(model, 0).save()
        mlp_output = get_mlp_output(model, 0).save()
        layer_output = get_layer_output(model, 0).save()
        logits = project_on_vocab(model, layer_output)
        loss = logits.sum()
        with loss.backward():
            layer_output_grad = layer_output.grad.save()
            mlp_output_grad = mlp_output.grad.save()
            attn_output_grad = attn_output.grad.save()
            layer_input_grad = layer_input.grad.save()
        # Assert gradient shape
        for grad, tensor, name in [
            (layer_output_grad, layer_output, "layer_output"),
            (mlp_output_grad, mlp_output, "mlp_output"),
            (attn_output_grad, attn_output, "attn_output"),
            (layer_input_grad, layer_input, "layer_input"),
        ]:
            assert grad is not None, f"No gradient computed for {name}!"
            assert (
                grad.shape == tensor.shape
            ), f"Gradient shape mismatch for {name}: {grad.shape} != {tensor.shape}"


def test_grad_from_mlp(llama_like_model_name):
    model = TransformersModel(llama_like_model_name, task="text-generation", device_map="auto")
    prompt = "Hello, world!"
    if get_num_layers(model) < 2:
        pytest.skip("Model has less than 2 layers")
    with model.trace(prompt):
        layer_input = get_layer_input(model, 0).save()
        attn_output = get_attention_output(model, 0).save()
        mlp_output = get_mlp_output(model, 0).save()
        layer_output = get_layer_output(model, 0).save()
        next_mlp_output = get_mlp_output(model, 1).save()
        loss = next_mlp_output.sum()
        with loss.backward():
            layer_output_grad = layer_output.grad.save()
            mlp_output_grad = mlp_output.grad.save()
            attn_output_grad = attn_output.grad.save()
            layer_input_grad = layer_input.grad.save()

    for grad, tensor, name in [
        (layer_output_grad, layer_output, "layer_output"),
        (mlp_output_grad, mlp_output, "mlp_output"),
        (attn_output_grad, attn_output, "attn_output"),
        (layer_input_grad, layer_input, "layer_input"),
    ]:
        assert grad is not None, f"No gradient computed for {name}!"
        assert (
            grad.shape == tensor.shape
        ), f"Gradient shape mismatch for {name}: {grad.shape} != {tensor.shape}"


def test_cache_with_renamed_modules(llama_like_model_name):
    """Test that nnsight cache supports renamed module access"""
    pytest.skip(
        "Cache is not supported yet due to a nnsight renaming issue."
    )  # TODO: Update once nnsight is fixed
    with th.no_grad():
        model = TransformersModel(llama_like_model_name, task="text-generation", device_map="auto")
        prompt = "Hello, world!"
        num_layers = get_num_layers(model)

        # Cache layers using renamed module references
        with model.trace(prompt) as tracer:
            layers_to_cache = [get_layer(model, i) for i in range(0, num_layers, 2)]
            cache = tracer.cache(modules=layers_to_cache).save()

        # Access cached modules using renamed names (attribute notation)
        layer_0_output_attr = cache.model.layers[0].output
        assert layer_0_output_attr[0].shape[0] == 1  # batch size

        # Access using dictionary notation with renamed path
        layer_0_output_dict = cache["model.layers.0"].output
        assert th.allclose(layer_0_output_attr[0], layer_0_output_dict[0])
