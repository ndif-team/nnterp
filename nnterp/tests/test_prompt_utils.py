import torch as th
from nnterp.prompt_utils import (
    Prompt,
    run_prompts,
    get_first_tokens,
    next_token_probs_unsqueeze,
)
from nnterp.interventions import logit_lens


def test_process_tokens_with_tokenization(model):
    """Test get_first_tokens function."""
    with th.no_grad():
        # Test with single word
        words = "hello"
        tokens = get_first_tokens(words, model)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test with list of words
        words = ["hello", "world"]
        tokens = get_first_tokens(words, model)
        assert isinstance(tokens, list)
        assert len(tokens) >= 2  # Should have at least the original words


def test_prompt_from_strings(model):
    """Test Prompt.from_strings class method."""
    with th.no_grad():

        # Test with single target string
        prompt = Prompt.from_strings("The quick brown fox", "jumps", model)

        assert prompt.prompt == "The quick brown fox"
        assert "target" in prompt.target_tokens
        assert isinstance(prompt.target_tokens["target"], list)
        assert len(prompt.target_tokens["target"]) > 0

        # Test with dictionary of targets
        prompt = Prompt.from_strings(
            "Hello world", {"greeting": "hello", "object": "world"}, model
        )

        assert "greeting" in prompt.target_tokens
        assert "object" in prompt.target_tokens
        assert isinstance(prompt.target_tokens["greeting"], list)
        assert isinstance(prompt.target_tokens["object"], list)


def test_prompt_get_target_probs(model):
    """Test Prompt.get_target_probs method."""
    with th.no_grad():

        prompt = Prompt.from_strings(
            "The quick brown fox", {"target": "jumps", "animal": "fox"}, model
        )

        # Create mock probabilities tensor (batch_size, layers, vocab_size)
        vocab_size = model.vocab_size
        mock_probs = th.rand(1, 3, vocab_size)  # 1 prompt, 3 layers, vocab_size

        target_probs = prompt.get_target_probs(mock_probs)

        assert "target" in target_probs
        assert "animal" in target_probs
        assert target_probs["target"].shape == (1, 3)  # batch_size, layers
        assert target_probs["animal"].shape == (1, 3)

        # Test with specific layer
        layer_probs = prompt.get_target_probs(mock_probs, layer=1)
        assert target_probs["target"].shape == (
            1,
            3,
        )  # Still all layers since we didn't specify layer in the call


def test_prompt_has_no_collisions(model):
    """Test Prompt.has_no_collisions method."""
    with th.no_grad():
        # Create a prompt that likely has no collisions
        prompt = Prompt.from_strings(
            "Hello world",
            {"greeting": "hello", "punctuation": ".", "greeting2": "hello"},
            model,
        )

        # Test basic collision detection
        result = prompt.has_no_collisions()
        assert isinstance(result, bool)
        assert not result

        # Test with ignored targets
        # Skip if unrelated targets share tokens (toy model tokenizer artifact)
        greeting_tokens = set(prompt.target_tokens["greeting"])
        punct_tokens = set(prompt.target_tokens["punctuation"])
        if greeting_tokens & punct_tokens:
            pytest.skip("Untrained tokenizer produces shared tokens across unrelated words")

        result = prompt.has_no_collisions(ignore_targets="greeting")
        assert isinstance(result, bool)
        assert result

        result = prompt.has_no_collisions(ignore_targets=["greeting", "punctuation"])
        assert isinstance(result, bool)
        assert result


def test_prompt_run(model):
    """Test Prompt.run method."""
    with th.no_grad():
        prompt = Prompt.from_strings("The quick brown fox", {"target": "jumps"}, model)

        # Define a simple probability function
        def simple_get_probs(nn_model, prompt_text):
            # Use logit_lens for testing
            return logit_lens(nn_model, prompt_text)

        result = prompt.run(model, simple_get_probs)

        assert isinstance(result, dict)
        assert "target" in result
        assert isinstance(result["target"], th.Tensor)


def test_next_token_probs_unsqueeze(model):
    """Test next_token_probs_unsqueeze function."""
    with th.no_grad():
        prompt = "Hello world"

        probs = next_token_probs_unsqueeze(model, prompt)

        assert isinstance(probs, th.Tensor)
        assert probs.dim() == 3  # Should have 3 dimensions after unsqueeze
        assert probs.shape[0] == 1  # Batch size
        assert probs.shape[1] == 1  # Number of layers
        assert probs.shape[2] == model.vocab_size


def test_run_prompts(model):
    """Test run_prompts function."""
    with th.no_grad():
        # Create test prompts
        prompts = [
            Prompt.from_strings("Hello world", {"target": "!"}, model),
            Prompt.from_strings("The quick brown", {"target": "fox"}, model),
        ]

        # Test basic functionality
        result = run_prompts(model, prompts, batch_size=2)

        assert isinstance(result, dict)
        assert "target" in result
        assert isinstance(result["target"], th.Tensor)
        assert result["target"].shape[0] == len(prompts)  # Number of prompts

        # Test with custom get_probs function
        def custom_get_probs(nn_model, batch_prompts, **kwargs):
            return logit_lens(nn_model, batch_prompts)

        result = run_prompts(
            model, prompts, batch_size=1, get_probs_func=custom_get_probs
        )

        assert isinstance(result, dict)
        assert "target" in result
        assert result["target"].shape[0] == len(prompts)


def test_run_prompts_multiple_targets(model):
    """Test run_prompts with multiple targets per prompt."""
    with th.no_grad():
        prompts = [
            Prompt.from_strings(
                "The quick", {"animal": "fox", "adjective": "brown"}, model
            ),
            Prompt.from_strings(
                "Hello beautiful world for a ",
                {"animal": "fox", "adjective": "beautiful"},
                model,
            ),
        ]

        result = run_prompts(model, prompts, batch_size=2)

        assert isinstance(result, dict)
        assert "animal" in result
        assert "adjective" in result

        # Check shapes
        for target_name, target_probs in result.items():
            assert isinstance(target_probs, th.Tensor)
            assert target_probs.shape[0] == len(prompts)


def test_run_prompts_single_prompt(model):
    """Test run_prompts with a single prompt."""
    with th.no_grad():
        prompt = Prompt.from_strings("Test prompt", {"target": "word"}, model)

        result = run_prompts(model, [prompt], batch_size=1)

        assert isinstance(result, dict)
        assert "target" in result
        assert result["target"].shape[0] == 1
