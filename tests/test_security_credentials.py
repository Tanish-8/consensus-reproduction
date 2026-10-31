import os
import subprocess
import sys
import pytest

from core.model import Model


def test_nvidia_script_fails_cleanly_without_credentials():
    """Verify test_nvidia.py exits with error and does not expose secrets when key is unset."""
    env = os.environ.copy()
    env.pop("NVIDIA_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "test_nvidia.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    # Must report clear error message
    assert "NVIDIA_API_KEY environment variable is not set" in result.stderr
    # Must not leak any dummy or real key pattern
    assert "nvapi-" not in result.stdout
    assert "nvapi-" not in result.stderr


def test_openrouter_script_fails_cleanly_without_credentials():
    """Verify test_openrouter.py exits with error and does not expose secrets when key is unset."""
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "test_openrouter.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "OPENROUTER_API_KEY environment variable is not set" in (result.stderr + result.stdout)


def test_model_initialization_missing_keys_raises_runtime_error(monkeypatch):
    """Verify core.model.Model raises clear RuntimeError when API keys are absent."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY environment variable is not set"):
        Model("nvidia:meta/llama-3.1-8b-instruct", temperature=0.0, top_p=1.0, num_ctx=15000)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY environment variable is not set"):
        Model("openrouter:openrouter/free", temperature=0.0, top_p=1.0, num_ctx=15000)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY environment variable is not set"):
        Model("openai:gpt-4", temperature=0.0, top_p=1.0, num_ctx=15000)

    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY environment variable is not set"):
        Model("together:meta-llama/Llama-3-8b-chat-hf", temperature=0.0, top_p=1.0, num_ctx=15000)
