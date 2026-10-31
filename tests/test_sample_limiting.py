import os
import yaml
import asyncio
import pytest

from run_consensus import validate_max_samples, run_parallel_samples


def test_validate_max_samples_valid_inputs():
    """Verify validate_max_samples handles None and positive integers correctly."""
    assert validate_max_samples(None) is None
    assert validate_max_samples(1) == 1
    assert validate_max_samples(42) == 42
    assert validate_max_samples("1") == 1
    assert validate_max_samples("10") == 10


def test_validate_max_samples_rejects_zero_negative_and_invalid():
    """Verify validate_max_samples strictly rejects 0, negative values, and non-integer types."""
    with pytest.raises(ValueError, match="positive integer"):
        validate_max_samples(0)

    with pytest.raises(ValueError, match="positive integer"):
        validate_max_samples(-1)

    with pytest.raises(ValueError, match="positive integer"):
        validate_max_samples(-100)

    with pytest.raises(ValueError, match="positive integer"):
        validate_max_samples("invalid_number")

    with pytest.raises(ValueError, match="positive integer"):
        validate_max_samples([])


def test_run_parallel_samples_one_sample_preserves_identity(monkeypatch, tmp_path):
    """
    Verify max_samples=1 processes exactly one real WESAD sample, preserving its
    original index (0), ground truth label ('baseline'), and log directory path.
    """
    processed_samples = []

    async def mock_run_sample(
        idx,
        test_sample,
        examples,
        modalities,
        task_info,
        classes_info,
        modality_info_dict,
        model_pool,
        config,
    ):
        ground_truth = test_sample["label"]
        sample_log_path = os.path.join(config["log_path"], f"{idx}_{ground_truth}")
        processed_samples.append({
            "idx": idx,
            "label": ground_truth,
            "sample_log_path": sample_log_path,
            "features_count": len(test_sample["features"]),
        })

    import run_consensus
    monkeypatch.setattr(run_consensus, "run_sample", mock_run_sample)

    config = {
        "data_path": "./data/WESAD/split/default_ex_1_seed_0",
        "log_path": str(tmp_path / "logs"),
    }

    asyncio.run(run_parallel_samples(config, model_pool=None, max_samples=1))

    # Must process exactly one sample
    assert len(processed_samples) == 1

    sample = processed_samples[0]
    # Original index preserved
    assert sample["idx"] == 0
    # Original ground truth label preserved
    assert sample["label"] == "baseline"
    # Log path correctly associated
    expected_log_path = os.path.join(str(tmp_path / "logs"), "0_baseline")
    assert sample["sample_log_path"] == expected_log_path
    # Has real multimodal sensor features
    assert sample["features_count"] > 100


def test_run_parallel_samples_default_none_processes_all(monkeypatch, tmp_path):
    """Verify max_samples=None (default) processes all 90 samples in the WESAD split."""
    processed_indices = []

    async def mock_run_sample(
        idx,
        test_sample,
        examples,
        modalities,
        task_info,
        classes_info,
        modality_info_dict,
        model_pool,
        config,
    ):
        processed_indices.append(idx)

    import run_consensus
    monkeypatch.setattr(run_consensus, "run_sample", mock_run_sample)

    config = {
        "data_path": "./data/WESAD/split/default_ex_1_seed_0",
        "log_path": str(tmp_path / "logs"),
    }

    asyncio.run(run_parallel_samples(config, model_pool=None, max_samples=None))

    assert len(processed_indices) == 90
    assert sorted(processed_indices) == list(range(90))


def test_run_parallel_samples_rejects_non_positive_values(tmp_path):
    """Verify run_parallel_samples rejects non-positive max_samples."""
    config = {
        "data_path": "./data/WESAD/split/default_ex_1_seed_0",
        "log_path": str(tmp_path / "logs"),
    }

    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(run_parallel_samples(config, model_pool=None, max_samples=0))

    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(run_parallel_samples(config, model_pool=None, max_samples=-5))


def test_run_reads_max_samples_from_yaml(monkeypatch, tmp_path):
    """Verify run() respects max_samples specified in the configuration YAML."""
    processed_indices = []

    async def mock_run_sample(
        idx,
        test_sample,
        examples,
        modalities,
        task_info,
        classes_info,
        modality_info_dict,
        model_pool,
        config,
    ):
        processed_indices.append(idx)

    import run_consensus
    monkeypatch.setattr(run_consensus, "run_sample", mock_run_sample)
    monkeypatch.setattr(run_consensus, "load_models", lambda *args, **kwargs: None)

    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(
        yaml.dump({
            "seed": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "use_large_context": False,
            "models": ["openrouter:openrouter/free"],
            "data_path": "./data/WESAD/split/default_ex_1_seed_0",
            "log_path": str(tmp_path / "logs"),
            "max_samples": 1,
        }),
        encoding="utf-8",
    )

    run_consensus.run(str(yaml_file))

    assert len(processed_indices) == 1
    assert processed_indices[0] == 0
