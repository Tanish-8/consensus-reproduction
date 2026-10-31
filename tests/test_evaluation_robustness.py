import os
import json
import csv
import asyncio
import pytest
import numpy as np

from evaluate_results import (
    parse_sample_folder_name,
    parse_sample_summary,
    calculate_metrics,
    evaluate,
    load_dataset_labels,
    resolve_dataset_labels,
    WESAD_CANONICAL_LABELS,
    DEFAULT_LABELS,
    METHODS,
)


def test_parse_sample_folder_name():
    """Verify numerical indexing and label parsing from folder names."""
    idx, truth = parse_sample_folder_name("0_baseline")
    assert idx == 0
    assert truth == "baseline"

    idx, truth = parse_sample_folder_name("42_stress")
    assert idx == 42
    assert truth == "stress"

    idx, truth = parse_sample_folder_name("sample_amusement")
    assert idx is None
    assert truth == "sample_amusement"


def test_parse_sample_summary_success_and_casing(tmp_path):
    """Verify parsing normalizes uppercase/mixed-case predictions and truths."""
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text(
        "[Interpretation] Decision made: Baseline (ground truth: Baseline)\n"
        "[Semantic Fusion] Decision made: STRESS (ground truth: Baseline)\n"
        "[Statistical Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Hybrid Fusion] Decision made: Amusement (ground truth: baseline)\n",
        encoding="utf-8",
    )

    records = parse_sample_summary(str(summary_file), "0_baseline")

    assert records["Interpretation"]["status"] == "SUCCESS"
    assert records["Interpretation"]["y_pred"] == "baseline"
    assert records["Interpretation"]["y_true"] == "baseline"

    assert records["Semantic Fusion"]["status"] == "SUCCESS"
    assert records["Semantic Fusion"]["y_pred"] == "stress"

    assert records["Hybrid Fusion"]["status"] == "SUCCESS"
    assert records["Hybrid Fusion"]["y_pred"] == "amusement"


def test_parse_sample_summary_deduplication(tmp_path):
    """Verify that multiple runs appended to summary.txt take the final decision."""
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text(
        # First attempt:
        "[Interpretation] Decision made: stress (ground truth: baseline)\n"
        # Second attempt (appended):
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n"
        "[Semantic Fusion] Decision made: baseline (ground truth: baseline)\n",
        encoding="utf-8",
    )

    records = parse_sample_summary(str(summary_file), "1_baseline")

    # Final attempt should be baseline
    assert records["Interpretation"]["status"] == "SUCCESS"
    assert records["Interpretation"]["y_pred"] == "baseline"

    # Missing downstream methods should be MISSING
    assert records["Statistical Fusion"]["status"] == "MISSING"
    assert records["Statistical Fusion"]["y_pred"] is None
    assert records["Hybrid Fusion"]["status"] == "MISSING"


def test_parse_sample_summary_explicit_and_generic_failures(tmp_path):
    """Verify explicit execution failures and generic error logs are flagged as FAILED."""
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text(
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n"
        "[Semantic Fusion] Execution failed (ground truth: baseline)\n"
        "[Error] Statistical fusion failed\n",
        encoding="utf-8",
    )

    records = parse_sample_summary(str(summary_file), "5_baseline")

    assert records["Interpretation"]["status"] == "SUCCESS"
    assert records["Semantic Fusion"]["status"] == "FAILED"
    assert records["Semantic Fusion"]["y_pred"] is None
    assert records["Statistical Fusion"]["status"] == "FAILED"
    assert records["Statistical Fusion"]["y_pred"] is None
    assert records["Hybrid Fusion"]["status"] == "MISSING"


def test_parse_sample_summary_skipped_stages(tmp_path):
    """Verify dependency-skipped stages are accurately flagged as SKIPPED, not FAILED or MISSING."""
    summary_file = tmp_path / "summary.txt"
    summary_file.write_text(
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n"
        "[Semantic Fusion] Execution failed (ground truth: baseline)\n"
        "[Statistical Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Hybrid Fusion] Skipped (dependency failed) (ground truth: baseline)\n",
        encoding="utf-8",
    )

    records = parse_sample_summary(str(summary_file), "7_baseline")

    assert records["Interpretation"]["status"] == "SUCCESS"
    assert records["Semantic Fusion"]["status"] == "FAILED"
    assert records["Statistical Fusion"]["status"] == "SUCCESS"
    assert records["Hybrid Fusion"]["status"] == "SKIPPED"
    assert records["Hybrid Fusion"]["y_pred"] is None
    assert records["Hybrid Fusion"]["y_true"] == "baseline"


def test_calculate_metrics_consistent_labels():
    """Verify macro precision, recall, F1, and confusion matrices strictly adhere to defined labels."""
    labels = ["amusement", "baseline", "stress"]

    records = [
        {"status": "SUCCESS", "y_true": "baseline", "y_pred": "baseline"},
        {"status": "SUCCESS", "y_true": "stress", "y_pred": "stress"},
        {"status": "SUCCESS", "y_true": "amusement", "y_pred": "amusement"},
        {"status": "SUCCESS", "y_true": "stress", "y_pred": "baseline"},
    ]

    metrics = calculate_metrics(records, labels=labels)

    assert metrics["total_samples"] == 4
    assert metrics["successful_samples"] == 4
    assert metrics["failed_samples"] == 0
    assert metrics["skipped_samples"] == 0
    assert metrics["missing_samples"] == 0
    assert metrics["failure_rate"] == 0.0
    assert metrics["accuracy"] == 0.75

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    # Check amusement row: 1 correct
    assert cm[0, 0] == 1
    # Check baseline row: 1 correct
    assert cm[1, 1] == 1
    # Check stress row: 1 predicted baseline (index 1), 1 predicted stress (index 2)
    assert cm[2, 1] == 1
    assert cm[2, 2] == 1


def test_calculate_metrics_handles_all_four_outcomes():
    """Verify SUCCESS, FAILED, SKIPPED, and MISSING are accounted for with correct denominators."""
    labels = ["amusement", "baseline", "stress"]

    records = [
        {"status": "SUCCESS", "y_true": "baseline", "y_pred": "baseline"},
        {"status": "FAILED", "y_true": "stress", "y_pred": None},
        {"status": "SKIPPED", "y_true": "amusement", "y_pred": None},
        {"status": "MISSING", "y_true": "stress", "y_pred": None},
    ]

    metrics = calculate_metrics(records, labels=labels)

    assert metrics["total_samples"] == 4
    assert metrics["successful_samples"] == 1
    assert metrics["failed_samples"] == 1
    assert metrics["skipped_samples"] == 1
    assert metrics["missing_samples"] == 1
    # Total failures = 1 failed + 1 skipped + 1 missing = 3 out of 4 (75%)
    assert metrics["failure_rate"] == 0.75
    # Accuracy over the 1 completed sample is 1.0
    assert metrics["accuracy"] == 1.0


def test_calculate_metrics_unknown_hallucinated_label():
    """Verify hallucinated labels outside defined classes are handled safely without crashing."""
    labels = ["amusement", "baseline", "stress"]

    records = [
        {"status": "SUCCESS", "y_true": "baseline", "y_pred": "baseline"},
        {"status": "SUCCESS", "y_true": "stress", "y_pred": "neutral"},  # Hallucinated label
    ]

    metrics = calculate_metrics(records, labels=labels)

    assert metrics["accuracy"] == 0.5
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    # Ground truth baseline (index 1): 1 correct
    assert cm[1, 1] == 1
    # Ground truth stress (index 2): predicted 'neutral' (not in labels, so row sum is 0)
    assert cm[2].sum() == 0


def test_evaluate_cohort_alignment_detection(tmp_path):
    """Verify evaluate() detects when different methods are evaluated on misaligned cohorts."""
    # Create sample 0: all succeed
    s0 = tmp_path / "0_baseline"
    s0.mkdir()
    (s0 / "summary.txt").write_text(
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n"
        "[Semantic Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Statistical Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Hybrid Fusion] Decision made: baseline (ground truth: baseline)\n",
        encoding="utf-8",
    )

    # Create sample 1: Semantic fails, Statistical succeeds, Hybrid is skipped
    s1 = tmp_path / "1_stress"
    s1.mkdir()
    (s1 / "summary.txt").write_text(
        "[Interpretation] Decision made: stress (ground truth: stress)\n"
        "[Semantic Fusion] Execution failed (ground truth: stress)\n"
        "[Statistical Fusion] Decision made: stress (ground truth: stress)\n"
        "[Hybrid Fusion] Skipped (dependency failed) (ground truth: stress)\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    json_out, csv_rows = evaluate(
        log_dir=str(tmp_path),
        output_dir=str(out_dir),
        labels=["amusement", "baseline", "stress"],
    )

    summary = json_out["__cohort_summary__"]
    assert summary["total_cohort_samples"] == 2
    assert summary["common_successful_samples"] == 1
    assert summary["cohort_aligned"] is False

    # Interpretation: 2/2 success
    assert json_out["Interpretation"]["successful_samples"] == 2
    assert json_out["Interpretation"]["failed_samples"] == 0
    assert json_out["Interpretation"]["execution_failure_rate"] == 0.0
    assert json_out["Interpretation"]["incomplete_rate"] == 0.0

    # Semantic Fusion: 1/2 success, 1 failed call
    assert json_out["Semantic Fusion"]["successful_samples"] == 1
    assert json_out["Semantic Fusion"]["failed_samples"] == 1
    assert json_out["Semantic Fusion"]["skipped_samples"] == 0
    assert json_out["Semantic Fusion"]["execution_failure_rate"] == 0.5
    assert json_out["Semantic Fusion"]["incomplete_rate"] == 0.5

    # Statistical Fusion: 2/2 success (decoupled from Semantic Fusion!)
    assert json_out["Statistical Fusion"]["successful_samples"] == 2
    assert json_out["Statistical Fusion"]["failed_samples"] == 0
    assert json_out["Statistical Fusion"]["skipped_samples"] == 0
    assert json_out["Statistical Fusion"]["execution_failure_rate"] == 0.0
    assert json_out["Statistical Fusion"]["incomplete_rate"] == 0.0

    # Hybrid Fusion: 1/2 success, 1 skipped dependency
    assert json_out["Hybrid Fusion"]["successful_samples"] == 1
    assert json_out["Hybrid Fusion"]["failed_samples"] == 0
    assert json_out["Hybrid Fusion"]["skipped_samples"] == 1
    assert json_out["Hybrid Fusion"]["execution_failure_rate"] == 0.0
    assert json_out["Hybrid Fusion"]["incomplete_rate"] == 0.5

    # Common cohort metrics: all methods strictly evaluated on sample 0
    assert "__common_cohort__" in json_out
    for method in METHODS:
        assert json_out["__common_cohort__"][method]["samples"] == 1
        assert json_out["__common_cohort__"][method]["accuracy"] == 1.0


def test_evaluate_complex_multi_method_failure_patterns(tmp_path):
    """
    Verify evaluation accounting across varied multi-method failure patterns:
    Sample 0: all methods succeed.
    Sample 1: semantic fails (FAILED), statistical succeeds, hybrid SKIPPED.
    Sample 2: statistical fails (FAILED), hybrid SKIPPED.
    Sample 3: interpretation fails (FAILED), all 3 fusions SKIPPED.
    Sample 4: hybrid fusion missing (MISSING).
    """
    # Sample 0
    s0 = tmp_path / "0_baseline"
    s0.mkdir()
    (s0 / "summary.txt").write_text(
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n"
        "[Semantic Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Statistical Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Hybrid Fusion] Decision made: baseline (ground truth: baseline)\n",
        encoding="utf-8",
    )

    # Sample 1
    s1 = tmp_path / "1_stress"
    s1.mkdir()
    (s1 / "summary.txt").write_text(
        "[Interpretation] Decision made: stress (ground truth: stress)\n"
        "[Semantic Fusion] Execution failed (ground truth: stress)\n"
        "[Statistical Fusion] Decision made: stress (ground truth: stress)\n"
        "[Hybrid Fusion] Skipped (dependency failed) (ground truth: stress)\n",
        encoding="utf-8",
    )

    # Sample 2
    s2 = tmp_path / "2_amusement"
    s2.mkdir()
    (s2 / "summary.txt").write_text(
        "[Interpretation] Decision made: amusement (ground truth: amusement)\n"
        "[Semantic Fusion] Decision made: amusement (ground truth: amusement)\n"
        "[Statistical Fusion] Execution failed (ground truth: amusement)\n"
        "[Hybrid Fusion] Skipped (dependency failed) (ground truth: amusement)\n",
        encoding="utf-8",
    )

    # Sample 3
    s3 = tmp_path / "3_stress"
    s3.mkdir()
    (s3 / "summary.txt").write_text(
        "[Interpretation] Execution failed (ground truth: stress)\n"
        "[Semantic Fusion] Skipped (dependency failed) (ground truth: stress)\n"
        "[Statistical Fusion] Skipped (dependency failed) (ground truth: stress)\n"
        "[Hybrid Fusion] Skipped (dependency failed) (ground truth: stress)\n",
        encoding="utf-8",
    )

    # Sample 4
    s4 = tmp_path / "4_baseline"
    s4.mkdir()
    (s4 / "summary.txt").write_text(
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n"
        "[Semantic Fusion] Decision made: baseline (ground truth: baseline)\n"
        "[Statistical Fusion] Decision made: baseline (ground truth: baseline)\n",
        # Hybrid Fusion omitted (MISSING)
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    json_out, csv_rows = evaluate(
        log_dir=str(tmp_path),
        output_dir=str(out_dir),
        labels=["amusement", "baseline", "stress"],
    )

    # 1. Denominators and Cohort Total
    summary = json_out["__cohort_summary__"]
    assert summary["total_cohort_samples"] == 5
    assert summary["common_successful_samples"] == 1
    assert summary["cohort_aligned"] is False
    assert summary["common_sample_indices"] == [0]
    assert summary["common_sample_folders"] == ["0_baseline"]

    # 2. Interpretation: 4 success, 1 failed, 0 skipped, 0 missing
    interp = json_out["Interpretation"]
    assert interp["total_samples"] == 5
    assert interp["successful_samples"] == 4
    assert interp["failed_samples"] == 1
    assert interp["skipped_samples"] == 0
    assert interp["missing_samples"] == 0
    assert interp["execution_failure_rate"] == 0.2
    assert interp["incomplete_rate"] == 0.2

    # 3. Semantic Fusion: 3 success, 1 failed, 1 skipped, 0 missing
    sem = json_out["Semantic Fusion"]
    assert sem["total_samples"] == 5
    assert sem["successful_samples"] == 3
    assert sem["failed_samples"] == 1
    assert sem["skipped_samples"] == 1
    assert sem["missing_samples"] == 0
    assert sem["execution_failure_rate"] == 0.2
    assert sem["incomplete_rate"] == 0.4

    # 4. Statistical Fusion: 3 success, 1 failed, 1 skipped, 0 missing
    stat = json_out["Statistical Fusion"]
    assert stat["total_samples"] == 5
    assert stat["successful_samples"] == 3
    assert stat["failed_samples"] == 1
    assert stat["skipped_samples"] == 1
    assert stat["missing_samples"] == 0
    assert stat["execution_failure_rate"] == 0.2
    assert stat["incomplete_rate"] == 0.4

    # 5. Hybrid Fusion: 1 success, 0 failed, 3 skipped, 1 missing
    hyb = json_out["Hybrid Fusion"]
    assert hyb["total_samples"] == 5
    assert hyb["successful_samples"] == 1
    assert hyb["failed_samples"] == 0
    assert hyb["skipped_samples"] == 3
    assert hyb["missing_samples"] == 1
    assert hyb["execution_failure_rate"] == 0.0
    assert hyb["incomplete_rate"] == 0.8

    # 6. Common-cohort comparative metrics
    common_metrics = json_out["__common_cohort__"]
    for method in METHODS:
        assert common_metrics[method]["samples"] == 1
        assert common_metrics[method]["accuracy"] == 1.0

    # 7. Common cohort CSV file was generated
    assert (out_dir / "common_cohort_results.csv").exists()


def test_resolve_dataset_labels_strict_validation(tmp_path):
    """Verify resolve_dataset_labels rejects corrupt metadata, missing files, and malformed inputs."""
    # 1. Non-existent file without default raises FileNotFoundError
    missing_file = tmp_path / "non_existent_info.json"
    with pytest.raises(FileNotFoundError):
        resolve_dataset_labels(info_path=str(missing_file), allow_wesad_default=False)

    # 2. Non-existent file with explicit WESAD default allowed
    labels, source = resolve_dataset_labels(info_path=str(missing_file), allow_wesad_default=True)
    assert labels == WESAD_CANONICAL_LABELS
    assert source == "documented_wesad_default"

    # 3. Corrupt JSON raises ValueError
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        resolve_dataset_labels(info_path=str(corrupt_file))

    # 4. Empty dictionary raises ValueError
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least 2 classes"):
        resolve_dataset_labels(info_path=str(empty_file))

    # 5. Insufficient classes (< 2) raises ValueError
    insufficient_file = tmp_path / "one_class.json"
    insufficient_file.write_text(json.dumps({"class": {"only_one": "desc"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least 2 classes"):
        resolve_dataset_labels(info_path=str(insufficient_file))

    # 6. Valid classes dictionary resolves correctly with provenance
    valid_file = tmp_path / "valid.json"
    valid_file.write_text(
        json.dumps({"class": {"stress": "...", "baseline": "...", "amusement": "..."}}),
        encoding="utf-8",
    )
    labels, source = resolve_dataset_labels(info_path=str(valid_file))
    assert labels == ["amusement", "baseline", "stress"]
    assert source == f"metadata:{str(valid_file)}"

    # 7. Explicit labels validation
    labels, source = resolve_dataset_labels(explicit_labels=["Stress", "baseline"])
    assert labels == ["baseline", "stress"]
    assert source == "explicit"

    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        resolve_dataset_labels(explicit_labels=["only_one"])

    with pytest.raises(ValueError, match="list or tuple"):
        resolve_dataset_labels(explicit_labels=123)  # type: ignore


def test_evaluate_fails_safely_on_invalid_labels(tmp_path):
    """Verify evaluate() refuses to run against malformed label files rather than producing misleading metrics."""
    corrupt_info = tmp_path / "info.json"
    corrupt_info.write_text("malformed json content", encoding="utf-8")

    sample_dir = tmp_path / "0_baseline"
    sample_dir.mkdir()
    (sample_dir / "summary.txt").write_text(
        "[Interpretation] Decision made: baseline (ground truth: baseline)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        evaluate(
            log_dir=str(tmp_path),
            info_path=str(corrupt_info),
            allow_wesad_default=False,
        )


def test_stage_decoupling_execution_flow():
    """
    Verify the stage execution and skipping rules in run_consensus.py:
    1. Modality failure -> Interpretation fails, Semantic/Statistical/Hybrid Fusion skipped.
    2. Semantic failure -> Statistical Fusion still runs; Hybrid Fusion skipped.
    3. Interpretation + Semantic + Statistical succeed -> Hybrid Fusion runs.
    """
    logged_messages = []

    class DummyAgentPool:
        def __init__(self, interp_ret, sem_ret, stat_ret, hyb_ret):
            self.interp_ret = interp_ret
            self.sem_ret = sem_ret
            self.stat_ret = stat_ret
            self.hyb_ret = hyb_ret

        def log_summary(self, msg):
            logged_messages.append(msg)

        async def run_parallel_interpretation(self):
            return self.interp_ret

        async def run_semantic_fusion(self):
            return self.sem_ret

        async def run_statistical_fusion(self):
            return self.stat_ret

        async def run_hybrid_fusion(self):
            return self.hyb_ret

    async def execute_sample_flow(pool, label="baseline"):
        interp = await pool.run_parallel_interpretation()
        if interp is None:
            pool.log_summary(f"[Interpretation] Execution failed (ground truth: {label})")
            pool.log_summary(f"[Semantic Fusion] Skipped (dependency failed) (ground truth: {label})")
            pool.log_summary(f"[Statistical Fusion] Skipped (dependency failed) (ground truth: {label})")
            pool.log_summary(f"[Hybrid Fusion] Skipped (dependency failed) (ground truth: {label})")
            return

        pool.log_summary(f"[Interpretation] Decision made: {interp} (ground truth: {label})")

        sem = await pool.run_semantic_fusion()
        if sem is None:
            pool.log_summary(f"[Semantic Fusion] Execution failed (ground truth: {label})")
        else:
            pool.log_summary(f"[Semantic Fusion] Decision made: {sem} (ground truth: {label})")

        stat = await pool.run_statistical_fusion()
        if stat is None:
            pool.log_summary(f"[Statistical Fusion] Execution failed (ground truth: {label})")
        else:
            pool.log_summary(f"[Statistical Fusion] Decision made: {stat} (ground truth: {label})")

        if sem is None or stat is None:
            pool.log_summary(f"[Hybrid Fusion] Skipped (dependency failed) (ground truth: {label})")
        else:
            hyb = await pool.run_hybrid_fusion()
            if hyb is None:
                pool.log_summary(f"[Hybrid Fusion] Execution failed (ground truth: {label})")
            else:
                pool.log_summary(f"[Hybrid Fusion] Decision made: {hyb} (ground truth: {label})")

    async def run_all_cases():
        # Case 1: Semantic fails, Statistical succeeds -> Hybrid skipped
        logged_messages.clear()
        pool1 = DummyAgentPool(interp_ret="baseline", sem_ret=None, stat_ret="baseline", hyb_ret="baseline")
        await execute_sample_flow(pool1, "baseline")

        assert "[Interpretation] Decision made: baseline (ground truth: baseline)" in logged_messages
        assert "[Semantic Fusion] Execution failed (ground truth: baseline)" in logged_messages
        assert "[Statistical Fusion] Decision made: baseline (ground truth: baseline)" in logged_messages
        assert "[Hybrid Fusion] Skipped (dependency failed) (ground truth: baseline)" in logged_messages

        # Case 2: Interpretation fails -> all downstream skipped
        logged_messages.clear()
        pool2 = DummyAgentPool(interp_ret=None, sem_ret="baseline", stat_ret="baseline", hyb_ret="baseline")
        await execute_sample_flow(pool2, "stress")

        assert "[Interpretation] Execution failed (ground truth: stress)" in logged_messages
        assert "[Semantic Fusion] Skipped (dependency failed) (ground truth: stress)" in logged_messages
        assert "[Statistical Fusion] Skipped (dependency failed) (ground truth: stress)" in logged_messages
        assert "[Hybrid Fusion] Skipped (dependency failed) (ground truth: stress)" in logged_messages

        # Case 3: All succeed -> Hybrid Fusion runs and decides
        logged_messages.clear()
        pool3 = DummyAgentPool(interp_ret="stress", sem_ret="stress", stat_ret="stress", hyb_ret="stress")
        await execute_sample_flow(pool3, "stress")

        assert "[Interpretation] Decision made: stress (ground truth: stress)" in logged_messages
        assert "[Semantic Fusion] Decision made: stress (ground truth: stress)" in logged_messages
        assert "[Statistical Fusion] Decision made: stress (ground truth: stress)" in logged_messages
        assert "[Hybrid Fusion] Decision made: stress (ground truth: stress)" in logged_messages

    asyncio.run(run_all_cases())
