import os
import re
import json
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

LOG_DIR = "./logs/WESAD_Llama31_NVIDIA"

METHODS = [
    "Interpretation",
    "Semantic Fusion",
    "Statistical Fusion",
    "Hybrid Fusion",
]

WESAD_CANONICAL_LABELS = [
    "amusement",
    "baseline",
    "stress",
]

DEFAULT_LABELS = list(WESAD_CANONICAL_LABELS)

DECISION_PATTERN = re.compile(
    r"\[(.*?)\]\sDecision made:\s(.*?)\s\(ground truth:\s(.*?)\)"
)

SKIPPED_PATTERN = re.compile(
    r"\[(.*?)\]\sSkipped(?:\s\(.*?\))?(?:\s\(ground truth:\s(.*?)\))?"
)

FAILURE_PATTERN = re.compile(
    r"\[(.*?)\]\s(?:Execution failed|Failed)(?:\s\(ground truth:\s(.*?)\))?"
)


def resolve_dataset_labels(
    info_path: Optional[str] = None,
    explicit_labels: Optional[List[str]] = None,
    allow_wesad_default: bool = False,
) -> Tuple[List[str], str]:
    """
    Resolve and strictly validate canonical class labels with provenance tracking.

    Returns:
        Tuple of (sorted_distinct_lowercase_labels, label_source)

    Raises:
        ValueError: If labels are unknown, malformed, or contain fewer than 2 distinct classes.
        FileNotFoundError: If info_path is given but does not exist and allow_wesad_default is False.
    """
    if explicit_labels is not None:
        if not isinstance(explicit_labels, (list, tuple)):
            raise ValueError(
                f"Explicit labels must be a list or tuple of strings, got {type(explicit_labels).__name__}"
            )
        cleaned = sorted(list(set(str(k).strip().lower() for k in explicit_labels if str(k).strip())))
        if len(cleaned) < 2:
            raise ValueError(
                f"Explicit labels must contain at least 2 distinct classes, got: {explicit_labels}"
            )
        return cleaned, "explicit"

    if info_path is not None:
        if not os.path.exists(info_path):
            if allow_wesad_default:
                return list(WESAD_CANONICAL_LABELS), "documented_wesad_default"
            raise FileNotFoundError(f"Dataset label metadata file not found: '{info_path}'")

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Malformed label metadata file '{info_path}': invalid JSON ({e})") from e

        candidate = None
        if isinstance(data, dict):
            if "class" in data and isinstance(data["class"], dict):
                candidate = [str(k).strip().lower() for k in data["class"].keys() if str(k).strip()]
            elif "classes" in data and isinstance(data["classes"], list):
                candidate = [str(k).strip().lower() for k in data["classes"] if str(k).strip()]

        if not candidate or len(set(candidate)) < 2:
            raise ValueError(
                f"Malformed label metadata in '{info_path}': expected at least 2 classes in 'class' dict or 'classes' list, found: {candidate}"
            )

        return sorted(list(set(candidate))), f"metadata:{info_path}"

    if allow_wesad_default:
        return list(WESAD_CANONICAL_LABELS), "documented_wesad_default"

    raise ValueError(
        "Dataset class labels are unknown. Provide explicit labels, a valid metadata path, or explicitly enable allow_wesad_default."
    )


def load_dataset_labels(
    info_path: str = "./data/WESAD/info.json",
    allow_wesad_default: bool = False,
) -> List[str]:
    """Load canonical class labels from dataset metadata if available."""
    labels, _ = resolve_dataset_labels(info_path=info_path, allow_wesad_default=allow_wesad_default)
    return labels


def parse_sample_folder_name(folder_name: str) -> Tuple[Optional[int], str]:
    """Extract sample index and ground truth from a folder name like '42_stress'."""
    parts = folder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return int(parts[0]), parts[1].strip().lower()
    return None, folder_name.strip().lower()


def parse_sample_summary(
    summary_path: str,
    folder_name: str,
    methods: List[str] = METHODS,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse a sample's summary.txt to extract the final status and prediction for each method.
    Returns:
        {
            method: {
                "status": "SUCCESS" | "FAILED" | "MISSING",
                "y_true": str,
                "y_pred": Optional[str]
            }
        }
    """
    sample_idx, folder_truth = parse_sample_folder_name(folder_name)
    sample_records: Dict[str, Dict[str, Any]] = {}

    # Initialize all methods as MISSING by default
    for method in methods:
        sample_records[method] = {
            "status": "MISSING",
            "y_true": folder_truth,
            "y_pred": None,
        }

    if not os.path.exists(summary_path):
        return sample_records

    with open(summary_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Track the last recorded event for each method in this file
    for line in lines:
        line_clean = line.strip()

        # Check for successful decision
        dec_match = DECISION_PATTERN.search(line_clean)
        if dec_match:
            method = dec_match.group(1).strip()
            if method in methods:
                pred = dec_match.group(2).strip().lower()
                truth = dec_match.group(3).strip().lower()
                sample_records[method] = {
                    "status": "SUCCESS",
                    "y_true": truth or folder_truth,
                    "y_pred": pred,
                }
            continue

        # Check for skipped stage (dependency failure)
        skip_match = SKIPPED_PATTERN.search(line_clean)
        if skip_match:
            method = skip_match.group(1).strip()
            if method in methods:
                truth = skip_match.group(2).strip().lower() if skip_match.group(2) else folder_truth
                sample_records[method] = {
                    "status": "SKIPPED",
                    "y_true": truth or folder_truth,
                    "y_pred": None,
                }
            continue

        # Check for explicit failure
        fail_match = FAILURE_PATTERN.search(line_clean)
        if fail_match:
            method = fail_match.group(1).strip()
            if method in methods:
                truth = fail_match.group(2).strip().lower() if fail_match.group(2) else folder_truth
                sample_records[method] = {
                    "status": "FAILED",
                    "y_true": truth or folder_truth,
                    "y_pred": None,
                }
            continue

        # Check for generic error logs
        if "[Error] Failed to interpret" in line_clean:
            if "Interpretation" in methods:
                sample_records["Interpretation"] = {
                    "status": "FAILED",
                    "y_true": folder_truth,
                    "y_pred": None,
                }
        elif "[Error] Semantic fusion failed" in line_clean:
            if "Semantic Fusion" in methods:
                sample_records["Semantic Fusion"] = {
                    "status": "FAILED",
                    "y_true": folder_truth,
                    "y_pred": None,
                }
        elif "[Error] Statistical fusion failed" in line_clean:
            if "Statistical Fusion" in methods:
                sample_records["Statistical Fusion"] = {
                    "status": "FAILED",
                    "y_true": folder_truth,
                    "y_pred": None,
                }
        elif "[Error] Hybrid fusion failed" in line_clean:
            if "Hybrid Fusion" in methods:
                sample_records["Hybrid Fusion"] = {
                    "status": "FAILED",
                    "y_true": folder_truth,
                    "y_pred": None,
                }

    return sample_records


def calculate_metrics(
    records: List[Dict[str, Any]],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Compute classification metrics and transparent failure/completion statistics.
    Explicitly uses `labels` for macro-F1 and confusion matrix consistency.
    """
    total_samples = len(records)
    successful_records = [r for r in records if r["status"] == "SUCCESS"]
    failed_records = [r for r in records if r["status"] == "FAILED"]
    skipped_records = [r for r in records if r["status"] == "SKIPPED"]
    missing_records = [r for r in records if r["status"] == "MISSING"]

    failed_count = len(failed_records)
    skipped_count = len(skipped_records)
    missing_count = len(missing_records)
    incomplete_count = failed_count + skipped_count + missing_count

    execution_failure_rate = (failed_count / total_samples) if total_samples > 0 else 0.0
    skipped_rate = (skipped_count / total_samples) if total_samples > 0 else 0.0
    missing_rate = (missing_count / total_samples) if total_samples > 0 else 0.0
    incomplete_rate = (incomplete_count / total_samples) if total_samples > 0 else 0.0

    if not successful_records:
        empty_cm = np.zeros((len(labels), len(labels)), dtype=int).tolist()
        return {
            "samples": 0,
            "total_samples": total_samples,
            "successful_samples": 0,
            "failed_samples": failed_count,
            "skipped_samples": skipped_count,
            "missing_samples": missing_count,
            "incomplete_samples": incomplete_count,
            "execution_failure_rate": round(execution_failure_rate, 4),
            "skipped_rate": round(skipped_rate, 4),
            "missing_rate": round(missing_rate, 4),
            "incomplete_rate": round(incomplete_rate, 4),
            "failure_rate": round(incomplete_rate, 4),  # compatibility alias
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": empty_cm,
            "labels": labels,
        }

    y_true = [r["y_true"] for r in successful_records]
    y_pred = [r["y_pred"] for r in successful_records]

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "samples": len(y_true),
        "total_samples": total_samples,
        "successful_samples": len(successful_records),
        "failed_samples": failed_count,
        "skipped_samples": skipped_count,
        "missing_samples": missing_count,
        "incomplete_samples": incomplete_count,
        "execution_failure_rate": round(execution_failure_rate, 4),
        "skipped_rate": round(skipped_rate, 4),
        "missing_rate": round(missing_rate, 4),
        "incomplete_rate": round(incomplete_rate, 4),
        "failure_rate": round(incomplete_rate, 4),  # compatibility alias
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def evaluate(
    log_dir: str = LOG_DIR,
    methods: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    info_path: Optional[str] = "./data/WESAD/info.json",
    allow_wesad_default: bool = True,
    output_dir: str = ".",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation across all sample folders in log_dir, outputting metrics and matrices.
    """
    if methods is None:
        methods = list(METHODS)

    if labels is not None:
        resolved_labels, label_source = resolve_dataset_labels(explicit_labels=labels)
    else:
        resolved_labels, label_source = resolve_dataset_labels(
            info_path=info_path,
            allow_wesad_default=allow_wesad_default,
        )

    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Discover and numerically sort sample folders
    all_entries = os.listdir(log_dir)
    sample_folders = []
    for entry in all_entries:
        entry_path = os.path.join(log_dir, entry)
        if os.path.isdir(entry_path):
            has_summary = os.path.exists(os.path.join(entry_path, "summary.txt"))
            is_sample_name = parse_sample_folder_name(entry)[0] is not None
            if has_summary or is_sample_name:
                sample_folders.append(entry)

    def sort_key(name: str):
        idx, _ = parse_sample_folder_name(name)
        return (0, idx) if idx is not None else (1, name)

    sample_folders.sort(key=sort_key)
    total_cohort = len(sample_folders)

    # Collect per-sample records for each method
    method_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    common_success_samples = set(range(total_cohort))

    for sample_i, folder in enumerate(sample_folders):
        summary_path = os.path.join(log_dir, folder, "summary.txt")
        records = parse_sample_summary(summary_path, folder, methods=methods)

        for method in methods:
            rec = records[method]
            rec["folder"] = folder
            method_records[method].append(rec)
            if rec["status"] != "SUCCESS":
                common_success_samples.discard(sample_i)

    # Calculate metrics
    json_output: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"Evaluation Cohort: {total_cohort} sample folders in '{log_dir}'")
    print(f"Common Complete Samples across all methods: {len(common_success_samples)}/{total_cohort}")
    print(f"Class Labels ({label_source}): {resolved_labels}")
    print("=" * 80)

    for method in methods:
        recs = method_records[method]
        metrics = calculate_metrics(recs, labels=resolved_labels)
        json_output[method] = metrics

        acc = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1_score"]
        cm = metrics["confusion_matrix"]

        print(f"\n{method}")
        print("-" * 40)
        print(f"Total Cohort       : {metrics['total_samples']}")
        print(f"Successful         : {metrics['successful_samples']}")
        print(f"Execution Failures : {metrics['failed_samples']} ({metrics['execution_failure_rate'] * 100:.2f}%)")
        print(f"Skipped Dep Errors : {metrics['skipped_samples']} ({metrics['skipped_rate'] * 100:.2f}%)")
        print(f"Missing Outputs    : {metrics['missing_samples']} ({metrics['missing_rate'] * 100:.2f}%)")
        print(f"Incomplete Rate    : {metrics['incomplete_rate'] * 100:.2f}%")
        print(f"Accuracy           : {acc:.4f}")
        print(f"Macro Prec         : {precision:.4f}")
        print(f"Macro Recall       : {recall:.4f}")
        print(f"Macro F1           : {f1:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(cm))

        csv_rows.append({
            "Method": method,
            "Samples": metrics["samples"],
            "Total_Samples": metrics["total_samples"],
            "Failed": metrics["failed_samples"],
            "Skipped": metrics["skipped_samples"],
            "Missing": metrics["missing_samples"],
            "Execution_Failure_Rate": metrics["execution_failure_rate"],
            "Incomplete_Rate": metrics["incomplete_rate"],
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        })

        # Save confusion matrix CSV
        cm_filename = f"{method.lower().replace(' ', '_')}_confusion_matrix.csv"
        cm_path = os.path.join(output_dir, cm_filename)
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + resolved_labels)
            for label, row in zip(resolved_labels, cm):
                writer.writerow([label] + list(row))

    common_indices = sorted(list(common_success_samples))
    is_aligned = (len(common_indices) == total_cohort)

    # Calculate fair direct comparisons restricted strictly to common successful sample cohort
    common_cohort_metrics: Dict[str, Any] = {}
    common_csv_rows: List[Dict[str, Any]] = []

    for method in methods:
        common_recs = [method_records[method][i] for i in common_indices]
        c_metrics = calculate_metrics(common_recs, labels=resolved_labels)
        common_cohort_metrics[method] = c_metrics
        common_csv_rows.append({
            "Method": method,
            "Samples": c_metrics["samples"],
            "Total_Samples": c_metrics["total_samples"],
            "Failed": c_metrics["failed_samples"],
            "Skipped": c_metrics["skipped_samples"],
            "Missing": c_metrics["missing_samples"],
            "Execution_Failure_Rate": c_metrics["execution_failure_rate"],
            "Incomplete_Rate": c_metrics["incomplete_rate"],
            "Accuracy": round(c_metrics["accuracy"], 4),
            "Precision": round(c_metrics["precision"], 4),
            "Recall": round(c_metrics["recall"], 4),
            "F1": round(c_metrics["f1_score"], 4),
        })

    if not is_aligned:
        print("\n" + "=" * 80)
        print(f"COMMON COHORT DIRECT COMPARISON ({len(common_indices)}/{total_cohort} shared successful samples)")
        print("=" * 80)
        for method in methods:
            cm_m = common_cohort_metrics[method]
            print(
                f"{method:<20} | Acc: {cm_m['accuracy']:.4f} | Prec: {cm_m['precision']:.4f} | "
                f"Rec: {cm_m['recall']:.4f} | F1: {cm_m['f1_score']:.4f}"
            )

        common_csv_path = os.path.join(output_dir, "common_cohort_results.csv")
        with open(common_csv_path, "w", newline="", encoding="utf-8") as f:
            c_writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Method",
                    "Samples",
                    "Total_Samples",
                    "Failed",
                    "Skipped",
                    "Missing",
                    "Execution_Failure_Rate",
                    "Incomplete_Rate",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                ],
            )
            c_writer.writeheader()
            c_writer.writerows(common_csv_rows)

    # Add metadata summary to json_output
    json_output["__cohort_summary__"] = {
        "log_dir": log_dir,
        "total_cohort_samples": total_cohort,
        "common_successful_samples": len(common_indices),
        "cohort_aligned": is_aligned,
        "common_sample_indices": common_indices,
        "common_sample_folders": [sample_folders[i] for i in common_indices],
        "labels": resolved_labels,
        "label_source": label_source,
    }
    json_output["__common_cohort__"] = common_cohort_metrics

    # Save JSON results
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)

    # Save CSV results
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Samples",
                "Total_Samples",
                "Failed",
                "Skipped",
                "Missing",
                "Execution_Failure_Rate",
                "Incomplete_Rate",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 80)
    print("Saved:")
    print(f" - {json_path}")
    print(f" - {csv_path}")
    print("Done.\n")

    return json_output, csv_rows


if __name__ == "__main__":
    evaluate()
