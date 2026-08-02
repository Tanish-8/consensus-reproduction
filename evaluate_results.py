import os
import re
import json
import csv
from collections import defaultdict

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

results = defaultdict(lambda: {"y_true": [], "y_pred": []})

pattern = re.compile(
    r"\[(.*?)\]\sDecision made:\s(.*?)\s\(ground truth:\s(.*?)\)"
)

for folder in sorted(os.listdir(LOG_DIR)):
    summary_path = os.path.join(LOG_DIR, folder, "summary.txt")

    if not os.path.exists(summary_path):
        continue

    with open(summary_path, "r", encoding="utf-8") as f:

        last_prediction = {}

        for line in f:
            match = pattern.search(line)

            if not match:
                continue

            method = match.group(1).strip()

            if method not in METHODS:
                continue

            pred = match.group(2).strip()
            truth = match.group(3).strip()

            last_prediction[method] = (truth, pred)

        for method, (truth, pred) in last_prediction.items():
            results[method]["y_true"].append(truth)
            results[method]["y_pred"].append(pred)
            
labels = sorted(
    list(
        set(
            sum([results[m]["y_true"] for m in METHODS], [])
        )
    )
)

json_output = {}
csv_rows = []

print("=" * 70)

for method in METHODS:

    y_true = results[method]["y_true"]
    y_pred = results[method]["y_pred"]

    if len(y_true) == 0:
        continue

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{method}")
    print("-" * 40)
    print(f"Samples   : {len(y_true)}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    json_output[method] = {
        "samples": len(y_true),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
    }

    csv_rows.append({
        "Method": method,
        "Samples": len(y_true),
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    })

    cm_file = f"{method.lower().replace(' ', '_')}_confusion_matrix.csv"

    with open(cm_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([""] + labels)

        for label, row in zip(labels, cm):
            writer.writerow([label] + list(row))

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4)

with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Method",
            "Samples",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ],
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print("\nSaved:")
print("evaluation_results.json")
print("evaluation_results.csv")

print("\nDone.")