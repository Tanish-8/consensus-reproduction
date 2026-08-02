# Reproduction Report

## Paper

**Title:** ConSensus: Multi-Agent Collaboration for Multimodal Sensing

**Conference:** ACL 2026 Findings

**Authors:**
- Hyungjun Yoon
- Mohammad Malekzadeh
- Sung-Ju Lee
- Fahim Kawsar
- Lorena Qendro

---

# Objective

The objective of this project was to reproduce the WESAD experiments of the ConSensus framework using the publicly available source code and evaluate its performance using the Llama-3.1-8B-Instruct model served through NVIDIA Build.

---

# Environment

Operating System:
- Windows

Python Version:
- Python 3.12

Dataset:
- WESAD

LLM:
- Llama-3.1-8B-Instruct

Inference Provider:
- NVIDIA Build API

Temperature:
- 0.0

Top-p:
- 1.0

Seed:
- 0

---

## Project Structure

```
multi-agent-collaboration-for-multimodal-sensing/
│
├── baselines/
├── config/
│   ├── example.yaml
│   └── wesad_nvidia.yaml
│
├── core/
├── preprocess/
├── data/
├── logs/
│
├── run_consensus.py
├── evaluate_results.py
├── REPRODUCTION_REPORT.md
└── requirements.txt
```

---

# Dataset Preparation

1. Downloaded the WESAD dataset.
2. Successfully preprocessed the dataset.
3. Split the dataset using the repository's split script.
4. Generated a test dataset containing 90 evaluation samples.

---

## Reproduction Procedure

The following steps were performed to reproduce the ConSensus experiments on the WESAD dataset.

### Step 1: Repository Setup

- Cloned the official ConSensus GitHub repository.
- Created a Python 3.12 virtual environment.
- Installed all required dependencies using:

```bash
pip install -r requirements.txt
```

### Step 2: Dataset Preparation

- Downloaded the WESAD dataset.
- Preprocessed the raw dataset using:

```bash
python preprocess/preprocess_WESAD.py \
    --path <WESAD_PATH> \
    --out_dir ./data/WESAD
```

- Split the processed dataset into train/examples/test using:

```bash
python preprocess/split.py \
    --data_path ./data/WESAD \
    --test_per_class 30 \
    --examples_per_class 1 \
    --output_path ./data/WESAD/split
```

### Step 3: Model Configuration

- Created a new configuration file (`config/wesad_nvidia.yaml`).
- Configured the experiment to use:
  - NVIDIA Build API
  - Llama-3.1-8B-Instruct
  - Temperature = 0.0
  - Top-p = 1.0

### Step 4: Running the Experiment

Executed the experiment using:

```bash
python run_consensus.py --config_path config/wesad_nvidia.yaml
```

### Step 5: Evaluation

After inference completed, generated the evaluation metrics using:

```bash
python evaluate_results.py
```

The evaluation script computed:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrices

and exported the results as JSON and CSV files.

---

# Code Modifications

The following modifications were required to successfully reproduce the experiments.

### 1. Python Compatibility

Modified the WESAD preprocessing script to correctly load legacy pickle files using Latin-1 encoding.

### 2. NVIDIA API Integration

Added support for NVIDIA Build API inside `core/model.py`.

### 3. Configuration

Created a new configuration file:

config/wesad_nvidia.yaml

to run experiments using NVIDIA-hosted Llama-3.1-8B-Instruct.

### 4. Evaluation

Implemented a new evaluation script (`evaluate_results.py`) because the original repository did not include evaluation code for computing final metrics.

---

## Repository Modifications

The original repository required several modifications before the experiments could be successfully reproduced.

| Modified File                     | Changes Made                                                                                  |
|:----------------------------------|:----------------------------------------------------------------------------------------------|
| `preprocess/preprocess_WESAD.py`  | • Fixed legacy WESAD pickle loading using `pickle.load(..., encoding="latin1")`.<br>• Improved Python 3 compatibility. |
| `core/model.py`                   | • Added NVIDIA Build API support.<br>• Implemented retry logic with exponential backoff for HTTP 429 errors.<br>• Added a short delay before asynchronous API requests to reduce rate limiting. |
| `config/wesad_nvidia.yaml`        | • Created a configuration file for running Llama-3.1-8B-Instruct on NVIDIA Build. |
| `evaluate_results.py`             | • Implemented the evaluation pipeline.<br>• Computed Accuracy, Precision, Recall, Macro F1-score, and Confusion Matrices.<br>• Exported results to JSON and CSV files. |

---

# Experimental Results

| Method                     | Accuracy | Precision | Recall | Macro F1 |
|:---------------------------|---------:|----------:|-------:|---------:|
| Interpretation             | 90.00%   | 90.38%    | 90.00% | 89.85%   |
| Semantic Fusion            | 83.33%   | 85.28%    | 83.33% | 82.24%   |
| Statistical Fusion         | 90.00%   | 90.38%    | 90.00% | 89.85%   |
| Hybrid Fusion (ConSensus)  | 86.67%   | 87.08%    | 86.67% | 86.35%   |

---

## Comparison with the Paper

The reproduced experiments were compared against the results reported in Table 2 of the paper for the WESAD dataset using Llama-3.1-8B-Instruct.

| Method                    | Paper Accuracy | Reproduced Accuracy | Difference |
|:--------------------------|---------------:|--------------------:|-----------:|
| Semantic Fusion           | 77.3%          | 83.33%              | +6.03%     |
| Statistical Fusion        | 78.7%          | 90.00%              | +11.30%    |
| Hybrid Fusion (ConSensus) | 78.0%          | 86.67%              | +8.67%     |

The reproduced accuracies were consistently higher than those reported in the paper.

Several factors may explain these differences:

- The experiments were executed using the NVIDIA Build API instead of the original inference environment.
- Minor differences may exist in dependency versions and software libraries.
- Python compatibility fixes were required for loading the WESAD dataset.
- The original repository did not provide an evaluation pipeline; a custom evaluation script was implemented.
- Small implementation or prompt formatting differences may influence LLM predictions.

---

## Challenges Encountered

During the reproduction process, several technical issues were encountered and resolved:

1. The WESAD dataset uses legacy Python pickle serialization that is incompatible with modern Python versions without specifying `encoding="latin1"`.

2. The original repository did not include native support for NVIDIA Build. Support was implemented by extending the model loading pipeline.

3. Free NVIDIA Build endpoints occasionally returned HTTP 429 (Too Many Requests). Retry logic with exponential backoff was added to improve robustness.

4. The repository generated per-sample predictions but did not provide an evaluation script for computing aggregate metrics. A separate evaluation script was implemented.

5. The initial evaluation counted duplicate Interpretation decisions. The evaluation script was corrected to use only the final decision for each sample.

---

## Future Work

Possible extensions include:

- Reproduce experiments using GPT-OSS-20B.
- Reproduce experiments using GPT-OSS-120B.
- Evaluate additional datasets from the paper (PAMAP2, SleepEDF, ActionSense, MMFit).
- Compare NVIDIA Build with local inference using Ollama.
- Reproduce all baselines (Single-Agent, Debate, MAD, CMD, ReConcile) and compare against the published results.

---

## Conclusion

This work successfully reproduced the ConSensus inference pipeline on the WESAD dataset using Llama-3.1-8B-Instruct hosted through NVIDIA Build.

The repository was successfully adapted to support NVIDIA Build, compatibility issues with the WESAD dataset were resolved, and an evaluation pipeline was implemented to compute the final classification metrics.

The reproduced experiments achieved competitive performance across all fusion strategies. Although the obtained accuracies differed from those reported in the paper, the complete experimental pipeline was successfully executed, evaluated, and documented.

The developed evaluation framework and documented repository modifications provide a reproducible foundation for extending the experiments to additional datasets and language models in future work.

---

## Appendix

The following files were generated during the reproduction and evaluation process.

| Generated File                              | Description                                                 |
|:--------------------------------------------|:------------------------------------------------------------|
| `evaluation_results.json`                   | Stores the overall evaluation metrics in JSON format.         |
| `evaluation_results.csv`                    | Stores the overall evaluation metrics in CSV format.          |
| `interpretation_confusion_matrix.csv`       | Confusion matrix for the Interpretation method.             |
| `semantic_fusion_confusion_matrix.csv`      | Confusion matrix for the Semantic Fusion method.            |
| `statistical_fusion_confusion_matrix.csv`   | Confusion matrix for the Statistical Fusion method.         |
| `hybrid_fusion_confusion_matrix.csv`        | Confusion matrix for the Hybrid Fusion (ConSensus) method.  |

**Experiment Logs**

The detailed execution logs for each evaluation sample are stored in:

```text
logs/
└── WESAD_Llama31_NVIDIA/
```

Each sample directory contains:
- Agent interaction logs
- Intermediate reasoning
- Fusion decisions
- Final prediction summary