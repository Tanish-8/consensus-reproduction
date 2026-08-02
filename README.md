# ConSensus: Multi-Agent Collaboration for Multimodal Sensing (ACL '26)
This is the official Python implementation of "ConSensus: Multi-Agent Collaboration for Multimodal Sensing (ACL '26 Findings, long paper)" by
[Hyungjun Yoon](https://hjyoon.com/),
[Mohammad Malekzadeh](https://mmalekzadeh.github.io/),
[Sung-Ju Lee](https://sites.google.com/site/wewantsj/),
[Fahim Kawsar](https://fahim-kawsar.net/), and
[Lorena Qendro](https://lorenaqendro.github.io/).

[![arXiv](https://img.shields.io/badge/arXiv-2601.06453-b31b1b.svg)](https://arxiv.org/abs/2601.06453)

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) (recommended) or Python 3.12+
- [Ollama](https://ollama.com/) (for local model inference)

### Installation

1. Create a conda environment:
```bash
conda create -n consensus python=3.12
conda activate consensus
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (as needed):
   - For OpenAI: `export OPENAI_API_KEY=your_key`
   - For Together AI: `export TOGETHER_API_KEY=your_key`
   - For Ollama: see [Ollama Setup](#ollama-setup) below

## Data Preprocessing

### Supported Datasets

The following datasets are supported:
- **WESAD**: `preprocess_WESAD.py`
- **ActionSense**: `preprocess_ActionSense.py`
- **PAMAP2**: `preprocess_PAMAP2.py`
- **SleepEDF**: `preprocess_SleepEDF.py`
- **MMFit**: `preprocess_MMFit.py`

### Preprocessing Steps

1. **Process raw data:**
```bash
python preprocess/preprocess_<dataset>.py --path <data_path> --out_dir <processed_output_path>
```

2. **Split processed data:**
```bash
python preprocess/split.py \
    --data_path <processed_output_path> \
    --test_per_class 30 \
    --examples_per_class 1 \
    --output_path <split_output_path>
```

### Simulating Missing Modalities

To simulate missing modalities (e.g., 30% missing):

1. **Process with missing ratio:**
```bash
python preprocess/preprocess_<dataset>.py \
    --path <data_path> \
    --out_dir <processed_output_path> \
    --missing_ratio 0.3
```

2. **Split with missing flag:**
```bash
python preprocess/split.py \
    --data_path <processed_output_path> \
    --test_per_class 30 \
    --examples_per_class 1 \
    --output_path <split_output_path> \
    --missing
```

## Running Experiments

### Model Configuration

The configuration file (YAML) controls all experiment parameters. An example is provided in `config/example.yaml`.

#### Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `seed` | Random seed for reproducibility | `0` |
| `temperature` | Sampling temperature. `0.0` for deterministic (greedy) decoding, `0.7` for self-consistency. | `0.0` |
| `top_p` | Nucleus sampling probability. `1.0` disables nucleus filtering (uses full distribution). | `1.0` |
| `use_large_context` | Context window size. `false` = 15,000 tokens, `true` = 30,000 tokens. Enable for large datasets. | `false` |
| `num_instances` | Number of sampled instances (only for self-consistency baseline). | `3` |
| `models` | List of model strings specifying which LLMs to use (see [Supported Providers](#supported-providers)). | — |
| `data_path` | Path to the preprocessed and split data directory. | — |
| `log_path` | Directory where logs and results will be saved. | — |

#### Inference Parameters for Reproducibility

To reproduce the results reported in the paper, use the following settings:

| Parameter | Value | Notes |
|---|---|---|
| Temperature | `0.0` | Greedy decoding for deterministic outputs |
| Top-p | `1.0` | No nucleus filtering |
| Context length | 15,000 tokens | `use_large_context: false` (30,000 for large datasets) |
| Seed | `0` | Controls data shuffling order |

#### Supported Providers

Models are specified as `<provider>:<model_name>` strings in the `models` list:

- **Ollama** (local): `ollama:<model_name>` — e.g., `ollama:llama3`, `ollama:llama3.1:70b`
- **OpenAI**: `openai:<model_name>` — e.g., `openai:gpt-4`; requires `OPENAI_API_KEY`
- **Together AI**: `together:<model_name>` — e.g., `together:meta-llama/Llama-3-8b-chat-hf`; requires `TOGETHER_API_KEY`

For Ollama with a remote server, use the URL syntax:
```
ollama:url:<host>:<port>/<model_name>
```
For example: `ollama:url:http://192.168.1.100:11434/llama3`

#### Example Configuration (Ollama)

```yaml
seed: 0
temperature: 0.0
top_p: 1.0
use_large_context: false
models:
- ollama:llama3
- ollama:llama3
- ollama:llama3
data_path: ./data/WESAD/processed
log_path: ./logs/WESAD_llama3
```

#### Example Configuration (OpenAI)

```yaml
seed: 0
temperature: 0.0
top_p: 1.0
use_large_context: false
models:
- openai:gpt-4
- openai:gpt-4
- openai:gpt-4
data_path: ./data/WESAD/processed
log_path: ./logs/WESAD_gpt4
```

### Ollama Setup

[Ollama](https://ollama.com/) allows running open-source LLMs locally. Follow these steps to set it up for ConSensus:

1. **Install Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. **Start the Ollama server:**
```bash
ollama serve
```
This starts the server at `http://localhost:11434` by default. Keep this running in a separate terminal.

3. **Pull a model:**
```bash
ollama pull llama3
```
Replace `llama3` with any supported model. Common choices:
  - `llama3` — Llama 3 8B
  - `llama3.1:70b` — Llama 3.1 70B (requires ~40GB VRAM)
  - `gemma2` — Google Gemma 2
  - `qwen2.5` — Qwen 2.5

4. **Verify the model is available:**
```bash
ollama list
```

5. **Run the experiment** with the Ollama model in your config:
```yaml
models:
- ollama:llama3
- ollama:llama3
- ollama:llama3
```

#### Advanced Ollama Configuration

- **Custom host/port:** Set the `OLLAMA_HOST` environment variable before starting the server:
  ```bash
  OLLAMA_HOST=0.0.0.0:11434 ollama serve
  ```
  This is useful for serving Ollama on a remote machine accessible over the network.

- **GPU selection:** Use `CUDA_VISIBLE_DEVICES` to control which GPUs Ollama uses:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 ollama serve
  ```

- **Remote Ollama server:** If Ollama runs on a different machine, use the URL syntax in the config:
  ```yaml
  models:
  - ollama:url:http://192.168.1.100:11434/llama3
  ```

- **Context length:** The `use_large_context` config parameter controls the context window passed to Ollama (`num_ctx`). This overrides Ollama's default of 2,048 tokens, setting it to 15,000 (default) or 30,000 (large) to accommodate the multi-agent prompts.

### Running ConSensus

```bash
python run_consensus.py --config_path <config_path>
```

Final results are stored in the `log_path` specified in the config file.

### Running Baselines

The following baseline methods are available:
- **Single Agent**: `baselines/single_agent/run.py`
- **Self-Consistency**: `baselines/self_consistency/run.py`
- **Self-Refine**: `baselines/self_refine/run.py`
- **Debate**: `baselines/debate/run.py`
- **MAD**: `baselines/mad/run.py`
- **CMD**: `baselines/cmd/run.py`
- **ReConcile**: `baselines/reconcile/run.py`

Run any baseline with:
```bash
python baselines/<baseline_name>/run.py --config_path <config_path>
```

## Reproduction

This repository was reproduced using the WESAD dataset and Llama-3.1-8B-Instruct hosted through NVIDIA Build.

The reproduction includes:
- NVIDIA Build integration
- WESAD compatibility fixes
- Evaluation pipeline
- Experimental results

For complete details, see `REPRODUCTION_REPORT.md`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yoon2026consensus,
  title={ConSensus: Multi-Agent Collaboration for Multimodal Sensing},
  author={Yoon, Hyungjun and Malekzadeh, Mohammad and Lee, Sung-Ju and Kawsar, Fahim and Qendro, Lorena},
  journal={arXiv preprint arXiv:2601.06453},
  year={2026}
}
```
