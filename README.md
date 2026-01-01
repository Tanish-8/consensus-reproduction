# Official implementation of ConSensus: Multi-Agent Collaboration for Multimodal Sensing

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) (recommended) or Python 3.12+

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
   - For Ollama: configure locally (see model configuration)

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

2. **Split data for training/testing:**
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

The current implementation supports inference using:
- **Ollama**: Local models (e.g., `ollama:llama3`)
- **OpenAI**: OpenAI API models (e.g., `openai:gpt-4`)
- **Together AI**: Together AI models (e.g., `together:meta-llama/Llama-3-8b-chat-hf`)

API keys must be set as environment variables before running.

### Running ConSensus

```bash
python run_consensus.py --config_path <config_path>
```

An example configuration file is provided in `config/example.yaml`. The configuration includes:
- `seed`: Random seed for reproducibility
- `temperature`: Sampling temperature (0.0 for deterministic, 0.7 for self-consistency)
- `use_large_context`: Enable for large datasets
- `models`: List of model configurations to use
- `data_path`: Path to processed data
- `log_path`: Where logs and results will be saved

Final results are stored in the `log_path` specified in the config file.

### Running Baselines

The following baseline methods are available:
- **Single Agent**: `baselines/single_agent/run.py`
- **Self-Consistency**: `baselines/self_consistency/run.py`
- **Self-Refine**: `baselines/self_refine/run.py`
- **Debate**: `baselines/debate/run.py`
- **Judge**: `baselines/judge/run.py`
- **CMD**: `baselines/cmd/run.py`
- **Reconcile**: `baselines/reconcile/run.py`

Run any baseline with:
```bash
python baselines/<baseline_name>/run.py --config_path <config_path>
```
