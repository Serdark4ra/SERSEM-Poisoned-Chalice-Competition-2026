# SERSEM: Selective Entropy-Weighted Scoring for Membership Inference in Code Language Models

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3803437.3807734-blue)](https://doi.org/10.1145/3803437.3807734)

This repository contains the replication package for our submission to the [Poisoned Chalice Competition](https://poisonedchalice.github.io), which evaluates membership inference attacks against code language models.

## Paper and Competition

- **Competition**: [Poisoned Chalice: Code Model Privacy Challenge](https://poisonedchalice.github.io)
- **Paper**: *SERSEM: Selective Entropy-Weighted Scoring for Membership Inference in Code Language Models*
- **Method**: Our approach combines linear probes trained on internal model activations (LUMIA) with code-specific anomaly detection features to achieve state-of-the-art membership inference performance.

## Repository Structure

```
SERSEM-Poisoned-Chalice-Competition-2026/
├── README.md                      # This file
├── run_pipeline.sh                # Main execution script
├── .env                           # Configuration file (customize before running)
│
├── baselines/                     # Baseline attacks 
│   ├── Loss.py
│   ├── MIAttack.py
│   ├── MinKProbAttack.py
│   ├── Pac.py
│   ├── process.py
│   ├── requirements.txt
│   └── run.py
│   └── results/                    # Results from baselines run on our test set
│
├── code/                          # Source code
│   ├── run.py                     # Main experiment runner
│   ├── LumiaAttack.py             # LUMIA attack implementation
│   ├── MIAttack.py                # Base attack interface
│   ├── ASTExtractor.py            # AST-based feature extraction
│   ├── LinterExtractor.py         # Linter-based feature extraction
│   ├── transformers_compat.py     # Compatibility utilities
│   └── plot_results.py            # Visualization utilities
│
├── data/                          # Pre-computed results from paper
│   ├── 3b_train_test/             # Results for StarCoder2-3B model
│   │   ├── eval_results.parquet   # Evaluation dataset with scores
│   │   └── shadow_dataset.parquet # Training dataset for probes
│   └── 7b_train_test/             # Results for StarCoder2-7B model
│       ├── eval_results.parquet
│       └── shadow_dataset.parquet
│
├── probes_bigcode_starcoder2-3b.pkl  # Pre-trained probes for StarCoder2-3B
```

### Data Files

The `data/` directory contains the exact datasets used in our paper submissions:

- **`3b_train_test/`**: Results for `bigcode/starcoder2-3b` model
- **`7b_train_test/`**: Results for `bigcode/starcoder2-7b` model

Each subdirectory contains:
- **`eval_results.parquet`**: Evaluation samples with membership scores and ground truth labels
- **`shadow_dataset.parquet`**: Shadow (training) samples used to train the attack probes

These parquet files enable independent evaluation and analysis of our results without re-running the full pipeline.

## Prerequisites

### Software Requirements
- Python 3.11 or higher
- Git

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Before running experiments, configure the `.env` file:

**Configuration Options:**

```bash
# Model to evaluate (HuggingFace model ID)
PIPELINE_MODEL_PATH="bigcode/starcoder2-3b"

# Dataset (competition dataset)
PIPELINE_DATASET_PATH="AISE-TUDelft/Poisoned-Chalice"

# GPU device index (use nvidia-smi to check available GPUs)
GPU_INDEX="0"

# HuggingFace repository containing pre-trained probes
HF_REPO_ID="Serdark4r/SERSEM-Poisoned-Chalice-Competition-2026"
```

## How to Reproduce Results

Reproduce the full paper results:

```bash
# Run the pipeline
./run_pipeline.sh
```

## Output and Results

### Results Location

The pipeline creates a timestamped results directory:

```
pipeline_results/
└── replication_{model_name}_{timestamp}/
    ├── eval_results.parquet       # Evaluation scores
    ├── shadow_dataset.parquet     # Training data
    ├── metadata.json              # Run configuration
    ├── probes_{model_name}.pkl    # Trained probes
    ├── plots/                     # Visualizations
    └── code/                      # Source code snapshot
```

### Understanding Results

**`eval_results.parquet`** contains:
- `content`: Source code samples
- `membership`: Ground truth label ('member' or 'non-member')
- `is_member`: Binary label (1 = member, 0 = non-member)
- `language`: Programming language (Go, Java, Python, Ruby, Rust)
- `split_role`: Dataset role ('eval' for evaluation set)
- `lumia_score`: Membership inference score (higher = more likely member)
- `lumia_confidence_vector`: Per-layer confidence scores

**`metadata.json`** contains:
- Experiment timestamp and configuration
- Model and dataset information
- Sample counts and random seed
- Attacks executed

### Performance Metrics

The pipeline automatically computes and displays:
- **Overall AUC**: Area Under the ROC Curve across all languages
- **Per-language AUC**: Individual performance for each of 5 languages
- **ROC Curves**: Visual plots saved in `plots/` directory

## Pipeline Execution Flow

The `run_pipeline.sh` script executes the following steps:

1. **Configuration Loading**: Sources `.env` for parameters
2. **Probe Check**:
   - Checks for local probe file
   - If not found, attempts to download from HuggingFace
   - If download fails, will train new probes during execution
3. **Evaluation**: Runs `code/run.py` with specified parameters
4. **Results Generation**: Creates parquet files with scores
5. **Visualization**: Generates ROC curves and plots
6. **Output**: Saves everything to timestamped directory

### Command-line Override

You can override `.env` settings via environment variables:

```bash
# Run with different model without editing .env
PIPELINE_MODEL_PATH="JetBrains/Mellum-4b-base" ./run_pipeline.sh

# Run on different GPU
GPU_INDEX=1 ./run_pipeline.sh
```

## Analyzing Pre-computed Results

The `data/` directory contains results from our paper. You can analyze these without re-running.

## Troubleshooting

### Model Download Issues

If model download fails:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

**Note**: This is a research artifact. The membership inference attack implemented here is intended for studying privacy risks in code language models and developing defenses. Please use responsibly and ethically.

## How to Cite

If you use this work, please cite:

```bibtex
@inproceedings{10.1145/3803437.3807734,
author = {Dikici, K{\i}van{\c c} Kuzey and Kara, Serdar and {\c C}a{\u g}lar, Semih and T{\"u}z{\"u}n, Eray and Sav, Sinem},
title = {SERSEM: Selective Entropy-Weighted Scoring for Membership Inference in Code Language Models},
year = {2026},
isbn = {9798400726361},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3803437.3807734},
doi = {10.1145/3803437.3807734},
booktitle = {Proceedings of the 34th ACM International Conference on the Foundations of Software Engineering},
pages = {1456–1459},
numpages = {4},
keywords = {large language models, privacy, memorization, membership inference, data leakage, selective entropy scoring},
location = {Concordia University, Montreal, QC, Canada},
series = {FSE Companion '26}
}
```
