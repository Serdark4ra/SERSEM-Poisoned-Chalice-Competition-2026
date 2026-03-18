#!/bin/bash
set -e

# Ensure we are in the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
    echo "Sourcing .env..."
    source .env
else
    echo "Warning: .env not found. Using defaults or environment variables. (Tip: copy sample.env to .env)"
fi

# Set defaults
MODEL=${PIPELINE_MODEL_PATH:-"bigcode/starcoder2-3b"}
DATASET=${PIPELINE_DATASET_PATH:-"AISE-TUDelft/Poisoned-Chalice"}
GPU=${GPU_INDEX:-0}
SAMPLES=${SAMPLES_PER_LANGUAGE:-5000}
HF_REPO=${HF_REPO_ID:-"Serdark4r/SERSEM-Poisoned-Chalice-Competition-2026"}

echo "=================================================="
echo "Starting replication pipeline with parameters:"
echo "Model:                $MODEL"
echo "Dataset:              $DATASET"
echo "GPU Index:            $GPU"
echo "Samples Per Language: $SAMPLES"
echo "HF Repo ID:           $HF_REPO"
echo "=================================================="

export CUDA_VISIBLE_DEVICES=$GPU

# Check for the probe file
# Always use probes_bigcode_starcoder2-3b.pkl regardless of the model
PROBE_FILE="probes_bigcode_starcoder2-3b.pkl"
EXPECTED_PROBE_NAME="probes_$(echo "$MODEL" | tr '/' '_').pkl"

# First, ensure we have the starcoder2-3b probe file
if [ ! -f "$PROBE_FILE" ]; then
    echo "Probe not found locally. Downloading $PROBE_FILE from Hugging Face hub ($HF_REPO)..."
    python -c "
import sys
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(repo_id='$HF_REPO', filename='$PROBE_FILE', local_dir='.')
    print(f'Successfully downloaded probe to {path}')
except Exception as e:
    print(f'\n[Error] Failed to download $PROBE_FILE from Hugging Face repo $HF_REPO.')
    print(f'Details: {e}')
    sys.exit(1)
"
else
    echo "Found probe file locally: $PROBE_FILE"
fi

# If the model expects a different probe name, create a symlink
if [ "$EXPECTED_PROBE_NAME" != "$PROBE_FILE" ] && [ ! -f "$EXPECTED_PROBE_NAME" ]; then
    echo "Linking $PROBE_FILE to expected name: $EXPECTED_PROBE_NAME"
    ln -s "$PROBE_FILE" "$EXPECTED_PROBE_NAME"
fi

echo "Running evaluation script in code/..."
cd code

python run.py \
    --model_name "$MODEL" \
    --dataset "$DATASET" \
    --samples_per_language "$SAMPLES" \
    --lumia_cache_dir ".." \
    --output_dir "../pipeline_results" \
    --use_bf16

# Wait for completion, then find the latest results directory
LATEST_RESULTS_DIR=$(ls -td ../pipeline_results/replication_* 2>/dev/null | head -1 || true)

if [ -n "$LATEST_RESULTS_DIR" ]; then
    echo "Found evaluation results at: $LATEST_RESULTS_DIR"
    
    echo "Generating ROC plots..."
    python plot_results.py \
        --input_path "${LATEST_RESULTS_DIR}/eval_results.parquet" \
        --output_dir "${LATEST_RESULTS_DIR}/plots" \
        --title_prefix "StarCoder2-3b"
        
    echo "Pipeline complete. Check ${LATEST_RESULTS_DIR}/plots for the generated figures."
else
    echo "Error: Could not find output results directory."
    exit 1
fi
