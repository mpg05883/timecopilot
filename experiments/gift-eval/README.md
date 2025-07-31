# TimeCopilot GIFT-Eval Experiments

This project demonstrates the evaluation of a foundation model ensemble built using the [TimeCopilot](https://timecopilot.dev) library on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmark.

TimeCopilot is an open‑source AI agent for time series forecasting that provides a unified interface to multiple forecasting approaches, from foundation models to classical statistical, machine learning, and deep learning methods, along with built‑in ensemble capabilities for robust and explainable forecasting.

## Model Description

This ensemble leverages [**TimeCopilot's MedianEnsemble**](https://timecopilot.dev/api/models/ensembles/#timecopilot.models.ensembles.median.MedianEnsemble) feature, which combines five state-of-the-art foundation models:

- [**Chronos** (Amazon)](https://timecopilot.dev/api/models/foundational/models/#timecopilot.models.foundational.chronos.Chronos).
- [**Moirai** (Salesforce)](https://timecopilot.dev/api/models/foundational/models/#timecopilot.models.foundational.moirai.Moirai).
- [**TimesFM** (Google)](https://timecopilot.dev/api/models/foundational/models/#timecopilot.models.foundational.timesfm.TimesFM).
- [**TiRex** (NX-AI)](https://timecopilot.dev/api/models/foundational/models/#timecopilot.models.foundational.tirex.TiRex).
- [**Toto** (DataDog)](https://timecopilot.dev/api/models/foundational/models/#timecopilot.models.foundational.toto.Toto).


## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS CLI configured (for distributed evaluation)
- [Modal](https://modal.com/) account (for distributed evaluation)

### Installation

```bash
# Install dependencies
uv sync
```

## Dataset Management

### Download GIFT-Eval Dataset

```bash
# Download the complete GIFT-Eval dataset
make download-gift-eval-data
```

This downloads all 97 dataset configurations to `./data/gift-eval/`.

### Upload to S3 (Optional)

For distributed evaluation, upload the dataset to S3:

```bash
# Upload dataset to S3 for distributed access
make upload-data-to-s3
```

## Evaluation Methods

### 1. Local Evaluation

Run evaluation on a single dataset locally:

```bash
uv run python -m src.run_timecopilot \
  --dataset-name "m4_weekly" \
  --term "short" \
  --output-path "./results/timecopilot/" \
  --storage-path "./data/gift-eval"
```

**Parameters:**
- `--dataset-name`: GIFT-Eval dataset name (e.g., "m4_weekly", "bizitobs_l2c/H")
- `--term`: Forecasting horizon ("short", "medium", "long")
- `--output-path`: Directory to save evaluation results
- `--storage-path`: Path to GIFT-Eval dataset

### 2. Distributed Evaluation (Recommended)

Evaluate all 97 dataset configurations in parallel using [modal](https://modal.com/):

```bash
# Run distributed evaluation on Modal cloud
uv run modal run --detach -m src.run_modal::main
```

This creates one GPU job per dataset configuration, significantly reducing evaluation time.

**Infrastructure:**
- **GPU**: A10G per job
- **CPU**: 8 cores per job  
- **Timeout**: 3 hours per job
- **Storage**: S3 bucket for data and results

### 3. Collect Results

Download and consolidate results from distributed evaluation:

```bash
# Download all results from S3 and create consolidated CSV
uv run python -m src.download_results
```

Results are saved to `results/timecopilot/all_results.csv` in GIFT-Eval format.