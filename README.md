# LogiEval - Large Language Model Logical Reasoning Evaluation

A high-performance evaluation framework for testing large language models on logical reasoning datasets including LogiQA2.0, LogiQA, and ReClor.

## Features

- 🚀 **High Performance**: Optimized with vLLM for fast model inference
- 📊 **Multiple Datasets**: Support for LogiQA2.0, LogiQA, and ReClor datasets
- 🎯 **Multiple Sampling Methods**: Direct sampling, Best-of-N (BoN), and majority voting
- 💾 **Result Persistence**: Save intermediate results and outputs as JSON files
- 🔧 **Flexible Configuration**: Easy-to-use configuration system

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Evaluation

```bash
python evaluate.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --datasets "logiqa2" "logiqa" "reclor" \
    --sampling_method "direct" \
    --batch_size 8 \
    --output_dir "./results"
```

### Advanced Evaluation with Multiple Sampling

```bash
python evaluate.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --datasets "logiqa2" "logiqa" "reclor" \
    --sampling_method "bon" \
    --num_samples 5 \
    --batch_size 4 \
    --use_vllm \
    --output_dir "./results"
```

## Configuration

The evaluation framework supports various configuration options:

- `--model_name`: HuggingFace model identifier
- `--datasets`: List of datasets to evaluate (logiqa2, logiqa, reclor)
- `--sampling_method`: Sampling strategy (direct, bon, majority_vote)
- `--num_samples`: Number of samples for BoN and majority voting
- `--batch_size`: Batch size for inference
- `--use_vllm`: Enable vLLM acceleration
- `--temperature`: Sampling temperature
- `--max_tokens`: Maximum generation length
- `--output_dir`: Directory to save results

## Project Structure

```
LogiEval/
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── logiqa2.py
│   │   ├── logiqa.py
│   │   └── reclor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hf_model.py
│   │   └── vllm_model.py
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── direct.py
│   │   ├── bon.py
│   │   └── majority_vote.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── metrics.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Results

Evaluation results are saved in JSON format with the following structure:

```json
{
    "config": {...},
    "results": {
        "dataset_name": {
            "accuracy": 0.85,
            "total_samples": 1000,
            "correct_predictions": 850,
            "detailed_results": [...]
        }
    },
    "intermediate_outputs": [...]
}
```
