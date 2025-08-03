"""Main evaluation script for LogiEval"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import project modules
from src.utils.config import EvaluationConfig
from src.utils.metrics import EvaluationResults
from src.datasets.logiqa2 import LogiQA2Dataset
from src.datasets.logiqa import LogiQADataset
from src.datasets.reclor import ReClorDataset
from src.datasets.ar_lsat import ARLSATDataset
from src.models.hf_model import HuggingFaceModel
from src.models.vllm_model import VLLMModel
from src.sampling.direct import DirectSampling
from src.sampling.bon import BestOfNSampling
from src.sampling.majority_vote import MajorityVoteSampling


def create_model(config: EvaluationConfig):
    """Create model instance based on configuration"""
    if config.use_vllm:
        try:
            return VLLMModel(
                model_name=config.model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization
            )
        except ImportError:
            print("vLLM not available, falling back to HuggingFace model")
            return HuggingFaceModel(model_name=config.model_name)
    else:
        return HuggingFaceModel(model_name=config.model_name)


def create_dataset(dataset_name: str, split: str = "test"):
    """Create dataset instance based on name"""
    dataset_map = {
        "logiqa2": LogiQA2Dataset,
        "logiqa": LogiQADataset,
        "reclor": ReClorDataset,
        "ar_lsat": ARLSATDataset
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_name](split=split)


def create_sampler(sampling_method: str, model, config: EvaluationConfig):
    """Create sampling strategy based on method"""
    from src.models.base import GenerationConfig
    
    gen_config = GenerationConfig(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p
    )
    
    if sampling_method == "direct":
        return DirectSampling(model, gen_config)
    elif sampling_method == "bon":
        return BestOfNSampling(model, gen_config, config.num_samples)
    elif sampling_method == "majority_vote":
        return MajorityVoteSampling(model, gen_config, config.num_samples)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")


def evaluate_dataset(dataset, sampler, config: EvaluationConfig) -> EvaluationResults:
    """Evaluate model on a single dataset"""
    print(f"Evaluating on {dataset.name} dataset ({len(dataset)} examples)")
    
    results = EvaluationResults(dataset.name)
    
    # Process examples in batches
    batch_size = config.batch_size
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_examples = dataset.get_subset(start_idx, end_idx)
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_examples)} examples)")
        
        try:
            # Generate samples for batch
            batch_results = sampler.sample(batch_examples)
            
            # Process results
            for result in batch_results:
                example = dataset.get_example_by_id(result['example_id'])
                
                # Extract prediction and raw output
                prediction = result['prediction']
                
                if config.sampling_method == "direct":
                    raw_output = result['response'].text
                    intermediate_output = {
                        'response': result['response'].text,
                        'finish_reason': result['response'].finish_reason
                    }
                elif config.sampling_method == "bon":
                    raw_output = [resp.text for resp in result['responses']]
                    intermediate_output = {
                        'responses': [resp.text for resp in result['responses']],
                        'best_response': result['best_response'].text,
                        'scores': result['scores']
                    }
                elif config.sampling_method == "majority_vote":
                    raw_output = [resp.text for resp in result['responses']]
                    intermediate_output = {
                        'responses': [resp.text for resp in result['responses']],
                        'predictions': result['predictions'],
                        'vote_counts': result['vote_counts']
                    }
                
                # Add to results
                results.add_result(
                    sample_id=result['example_id'],
                    prediction=prediction,
                    target=result['target'],
                    raw_output=raw_output,
                    intermediate_output=intermediate_output if config.save_intermediate else None,
                    question=example.question,
                    choices=example.choices,
                    context=example.context
                )
        
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            # Add error results for the batch
            for example in batch_examples:
                results.add_result(
                    sample_id=example.id,
                    prediction="ERROR",
                    target=example.answer,
                    raw_output="",
                    intermediate_output={"error": str(e)} if config.save_intermediate else None,
                    question=example.question,
                    choices=example.choices,
                    context=example.context
                )
    
    return results


def save_results(config: EvaluationConfig, dataset_results: Dict[str, EvaluationResults], 
                execution_time: float):
    """Save evaluation results to JSON"""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = config.model_name.replace("/", "_").replace(":", "_")
    
    # Prepare results dictionary
    results_dict = {
        "config": config.to_dict(),
        "execution_time_seconds": execution_time,
        "timestamp": timestamp,
        "results": {}
    }
    
    # Add dataset results
    overall_metrics = {}
    for dataset_name, result in dataset_results.items():
        dataset_dict = result.to_dict()
        results_dict["results"][dataset_name] = dataset_dict
        
        # Collect overall metrics
        metrics = dataset_dict["metrics"]
        for metric_name, value in metrics.items():
            if metric_name not in overall_metrics:
                overall_metrics[metric_name] = []
            overall_metrics[metric_name].append(value)
    
    # Calculate overall averages
    results_dict["overall_metrics"] = {}
    for metric_name, values in overall_metrics.items():
        if isinstance(values[0], (int, float)):
            results_dict["overall_metrics"][f"avg_{metric_name}"] = sum(values) / len(values)
        results_dict["overall_metrics"][f"total_{metric_name}"] = sum(values) if isinstance(values[0], int) else values
    
    # Save results
    filename = f"evaluation_{model_name_safe}_{config.sampling_method}_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    return output_path


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate LLM on logical reasoning datasets")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model identifier")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM for acceleration")
    
    # Dataset arguments
    parser.add_argument("--datasets", nargs="+", 
                       choices=["logiqa2", "logiqa", "reclor","ar_lsat"],
                       default=["logiqa2", "logiqa", "reclor","ar_lsat"],
                       help="Datasets to evaluate on")
    
    # Sampling arguments
    parser.add_argument("--sampling_method", type=str,
                       choices=["direct", "bon", "majority_vote"],
                       default="direct",
                       help="Sampling strategy")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples for BoN/majority vote")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="Save intermediate outputs")
    
    # vLLM specific arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization for vLLM")
    
    # Configuration file
    parser.add_argument("--config", type=str,
                       help="Path to JSON configuration file")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = EvaluationConfig.from_json(args.config)
    else:
        config = EvaluationConfig(
            model_name=args.model_name,
            use_vllm=args.use_vllm,
            datasets=args.datasets,
            sampling_method=args.sampling_method,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_intermediate=args.save_intermediate,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    
    # Validate configuration
    config.validate()
    
    print("="*60)
    print("LogiEval - Large Language Model Logical Reasoning Evaluation")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Datasets: {config.datasets}")
    print(f"Sampling: {config.sampling_method}")
    if config.sampling_method in ["bon", "majority_vote"]:
        print(f"Num samples: {config.num_samples}")
    print(f"Use vLLM: {config.use_vllm}")
    print(f"Batch size: {config.batch_size}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Create model
        print("Loading model...")
        model = create_model(config)
        model.load_model()
        print(f"Model loaded: {model.get_model_info()}")
        
        # Create sampler
        sampler = create_sampler(config.sampling_method, model, config)
        
        # Evaluate on each dataset
        dataset_results = {}
        for dataset_name in config.datasets:
            print(f"\n--- Evaluating on {dataset_name} ---")
            try:
                # Load dataset
                dataset = create_dataset(dataset_name)
                print(f"Dataset statistics: {dataset.get_statistics()}")
                
                # Evaluate
                results = evaluate_dataset(dataset, sampler, config)
                dataset_results[dataset_name] = results
                
                # Print metrics
                metrics = results.get_metrics()
                print(f"Results for {dataset_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
                
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                # Create empty results for failed dataset
                dataset_results[dataset_name] = EvaluationResults(dataset_name)
        
        # Calculate total time
        execution_time = time.time() - start_time
        
        # Save results
        output_path = save_results(config, dataset_results, execution_time)
        
        print(f"\n--- Evaluation Complete ---")
        print(f"Total time: {execution_time:.2f} seconds")
        print(f"Results saved to: {output_path}")
        
        # Print summary
        print("\n--- Summary ---")
        for dataset_name, results in dataset_results.items():
            metrics = results.get_metrics()
            if metrics:
                accuracy = metrics.get('accuracy', 0)
                total = metrics.get('total_samples', 0)
                print(f"{dataset_name}: {accuracy:.4f} accuracy ({total} samples)")
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
