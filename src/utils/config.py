"""Configuration management for LogiEval"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Model settings
    model_name: str
    use_vllm: bool = False
    
    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    
    # Evaluation settings
    datasets: List[str] = None
    sampling_method: str = "direct"  # direct, bon, majority_vote
    num_samples: int = 1
    batch_size: int = 8
    
    # Output settings
    output_dir: str = "./results"
    save_intermediate: bool = True
    
    # vLLM specific settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["logiqa2", "logiqa", "reclor"]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "EvaluationConfig":
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_json(self, json_path: str) -> None:
        """Save config to JSON file"""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        valid_datasets = {"logiqa2", "logiqa", "reclor","ar_lsat"}
        valid_sampling = {"direct", "bon", "majority_vote"}
        
        # Validate datasets
        for dataset in self.datasets:
            if dataset not in valid_datasets:
                raise ValueError(f"Invalid dataset: {dataset}. Must be one of {valid_datasets}")
        
        # Validate sampling method
        if self.sampling_method not in valid_sampling:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}. Must be one of {valid_sampling}")
        
        # Validate parameters
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        
        if self.sampling_method in ["bon", "majority_vote"] and self.num_samples <= 1:
            raise ValueError(f"num_samples must be > 1 for {self.sampling_method} sampling")
