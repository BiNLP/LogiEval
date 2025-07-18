"""Base model interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class ModelResponse:
    """Response from model generation"""
    text: str
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class BaseModel(ABC):
    """Abstract base class for language models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def generate(self, prompts: Union[str, List[str]], 
                config: GenerationConfig = None) -> Union[ModelResponse, List[ModelResponse]]:
        """Generate text from prompts"""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], 
                      config: GenerationConfig = None) -> List[ModelResponse]:
        """Generate text for a batch of prompts"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "loaded": self.is_loaded(),
            "model_type": self.__class__.__name__
        }
