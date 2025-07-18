"""vLLM model implementation for high-performance inference"""

from typing import List, Union, Optional, Dict, Any
import json

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from .base import BaseModel, GenerationConfig, ModelResponse


class VLLMModel(BaseModel):
    """vLLM model implementation for high-performance inference"""
    
    def __init__(self, model_name: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9, **kwargs):
        super().__init__(model_name, **kwargs)
        
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
    def load_model(self) -> None:
        """Load vLLM model"""
        print(f"Loading vLLM model: {self.model_name}")
        
        try:
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True
            )
            print("vLLM model loaded successfully")
            
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], 
                config: GenerationConfig = None) -> Union[ModelResponse, List[ModelResponse]]:
        """Generate text from prompts"""
        if not self.is_loaded():
            self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        responses = self.generate_batch(prompts, config)
        
        return responses[0] if is_single else responses
    
    def generate_batch(self, prompts: List[str], 
                      config: GenerationConfig = None) -> List[ModelResponse]:
        """Generate text for a batch of prompts"""
        if not self.is_loaded():
            self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        # Create vLLM sampling parameters
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences if config.stop_sequences else None
        )
        
        if config.top_k > 0:
            sampling_params.top_k = config.top_k
        
        if config.repetition_penalty != 1.0:
            sampling_params.repetition_penalty = config.repetition_penalty
        
        try:
            # Generate with vLLM
            outputs = self.model.generate(prompts, sampling_params)
            
            responses = []
            for output in outputs:
                # Extract the best completion
                if output.outputs:
                    generated_text = output.outputs[0].text
                    finish_reason = output.outputs[0].finish_reason
                    
                    # Extract logprobs if available
                    logprobs = None
                    if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
                        logprobs = [token.logprob for token in output.outputs[0].logprobs]
                else:
                    generated_text = ""
                    finish_reason = "error"
                    logprobs = None
                
                response = ModelResponse(
                    text=generated_text.strip(),
                    logprobs=logprobs,
                    finish_reason=finish_reason
                )
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error during vLLM generation: {e}")
            # Return empty responses on error
            return [ModelResponse(text="", finish_reason="error") for _ in prompts]
    
    def generate_multiple_samples(self, prompts: List[str], num_samples: int,
                                 config: GenerationConfig = None) -> List[List[ModelResponse]]:
        """Generate multiple samples for each prompt (useful for BoN sampling)"""
        if not self.is_loaded():
            self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        # Create sampling parameters for multiple samples
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=num_samples,  # Generate multiple samples
            stop=config.stop_sequences if config.stop_sequences else None
        )
        
        if config.top_k > 0:
            sampling_params.top_k = config.top_k
        
        try:
            outputs = self.model.generate(prompts, sampling_params)
            
            all_responses = []
            for output in outputs:
                prompt_responses = []
                for completion in output.outputs:
                    response = ModelResponse(
                        text=completion.text.strip(),
                        logprobs=[token.logprob for token in completion.logprobs] if hasattr(completion, 'logprobs') and completion.logprobs else None,
                        finish_reason=completion.finish_reason
                    )
                    prompt_responses.append(response)
                all_responses.append(prompt_responses)
            
            return all_responses
            
        except Exception as e:
            print(f"Error during vLLM multi-sample generation: {e}")
            # Return empty responses on error
            return [[ModelResponse(text="", finish_reason="error") for _ in range(num_samples)] for _ in prompts]
