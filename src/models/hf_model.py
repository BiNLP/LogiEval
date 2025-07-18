"""HuggingFace Transformers model implementation"""

import torch
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base import BaseModel, GenerationConfig, ModelResponse


class HuggingFaceModel(BaseModel):
    """HuggingFace Transformers model implementation"""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.pipeline = None
        
    def load_model(self) -> None:
        """Load HuggingFace model and tokenizer"""
        print(f"Loading HuggingFace model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
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
        
        responses = []
        
        generation_kwargs = {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False
        }
        
        if config.top_k > 0:
            generation_kwargs["top_k"] = config.top_k
        
        if config.repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = config.repetition_penalty
        
        try:
            # Process prompts in batch
            outputs = self.pipeline(
                prompts,
                batch_size=len(prompts),
                **generation_kwargs
            )
            
            for i, output in enumerate(outputs):
                # Extract generated text
                if isinstance(output, list) and len(output) > 0:
                    generated_text = output[0].get("generated_text", "")
                else:
                    generated_text = output.get("generated_text", "")
                
                response = ModelResponse(
                    text=generated_text.strip(),
                    finish_reason="stop"
                )
                responses.append(response)
                
        except Exception as e:
            print(f"Error during generation: {e}")
            # Return empty responses on error
            for _ in prompts:
                responses.append(ModelResponse(text="", finish_reason="error"))
        
        return responses
