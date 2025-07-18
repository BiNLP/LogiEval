"""Majority vote sampling strategy"""

from typing import List, Dict, Any
from ..models.base import BaseModel, GenerationConfig, ModelResponse
from ..models.vllm_model import VLLMModel
from ..datasets.base import LogicalReasoningExample


class MajorityVoteSampling:
    """Majority vote sampling strategy - generate N samples and take majority vote"""
    
    def __init__(self, model: BaseModel, config: GenerationConfig = None, num_samples: int = 5):
        self.model = model
        self.config = config or GenerationConfig()
        self.num_samples = num_samples
        
        # For majority vote, we want some randomness
        if self.config.temperature == 0.0:
            self.config.temperature = 0.7
    
    def sample(self, examples: List[LogicalReasoningExample]) -> List[Dict[str, Any]]:
        """
        Generate N samples for each example and take majority vote
        
        Returns:
            List of results with format:
            {
                'example_id': str,
                'prompt': str,
                'responses': List[ModelResponse],
                'predictions': List[str],
                'majority_prediction': str,
                'target': str,
                'vote_counts': Dict[str, int]
            }
        """
        # Prepare prompts
        prompts = [example.format_prompt() for example in examples]
        
        # Generate multiple samples
        if isinstance(self.model, VLLMModel):
            # Use vLLM's efficient multi-sample generation
            all_responses = self.model.generate_multiple_samples(prompts, self.num_samples, self.config)
        else:
            # Generate samples individually for other model types
            all_responses = []
            for prompt in prompts:
                responses = []
                for _ in range(self.num_samples):
                    response = self.model.generate(prompt, self.config)
                    responses.append(response)
                all_responses.append(responses)
        
        # Process results
        results = []
        for example, prompt, responses in zip(examples, prompts, all_responses):
            # Extract predictions from all responses
            predictions = [
                self._extract_prediction(resp.text, example.choices) 
                for resp in responses
            ]
            
            # Perform majority vote
            majority_prediction, vote_counts = self._majority_vote(predictions, example.choices)
            
            result = {
                'example_id': example.id,
                'prompt': prompt,
                'responses': responses,
                'predictions': predictions,
                'majority_prediction': majority_prediction,
                'target': example.answer,
                'vote_counts': vote_counts
            }
            results.append(result)
        
        return results
    
    def _majority_vote(self, predictions: List[str], choices: List[str]) -> tuple[str, Dict[str, int]]:
        """
        Perform majority vote on predictions
        
        Returns:
            (majority_prediction, vote_counts)
        """
        from ..utils.metrics import majority_vote
        
        # Get choice labels based on number of choices
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
        
        # Count votes
        vote_counts = {}
        for pred in predictions:
            if pred in vote_counts:
                vote_counts[pred] += 1
            else:
                vote_counts[pred] = 1
        
        # Get majority vote
        majority_prediction = majority_vote(predictions, choice_labels)
        
        return majority_prediction, vote_counts
    
    def _extract_prediction(self, response_text: str, choices: List[str]) -> str:
        """Extract prediction from response text"""
        from ..utils.metrics import extract_answer_choice
        
        # Get choice labels based on number of choices
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
        return extract_answer_choice(response_text, choice_labels)
