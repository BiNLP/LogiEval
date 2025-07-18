"""Best-of-N (BoN) sampling strategy"""

from typing import List, Dict, Any
from ..models.base import BaseModel, GenerationConfig, ModelResponse
from ..models.vllm_model import VLLMModel
from ..datasets.base import LogicalReasoningExample


class BestOfNSampling:
    """Best-of-N sampling strategy - generate N samples and select best based on score"""
    
    def __init__(self, model: BaseModel, config: GenerationConfig = None, num_samples: int = 5):
        self.model = model
        self.config = config or GenerationConfig()
        self.num_samples = num_samples
        
        # For BoN, we want some randomness
        if self.config.temperature == 0.0:
            self.config.temperature = 0.7
    
    def sample(self, examples: List[LogicalReasoningExample]) -> List[Dict[str, Any]]:
        """
        Generate N samples for each example and select the best one
        
        Returns:
            List of results with format:
            {
                'example_id': str,
                'prompt': str,
                'responses': List[ModelResponse],
                'best_response': ModelResponse,
                'prediction': str,
                'target': str,
                'scores': List[float]
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
            # Score each response
            scores = [self._score_response(resp, example) for resp in responses]
            
            # Select best response
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_response = responses[best_idx]
            
            result = {
                'example_id': example.id,
                'prompt': prompt,
                'responses': responses,
                'best_response': best_response,
                'prediction': self._extract_prediction(best_response.text, example.choices),
                'target': example.answer,
                'scores': scores
            }
            results.append(result)
        
        return results
    
    def _score_response(self, response: ModelResponse, example: LogicalReasoningExample) -> float:
        """
        Score a response. Higher scores are better.
        
        This is a simple scoring function that can be extended with more sophisticated
        scoring methods like reward models, confidence estimation, etc.
        """
        score = 0.0
        
        # Length penalty (prefer concise answers)
        length_penalty = max(0, 1.0 - len(response.text) / 1000.0)
        score += length_penalty * 0.1
        
        # Valid answer bonus
        prediction = self._extract_prediction(response.text, example.choices)
        if prediction != 'INVALID':
            score += 1.0
        
        # Logprob score (if available)
        if response.logprobs:
            avg_logprob = sum(response.logprobs) / len(response.logprobs)
            score += avg_logprob * 0.1  # Normalize logprobs
        
        # Confidence indicators in text
        confidence_keywords = ['clearly', 'obviously', 'definitely', 'certainly']
        for keyword in confidence_keywords:
            if keyword.lower() in response.text.lower():
                score += 0.1
        
        return score
    
    def _extract_prediction(self, response_text: str, choices: List[str]) -> str:
        """Extract prediction from response text"""
        from ..utils.metrics import extract_answer_choice
        
        # Get choice labels based on number of choices
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
        return extract_answer_choice(response_text, choice_labels)
