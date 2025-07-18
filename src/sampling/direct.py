"""Direct sampling strategy"""

from typing import List, Dict, Any
from ..models.base import BaseModel, GenerationConfig, ModelResponse
from ..datasets.base import LogicalReasoningExample


class DirectSampling:
    """Direct sampling strategy - single sample per prompt"""
    
    def __init__(self, model: BaseModel, config: GenerationConfig = None):
        self.model = model
        self.config = config or GenerationConfig()
    
    def sample(self, examples: List[LogicalReasoningExample]) -> List[Dict[str, Any]]:
        """
        Generate single samples for each example
        
        Returns:
            List of results with format:
            {
                'example_id': str,
                'prompt': str,
                'response': ModelResponse,
                'prediction': str,
                'target': str
            }
        """
        # Prepare prompts
        prompts = [example.format_prompt() for example in examples]
        
        # Generate responses
        responses = self.model.generate_batch(prompts, self.config)
        
        # Process results
        results = []
        for example, prompt, response in zip(examples, prompts, responses):
            result = {
                'example_id': example.id,
                'prompt': prompt,
                'response': response,
                'prediction': self._extract_prediction(response.text, example.choices),
                'target': example.answer
            }
            results.append(result)
        
        return results
    
    def _extract_prediction(self, response_text: str, choices: List[str]) -> str:
        """Extract prediction from response text"""
        from ..utils.metrics import extract_answer_choice
        
        # Get choice labels based on number of choices
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
        return extract_answer_choice(response_text, choice_labels)
