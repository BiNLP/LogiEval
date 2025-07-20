"""Evaluation metrics and scoring functions"""

from typing import List, Dict, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate accuracy score"""
    return accuracy_score(targets, predictions)


def calculate_detailed_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate detailed classification metrics"""
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def extract_answer_choice(text: str, choices: List[str] = None) -> str:
    """
    Extract answer choice from model output.
    
    Args:
        text: Model output text
        choices: List of valid choices (e.g., ['A', 'B', 'C', 'D'])
    
    Returns:
        Extracted choice or 'INVALID' if no valid choice found
    """
    if choices is None:
        choices = ['A', 'B', 'C', 'D']
    
    text = text.strip()
    
    # # Direct match
    # for choice in choices:
    #     if text == choice:
    #         return choice
    
    # # Look for choice at the beginning
    # for choice in choices:
    #     if text.startswith(choice):
    #         return choice
    
    # Look for explicit answer patterns (more specific first)
    # patterns = [
    #     # "Answer: A" or "Answer is A" or "Answer A"
    #     r'(?:answer|choice)(?:\s*is)?\s*:?\s*([A-D])\b',
    #     # "The answer is A"
    #     r'(?:the\s+)?answer\s+is\s+([A-D])\b',
    #     # "I choose A" or "My choice is A"
    #     r'(?:I\s+choose|my\s+choice\s+is|I\s+select)\s+([A-D])\b',
    #     "(A)" or "[A]"
    #     # r'[\(\[]([A-D])[\)\]]',
    #     # At the very beginning or end of text
    #     r'^([A-D])\b',
    #     r'\b([A-D])$',
    #     # After common conclusion words
    #     r'(?:therefore|thus|so|hence|conclusion|result).*?([A-D])\b',
    #     # Latex boxed format, e.g., \boxed{A} or \boxed{B}
    #     r'\\boxed\{\{?([A-Za-z])\}?\}'
    # ]
    # import re
    # # import pdb; pdb.set_trace()
    # for pattern in patterns:
    #     match = re.search(pattern, text, re.IGNORECASE)
    #     if match:
    #         choice = match.group(1).upper()
    #         # if choice in choices:
    #         #     return choice
    #         return choice
    import re
    # import pdb; pdb.set_trace()
    matches = re.findall(r'\\boxed\{\{?([A-Za-z])\}?\}', text, re.IGNORECASE)
    if matches:
        predition = [letter.upper() for letter in matches][-1]
        return predition
    
    
    return 'INVALID'


def majority_vote(responses: List[str], choices: List[str] = None) -> str:
    """
    Perform majority voting on a list of responses.
    
    Args:
        responses: List of model responses
        choices: List of valid choices
    
    Returns:
        Most common valid choice, or 'INVALID' if no valid choices
    """
    if choices is None:
        choices = ['A', 'B', 'C', 'D']
    
    # Extract choices from responses
    extracted_choices = [extract_answer_choice(resp, choices) for resp in responses]
    
    # Count valid choices
    choice_counts = {}
    for choice in extracted_choices:
        if choice != 'INVALID':
            choice_counts[choice] = choice_counts.get(choice, 0) + 1
    
    if not choice_counts:
        return 'INVALID'
    
    # Return most common choice
    return max(choice_counts, key=choice_counts.get)


def select_best_of_n(responses: List[str], scores: List[float], choices: List[str] = None) -> str:
    """
    Select the best response from N candidates based on scores.
    
    Args:
        responses: List of model responses
        scores: List of confidence scores
        choices: List of valid choices
    
    Returns:
        Best valid response choice
    """
    if choices is None:
        choices = ['A', 'B', 'C', 'D']
    
    # Pair responses with scores and sort by score (descending)
    response_score_pairs = list(zip(responses, scores))
    response_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return the first valid choice
    for response, _ in response_score_pairs:
        choice = extract_answer_choice(response, choices)
        if choice != 'INVALID':
            return choice
    
    return 'INVALID'


class EvaluationResults:
    """Container for evaluation results"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.predictions = []
        self.targets = []
        self.raw_outputs = []
        self.intermediate_outputs = []
        self.sample_ids = []
    
    def add_result(self, sample_id: str, prediction: str, target: str, 
                   raw_output: Union[str, List[str]], intermediate_output: Any = None):
        """Add a single evaluation result"""
        self.sample_ids.append(sample_id)
        self.predictions.append(prediction)
        self.targets.append(target)
        self.raw_outputs.append(raw_output)
        self.intermediate_outputs.append(intermediate_output)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return evaluation metrics"""
        if not self.predictions:
            return {}
        
        metrics = calculate_detailed_metrics(self.predictions, self.targets)
        metrics.update({
            "total_samples": len(self.predictions),
            "correct_predictions": sum(1 for p, t in zip(self.predictions, self.targets) if p == t),
            "invalid_predictions": sum(1 for p in self.predictions if p == 'INVALID')
        })
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization"""
        return {
            "dataset_name": self.dataset_name,
            "metrics": self.get_metrics(),
            "detailed_results": [
                {
                    "sample_id": sid,
                    "prediction": pred,
                    "target": target,
                    "raw_output": raw,
                    "intermediate_output": inter,
                    "correct": pred == target
                }
                for sid, pred, target, raw, inter in zip(
                    self.sample_ids, self.predictions, self.targets,
                    self.raw_outputs, self.intermediate_outputs
                )
            ]
        }
