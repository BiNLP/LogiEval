"""Base dataset interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass


@dataclass
class LogicalReasoningExample:
    """Standard format for logical reasoning examples"""
    id: str
    question: str
    choices: List[str]
    answer: str
    context: Optional[str] = None
    explanation: Optional[str] = None
    
    def format_prompt(self, prompt_template: str = None) -> str:
        """Format the example as a prompt for the model"""
        if prompt_template is None:
            prompt_template = self.get_default_prompt_template()
        
        return prompt_template.format(
            context=self.context or "",
            question=self.question,
            choices=self.format_choices(),
            answer=self.answer
        )
    
    def format_choices(self) -> str:
        """Format choices as A) ... B) ... etc."""
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        formatted = []
        for i, choice in enumerate(self.choices):
            if i < len(choice_labels):
                formatted.append(f"{choice_labels[i]}) {choice}")
        return "\n".join(formatted)
    
    @staticmethod
    def get_default_prompt_template() -> str:
        """Get default prompt template"""
        return """Given the following logical reasoning problem, choose the best answer.

{context}

Question: {question}

{choices}

Answer:"""


class BaseDataset(ABC):
    """Abstract base class for logical reasoning datasets"""
    
    def __init__(self, split: str = "test"):
        self.split = split
        self.examples: List[LogicalReasoningExample] = []
        self.name = self.__class__.__name__.lower()
    
    @abstractmethod
    def load_data(self) -> None:
        """Load dataset from source"""
        pass
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> LogicalReasoningExample:
        return self.examples[idx]
    
    def __iter__(self) -> Iterator[LogicalReasoningExample]:
        return iter(self.examples)
    
    def get_subset(self, start: int = 0, end: int = None) -> List[LogicalReasoningExample]:
        """Get a subset of examples"""
        if end is None:
            end = len(self.examples)
        return self.examples[start:end]
    
    def get_example_by_id(self, example_id: str) -> Optional[LogicalReasoningExample]:
        """Get example by ID"""
        for example in self.examples:
            if example.id == example_id:
                return example
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.examples:
            return {}
        
        num_choices = [len(ex.choices) for ex in self.examples]
        answer_distribution = {}
        
        for ex in self.examples:
            if ex.answer in answer_distribution:
                answer_distribution[ex.answer] += 1
            else:
                answer_distribution[ex.answer] = 1
        
        return {
            "name": self.name,
            "split": self.split,
            "total_examples": len(self.examples),
            "avg_choices": sum(num_choices) / len(num_choices) if num_choices else 0,
            "min_choices": min(num_choices) if num_choices else 0,
            "max_choices": max(num_choices) if num_choices else 0,
            "answer_distribution": answer_distribution
        }
