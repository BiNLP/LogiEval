"""LogiQA 2.0 dataset loader"""

from typing import List
from datasets import load_dataset
from .base import BaseDataset, LogicalReasoningExample
import json


class LogiQA2Dataset(BaseDataset):
    """LogiQA 2.0 dataset from datatune/LogiQA2.0"""
    
    def __init__(self, split: str = "test"):
        super().__init__(split)
        self.name = "logiqa2"
        self.load_data()
    
    def load_data(self) -> None:
        """Load LogiQA 2.0 dataset from HuggingFace"""
        # import pdb;pdb.set_trace
        try:
            dataset = load_dataset("datatune/LogiQA2.0", split=self.split)
        except Exception as e:
            print(f"Error loading LogiQA2.0 dataset: {e}")
            print("Trying alternative split names...")
            # Try common split names
            for alt_split in ["validation", "dev"]:
                try:
                    dataset = load_dataset("datatune/LogiQA2.0", split=alt_split)
                    print(f"Successfully loaded with split: {alt_split}")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not load LogiQA2.0 dataset with split: {self.split}")
        
        self.examples = []
        for i, item in enumerate(dataset):
            # Adapt to the actual structure of LogiQA2.0
            # Common fields: text, options, answer, id
            example_id = item.get('id', f"logiqa2_{i}")
            
            item = json.loads(item['text'])
            # Extract question and context
            question = item.get('question', '')
            context = item.get('text', '')
            
            # Extract choices
            choices = item.get('options', item.get('choices', []))
            if isinstance(choices, str):
                # If choices is a string, try to parse it
                choices = self._parse_choices_string(choices)
            
            # Extract answer
            answer = item.get('answer', item.get('label', ''))
            if isinstance(answer, int):
                # Convert integer answer to letter
                answer = chr(ord('A') + answer) if 0 <= answer < 26 else str(answer)
            
            example = LogicalReasoningExample(
                id=str(example_id),
                question=question,
                choices=choices,
                answer=str(answer),
                context=context if context else None
            )
            
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from LogiQA2.0 {self.split} split")
    
    def _parse_choices_string(self, choices_str: str) -> List[str]:
        """Parse choices from string format"""
        # Handle different choice formats
        import re
        
        # Try to split by common patterns
        patterns = [
            r'[A-Z]\)\s*([^A-Z]+?)(?=[A-Z]\)|$)',  # A) choice B) choice
            r'[A-Z]:\s*([^A-Z]+?)(?=[A-Z]:|$)',    # A: choice B: choice
            r'[A-Z]\.\s*([^A-Z]+?)(?=[A-Z]\.|$)',  # A. choice B. choice
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, choices_str, re.DOTALL)
            if matches:
                return [match.strip() for match in matches]
        
        # Fallback: split by newlines or common separators
        choices = re.split(r'[;\n]', choices_str)
        return [choice.strip() for choice in choices if choice.strip()]
