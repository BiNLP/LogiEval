"""LogiQA dataset loader"""

from typing import List
from datasets import load_dataset
from .base import BaseDataset, LogicalReasoningExample


class LogiQADataset(BaseDataset):
    """LogiQA dataset from lucasmccabe/logiqa"""
    
    def __init__(self, split: str = "test"):
        super().__init__(split)
        self.name = "logiqa"
        self.load_data()
    
    def load_data(self) -> None:
        """Load LogiQA dataset from HuggingFace"""
        try:
            dataset = load_dataset("lucasmccabe/logiqa", split=self.split)
        except Exception as e:
            print(f"Error loading LogiQA dataset: {e}")
            print("Trying alternative split names...")
            # Try common split names
            for alt_split in ["test", "validation", "dev", "train"]:
                try:
                    dataset = load_dataset("lucasmccabe/logiqa", split=alt_split)
                    print(f"Successfully loaded with split: {alt_split}")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not load LogiQA dataset with split: {self.split}")
        
        self.examples = []
        for i, item in enumerate(dataset):
            # Adapt to the actual structure of LogiQA
            example_id = item.get('id', f"logiqa_{i}")
            
            # Extract question and context
            question = item.get('question', item.get('text', ''))
            context = item.get('context', item.get('passage', ''))
            
            # Extract choices
            choices = []
            choice_keys = ['option_a', 'option_b', 'option_c', 'option_d']
            for key in choice_keys:
                if key in item and item[key]:
                    choices.append(item[key])
            
            # If no individual options, try 'options' or 'choices'
            if not choices:
                choices = item.get('options', item.get('choices', []))
                if isinstance(choices, str):
                    choices = self._parse_choices_string(choices)
            
            # Extract answer
            answer = item.get('answer', item.get('label', ''))
            if isinstance(answer, int):
                # Convert integer answer to letter
                answer = chr(ord('A') + answer) if 0 <= answer < 26 else str(answer)
            
            # Ensure we have valid data
            if not question or not choices:
                continue
            
            example = LogicalReasoningExample(
                id=str(example_id),
                question=question,
                choices=choices,
                answer=str(answer),
                context=context if context else None
            )
            
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from LogiQA {self.split} split")
    
    def _parse_choices_string(self, choices_str: str) -> List[str]:
        """Parse choices from string format"""
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
