"""Quick test script to validate the evaluation framework"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import EvaluationConfig
from src.datasets.logiqa2 import LogiQA2Dataset
from src.utils.metrics import extract_answer_choice, majority_vote


def test_config():
    """Test configuration creation and validation"""
    print("Testing configuration...")
    
    config = EvaluationConfig(
        model_name="gpt2",
        datasets=["logiqa2"],
        sampling_method="direct"
    )
    
    config.validate()
    print("✓ Configuration validation passed")
    
    # Test invalid config
    try:
        invalid_config = EvaluationConfig(
            model_name="gpt2",
            datasets=["invalid_dataset"],
            sampling_method="direct"
        )
        invalid_config.validate()
        print("✗ Should have failed validation")
    except ValueError:
        print("✓ Invalid configuration correctly rejected")


def test_answer_extraction():
    """Test answer extraction from model outputs"""
    print("\nTesting answer extraction...")
    
    test_cases = [
        ("A", "A"),
        ("The answer is B", "B"),
        ("I think the correct choice is C.", "C"),
        ("Answer: D", "D"),
        ("Based on the logic, A) is correct", "A"),
        ("This is clearly option B)", "B"),
        ("Invalid response", "INVALID")
    ]
    
    for text, expected in test_cases:
        result = extract_answer_choice(text)
        if result == expected:
            print(f"✓ '{text}' -> '{result}'")
        else:
            print(f"✗ '{text}' -> '{result}' (expected '{expected}')")


def test_majority_vote():
    """Test majority voting"""
    print("\nTesting majority vote...")
    
    responses = ["A", "B", "A", "A", "C"]
    result = majority_vote(responses)
    expected = "A"
    
    if result == expected:
        print(f"✓ Majority vote: {responses} -> {result}")
    else:
        print(f"✗ Majority vote: {responses} -> {result} (expected {expected})")


def test_dataset_loading():
    """Test dataset loading (requires internet connection)"""
    print("\nTesting dataset loading...")
    
    try:
        # Try to load a small subset for testing
        print("Attempting to load LogiQA2 dataset...")
        dataset = LogiQA2Dataset()
        
        if len(dataset) > 0:
            print(f"✓ LogiQA2 dataset loaded: {len(dataset)} examples")
            
            # Test example formatting
            example = dataset[0]
            prompt = example.format_prompt()
            print(f"✓ Example formatting works")
            print(f"Sample prompt length: {len(prompt)} characters")
        else:
            print("✗ Dataset loaded but is empty")
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        print("Note: This might be due to network issues or dataset availability")


if __name__ == "__main__":
    print("LogiEval Framework Test")
    print("=" * 40)
    
    test_config()
    test_answer_extraction()
    test_majority_vote()
    test_dataset_loading()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nTo run a full evaluation, use:")
    print("python evaluate.py --model_name gpt2 --datasets logiqa2 --batch_size 2")
