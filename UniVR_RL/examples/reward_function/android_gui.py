# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Number Game Reward Function

Scoring rules:
- Selecting the correct number: +1.0
- Selecting the wrong number: 0.0

Input format:
reward_input = {
    "response": "1",  # model output answer (0/1/2)
    "response_length": 10,  # response length in tokens
    "ground_truth": "1"  # correct answer (0/1/2)
}

Output format:
{
    "overall": 1.0,  # total score (required field)
    "accuracy": 1.0  # accuracy (optional, for monitoring)
}
"""

import re
from typing import Any


# Metadata - required by the EasyR1 framework
REWARD_NAME = "number_game"
REWARD_TYPE = "batch"  # batch processing mode


def extract_answer(response: str) -> str:
    """
    Extract the answer index from a model response.

    Args:
        response: Raw model response string.

    Returns:
        "0", "1", "2", or "" on extraction failure.
    """
    # Case 1: response is itself a single digit
    response = response.strip()
    if response in ["0", "1", "2"]:
        return response

    # Case 2: response contains extra text — extract the first occurrence of 0/1/2
    match = re.search(r"[012]", response)
    if match:
        return match.group(0)

    # Extraction failed
    return ""


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    Compute scores for a batch of samples.

    Args:
        reward_inputs: List of sample dicts, each containing:
            - response: model response string
            - response_length: response length in tokens
            - ground_truth: correct answer

    Returns:
        List of score dicts, each containing:
            - overall: total score (1.0 = correct, 0.0 = wrong)
            - accuracy: accuracy (same as overall, for monitoring)
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")

        # Extract predicted answer
        predicted = extract_answer(response)

        # Compute score
        if predicted == ground_truth:
            score = 1.0
        else:
            score = 0.0

        # Return format: must include the 'overall' field
        scores.append({"overall": score, "accuracy": score})

    return scores


# Test cases
if __name__ == "__main__":
    test_cases = [
        # Perfect match
        {"response": "0", "response_length": 1, "ground_truth": "0"},
        {"response": "1", "response_length": 1, "ground_truth": "1"},
        {"response": "2", "response_length": 1, "ground_truth": "2"},
        # Response with extra text
        {"response": "The answer is 1", "response_length": 15, "ground_truth": "1"},
        {"response": "I choose option 2", "response_length": 18, "ground_truth": "2"},
        # Wrong answer
        {"response": "0", "response_length": 1, "ground_truth": "1"},
        {"response": "2", "response_length": 1, "ground_truth": "0"},
        # Extraction failure
        {"response": "I don't know", "response_length": 12, "ground_truth": "1"},
        {"response": "", "response_length": 0, "ground_truth": "2"},
    ]

    scores = compute_score(test_cases)

    print("Reward Function Test Results:")
    print("=" * 60)
    for i, (test, score) in enumerate(zip(test_cases, scores), 1):
        print(f"{i}. Response: {test['response']!r}")
        print(f"   Ground Truth: {test['ground_truth']!r}")
        print(f"   Score: {score}")
        print()
