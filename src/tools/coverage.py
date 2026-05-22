import os
import sys
import json
import pathlib
from typing import List
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.tools.tool import Tool

class CoverageChecker(Tool):
    def __init__(self, min_evidence_size: int):
        """min_evidence_size: Minimum number of words to be modified by each action."""
        # attribute to aggregate unique substrings counts across input strings.
        self.substring_counter = defaultdict(lambda: 0) 
        self.min_evidence_size = min_evidence_size

    def reset_substring_counter(self):
        self.substring_counter = defaultdict(lambda: 0)

    def get_unique_substrings(self, s: str, k: int):
        if k <= 0 or k > len(s):
            return set()  # no valid substrings possible
        return {s[i:i+k] for i in range(len(s) - k + 1)}
    
    def get_candidate_alphas(self):
        candidate_alphas = {}
        print(self.substring_counter)
        for substring, count in self.substring_counter.items():
            if count >= self.min_evidence_size:
                candidate_alphas[substring] = count 

        return candidate_alphas

    def __call__(self, input_strings: list[str], substring_size: int=1):
        # reset substring counter before processing a new input.
        self.reset_substring_counter()
        
        for input_string in input_strings:
            substrings = list(self.get_unique_substrings(s=input_string, k=substring_size))
            for substring in substrings:
                self.substring_counter[substring] += 1
        
        return self.get_candidate_alphas()
        

def test_coverage_checker():
    input_strings = ["aabcd", "baacd", "cdaab"]
    coverage_checker_tool = CoverageChecker(min_evidence_size=2)
    candidate_alphas = coverage_checker_tool(input_strings, substring_size=2)
    # get candidate alphas as computed by the Coverage Checker tool.
    print(candidate_alphas)

# main
if __name__ == "__main__":
    test_coverage_checker()