"""
Permutation Evaluation Package for PBEBench.

This package provides tools for testing LLM understanding of program ordering
by creating permuted datasets and evaluating model ability to unscramble them.

Key insight from BFCC relations (Kiparsky 1968, 1971):
- Feeding (F): Program A's output creates context for Program B
- Bleeding (B): Program A removes context that Program B needs
- When programs have F/B relations, order can matter for the output

Modules:
- core: Core data structures and program execution
- permutation: Logic for finding valid permutations
- dataset: Dataset creation utilities
- evaluate: LLM evaluation framework
- prompts: High-quality evaluation prompts
"""

from .core import apply_programs, PBEInstance
from .permutation import find_valid_permutation, PermutedInstance
from .dataset import create_permuted_dataset
from .evaluate import run_evaluation, EvalResult

__all__ = [
    "apply_programs",
    "PBEInstance", 
    "find_valid_permutation",
    "PermutedInstance",
    "create_permuted_dataset",
    "run_evaluation",
    "EvalResult",
]


