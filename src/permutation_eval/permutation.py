"""
Permutation generation and validation logic.

The key challenge: not all permutations change the output.
We need to find permutations that provably produce different results.
"""

import os
import sys
import random
import pathlib
from dataclasses import dataclass, field
from typing import Optional
from itertools import permutations as all_permutations

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.permutation_eval.core import PBEInstance, apply_programs


@dataclass
class PermutedInstance:
    """
    A PBE instance with scrambled program order.
    
    The task: given (inputs, outputs, permuted_programs), 
    determine the correct ordering to produce outputs from inputs.
    """
    id: str
    inputs: list[str]
    outputs: list[str]  # Outputs under ORIGINAL ordering
    permuted_programs: list[str]  # Programs in scrambled order
    permutation: list[int]  # The permutation applied
    ground_truth_order: list[int]  # Indices to recover original
    original_programs: list[str]  # For reference
    cascade_length: int
    bfcc_category: str
    bfcc_string: str = ""
    prompt: str = ""
    num_valid_solutions: Optional[int] = None  # How many orderings produce valid outputs
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "permuted_programs": self.permuted_programs,
            "permutation": self.permutation,
            "ground_truth_order": self.ground_truth_order,
            "original_programs": self.original_programs,
            "cascade_length": self.cascade_length,
            "bfcc_category": self.bfcc_category,
            "bfcc_string": self.bfcc_string,
            "num_valid_solutions": self.num_valid_solutions,
            "prompt": self.prompt,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PermutedInstance":
        """Create from dict, ignoring unknown keys for forward compatibility."""
        known_fields = {
            "id", "inputs", "outputs", "permuted_programs", "permutation",
            "ground_truth_order", "original_programs", "cascade_length",
            "bfcc_category", "bfcc_string", "num_valid_solutions", "prompt"
        }
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


def invert_permutation(perm: list[int]) -> list[int]:
    """
    Given permutation P, return inverse Q such that P[Q[i]] = i.
    
    If we permuted programs by perm, ground_truth_order tells us
    which index in the permuted list should come first, second, etc.
    """
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def outputs_differ(
    inputs: list[str],
    programs: list[str],
    perm: list[int],
    original_outputs: list[str]
) -> bool:
    """Check if a permutation produces different outputs."""
    permuted_programs = [programs[p] for p in perm]
    new_outputs = apply_programs(inputs, permuted_programs)
    return new_outputs != original_outputs


def count_valid_solutions(
    inputs: list[str],
    programs: list[str],
    target_outputs: list[str],
    max_factorial: int = 7
) -> Optional[int]:
    """
    Count how many permutations produce the target output.
    Returns None if cascade_length > max_factorial (too expensive).
    """
    n = len(programs)
    if n > max_factorial:
        return None
    
    count = 0
    for perm in all_permutations(range(n)):
        permuted = [programs[p] for p in perm]
        outputs = apply_programs(inputs, permuted)
        if outputs == target_outputs:
            count += 1
    return count


def count_unique_outputs(
    inputs: list[str],
    programs: list[str],
    max_factorial: int = 7
) -> Optional[int]:
    """
    Count unique outputs across all permutations.
    
    Only feasible for small cascade lengths due to factorial growth.
    Returns None if cascade_length > max_factorial.
    """
    n = len(programs)
    if n > max_factorial:
        return None
    
    seen = set()
    for perm in all_permutations(range(n)):
        permuted = [programs[p] for p in perm]
        outputs = apply_programs(inputs, permuted)
        seen.add(tuple(outputs))
    
    return len(seen)


def find_valid_permutation(
    instance: PBEInstance,
    strategy: str = "random",
    max_attempts: int = 100,
    seed: Optional[int] = None
) -> Optional[tuple[list[int], list[str]]]:
    """
    Find a permutation that produces different outputs.
    
    Args:
        instance: The PBE instance
        strategy: "random", "fb_swap", or "exhaustive"
        max_attempts: Max random attempts (for random strategy)
        seed: Random seed for reproducibility
    
    Returns:
        (permutation, permuted_programs) or None if not found
    
    Strategies:
        - random: Try random permutations until one works
        - fb_swap: Prioritize swapping F/B related pairs
        - exhaustive: Try all permutations (small cascades only)
    """
    if seed is not None:
        random.seed(seed)
    
    n = len(instance.programs)
    original_outputs = instance.outputs
    
    if strategy == "exhaustive" and n <= 7:
        return _find_exhaustive(instance)
    elif strategy == "fb_swap":
        result = _find_by_fb_swap(instance)
        if result:
            return result
        # Fall through to random if fb_swap fails
    
    # Random strategy
    for _ in range(max_attempts):
        perm = list(range(n))
        random.shuffle(perm)
        
        if perm == list(range(n)):
            continue
        
        if outputs_differ(instance.inputs, instance.programs, perm, original_outputs):
            permuted = [instance.programs[p] for p in perm]
            return perm, permuted
    
    return None


def _find_by_fb_swap(instance: PBEInstance) -> Optional[tuple[list[int], list[str]]]:
    """Try swapping F/B related pairs first."""
    fb_edges = instance.get_fb_edges()
    
    # Get unique pairs (avoid swapping same pair twice)
    seen_pairs = set()
    for i, rel, j in fb_edges:
        pair = (min(i, j), max(i, j))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            
            # Try swapping this pair
            perm = list(range(len(instance.programs)))
            perm[i], perm[j] = perm[j], perm[i]
            
            if outputs_differ(instance.inputs, instance.programs, perm, instance.outputs):
                permuted = [instance.programs[p] for p in perm]
                return perm, permuted
    
    return None


def _find_exhaustive(instance: PBEInstance) -> Optional[tuple[list[int], list[str]]]:
    """Try all permutations, return first that differs."""
    n = len(instance.programs)
    original_order = tuple(range(n))
    
    for perm in all_permutations(range(n)):
        if perm == original_order:
            continue
        
        perm_list = list(perm)
        if outputs_differ(instance.inputs, instance.programs, perm_list, instance.outputs):
            permuted = [instance.programs[p] for p in perm_list]
            return perm_list, permuted
    
    return None
