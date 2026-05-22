"""
Dataset creation for permutation evaluation.
"""

import os
import sys
import json
import random
import pathlib
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.permutation_eval.core import PBEInstance
from src.permutation_eval.permutation import (
    PermutedInstance, 
    find_valid_permutation, 
    invert_permutation,
    count_valid_solutions
)
from src.permutation_eval.prompts import format_unscramble_prompt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a permuted dataset for permutation evaluation from PBE data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file containing PBE instances."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSONL file for permuted instances."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of instances to process."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "fb_swap", "exhaustive"],
        help="Permutation search strategy."
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Maximum attempts when searching for a valid permutation."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum examples (input-output) pairs to be shown in the unscrambling prompt."
    )
    parser.add_argument(
        "--unique-solution-only",
        action="store_true",
        help="Skip instances with multiple valid orderings."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging."
    )

    return parser.parse_args()

@dataclass
class DatasetStats:
    """Statistics from dataset creation."""
    total_input: int = 0
    valid_permutations: int = 0
    skipped_verify_failed: int = 0
    skipped_no_valid_perm: int = 0
    skipped_ambiguous: int = 0
    by_cascade_length: dict = None
    by_bfcc_category: dict = None
    solution_counts: dict = None
    
    def __post_init__(self):
        if self.by_cascade_length is None:
            self.by_cascade_length = {}
        if self.by_bfcc_category is None:
            self.by_bfcc_category = {}
        if self.solution_counts is None:
            self.solution_counts = {}
    
    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "valid_permutations": self.valid_permutations,
            "skipped_verify_failed": self.skipped_verify_failed,
            "skipped_no_valid_perm": self.skipped_no_valid_perm,
            "skipped_ambiguous": self.skipped_ambiguous,
            "yield_rate": self.valid_permutations / self.total_input if self.total_input > 0 else 0,
            "by_cascade_length": self.by_cascade_length,
            "by_bfcc_category": self.by_bfcc_category,
            "solution_counts": self.solution_counts,
        }


def load_pbe_data(path: Path) -> list[dict]:
    """Load PBE instances from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            rec['programs'] = [p.replace("\\","") for p in rec['programs']]
            rec['original_programs'] = [p.replace("\\","") for p in rec['original_programs']]
            data.append(rec)

    return data


def create_permuted_dataset(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
    max_instances: Optional[int] = None,
    strategy: str = "random",
    max_attempts: int = 100,
    unique_solution_only: bool = False,
    verbose: bool = True,
    max_examples: int = 10,
) -> DatasetStats:
    """
    Create permuted dataset from existing PBE data.
    
    Args:
        unique_solution_only: If True, skip instances with multiple valid orderings
    """
    random.seed(seed)
    
    raw_data = load_pbe_data(input_path)
    if max_instances:
        raw_data = raw_data[:max_instances]
    
    stats = DatasetStats(total_input=len(raw_data))
    permuted_instances = []
    
    for idx, raw in enumerate(raw_data):
        instance = PBEInstance.from_dict(raw)
        
        # Verify original instance
        if not instance.verify():
            stats.skipped_verify_failed += 1
            continue
        
        # Calculate ambiguity (only possible for small cascades)
        num_solutions = None
        if instance.cascade_length <= 7:
            num_solutions = count_valid_solutions(
                instance.inputs, 
                instance.programs, 
                instance.outputs
            )
            
            # Filter if unique required
            if unique_solution_only and num_solutions is not None and num_solutions > 1:
                stats.skipped_ambiguous += 1
                continue
                
            if num_solutions:
                stats.solution_counts[num_solutions] = stats.solution_counts.get(num_solutions, 0) + 1

        # Find a valid permutation
        result = find_valid_permutation(
            instance, 
            strategy=strategy, 
            max_attempts=max_attempts,
            seed=seed + idx
        )
        
        if result is None:
            stats.skipped_no_valid_perm += 1
            continue
        
        perm, permuted_programs = result
        ground_truth = invert_permutation(perm)
        
        permuted = PermutedInstance(
            id=f"perm_{idx:05d}",
            inputs=instance.inputs,
            outputs=instance.outputs,
            permuted_programs=permuted_programs,
            permutation=perm,
            ground_truth_order=ground_truth,
            original_programs=instance.programs,
            cascade_length=instance.cascade_length,
            bfcc_category=instance.bfcc_category,
            bfcc_string=raw.get("bfcc_string", ""),
            num_valid_solutions=num_solutions,
            prompt=format_unscramble_prompt(
                inputs=instance.inputs,
                outputs=instance.outputs,
                permuted_programs=permuted_programs,
                max_examples=max_examples,
            ),
        )
        permuted_instances.append(permuted)
        stats.valid_permutations += 1
        
        # Update stratified stats
        cl = instance.cascade_length
        stats.by_cascade_length[cl] = stats.by_cascade_length.get(cl, 0) + 1
        
        cat = instance.bfcc_category
        stats.by_bfcc_category[cat] = stats.by_bfcc_category.get(cat, 0) + 1
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for inst in permuted_instances:
            f.write(json.dumps(inst.to_dict()) + "\n")
    
    # Write stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    
    if verbose:
        print(f"Created {len(permuted_instances)} permuted instances")
        print(f"  Yield rate: {stats.valid_permutations}/{stats.total_input} "
              f"({stats.valid_permutations/stats.total_input*100:.1f}%)")
        print(f"  Skipped (ambiguous): {stats.skipped_ambiguous}")
        print(f"  Skipped (no valid perm): {stats.skipped_no_valid_perm}")
    
    return stats

def main() -> None:
    args = parse_args()

    create_permuted_dataset(
        input_path=args.input,
        output_path=args.output,
        seed=args.seed,
        max_instances=args.max_instances,
        strategy=args.strategy,
        max_attempts=args.max_attempts,
        unique_solution_only=args.unique_solution_only,
        verbose=not args.quiet,
        max_examples=args.max_examples,
    )

# main
if __name__ == "__main__":
    main()
    # python src/permutation_eval/dataset.py --input "data/adaptive_balanced_1008_complete_promptsfile.jsonl" --output "data/adaptive_balanced_1008_permutation_promptsfile.jsonl" --max-attempts 10000 --seed 42 --strategy "fb_swap"