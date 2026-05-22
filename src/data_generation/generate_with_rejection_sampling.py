#!/usr/bin/env python3
# balanced_sampler.py - Implements rejection sampling for balanced program relationships

import os
import sys
import json
import random
import argparse
import pathlib
from tqdm import tqdm
from collections import defaultdict, Counter

# Ensure the module path is correctly set up
module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

# Import existing functionality from detector.py and generate.py
from src.data_generation.detector import process_programs
from src.data_generation.generate import sample_inputs, sample_program, write_jsonl

def sample_instance(vocab_chars, num_inputs=5, input_len_range=(2,6), env_len_range=(1,3), seq_len=3):
    """
    Generate a sample instance with inputs, outputs, programs, and their relationships.
    
    Args:
        vocab_chars (str): Characters to use in the vocabulary.
        num_inputs (int): Number of input strings to generate.
        input_len_range (tuple): Range of lengths for input strings.
        env_len_range (tuple): Range of lengths for environment strings.
        seq_len (int): Number of programs in the sequence.
        
    Returns:
        dict: A dictionary containing inputs, outputs, programs, and their relationships.
    """
    try:
        # Import inside the function as in the original code
        from src.data_generation.primitives import ProgramVocabulary, ProgramBFCCNode
        
        inputs = None
        outputs = None
        attempts = 0
        max_local_attempts = 50  # Prevent infinite loops
        
        while inputs == outputs and attempts < max_local_attempts:
            inputs = sample_inputs(vocab_chars=vocab_chars, num_inputs=num_inputs, input_len_range=input_len_range)
            outputs = inputs.copy()  # Create a copy to avoid reference issues
            programs = []
            for _ in range(seq_len):
                code, string_repr = sample_program(vocab_chars=vocab_chars, env_len_range=env_len_range)
                outputs = code(outputs)
                programs.append(string_repr)
            attempts += 1
            
        if attempts >= max_local_attempts:
            print(f"Warning: Failed to generate non-identical inputs and outputs after {max_local_attempts} attempts")
            return None
            
        # Process programs to get relationships
        bfcc_dag = process_programs(programs=programs)
            
        return {"inputs": inputs, "outputs": outputs, "programs": programs, "bfcc_dag": bfcc_dag}
    except Exception as e:
        print(f"Error generating sample instance: {e}")
        return None

class RelationshipClassifier:
    """Classifies program relationships into feeding, bleeding, counterfeeding, and counterbleeding."""
    
    @staticmethod
    def classify_instance(instance):
        """
        Classify the relationships in an instance.
        
        Args:
            instance (dict): An instance containing programs and their relationships.
            
        Returns:
            dict: A dictionary with counts of each relationship type.
        """
        if instance is None:
            return {"feeding": 0, "bleeding": 0, "counterfeeding": 0, "counterbleeding": 0, "neutral": 0}
            
        relationships = instance.get("bfcc_dag", [])
        
        # Initialize counters
        relation_counts = {
            "feeding": 0,
            "bleeding": 0,
            "counterfeeding": 0,
            "counterbleeding": 0,
            "neutral": 0
        }
        
        # Process each relationship
        for rel in relationships:
            if len(rel) >= 3:
                src_idx, rel_types, dst_idx = rel[0], rel[1], rel[2]
                
                # Counter-relationships occur when later programs (higher indices) 
                # affect earlier programs (lower indices)
                is_counter = src_idx > dst_idx  # e.g., 2→1, 2→0, or 1→0
                
                for rel_type in rel_types:
                    if rel_type == 'F':
                        if is_counter:
                            relation_counts["counterfeeding"] += 1
                        else:
                            relation_counts["feeding"] += 1
                    elif rel_type == 'B':
                        if is_counter:
                            relation_counts["counterbleeding"] += 1
                        else:
                            relation_counts["bleeding"] += 1
                    elif rel_type == 'N':
                        relation_counts["neutral"] += 1
        
        return relation_counts
    
    @staticmethod
    def get_dominant_relationship(counts):
        """
        Returns the dominant relationship type in an instance.
        
        Args:
            counts (dict): A dictionary with counts of each relationship type.
            
        Returns:
            str: The dominant relationship type.
        """
        # Exclude neutral from consideration unless it's the only relationship
        non_neutral = {k: v for k, v in counts.items() if k != "neutral" and v > 0}
        if non_neutral:
            return max(non_neutral.items(), key=lambda x: x[1])[0]
        return "neutral"

def generate_balanced_dataset(target_size, vocab_chars, min_seq_len=2, max_seq_len=5, max_attempts=100000):
    """
    Generate a dataset with balanced relationship types.
    
    Args:
        target_size (int): Total number of instances in the final dataset.
        vocab_chars (str): String of characters to use in the vocabulary.
        min_seq_len (int): Minimum sequence length.
        max_seq_len (int): Maximum sequence length.
        max_attempts (int): Maximum number of attempts before giving up.
        
    Returns:
        tuple: (List of instances with balanced relationship types, statistics dictionary)
    """
    # Target counts for each relationship type
    target_per_type = target_size // 4
    
    # Initialize datasets by relationship type
    datasets = {
        "feeding": [],
        "bleeding": [],
        "counterfeeding": [],
        "counterbleeding": []
    }
    
    # Track progress
    instances_generated = 0
    instances_needed = {rel_type: target_per_type for rel_type in datasets.keys()}
    all_relation_counts = defaultdict(int)  # Track all relationship types
    
    print("Generating balanced dataset...")
    
    # Generate instances until we have enough of each type or reach max attempts
    with tqdm(total=target_size) as pbar:
        while sum(instances_needed.values()) > 0 and instances_generated < max_attempts:
            # Generate a new instance
            seq_len = random.randint(min_seq_len, max_seq_len)
            instance = sample_instance(
                vocab_chars=vocab_chars,
                num_inputs=5,
                input_len_range=(2, 6),
                env_len_range=(1, 3),
                seq_len=seq_len
            )
            
            if instance is None:
                continue  # Skip this iteration if instance generation failed
                
            instances_generated += 1
            
            # Classify the instance
            relation_counts = RelationshipClassifier.classify_instance(instance)
            dominant_relation = RelationshipClassifier.get_dominant_relationship(relation_counts)
            
            # Update global counts
            for rel_type, count in relation_counts.items():
                all_relation_counts[rel_type] += count
            
            # If we still need instances of this type, add it to the dataset
            if dominant_relation in instances_needed and instances_needed[dominant_relation] > 0:
                datasets[dominant_relation].append(instance)
                instances_needed[dominant_relation] -= 1
                pbar.update(1)
            
            # Provide progress information every 100 instances
            if instances_generated % 100 == 0:
                remaining = sum(instances_needed.values())
                efficiency = (target_size - remaining) / instances_generated * 100 if instances_generated > 0 else 0
                print(f"\nGenerated {instances_generated} instances")
                print(f"Still need: {dict(instances_needed)}")
                print(f"Current distribution: {dict(all_relation_counts)}")
                print(f"Efficiency: {efficiency:.2f}%")
                
                # Early termination condition for very inefficient sampling
                if instances_generated >= 10000 and efficiency < 1:
                    print("Warning: Low efficiency detected. Consider adjusting parameters.")
    
    # Check if we reached max attempts
    if instances_generated >= max_attempts:
        print(f"Warning: Reached maximum number of attempts ({max_attempts}) before completing the dataset.")
    
    # Combine all datasets
    final_dataset = []
    for rel_type, instances in datasets.items():
        final_dataset.extend(instances)
        print(f"{rel_type}: {len(instances)} instances")
    
    # Shuffle the dataset
    random.shuffle(final_dataset)
    
    # Add generation statistics
    stats = {
        "total_generated": instances_generated,
        "total_kept": len(final_dataset),
        "efficiency": len(final_dataset) / instances_generated * 100 if instances_generated > 0 else 0,
        "relationship_distribution": {rel_type: len(instances) for rel_type, instances in datasets.items()},
        "relationship_counts": dict(all_relation_counts),
        "completed": sum(instances_needed.values()) == 0
    }
    
    print(f"Dataset generation complete.")
    print(f"Generated {instances_generated} instances to get {len(final_dataset)} balanced instances.")
    print(f"Efficiency: {stats['efficiency']:.2f}%")
    
    return final_dataset, stats

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a balanced dataset of program transformations")
    parser.add_argument("--vocab", type=str, default="abcdefghijkuvwxyz", 
                        help="Characters to use in the vocabulary (default: abcdefghijkuvwxyz)")
    parser.add_argument("--size", type=int, default=1000, 
                        help="Target dataset size (default: 1000)")
    parser.add_argument("--min-seq-len", type=int, default=2, 
                        help="Minimum sequence length (default: 2)")
    parser.add_argument("--max-seq-len", type=int, default=5, 
                        help="Maximum sequence length (default: 5)")
    parser.add_argument("--output", type=str, default="balanced_program_transformations_dataset.jsonl", 
                        help="Output file path (default: balanced_program_transformations_dataset.jsonl)")
    parser.add_argument("--stats", type=str, default="generation_stats.json", 
                        help="Statistics file path (default: generation_stats.json)")
    parser.add_argument("--max-attempts", type=int, default=100000, 
                        help="Maximum sampling attempts (default: 100000)")
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Vocabulary: {args.vocab}")
    print(f"  Target size: {args.size}")
    print(f"  Sequence length: {args.min_seq_len}-{args.max_seq_len}")
    print(f"  Output file: {args.output}")
    print(f"  Stats file: {args.stats}")
    print(f"  Max attempts: {args.max_attempts}")
    
    # Generate the balanced dataset
    dataset, stats = generate_balanced_dataset(
        target_size=args.size,
        vocab_chars=args.vocab,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        max_attempts=args.max_attempts
    )
    
    # Write dataset to file
    print(f"Writing {len(dataset)} samples to {args.output}")
    write_jsonl(dataset, args.output)
    
    # Write stats to file
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset written to {args.output}")
    print(f"Statistics written to {args.stats}")
    print("Complete!")