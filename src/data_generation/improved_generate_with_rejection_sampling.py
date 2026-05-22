#!/usr/bin/env python3
"""
Usage:
  python src/data_generation/improved_generate_with_rejection_sampling.py \
    --size 1008 \
    --min-seq-len 2 \
    --max-seq-len 5 \
    --output data/cascaded_checked_balanced_program_transformations_dataset.jsonl \
    --stats data/generation_stats.json \
    --vocab abcdefghijkuvwxyz \
    --dedupe  # use --no-dedupe to disable

Generates a balanced dataset over all 16 BFCC binary categories, with positional
pruning of degenerate programs and online category balancing.
"""

import os
import sys
import json
import random
import pathlib
import argparse
from tqdm import tqdm
from typing import List

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.detector import process_programs
from src.data_generation.generate import sample_inputs, ProgramVocabulary, ProgramBFCCNode

def substrings_fixed_length(s_l: list[str], c: int):
    result = []
    for s in s_l:
        n = len(s)
        for i in range(n - c + 1):
            result.append(s[i:i+c])
    return result

def sample_program(vocab_chars: str, input_examples: List[str], env_len_range: tuple[int, int]=(1,3)):
    program_vocab = ProgramVocabulary(list(vocab_chars))
    env_lens = list(range(env_len_range[0], env_len_range[1]+1))
    input_env_len = random.choice([1,2,3])
    input_chars_options = substrings_fixed_length(input_examples, c=input_env_len)
    input_chars = random.choice(input_chars_options)
    output_env_len = random.choice([1,2,3])
    output_chars = "".join(random.choices(vocab_chars, k=output_env_len))
    string_repr = f'replace("{input_chars}", "{output_chars}")'
    code = ProgramBFCCNode.from_string(string_repr=string_repr, vocabulary=program_vocab)
    return code, string_repr

def sample_program_with_min_k(current_strings: List[str], vocab_chars: str, env_len_range: tuple[int, int]=(1,3), k: int=1, max_tries: int=50):
    if not current_strings:
        return None, None
    
    program_vocab = ProgramVocabulary(list(vocab_chars))
    pattern_length = random.randint(env_len_range[0], env_len_range[1])
    coverage_for_length = {}
    
    for string_idx, s in enumerate(current_strings):
        for i in range(len(s) - pattern_length + 1):
            substring = s[i:i+pattern_length]
            if substring not in coverage_for_length:
                coverage_for_length[substring] = set()
            coverage_for_length[substring].add(string_idx)
    
    candidates = [sub for sub, indices in coverage_for_length.items() if len(indices) >= k]
    if not candidates:
        return None, None
    
    tries = 0
    while tries < max_tries:
        input_chars = random.choice(candidates)
        output_env_len = random.randint(env_len_range[0], env_len_range[1])
        output_chars = "".join(random.choices(vocab_chars, k=output_env_len))
        
        if output_chars == input_chars:
            tries += 1
            continue
        
        changed = 0
        for s in current_strings:
            new_s = s.replace(input_chars, output_chars)
            if new_s != s:
                changed += 1
        
        if changed >= k:
            string_repr = f'replace("{input_chars}", "{output_chars}")'
            code = ProgramBFCCNode.from_string(string_repr=string_repr, vocabulary=program_vocab)
            return code, string_repr
        
        tries += 1
    
    return None, None

def run_cascade(inputs, program_strings, vocab_chars):
    """
    Execute a cascade on the given inputs.
    """
    from src.data_generation.primitives import ProgramVocabulary, ProgramBFCCNode
    current = inputs[:]
    program_vocab = ProgramVocabulary(list(vocab_chars))
    for prog in program_strings:
        node = ProgramBFCCNode.from_string(prog, vocabulary=program_vocab)
        current = node(current)
    return current

def prune_degenerate_programs(inputs, program_strings, vocab_chars):
    """
    Drop any program that does not change any input at its position.
    Returns (pruned_programs, final_outputs, kept_indices, pruned_count).
    """
    from src.data_generation.primitives import ProgramVocabulary, ProgramBFCCNode
    program_vocab = ProgramVocabulary(list(vocab_chars))
    current = inputs[:]
    pruned = []
    kept_indices = []
    for idx, prog in enumerate(program_strings):
        node = ProgramBFCCNode.from_string(prog, vocabulary=program_vocab)
        next_outputs = node(current)
        if next_outputs != current:
            pruned.append(prog)
            kept_indices.append(idx)
            current = next_outputs
    return pruned, current, kept_indices, (len(program_strings) - len(pruned))

def sample_instance(vocab_chars, num_inputs=5, input_len_range=(2,6), env_len_range=(1,3), seq_len=3, min_cascade_len=1, min_modified_per_program: int=1):
    """
    Generate a sample instance with inputs, outputs, programs, and their relationships.
    
    Args:
        vocab_chars (str): Characters to use in the vocabulary.
        num_inputs (int): Number of input strings to generate.
        input_len_range (tuple): Range of lengths for input strings.
        env_len_range (tuple): Range of lengths for environment strings.
        seq_len (int): Number of programs in the sequence.
        min_cascade_len (int): Minimum length required for the pruned cascade.
        min_modified_per_program (int): Lower bound on number of strings each program must modify at its step.
        
    Returns:
        dict: A dictionary containing inputs, outputs, programs, and their relationships.
    """
    try:
        inputs = None
        outputs = None
        attempts = 0
        max_attempts = 50

        while attempts < max_attempts:
            inputs = sample_inputs(vocab_chars=vocab_chars, num_inputs=num_inputs, input_len_range=input_len_range)
            outputs = inputs.copy()
            raw_programs = []
            failed_step = False
            for _ in range(seq_len):
                code, string_repr = sample_program_with_min_k(current_strings=outputs, vocab_chars=vocab_chars, env_len_range=env_len_range, k=min_modified_per_program)
                if code is None:
                    failed_step = True
                    break
                outputs = code(outputs)
                raw_programs.append(string_repr)

            if failed_step:
                attempts += 1
                continue

            if inputs == outputs:
                attempts += 1
                continue

            pruned_programs, _, _, _ = prune_degenerate_programs(
                inputs, raw_programs, vocab_chars
            )

            if len(pruned_programs) == 0 or len(pruned_programs) < min_cascade_len:
                attempts += 1
                continue

            recon_outputs = run_cascade(inputs, pruned_programs, vocab_chars)
            if recon_outputs != outputs:
                raise RuntimeError("Post-prune conservation failed: pruned cascade outputs differ from pre-prune outputs")

            bfcc_dag = process_programs(programs=pruned_programs)

            return {
                "inputs": inputs,
                "outputs": outputs,
                "programs": pruned_programs,
                "bfcc_dag": bfcc_dag,
                "cascade_length": len(pruned_programs)
            }

        print(f"Warning: Failed to generate non-identical inputs/outputs after {max_attempts} attempts")
        return None
    except Exception as e:
        print(f"Error generating sample instance: {e}")
        return None

class RelationshipClassifierRelaxed:
    """Classifies program relationships into binary categories."""
    
    @staticmethod
    def classify_binary(instance):
        """
        Classify the relationships in an instance into binary categories.
        
        Args:
            instance (dict): An instance containing programs and relationships.
            
        Returns:
            dict: A dictionary with binary flags for each relationship type.
            str: A category string representing the binary pattern.
        """
        if instance is None:
            return {"F": 0, "B": 0, "CF": 0, "CB": 0}, "0000"
        
        # If we need to regenerate the BFCC DAG (uses original_programs)
        if "original_programs" in instance and instance.get("bfcc_dag") is None:
            instance["bfcc_dag"] = process_programs(instance["original_programs"])
            
        relationships = instance.get("bfcc_dag", [])
        
        # Initialize binary flags
        relation_binary = {
            "F": 0,   # Feeding
            "B": 0,   # Bleeding
            "CF": 0,  # Counterfeeding
            "CB": 0   # Counterbleeding
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
                            relation_binary["CF"] = 1
                        else:
                            relation_binary["F"] = 1
                    elif rel_type == 'B':
                        if is_counter:
                            relation_binary["CB"] = 1
                        else:
                            relation_binary["B"] = 1
        
        # Create a category string (e.g., "1010" for F=1, B=0, CF=1, CB=0)
        category = "".join(str(relation_binary[r]) for r in ["F", "B", "CF", "CB"])
        # reduce categories to binary presence ones. If more than one relationship type is present, randomly pick one as the representative.
        reduced_categories = []
        for i,bit in enumerate(category):
            if bit == "1":
                reduced_categories.append("0"*(i)+"1"+"0"*(3-i))
        if len(reduced_categories) == 0:
            reduced_categories = ["0000"]
        category = random.choice(reduced_categories)
        
        return relation_binary, category
    
    @staticmethod
    def get_all_categories():
        """
        Generate all possible binary categories for the four relationship types.
        
        Returns:
            list: All possible binary category strings.
        """
        categories = ["0000"]
        # Generate all 16 possibilities (2^4)
        for i in range(4):
            # Convert to 4-bit binary string with leading zeros
            cat = "0"*(3-i)+"1"+"0"*(i)
            categories.append(cat)
        
        return categories

class RelationshipClassifier:
    """Classifies program relationships into binary categories."""
    
    @staticmethod
    def classify_binary(instance):
        """
        Classify the relationships in an instance into binary categories.
        
        Args:
            instance (dict): An instance containing programs and relationships.
            
        Returns:
            dict: A dictionary with binary flags for each relationship type.
            str: A category string representing the binary pattern.
        """
        if instance is None:
            return {"F": 0, "B": 0, "CF": 0, "CB": 0}, "0000"
        
        # If we need to regenerate the BFCC DAG (uses original_programs)
        if "original_programs" in instance and instance.get("bfcc_dag") is None:
            instance["bfcc_dag"] = process_programs(instance["original_programs"])
            
        relationships = instance.get("bfcc_dag", [])
        
        # Initialize binary flags
        relation_binary = {
            "F": 0,   # Feeding
            "B": 0,   # Bleeding
            "CF": 0,  # Counterfeeding
            "CB": 0   # Counterbleeding
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
                            relation_binary["CF"] = 1
                        else:
                            relation_binary["F"] = 1
                    elif rel_type == 'B':
                        if is_counter:
                            relation_binary["CB"] = 1
                        else:
                            relation_binary["B"] = 1
        
        # Create a category string (e.g., "1010" for F=1, B=0, CF=1, CB=0)
        category = "".join(str(relation_binary[r]) for r in ["F", "B", "CF", "CB"])
        
        return relation_binary, category
    
    @staticmethod
    def get_all_categories():
        """
        Generate all possible binary categories for the four relationship types.
        
        Returns:
            list: All possible binary category strings.
        """
        categories = []
        # Generate all 16 possibilities (2^4)
        for i in range(16):
            # Convert to 4-bit binary string with leading zeros
            binary = format(i, '04b')
            categories.append(binary)
        return categories

def generate_bfcc_string(binary_flags):
    """
    Generate a human-readable string describing the BFCC relationships.
    
    Args:
        binary_flags (dict): Binary flags for each relationship type
        
    Returns:
        str: A string description of the relationships
    """
    components = []
    if binary_flags["F"] == 1:
        components.append("Feeding")
    if binary_flags["B"] == 1:
        components.append("Bleeding")
    if binary_flags["CF"] == 1:
        components.append("Counterfeeding")
    if binary_flags["CB"] == 1:
        components.append("Counterbleeding")
    
    if not components:
        return "No BFCC relationships"
    
    return ", ".join(components)

def generate_balanced_dataset(target_size, vocab_chars, min_seq_len=2, max_seq_len=5, max_attempts=100000, dedupe=True, num_inputs=5, input_len_range=(2, 6), relaxed_mode: bool=False, patience: int=100000, min_modified_per_program: int=1):
    """
    Generate a dataset with balanced binary categories and cascade lengths.
    
    Args:
        target_size (int): Total number of instances in the final dataset.
        vocab_chars (str): String of characters to use in the vocabulary.
        min_seq_len (int): Minimum sequence length.
        max_seq_len (int): Maximum sequence length.
        max_attempts (int): Maximum number of generation attempts.
        
    Returns:
        tuple: (List of instances with balanced categories, statistics dictionary)
    """
    # Get all possible binary categories
    rel_classifier = RelationshipClassifierRelaxed if relaxed_mode else RelationshipClassifier
    all_categories = rel_classifier.get_all_categories() if args.categories == "all" else list(args.categories)
    num_categories = len(all_categories)  # Should be 16 (for regular mode), 5 for relaxed mode.
    # print("categories:", all_categories)
    # exit()

    # Target count per category
    target_per_category = target_size // num_categories
    
    # Initialize dataset storage
    datasets_by_category = {category: [] for category in all_categories}
    
    # Track needed instances
    needed_by_category = {category: target_per_category for category in all_categories}
    
    # Optional dedupe store
    seen_keys = set()
    
    # Generate instances until we have enough or reach max attempts
    instances_generated = 0
    accepted_instances = 0
    
    print(f"Generating balanced dataset: {target_size} instances across {num_categories} categories")
    print(f"Target per category: ~{target_per_category}")
    
    # Online category-only gating
    steps_pbar = tqdm(total=max_attempts)
    with tqdm(total=target_size) as pbar:
        while sum(needed_by_category.values()) > 0 and instances_generated < patience if patience != -1 else instances_generated < max_attempts:
            seq_len = random.choice(list(range(min_seq_len, max_seq_len + 1)))
            instance = sample_instance(
                vocab_chars=vocab_chars,
                num_inputs=num_inputs,
                input_len_range=input_len_range,
                env_len_range=(1, 3),
                seq_len=seq_len,
                min_cascade_len=min_seq_len,
                min_modified_per_program=min_modified_per_program
            )
            instances_generated += 1
            steps_pbar.update(1)
            if instance is None: continue
            # Classify and annotate
            rel_classifier = RelationshipClassifierRelaxed if relaxed_mode else RelationshipClassifier
            binary_flags, category = rel_classifier.classify_binary(instance)
            instance["bfcc_string"] = generate_bfcc_string(binary_flags)
            instance["bfcc_category"] = category
            # Dedupe
            if dedupe:
                key = (tuple(instance["inputs"]), tuple(instance["outputs"]), tuple(instance["programs"]), instance["cascade_length"])
                if key in seen_keys: continue
                seen_keys.add(key)
            # Accept by category only
            if category in needed_by_category and needed_by_category[category] > 0:
                datasets_by_category[category].append(instance)
                needed_by_category[category] -= 1
                accepted_instances += 1
                pbar.update(1)
            # Logs (no pool-mode suggestions)
            if instances_generated % 100 == 0:
                efficiency = (accepted_instances / instances_generated) * 100 if instances_generated else 0.0
                print(f"\nGenerated: {instances_generated}, Accepted: {accepted_instances}")
                print(f"Efficiency: {efficiency:.2f}%")
                sorted_needs = sorted(needed_by_category.items(), key=lambda x: x[1], reverse=True)
                print("Top categories still needed:")
                for cat, need in sorted_needs[:5]:
                    if need > 0: print(f"  Category {cat}: {need} instances")

        # if you run out of patience
        if patience != -1:
            while instances_generated < max_attempts and sum(len(v) for v in datasets_by_category.values()) < target_size:
                seq_len = random.choice(list(range(min_seq_len, max_seq_len + 1)))
                instance = sample_instance(
                    vocab_chars=vocab_chars,
                    num_inputs=num_inputs,
                    input_len_range=input_len_range,
                    env_len_range=(1, 3),
                    seq_len=seq_len,
                    min_cascade_len=min_seq_len,
                    min_modified_per_program=min_modified_per_program
                )
                instances_generated += 1
                steps_pbar.update(1)
                if instance is None: continue
                # Classify and annotate
                rel_classifier = RelationshipClassifierRelaxed if relaxed_mode else RelationshipClassifier
                binary_flags, category = rel_classifier.classify_binary(instance)
                instance["bfcc_string"] = generate_bfcc_string(binary_flags)
                instance["bfcc_category"] = category
                # Dedupe
                if dedupe:
                    key = (tuple(instance["inputs"]), tuple(instance["outputs"]), tuple(instance["programs"]), instance["cascade_length"])
                    if key in seen_keys: continue
                    seen_keys.add(key)
                datasets_by_category[category].append(instance)
                accepted_instances += 1
                pbar.update(1)
                # Logs (no pool-mode suggestions)
                if instances_generated % 100 == 0:
                    efficiency = (accepted_instances / instances_generated) * 100 if instances_generated else 0.0
                    print(f"\nGenerated: {instances_generated}, Accepted: {accepted_instances}")
                    print(f"Efficiency: {efficiency:.2f}%")
                    sorted_needs = sorted(needed_by_category.items(), key=lambda x: x[1], reverse=True)
                    print("Top categories still needed:")
                    if patience is not None and instances_generated > patience:
                        print("\x1b[31;1mImpatient mode engaged!\x1b[0m")
                    for cat, need in sorted_needs[:5]:
                        if need > 0: print(f"  Category {cat}: {need} instances")
        
    # Combine all instances
    final_dataset = []
    for category_instances in datasets_by_category.values():
        final_dataset.extend(category_instances)
    
    # Shuffle the dataset
    random.shuffle(final_dataset)
    
    # Generate statistics
    stats = {
        "total_generated": instances_generated,
        "total_accepted": accepted_instances,
        "efficiency": (accepted_instances / instances_generated) * 100 if instances_generated > 0 else 0,
        "category_distribution": {cat: len(datasets_by_category[cat]) for cat in all_categories},
        "cascade_distribution": {l: sum(1 for inst in final_dataset if inst.get("cascade_length") == l) for l in range(min_seq_len, max_seq_len + 1)},
        "completed": sum(needed_by_category.values()) == 0
    }
    
    print(f"\nDataset generation complete.")
    print(f"Generated {instances_generated} instances to get {accepted_instances} balanced instances.")
    print(f"Efficiency: {stats['efficiency']:.2f}%")
    print("\nCategory distribution:")
    for cat, count in stats["category_distribution"].items():
        print(f"  {cat}: {count} instances")
    print("\nCascade length distribution:")
    for length, count in stats["cascade_distribution"].items():
        print(f"  Length {length}: {count} instances")
    
    return final_dataset, stats

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a balanced dataset over BFCC binary categories with cascade pruning")
    parser.add_argument("--vocab", type=str, default="abcdefghijkuvwxyz", 
                        help="Characters to use in the vocabulary (default: abcdefghijkuvwxyz)")
    parser.add_argument("--size", type=int, default=1000, 
                        help="Target dataset size (default: 1000)")
    parser.add_argument("--num-inputs", type=int, default=5, 
                        help="Number of input-output examples. (default: 5)")
    parser.add_argument("--min-seq-len", type=int, default=2, 
                        help="Minimum cascade length (default: 2)")
    parser.add_argument("--max-seq-len", type=int, default=5, 
                        help="Maximum cascade length (default: 5)")
    parser.add_argument("--output", type=str, default="data/cascaded_checked_balanced_program_transformations_dataset.jsonl", 
                        help="Output file path (default: data/cascaded_checked_balanced_program_transformations_dataset.jsonl)")
    parser.add_argument("--min-input-len", type=int, default=2, help="max #characters in inputs")
    parser.add_argument("--max-input-len", type=int, default=6, help="max #characters in outputs")
    parser.add_argument("--stats", type=str, default="data/generation_stats.json", 
                        help="Statistics file path (default: data/generation_stats.json)")
    parser.add_argument("--max-attempts", type=int, default=2000000, 
                        help="Maximum generation attempts (default: 2000000)")
    parser.add_argument("--categories", nargs="+", type=str , default="all", 
                        help="Use if only specific categories are to be considered.")
    parser.add_argument("--dedupe", action="store_true", default=True, help="Dedupe on (inputs, outputs, pruned_programs, length)")
    parser.add_argument("--patience", default=-1, type=int, help="drop all category constraints.")
    parser.add_argument("--no-dedupe", action="store_false", dest="dedupe")
    parser.add_argument("--relaxed-mode", action="store_true", help="relaxed constraints on relation types.")
    parser.add_argument("--min-modified-per-program", type=int, default=1, help="lower bound on #strings modified per program (per step)")
    return parser.parse_args()

# Write JSONL file with proper program representation
def write_formatted_jsonl(data, path: str):
    """
    Write data to a JSONL file with programs in the expected format with escaped quotes.
    
    Args:
        data (list): List of instances to write
        path (str): Output file path
    """
    with open(path, "w") as f:
        for rec in tqdm(data):
            # Remove original_programs before writing to file
            output_rec = {k: v for k, v in rec.items() if k != "original_programs"}
            f.write(json.dumps(output_rec) + "\n")
    
    # Display sample entry
    if data:
        print("\nSample entry from output:")
        sample = data[0]
        print(f"  BFCC Category: {sample.get('bfcc_category', 'N/A')}")
        print(f"  BFCC String: {sample.get('bfcc_string', 'N/A')}")
        print(f"  Cascade Length: {sample.get('cascade_length', 'N/A')}")
        print(f"  Programs: {sample.get('programs', [])[:2]}...")

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Vocabulary: {args.vocab}")
    print(f"  Target size: {args.size}")
    print(f"  Cascade length range: {args.min_seq_len}-{args.max_seq_len}")
    print(f"  Number of inputs: {args.num_inputs}")
    print(f"  Output file: {args.output}")
    print(f"  Stats file: {args.stats}")
    print(f"  Max attempts: {args.max_attempts}")
    print(f"  Min modified per program: {args.min_modified_per_program}")
    
    # Generate the balanced dataset
    dataset, stats = generate_balanced_dataset(
        target_size=args.size,
        vocab_chars=args.vocab, num_inputs=args.num_inputs,
        min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len,
        max_attempts=args.max_attempts, dedupe=args.dedupe,
        input_len_range=(args.min_input_len, args.max_input_len),
        relaxed_mode=args.relaxed_mode,
        patience=args.patience,
        min_modified_per_program=args.min_modified_per_program,
    )
    
    # Write the dataset to file with properly formatted programs
    print(f"Writing {len(dataset)} samples to {args.output}")
    write_formatted_jsonl(dataset, args.output)
    
    # Write stats to file
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset written to {args.output}")
    print(f"Statistics written to {args.stats}")
    print("Complete!")
