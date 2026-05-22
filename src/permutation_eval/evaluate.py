"""
Evaluation framework for measuring LLM unscrambling ability.
"""

import os
import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union
from tqdm import tqdm

module_path = str(Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.permutation_eval.core import apply_programs
from src.permutation_eval.permutation import PermutedInstance
from src.permutation_eval.prompts import format_unscramble_prompt


@dataclass
class EvalResult:
    """Result for a single evaluation instance."""
    id: str
    correct: bool
    predicted_order: Optional[list[int]]
    ground_truth_order: list[int]
    model_response: str
    cascade_length: int
    bfcc_category: str
    execution_verified: bool = False
    num_valid_solutions: Optional[int] = None
    is_unique_solution: bool = False
    is_null_response: bool=False


@dataclass
class EvalStats:
    """Aggregate evaluation statistics."""
    total: int = 0
    correct: int = 0
    null_responses: int = 0
    invalid_format: int = 0
    by_cascade_length: dict = field(default_factory=dict)
    by_bfcc_category: dict = field(default_factory=dict)
    
    # New: Track stats specifically for unique-solution instances
    unique_subset: dict = field(default_factory=lambda: {"total": 0, "correct": 0})
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "null_responses": self.null_responses, 
            "invalid_format": self.invalid_format,
            "unique_subset": self.unique_subset,
            "by_cascade_length": self.by_cascade_length,
            "by_bfcc_category": self.by_bfcc_category,
        }


def parse_ordering_from_response(response: str, n_programs: int) -> Optional[list[int]]:
    """Extract ordering from model response."""
    patterns = [
        r'```json\s*\n?\s*(\[[\d,\s]+\])\s*\n?```',
        r'```\s*\n?\s*(\[[\d,\s]+\])\s*\n?```',
        r'\*\*?(?:Answer|Final|Order(?:ing)?)[:\s]*\*?\*?\s*(\[[\d,\s]+\])',
        r'(\[[\d,\s]+\])\s*$',
        r'(\[[\d,\s]+\])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                ordering = json.loads(match.group(1))
                if (isinstance(ordering, list) and 
                    len(ordering) == n_programs and
                    set(ordering) == set(range(n_programs))):
                    return ordering
            except json.JSONDecodeError:
                continue
    return None

def strip_cot_from_response(text: str) -> str:
    # 1. Remove <think>...</think> sections if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    return text

    # # 2. Extract all fenced code blocks (``` ... ```)
    # code_blocks = re.findall(r"```[\s\S]*?```", text)

    # if code_blocks:
    #     last_block = code_blocks[-1]

    #     # Find where the last block starts
    #     start_index = text.rfind(last_block)

    #     # Keep from the last code block to the end (usually includes short explanation)
    #     cleaned = text[start_index:].strip()
    #     return cleaned

    # # 3. If no fenced code blocks exist, fallback:
    # #    Remove extra blank lines and return the last few lines.
    # lines = [line.strip() for line in text.splitlines() if line.strip()]

    # # Heuristic: keep the last 5 non-empty lines (final answer region)
    # return "\n".join(lines[-5:])

def verify_ordering_by_execution(
    inputs: list[str],
    outputs: list[str],
    permuted_programs: list[str],
    ordering: list[int]
) -> bool:
    """Verify if ordering produces correct outputs."""
    ordered_programs = [permuted_programs[i] for i in ordering]
    computed = apply_programs(inputs, ordered_programs)
    return computed == outputs


def query_openai(
    prompt: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0
) -> str:
    """Query OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def evaluate_instance(
    instance: PermutedInstance,
    model: str,
    query_fn=None
) -> EvalResult:
    """Evaluate a single instance."""
    if query_fn is None:
        query_fn = query_openai
    
    prompt = format_unscramble_prompt(
        instance.inputs,
        instance.outputs,
        instance.permuted_programs
    )
    
    response = query_fn(prompt, model)
    
    n_programs = len(instance.permuted_programs)
    predicted_order = parse_ordering_from_response(response, n_programs)
    
    correct = False
    execution_verified = False
    
    if predicted_order is not None:
        correct = verify_ordering_by_execution(
            instance.inputs,
            instance.outputs,
            instance.permuted_programs,
            predicted_order
        )
        execution_verified = True
    
    return EvalResult(
        id=instance.id,
        correct=correct,
        predicted_order=predicted_order,
        ground_truth_order=instance.ground_truth_order,
        model_response=response,
        cascade_length=instance.cascade_length,
        bfcc_category=instance.bfcc_category,
        execution_verified=execution_verified,
        num_valid_solutions=instance.num_valid_solutions,
        is_unique_solution=instance.num_valid_solutions == 1
    )

def evaluate_instance_without_inference(
        instance: PermutedInstance, 
        response: Union[str, None],
        strip_cot: bool=False,
    ) -> EvalResult:
    """Evaluate a single instance without querying a model."""
    n_programs = len(instance.permuted_programs)
    correct = False
    execution_verified = False
    predicted_order = None
    is_null_response = False

    if response is None: 
        is_null_response = True
        # correct is False and execution verified is False.
    else:
        if strip_cot: 
            if "</think>" not in response:
                is_null_response = True
            else:
                response = strip_cot_from_response(response)
                predicted_order = parse_ordering_from_response(response, n_programs)
                if predicted_order is not None:
                    correct = verify_ordering_by_execution(
                        instance.inputs,
                        instance.outputs,
                        instance.permuted_programs,
                        predicted_order
                    )
                    execution_verified = True
        else:
            predicted_order = parse_ordering_from_response(response, n_programs)
            if predicted_order is not None:
                correct = verify_ordering_by_execution(
                    instance.inputs,
                    instance.outputs,
                    instance.permuted_programs,
                    predicted_order
                )
                execution_verified = True
    
    return EvalResult(
        id=instance.id,
        correct=correct,
        predicted_order=predicted_order,
        ground_truth_order=instance.ground_truth_order,
        model_response=response,
        cascade_length=instance.cascade_length,
        bfcc_category=instance.bfcc_category,
        execution_verified=execution_verified,
        num_valid_solutions=instance.num_valid_solutions,
        is_unique_solution=instance.num_valid_solutions == 1,
        is_null_response=is_null_response
    )


def run_evaluation(
    data_path: Path,
    output_path: Path,
    model: str = "gpt-4o",
    max_instances: Optional[int] = None,
    query_fn=None,
    verbose: bool = True
) -> EvalStats:
    """Run full evaluation on permuted dataset."""
    with open(data_path) as f:
        data = [PermutedInstance.from_dict(json.loads(line)) for line in f]
    
    if max_instances:
        data = data[:max_instances]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    covered_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                covered_ids.add(rec["id"])
    
    stats = EvalStats()
    results = []
    
    pbar = tqdm(data, desc=f"Evaluating {model}", disable=not verbose)
    for instance in pbar:
        if instance.id in covered_ids:
            continue
        
        result = evaluate_instance(instance, model, query_fn)
        
        result_dict = {
            "id": result.id,
            "correct": result.correct,
            "predicted_order": result.predicted_order,
            "ground_truth_order": result.ground_truth_order,
            "model_response": result.model_response,
            "cascade_length": result.cascade_length,
            "bfcc_category": result.bfcc_category,
            "execution_verified": result.execution_verified,
            "num_valid_solutions": result.num_valid_solutions,
            "is_unique_solution": result.is_unique_solution,
        }
        results.append(result_dict)
        
        with open(output_path, "a") as f:
            f.write(json.dumps(result_dict) + "\n")
    
    # Compute full stats
    stats = EvalStats()
    for r in results:
        stats.total += 1
        if r["correct"]:
            stats.correct += 1
        if r["predicted_order"] is None:
            stats.invalid_format += 1
        
        # Unique subset stats
        if r.get("is_unique_solution"):
            stats.unique_subset["total"] += 1
            if r["correct"]:
                stats.unique_subset["correct"] += 1
        
        # By cascade length
        cl = r["cascade_length"]
        if cl not in stats.by_cascade_length:
            stats.by_cascade_length[cl] = {"total": 0, "correct": 0}
        stats.by_cascade_length[cl]["total"] += 1
        if r["correct"]:
            stats.by_cascade_length[cl]["correct"] += 1
            
        # By category
        cat = r["bfcc_category"]
        if cat not in stats.by_bfcc_category:
            stats.by_bfcc_category[cat] = {"total": 0, "correct": 0}
        stats.by_bfcc_category[cat]["total"] += 1
        if r["correct"]:
            stats.by_bfcc_category[cat]["correct"] += 1
    
    if verbose:
        print(f"\nRESULTS: {model}")
        print(f"Overall Accuracy: {stats.accuracy:.1%}")
        u_total = stats.unique_subset["total"]
        u_acc = stats.unique_subset["correct"] / u_total if u_total > 0 else 0
        print(f"Unique-Solution Accuracy: {u_acc:.1%} ({stats.unique_subset['correct']}/{u_total})")
        
    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump({"model": model, **stats.to_dict()}, f, indent=2)
    
    return stats

def run_evaluation_on_saved_outputs(
    output_path: str,
    strip_cot: bool = False,
    verbose: bool = True
) -> EvalStats:
    """Run full evaluation on permuted dataset."""
    with open(output_path) as f:
        data = [json.loads(line) for line in f]
    
    stats = EvalStats()
    results = []
    
    # pbar = tqdm(data, desc=f"Evaluating {model}", disable=not verbose)
    pbar = tqdm(data, disable=not verbose)
    for rec in pbar:
        instance = PermutedInstance.from_dict(rec['input'])
        response = rec['outputs'][0]
        # print(response)
        result = evaluate_instance_without_inference(instance, response, strip_cot=strip_cot)
        result_dict = {
            "id": result.id,
            "correct": result.correct,
            "predicted_order": result.predicted_order,
            "ground_truth_order": result.ground_truth_order,
            "model_response": result.model_response,
            "cascade_length": result.cascade_length,
            "bfcc_category": result.bfcc_category,
            "execution_verified": result.execution_verified,
            "num_valid_solutions": result.num_valid_solutions,
            "is_unique_solution": result.is_unique_solution,
            "is_null_response": result.is_null_response,
        }
        results.append(result_dict)
    
    # Compute full stats
    stats = EvalStats()
    for r in results:
        stats.total += 1
        if r["correct"]:
            stats.correct += 1
        if r["predicted_order"] is None:
            stats.invalid_format += 1
        if r["is_null_response"]:
            stats.null_responses += 1
        
        # Unique subset stats
        if r.get("is_unique_solution"):
            stats.unique_subset["total"] += 1
            if r["correct"]:
                stats.unique_subset["correct"] += 1
        
        # By cascade length
        cl = r["cascade_length"]
        if cl not in stats.by_cascade_length:
            stats.by_cascade_length[cl] = {"total": 0, "correct": 0}
        stats.by_cascade_length[cl]["total"] += 1
        if r["correct"]:
            stats.by_cascade_length[cl]["correct"] += 1
            
        # By category
        cat = r["bfcc_category"]
        if cat not in stats.by_bfcc_category:
            stats.by_bfcc_category[cat] = {"total": 0, "correct": 0}
        stats.by_bfcc_category[cat]["total"] += 1
        if r["correct"]:
            stats.by_bfcc_category[cat]["correct"] += 1
    
    if verbose:
        # print(f"\nRESULTS: {model}")
        print(f"Overall Accuracy: {stats.accuracy:.1%}")
        u_total = stats.unique_subset["total"]
        u_acc = stats.unique_subset["correct"] / u_total if u_total > 0 else 0
        print(f"Unique-Solution Accuracy: {u_acc:.1%} ({stats.unique_subset['correct']}/{u_total})")
        
    print(stats.to_dict())
    # Save stats
    # stats_path = output_path.with_suffix(".stats.json")
    # with open(stats_path, "w") as f:
    #     json.dump({"model": model, **stats.to_dict()}, f, indent=2)
    
    return stats

# main
if __name__ == "__main__":
    try:
        strip_cot = bool(sys.argv[2])
    except IndexError:
        strip_cot = False
    run_evaluation_on_saved_outputs(
        output_path=sys.argv[1], 
        strip_cot=strip_cot
    )