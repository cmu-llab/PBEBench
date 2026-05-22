"""
Core data structures and program execution utilities.
"""

import os
import re
import sys
import pathlib
from typing import Optional
from dataclasses import dataclass

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
# print(module_path)
sys.path.append(module_path)

from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

# Regex for parsing replace("A", "B") programs
_RE_REPLACE = re.compile(
    r"replace\(\s*([\'\"])\s*(.*?)\s*\1\s*,\s*([\'\"])\s*(.*?)\s*\3\s*\)"
)

ACCEPTABLE_VOCABULARY = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
program_vocab = ProgramVocabulary(ACCEPTABLE_VOCABULARY)

def parse_program(prog: str) -> tuple[str, str]:
    """
    Parse a program string of the form replace('A','B').
    
    Returns:
        (lhs, rhs) tuple of strings
    
    Raises:
        ValueError: if parsing fails
    """
    # prog = re.sub(r"(?<!['\"])(replace\([^)]*\))(?!['\"])", r'"\1"', prog)
    prog = prog.replace("\\","") # check "get_instance_complexities" in src/metrics/functional_correctness.py
    prog = ProgramBFCCNode.from_string(prog, vocabulary=program_vocab)

    return prog.alpha, prog.beta


def apply_single_program(inputs: list[str], prog: str) -> list[str]:
    """Apply a single replace program to a list of inputs."""
    lhs, rhs = parse_program(prog)
    return [s.replace(lhs, rhs) for s in inputs]


def apply_programs(inputs: list[str], programs: list[str]) -> list[str]:
    """
    Apply a sequence of replace programs to inputs.
    
    Programs are applied left-to-right, with each program
    operating on the output of the previous.
    
    Args:
        inputs: List of input strings
        programs: List of program strings like replace("A", "B")
    
    Returns:
        List of output strings after all programs applied
    """
    outputs = inputs.copy()
    for prog in programs:
        outputs = apply_single_program(outputs, prog)
    return outputs


@dataclass
class PBEInstance:
    """
    A Programming-by-Example instance.
    
    Attributes:
        inputs: Input strings
        outputs: Expected output strings
        programs: Ordered list of replace programs
        bfcc_dag: BFCC relation graph [(i, [relations], j), ...]
        cascade_length: Number of programs
        bfcc_category: Binary string encoding which relation types exist
    """
    inputs: list[str]
    outputs: list[str]
    programs: list[str]
    bfcc_dag: list
    cascade_length: int
    bfcc_category: str
    
    @classmethod
    def from_dict(cls, d: dict) -> "PBEInstance":
        """Create instance from dictionary (e.g., loaded from JSON)."""
        return cls(
            inputs=d["inputs"],
            outputs=d["outputs"],
            programs=d["programs"],
            bfcc_dag=d.get("bfcc_dag", []),
            cascade_length=d.get("cascade_length", len(d["programs"])),
            bfcc_category=d.get("bfcc_category", ""),
        )
    
    def verify(self) -> bool:
        """Verify that programs produce expected outputs."""
        computed = apply_programs(self.inputs, self.programs)
        return computed == self.outputs
    
    def get_fb_edges(self) -> list[tuple[int, str, int]]:
        """
        Extract edges with Feeding or Bleeding relations.
        
        Returns:
            List of (source_idx, relation_type, target_idx) tuples
        """
        edges = []
        for edge in self.bfcc_dag:
            i, relations, j = edge[0], edge[1], edge[2]
            for rel in relations:
                if rel in ("F", "B"):
                    edges.append((i, rel, j))
        return edges


