"""
Detector for BFCC relations between string-rewrite programs of the form
replace(a, b), faithful to the paper's formal definitions.

Assumptions and semantics (explicit):
- Replacement semantics match Python str.replace: global, left-to-right,
  non-overlapping occurrences; no boundary markers; no template/context.
- Relations are computed on ordered pairs of programs (i, j), i != j.
- This module emits F/B labels on each ordered pair; counter-relations
  (CF/CB) are derived by consumers from index order (i > j) and are not
  emitted as labels here to preserve the existing schema.

Notes:
- Parsing accepts both single and double quotes inside replace(·, ·).
- The mathematics of feeds()/bleeds() is unchanged and follows the paper.
"""

import json
import re
from itertools import permutations
from collections import Counter
from typing import List, Tuple


def substrings(s: str) -> set[str]:
    """
    Compute all non-empty substrings of a string.

    Parameters
    ----------
    s : str
        a string

    Returns
    -------
    set[str]
        A set of strings (substrings of `s`)
    """
    n = len(s)
    substrs = set()
    
    # Generate all possible substrings
    for start in range(n):
        for end in range(start + 1, n + 1):
            substring = s[start:end]
            substrs.add(substring)
            
    return substrs


def bag_of_substrings_dict(s: str) -> dict[str, int]:
    """
    Compute the bag (multiset) of substrings of a string.

    Parameters
    ----------
    s : str
        The input string

    Returns
    -------
        A Counter where keys are substrings and values are counts
    """
    bag = Counter()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            bag[s[i:j]] += 1
    return bag


def prefixes(s: str) -> set[str]:
    """
    Compute all prefixes of a string

    Parameters
    ----------
    s : str
        a string

    Returns
    -------
    Set
        A set of strings
    """
    n = len(s)
    prefixes = set()
    
    # Generate all prefixes
    for end in range(1, n + 1):
        prefix = s[:end]
        prefixes.add(prefix)
            
    return prefixes


def suffixes(s: str) -> set[str]:
    """
    Compute all suffixes of a string

    Parameters
    ----------
    s : str
        a string

    Returns
    -------
    Set
        A set of strings
    """
    n = len(s)
    suffixes = set()
    
    # Generate all suffixes
    for start in range(n):
        suffix = s[start:]
        suffixes.add(suffix)
            
    return suffixes


def overlap_difference(a1: str, a2: str, b1: str) -> set[str]:
    """
    Compute the set of overlaps between two strings `a2` and `b1` tha cannot be
    explained by substrings in `a1`

    Parameters
    ----------
    a1 : str
        the string representing the status quo
    a2 : str
        the first string compared
    b1 : str
        the second string compared
    
    Returns
    -------
    set[str]
        A set of strings (overlaps between `a2` and `b1` in excess about `a1`)
    """
    ol = (prefixes(a2) & suffixes(b1)) | (prefixes(b1) & suffixes(a2))
    bag = bag_of_substrings_dict(a1)
    return {s for s in ol if bag[s] < 1}


def feeds(a_in: str, a_out: str, b_in: str, b_out: str) -> bool:
    """
    Computes whether the first rewrite rule potentially feeds the second rewrite
    rule.

    Parameters
    ----------
    a_in : str
        The LHS of the first rule
    a_out : str
        The RHS of the first rule
    b_in : str
        The LHS of the second rule
    b_out str
        The RHS of the second rule

    Returns
    -------
    bool
        True if the first rule potentially feeds the second
    """
    if a_in == a_out:
        return False
    if a_out == '' and len(b_in) > 1 \
        or (a_out in substrings(b_in) and a_in.count(a_out) < 1) \
        or (b_in in substrings(a_out) and a_out.count(b_in) > a_in.count(b_in)) \
        or overlap_difference(a_in, a_out, b_in):
        return True
    else:
        return False


def bleeds(a_in: str, a_out: str, b_in: str, b_out: str) -> bool:
    """
    Computes whether the first rewrite rule potentially bleeds the second rewrite
    rule.

    Parameters
    ----------
    a_in : str
        The LHS of the first rule
    a_out : str
        The RHS of the first rule
    b_in : str
        The LHS of the second rule
    b_out str
        The RHS of the second rule

    Returns
    -------
    bool
        True if the first rule potentially bleeds the second
    """
    if a_in == a_out:
        return False
    if a_in == '' and len(b_in) > 1 \
        or (a_in in substrings(b_in) and a_out.count(a_in) < 1) \
        or (b_in in substrings(a_in) and a_out.count(b_in) < a_in.count(b_in)) \
        or overlap_difference(a_out, a_in, b_in):
        return True
    else:
        return False


_RE_REPLACE = re.compile(
    r"replace\(\s*([\'\"])\s*(.*?)\s*\1\s*,\s*([\'\"])\s*(.*?)\s*\3\s*\)")


def _parse_program(prog: str) -> Tuple[str, str]:
    """
    Parse a program string of the form replace('a','b') or replace("a","b").
    Returns (lhs, rhs). Raises ValueError if parsing fails.
    """
    m = _RE_REPLACE.search(prog)
    if not m:
        raise ValueError(f"Invalid program format (expected replace('a','b')): {prog}")
    lhs = m.group(2)
    rhs = m.group(4)
    return lhs, rhs


def process_programs(programs: List[str], include_neutral: bool = True) -> List[Tuple[int, List[str], int]]:
    """
    Compute relations (F/B) between each ordered pair of programs and return
    edges in the existing schema:
      [(src_idx, [labels], dst_idx)], labels ⊆ {'F','B'} and optionally 'N'.

    - includes both directions (i<j and i>j); CF/CB are derived by consumers
      based on index order and are not emitted here (schema preservation).
    - if include_neutral is True (default), emit 'N' when neither F nor B holds.
      If False, omit such edges.
    """
    relations: List[Tuple[int, List[str], int]] = []

    # Pre-parse programs once for clarity and robustness
    parsed = []
    for p in programs:
        a_in, a_out = _parse_program(p)
        parsed.append((a_in, a_out))

    for (i, (a_in, a_out)), (j, (b_in, b_out)) in permutations(enumerate(parsed), 2):
        labels: List[str] = []

        if feeds(a_in, a_out, b_in, b_out):
            labels.append('F')
        if bleeds(a_in, a_out, b_in, b_out):
            labels.append('B')

        if include_neutral:
            if not labels:
                labels.append('N')
            relations.append((i, labels, j))
        else:
            if labels:
                relations.append((i, labels, j))

    return relations


def process_file_and_save(file_path: str, output_path: str) -> None:
    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            programs = entry.get("programs", [])
            entry["bfcc_dag"] = process_programs(programs)
            outfile.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    file_path = "/Users/manavnitinkapadnis/Documents/GitHub/pbe-reasoning/data/example_auto_gen_data_manav.jsonl"
    output_path = "/Users/manavnitinkapadnis/Documents/GitHub/pbe-reasoning/data/example_auto_gen_data_manav_darsh_detected.jsonl"
    process_file_and_save(file_path, output_path)
