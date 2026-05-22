"""
High-quality prompts for permutation evaluation.

The prompt design emphasizes:
1. Clear task framing with concrete examples
2. Explanation of why order matters (feeding/bleeding)
3. Structured output format for reliable parsing
"""

import json
from typing import Optional

# Main evaluation prompt - emphasizes reasoning about program interactions
UNSCRAMBLE_PROMPT = '''You are solving a program ordering puzzle. Given input-output string pairs and a scrambled list of string replacement programs, determine the correct execution order.

## Background

Each program performs a Python string replacement: `replace("A", "B")` replaces all occurrences of "A" with "B".

**Why order matters:** Programs can interact in two key ways:
- **Feeding**: One program creates substrings that another program can match.  
  Example: `replace("a", "bc")` followed by `replace("bc", "x")` — the first creates "bc" for the second.
- **Bleeding**: One program removes substrings that another would have matched.  
  Example: `replace("ab", "x")` followed by `replace("a", "y")` — the first removes "a"s that the second needs.

## Your Task

**Inputs:** {inputs}

**Outputs:** {outputs}

**Scrambled Programs** (indices 0 to {n_minus_1}):
{programs_formatted}

Find the ordering `[i₀, i₁, ..., i_{n_minus_1}]` such that applying programs in that order transforms each input to its corresponding output.

## Approach

1. Trace through what each program does
2. Identify potential feeding/bleeding interactions
3. Reason about which programs must come before others
4. Verify your ordering produces the expected outputs

## Output Format

Provide your final answer as a JSON array of indices:

```json
[i0, i1, i2, ...]
```

Your ordering must be a permutation of [0, 1, ..., {n_minus_1}].'''


def format_unscramble_prompt(
    inputs: list[str],
    outputs: list[str],
    permuted_programs: list[str],
    max_examples: int = 10
) -> str:
    """
    Format the unscrambling prompt for a single instance.
    
    Args:
        inputs: Input strings
        outputs: Expected output strings
        permuted_programs: Programs in scrambled order
        max_examples: Maximum input/output pairs to show
    
    Returns:
        Formatted prompt string
    """
    n = len(permuted_programs)
    
    # Truncate examples if too many
    display_inputs = inputs[:max_examples]
    display_outputs = outputs[:max_examples]
    
    if len(inputs) > max_examples:
        inputs_str = json.dumps(display_inputs)[:-1] + ', ...]'
        outputs_str = json.dumps(display_outputs)[:-1] + ', ...]'
    else:
        inputs_str = json.dumps(display_inputs)
        outputs_str = json.dumps(display_outputs)
    
    programs_formatted = "\n".join(
        f"  [{i}]: `{prog}`"
        for i, prog in enumerate(permuted_programs)
    )
    
    return UNSCRAMBLE_PROMPT.format(
        inputs=inputs_str,
        outputs=outputs_str,
        n_minus_1=n - 1,
        programs_formatted=programs_formatted
    )


# Simpler prompt variant for comparison
UNSCRAMBLE_PROMPT_SIMPLE = '''Given these inputs: {inputs}
And these outputs: {outputs}

Put these programs in the correct order to transform inputs to outputs:
{programs_formatted}

Answer with a JSON array of indices, e.g., [2, 0, 1] means program 2 first, then 0, then 1.

```json
[your answer here]
```'''


def format_simple_prompt(
    inputs: list[str],
    outputs: list[str],
    permuted_programs: list[str],
    max_examples: int = 10
) -> str:
    """Format the simpler prompt variant."""
    display_inputs = inputs[:max_examples]
    display_outputs = outputs[:max_examples]
    
    programs_formatted = "\n".join(
        f"[{i}]: {prog}"
        for i, prog in enumerate(permuted_programs)
    )
    
    return UNSCRAMBLE_PROMPT_SIMPLE.format(
        inputs=json.dumps(display_inputs),
        outputs=json.dumps(display_outputs),
        programs_formatted=programs_formatted
    )


