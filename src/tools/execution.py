import os
import re
import sys
import json
import pathlib
from typing import List
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.tools.tool import Tool
from src.data_generation.primitives import ProgramBFCCEdge, ProgramBFCCInteractionDAG, ProgramVocabulary, ProgramBFCCNode

def extract_last_python_block(markdown_text: str) -> str:
    """
    Extracts first Python code block from a markdown string.

    Args:
        markdown_text (str): The markdown text containing code blocks.

    Returns:
        List[str]: A list of Python code blocks.
    """
    code_block_pattern = r"```python\s(.*?)```"
    matches = re.findall(code_block_pattern, markdown_text, re.DOTALL)
    try: return [block.strip() for block in matches][-1]
    except IndexError: return "[]"

def extract_program_str_from_last_python_block(model_response: str) -> str:
    try: pred = extract_last_python_block(model_response) 
    except TypeError as e:
        if model_response is None: pass
        else: print(e)
        pred = "replace('x','x')"

    # wrap any replace(...) that isn’t already quoted in single quotes
    # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
    pred = re.sub(
        r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
        r'"\1"',
        pred
    )

    return pred

class ExecutionTool(Tool):
    def __init__(self, vocab_chars: str):
        self.vocab_chars = vocab_chars
        self.vocabulary = ProgramVocabulary(vocab_chars)

    def response_parser(self, response: str):
        """Expect a Python code block enclosed in triple backticks with python in the first line (```python```)"""
        program_string = extract_program_str_from_last_python_block(response)
        cascade_code = ProgramBFCCNode.from_string(program_string, vocabulary=self.vocabulary)

        return cascade_code
        

    def __call__(self, response: str, input_strings: list[str]): 
        cascade_code = self.response_parser(response)
        output_strings = cascade_code(input_strings)

        return output_strings

def test_execution_tool():
    mock_response = """Some random reasoning
    
    Next Action Program
    ```python
    "replace('LX','L')"
    ```"""

    execution_tool = ExecutionTool(vocab_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print(execution_tool(mock_response, ["akLX", "aLXk", "akLXLX", "aklx", "akLx"]))

# main
if __name__ == "__main__":
    test_execution_tool()