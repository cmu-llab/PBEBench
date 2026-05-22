import os
import sys
import json
import pathlib
from typing import List
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.tools.tool import Tool

class RelationClassifierTool(Tool):
    def __init__(self):
        pass

    def __call__(self, pro):
        pass


def test_relation_classifier_tool():
    pass

# main
if __name__ == "__main__":
    test_relation_classifier_tool()