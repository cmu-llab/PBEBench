import os
import sys

prompt_v1 = """Follow the instructions below to solve the code completion task:

We will provide the input corpus and corresponding output corpus. Each element in the corpus is a string, and the output is transformed from the corresponding input using an ordered sequence of "replace" programs. You need to find the correctly constructed and ordered sequence of "replace" programs to transform the entire input corpus into the output corpus. Note that the programs can interact with each other in a way that reduces or increases the number of times they are applied on a given input based on where they are ordered in the sequence. This makes it very important to apply them in the correct order. 

The programs should be written using only the Python replace function. For example, for a program that replaces all occurrences of "ab" with "bc" it should be written as: ```replace('ab', 'bc')```

Here is an example of the full task:
### Inputs 
["abc", "ebc", "aba"]

### Outputs
["edc", "edc", "aba"]

### Program Sequence
```python
["replace('bc','dc')", "replace('ad','ed')"]
```

While generating the program sequence, you need to abide by the following restrictions:
1. Each program in the sequence should have the form "replace(A, B)", where A and B are both strings.
2. Both argument strings A and B in "replace(A, B)" should have <= {program_length} characters. A should have at least 1 character but B can be null (or "").
3. The maximum number of programs in a sequence is {program_num}
4. You should only consider the Python ‘replace’ function for specifying programs (each program is a Python replace function). You can not use any other Python modules or functions.  
5. Strictly follow the markdown style convention while presenting your final program sequence, and make sure to enclose it in the ```python``` markdown style code block. 

Now, please generate the sequence of programs corresponding to the following input corpus and output corpus:

### Inputs 
{inputs_list}

### Outputs
{outputs_list}

### Program Sequence
"""

pbebench_task_description_prompt = """We will provide the input corpus and corresponding output corpus. Each element in the corpus is a string, and the output is transformed from the corresponding input using an ordered sequence of "replace" programs. You need to find the correctly constructed and ordered sequence of "replace" programs to transform the entire input corpus into the output corpus. Note that the programs can interact with each other in a way that reduces or increases the number of times they are applied on a given input based on where they are ordered in the sequence. This makes it very important to apply them in the correct order. 

The programs should be written using only the Python replace function. For example, for a program that replaces all occurrences of "ab" with "bc" it should be written as: ```replace('ab', 'bc')```

Here is an example of the full task:
### Inputs 
["abc", "ebc", "aba"]

### Outputs
["edc", "edc", "aba"]

### Program Sequence
```python
["replace('bc','dc')", "replace('ad','ed')"]
```

While generating the program sequence, you need to abide by the following restrictions:
1. Each program in the sequence should have the form "replace(A, B)", where A and B are both strings.
2. Both argument strings A and B in "replace(A, B)" should have <= {program_length} characters. A should have at least 1 character but B can be null (or "").
3. The strings A and B in the program should only use characters from the following alphabet: {alphabet}. Using a symbol outside this set will make the program invalid.
4. The maximum number of programs in a sequence is {program_num}
5. You should only consider the Python ‘replace’ function for specifying programs (each program is a Python replace function). You can not use any other Python modules or functions.  
6. Strictly follow the markdown style convention while presenting your final program sequence, and make sure to enclose it in the ```python``` markdown style code block. """

pbebench_task_prompt = """Now, please generate the sequence of programs corresponding to the following input corpus and output corpus:

### Inputs 
{inputs_list}

### Outputs
{outputs_list}

### Program Sequence"""

unified_dsl_reasoning_prompt = """You need to solve a reasoning task by synthesizing DSL (domain-specific language) programs. The instructions are provided in two parts: a task description that explains the task and DSL structure, and a task prompt that specifies the particular reasoning problem.

# Task Description

{task_description_prompt}

# Task

{task_prompt}
"""

prompt_step = """Follow the instructions below to solve the code completion task:

We will provide the input corpus and corresponding output corpus. Each element in the corpus is a string, and the output is transformed from the corresponding input using an ordered sequence of "replace" programs. You need to find the correctly constructed and ordered sequence of "replace" programs to transform the entire input corpus into the output corpus. Note that the programs can interact with each other in a way that reduces or increases the number of times they are applied on a given input based on where they are ordered in the sequence. This makes it very important to apply them in the correct order. 

The programs should be written using only the Python replace function. For example, for a program that replaces all occurrences of "ab" with "bc" it should be written as: ```replace('ab', 'bc')```

Here is an example of the full task:
### Inputs 
["abc", "ebc", "aba"]

### Outputs
["edc", "edc", "aba"]

### Program Sequence
```python
["replace('bc','dc')", "replace('ad','ed')"]
```

While generating the program sequence, you need to abide by the following restrictions:
1. Each program in the sequence should have the form "replace(A, B)", where A and B are both strings.
2. Both argument strings A and B in "replace(A, B)" should have <= {program_length} characters. A should have at least 1 character but B can be null (or "").
3. The maximum number of programs required for a sequence is {program_num} but, you should try to get as close as possible with {step_size} programs. If you produce more programs, then only the first {step_size} programs will be considered.
4. You should only consider the Python ‘replace’ function for specifying programs (each program is a Python replace function). You can not use any other Python modules or functions.  
5. Strictly follow the markdown style convention while presenting your final program sequence, and make sure to enclose it in the ```python``` markdown style code block. 

Now, please generate the sequence of programs corresponding to the following input corpus and output corpus:

### Inputs 
{inputs_list}

### Outputs
{outputs_list}

### Program Sequence
"""

def wrap_task_prompt(task_prompt: str):
    return "Question: "+task_prompt.strip("\n")+"\nAnswer:"

class ReasoningGymPromptSplitter:
    """General dataset splitting class template for parsing reasoning gym prompts into task description prompt and task prompt. The task description prompt contains information about the DSL and the task structure, while the task prompt contains the specific reasoning problem to be solved. The split prompts can then be used in a unified DSL reasoning prompt format for training/evaluation."""
    split_phrase = None
    include_phrase_in = None
    def __init__(self):
        self.task_description_prompt = None
        assert self.include_phrase_in in [None, "task_prompt", "task_description"], "include_phrase_in should be either None, 'task_prompt' or 'task_description'"
    
    def set_task_description_prompt(self, prompt: str):
        """Splits the prompt into task description prompt and task prompt."""
        task_description_prompt, task_prompt = prompt.split(self.split_phrase, 1)
        if self.include_phrase_in == "task_description":
            self.task_description_prompt = task_description_prompt + self.split_phrase
            task_prompt = task_prompt
        elif self.include_phrase_in == "task_prompt":
            self.task_description_prompt = task_description_prompt
            task_prompt = self.split_phrase + task_prompt
        elif self.include_phrase_in is None:
            self.task_description_prompt = task_description_prompt
            task_prompt = task_prompt

        return wrap_task_prompt(task_prompt)
    
    def __call__(self, prompt: str):
        """Isolates the task prompt from the whole prompt."""
        if self.task_description_prompt is None:
            return self.set_task_description_prompt(prompt)
        if self.include_phrase_in == "task_description":
            return wrap_task_prompt(prompt.removeprefix(self.task_description_prompt))
        elif self.include_phrase_in == "task_prompt":
            return wrap_task_prompt(prompt.removeprefix(self.task_description_prompt))
        elif self.task_description_prompt is None: # the split phrase is not included anywhere.
            return wrap_task_prompt(prompt.removeprefix(self.task_description_prompt + self.split_phrase))

class ABPromptSpiltter(ReasoningGymPromptSplitter):
    split_phrase = "Now, consider the following program:"
    include_phrase_in = "task_prompt"

class AcrePromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "Do not use quotation marks in your answer."
    include_phrase_in = "task_description"

class AdvancedGeometryPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "For all geometry problems:"
    include_phrase_in = "task_description"
        
    def set_task_description_prompt(self, prompt: str):
        task_prompt, task_description_prompt = prompt.split(self.split_phrase, 1)
        self.task_description_prompt = self.split_phrase + task_description_prompt

        return wrap_task_prompt(task_prompt)

class AiwPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = None
    include_phrase_in = None

    def set_task_description_prompt(self, prompt: str):
        self.task_description_prompt = "You will be a given an arithemtic word problem with a simple numerical answer. Your task is to solve the problem and provide the final answer as a number. You should not include any explanation or reasoning steps in your answer, just the final numerical answer."
        task_prompt = prompt
        
        return wrap_task_prompt(task_prompt)

class Arc1DPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = None
    include_phrase_in = None

    def set_task_description_prompt(self, prompt: str):
        self.task_description_prompt = "You will be given a few examples showing a transformation over 1D grids. The examples will be paired input and output versions of the 1D grid and you need to infer the transformation and apply it on a novel 1D grid. You must only predict the output 1D grid after applying the transformation and nothing else as your final answer."
        task_prompt = prompt
        
        return wrap_task_prompt(prompt)
    
class ArcAGIPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = None
    include_phrase_in = None

    def set_task_description_prompt(self, prompt: str):
        self.task_description_prompt = "You will be given a few examples showing a transformation over 2D grids. The examples will be paired input and output versions of the 2D grid and you need to infer the transformation and apply it on a novel 2D grid. You must only predict the output 2D grid after applying the transformation and nothing else as your final answer."
        task_prompt = prompt
        
        return wrap_task_prompt(prompt)
    
class BaseConversionPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "If the target base is > 10, use lowercase letters a-z for digits above 9."
    include_phrase_in = "task_description"

class BasicArithmeticPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = None
    include_phrase_in = None

    def set_task_description_prompt(self, prompt: str):
        self.task_description_prompt = "You will be given an arithmetic expression involving addition, subtraction, multiplication, and division. Evaluate the expression using the standard order of operations (PEMDAS). Return only the final numerical result. Do not include any explanation, reasoning, or additional text."
        task_prompt = prompt
        
        return wrap_task_prompt(task_prompt)
    
class BinaryAlternationPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "Now, determine the minimum number of swaps to make the following binary string alternating:"
    include_phrase_in = "task_prompt"

class BinaryMatrixPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "Find the distance to the nearest 0 for each cell in the matrix below:"
    include_phrase_in = "task_prompt"

class BitwiseArithmeticPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "Reply only with the final hexidecimal value."
    include_phrase_in = "task_description"

class BoxnetPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = "Include an agent in the action plan only if it has a task to perform next."
    include_phrase_in = "task_description"

class CaesarCipherPromptSplitter(ReasoningGymPromptSplitter):
    split_phrase = None
    include_phrase_in = None

    def set_task_description_prompt(self, prompt: str):
        self.task_description_prompt = "The input is a message encrypted with a Caesar cipher using an unknown shift between 1 and 25. Determine the correct shift and decrypt the text into readable English. Preserve spacing and punctuation. Output only the decrypted plaintext, with no explanation."
        task_prompt = prompt
        
        return wrap_task_prompt(task_prompt)

REASONING_GYM_PROMPT_SPLITTERS = {
    "ab": ABPromptSpiltter(),
    "acre": AcrePromptSplitter(),
    "advanced_geometry": AdvancedGeometryPromptSplitter(),
    "aiw": AiwPromptSplitter(),
    "arc_1d": Arc1DPromptSplitter(),
    "arc_agi": ArcAGIPromptSplitter(),
    "base_conversion": BaseConversionPromptSplitter(),
    "basic_arithmetic": BasicArithmeticPromptSplitter(),
    "binary_alternation": BinaryAlternationPromptSplitter(),
    "binary_matrix": BinaryMatrixPromptSplitter(),
    "bitwise_arithmetic": BitwiseArithmeticPromptSplitter(), 
    "boxnet": BoxnetPromptSplitter(),
    "caesar_cipher": CaesarCipherPromptSplitter(),
}

def test_prompt_v1():
    program_num = 5
    program_length = 3
    inputs_list = ["vfxkh", "akeby", "xcd", "gda", "hx"]
    outputs_list = ["vfxkh", "akebh", "xcd", "gda", "hx"]
    prompt = prompt_v1.format(inputs_list=inputs_list, outputs_list=outputs_list, program_num=program_num, program_length=program_length)
    print(prompt)
    
def test_reasoning_gym_prompt_splitter(dataset_name):
    import reasoning_gym, copy
    
    data = list(reasoning_gym.create_dataset(dataset_name, size=10, seed=42))
    prompt_splitter = REASONING_GYM_PROMPT_SPLITTERS.get(dataset_name)
    if prompt_splitter is None:
        raise NotImplementedError(f"No prompt splitter implemented for dataset {dataset_name}")
    task_prompt = prompt_splitter(data[0]['question'])
    task_desc_prompt = copy.deepcopy(prompt_splitter.task_description_prompt)
    print(unified_dsl_reasoning_prompt.format(task_description_prompt=task_desc_prompt, task_prompt=task_prompt))
    for rec in data[1:]:
        task_prompt = prompt_splitter(rec['question'])
        assert task_desc_prompt == prompt_splitter.task_description_prompt, "Task description prompt should be the same across all examples in the dataset"

# main
if __name__ == "__main__":
    # test_reasoning_gym_prompt_splitter("ab")
    # print("-----------------------------")
    # test_reasoning_gym_prompt_splitter("acre")
    # print("-----------------------------")
    # test_reasoning_gym_prompt_splitter("advanced_geometry")
    # print("-----------------------------")
    # test_reasoning_gym_prompt_splitter("aiw")
    # print("-----------------------------")
    # test_reasoning_gym_prompt_splitter("arc_1d")
    # print("-----------------------------")
    # test_reasoning_gym_prompt_splitter("arc_agi")
    # print("-----------------------------")
    test_reasoning_gym_prompt_splitter("base_conversion")
    print("-----------------------------")
    test_reasoning_gym_prompt_splitter("basic_arithmetic") 
    print("-----------------------------")
    test_reasoning_gym_prompt_splitter("binary_alternation")
    print("-----------------------------")
    test_reasoning_gym_prompt_splitter("binary_matrix")
    print("-----------------------------")
    test_reasoning_gym_prompt_splitter("boxnet")
    print("-----------------------------")
    test_reasoning_gym_prompt_splitter("caesar_cipher")

    # test_prompt_v1()