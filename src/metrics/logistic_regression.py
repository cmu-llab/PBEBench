import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from difflib import SequenceMatcher
import re
import os

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_input_output_features(inputs, outputs):
    features = []
    for inp, out in zip(inputs, outputs):
        length_diff = len(out) - len(inp)
        similarity = string_similarity(inp, out)
        exact_match = int(inp == out)
        char_changes = sum(c1 != c2 for c1, c2 in zip(inp, out)) + abs(len(inp) - len(out))
        features.extend([length_diff, similarity, exact_match, char_changes])
    return features  # 5 inputs/outputs -> 20 features

def extract_bfcc_features(sample_input):
    features = []

    # cascade_length
    features.append(sample_input.get("cascade_length", 0))

    # bfcc_dag: count number of non-neutral ('N') relations
    dag = sample_input.get("bfcc_dag", [])
    non_neutral_count = sum(1 for _, rel, _ in dag if rel[0] != "N")
    features.append(non_neutral_count)

    # bfcc_category: convert to binary feature vector
    category_str = sample_input.get("bfcc_category", "")
    bfcc_category = [int(bit) for bit in category_str]
    features.extend(bfcc_category)

    return features

def extract_program_features(programs, outputs):
    features = []
    
    # Program length: the length of each program (number of characters)
    total_program_length = sum(len(p) for p in programs)
    features.append(total_program_length)

    # Number of replace operations: count "replace" occurrences in programs
    num_replaces = sum(p.lower().startswith("replace") for p in programs)
    features.append(num_replaces)

    # Program similarity: average similarity between program and each output
    program_similarity = np.mean([string_similarity(p, output) for p, output in zip(programs, outputs)])
    features.append(program_similarity)

    # Unique replacements: count distinct replacements
    unique_replacements = len(set(re.findall(r'replace\(([^,]+), ([^,]+)\)', ' '.join(programs))))
    features.append(unique_replacements)

    return features

# === Load dataset ===
data_path = '/home/darsha/Testing/pbe-reasoning/outputs/tagged_qwen3_32b_preds_cascaded_checked_balanced_program_transformations_dataset_2048_max_token_length.jsonl'  # Change this to your file path
samples = []
with open(data_path, 'r') as f:
    for line in f:
        samples.append(json.loads(line))

X = []
y = []

for sample in samples:
    input_data = sample["input"]["inputs"]
    output_data = sample["input"]["outputs"]
    programs = sample["input"]["programs"]

    # Feature 1: input-output transformations
    io_features = extract_input_output_features(input_data, output_data)

    # Feature 2: cascade + DAG + category
    bfcc_features = extract_bfcc_features(sample["input"])

    # Feature 3: program-level features
    program_features = extract_program_features(programs, output_data)  # Pass outputs here

    # Final feature vector
    features = io_features + bfcc_features + program_features
    X.append(features)
    y.append(sample["correct"])

X = np.array(X)
y = np.array(y)

# === Normalize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train classifier ===
clf = LogisticRegression()
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
