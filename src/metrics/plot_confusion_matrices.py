import os
import re
import sys
import json
import pathlib
from tqdm import tqdm
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.eval.utils import read_jsonl
from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary
from src.data_generation.improved_generate_with_rejection_sampling import RelationshipClassifier
from src.metrics.functional_correctness import extract_last_python_block, try_ast_literal_eval, eval_outputs, reward

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_cascade_len_confusion_matrix(confusion_matrix, normalize=True, dpi=300, title="Confusion Matrix", save_path=None):
    """
    Plot and save a confusion matrix given as a dict of dicts.
    Supports numeric or string labels.

    Args:
        confusion_matrix (dict): nested dict {gt: {pred: count}}
        normalize (bool): if True, normalize each row to sum to 1
        dpi (int): resolution for the plot
        title (str): optional title
        save_path (str): if provided, saves the figure instead of showing it
    """
    # Extract labels
    gt_labels = list(confusion_matrix.keys())
    pred_labels = list(next(iter(confusion_matrix.values())).keys())

    # Sort numerically if labels are ints
    if isinstance(gt_labels[0], int):
        gt_labels = sorted(gt_labels)
        pred_labels = sorted(pred_labels)

    # Build matrix
    mat = np.zeros((len(gt_labels), len(pred_labels)), dtype=float)
    for i, gt in enumerate(gt_labels):
        for j, pred in enumerate(pred_labels):
            mat[i, j] = confusion_matrix[gt].get(pred, 0)

    # Normalize rows if requested
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else None)

    # Labels
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(gt_labels)))
    ax.set_xticklabels(pred_labels, ha="right")
    ax.set_yticklabels(gt_labels)

    # Add values inside cells
    for i in range(len(gt_labels)):
        for j in range(len(pred_labels)):
            value = mat[i, j]
            if normalize:
                text_val = f"{value:.2f}" if value > 0 else ""
            else:
                text_val = str(int(value)) if value > 0 else ""
            ax.text(j, i, text_val, ha="center", va="center", color="black", fontsize=8)

    # Axis labels & title
    ax.set_xlabel("Predicted Cascade Length")
    ax.set_ylabel("Ground Truth Cascade Length")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_simple_confusion_matrix(confusion_matrix, save_path: str, normalize=True, dpi=200):
    """
    Plot a confusion matrix given as a dict of dicts.

    confusion_matrix[gt][pred] = count
    gt = ground truth labels (e.g. F-, B-, CF-, CB-, F+, B+, CF+, CB+)
    pred = predicted labels (same plus possibly "INVALID")

    Args:
        confusion_matrix (dict): nested dict {gt: {pred: count}}
        normalize (bool): if True, normalize each row to sum to 1
        dpi (int): resolution for the plot
    """

    # Extract labels
    gt_labels = list(confusion_matrix.keys())
    pred_labels = list(next(iter(confusion_matrix.values())).keys())

    # Put INVALID last if it exists
    gt_labels = [l for l in gt_labels if l != "INVALID"]
    pred_labels = [l for l in pred_labels if l != "INVALID"]
    if "INVALID" in next(iter(confusion_matrix.values())).keys():
        pred_labels.append("INVALID")

    # Build matrix
    mat = np.zeros((len(gt_labels), len(pred_labels)), dtype=float)
    for i, gt in enumerate(gt_labels):
        for j, pred in enumerate(pred_labels):
            mat[i, j] = confusion_matrix[gt].get(pred, 0)

    # Normalize rows if requested
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else None)

    # Labels
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(gt_labels)))
    ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    ax.set_yticklabels(gt_labels)

    # Add values inside cells
    for i in range(len(gt_labels)):
        for j in range(len(pred_labels)):
            value = mat[i, j]
            if normalize:
                text_val = f"{value:.2f}" if value > 0 else ""
            else:
                text_val = str(int(value)) if value > 0 else ""
            ax.text(j, i, text_val, ha="center", va="center", color="black", fontsize=8)

    # Title and colorbar
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth Label")
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(confusion_matrix, save_path: str, normalize: bool=True):
    """
    Plot a confusion matrix given as a dict of dicts.
    
    confusion_matrix[gt][pred] = count
    gt ∈ {16 binary strings of length 4}
    pred ∈ {16 binary strings of length 4} ∪ {"INVALID"}
    
    normalize: if True, normalize each row to sum to 1
    """

    # Sort categories to ensure consistent ordering
    gt_labels = sorted([k for k in confusion_matrix.keys() if k != "INVALID"])
    pred_labels = sorted([k for k in next(iter(confusion_matrix.values())).keys() if k != "INVALID"]) + ["INVALID"]

    # Build matrix
    mat = np.zeros((len(gt_labels), len(pred_labels)), dtype=float)
    for i, gt in enumerate(gt_labels):
        for j, pred in enumerate(pred_labels):
            mat[i, j] = confusion_matrix[gt].get(pred, 0)

    # Normalize rows if requested
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div by zero
        mat = mat / row_sums

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else None)

    # Labels
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(gt_labels)))
    ax.set_xticklabels(pred_labels, rotation=45, ha="right")
    ax.set_yticklabels(gt_labels)

    # Add values inside cells
    for i in range(len(gt_labels)):
        for j in range(len(pred_labels)):
            value = mat[i, j]
            if normalize:
                text_val = f"{value:.2f}" if value > 0 else ""
            else:
                text_val = str(int(value)) if value > 0 else ""
            ax.text(j, i, text_val, ha="center", va="center", color="black", fontsize=8)

    # Title and colorbar
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth Label")
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Example usage:
# plot_confusion_matrix(confusion_matrix)

vocab_chars = "abcdefghijkuvwxyz"
max_programs = 5

def pick_highest_reward_prog_seq(programs: list[str], inputs: list[str], outputs: list[str], max_programs: int=5):
    progs_and_rewards = []
    for prog_seq_string in programs:
        pred_cascade = try_ast_literal_eval(prog_seq_string)
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        progs_and_rewards.append((pred_cascade[:max_programs], reward(pred_outputs, inputs, outputs)))
    progs_and_rewards = sorted(progs_and_rewards, reverse=True, key=lambda x: x[1]) # sort by reward.
    # print(progs_and_rewards[0][0])

    return progs_and_rewards[0][0]

def load_model_predictions_multi(path: str, extract_method) -> list[dict]: 
    raw_data: list[dict] = read_jsonl(path) 
    loaded_data = []
    null_ctr = 0
    for rec in raw_data: 
        # extract the code‐block text
        rec['input']["predictions"] = []
        for output in rec['outputs']: 
            try: pred = extract_method(output) 
            except TypeError as e:
                if output is None:
                    null_ctr += 1
                else: print(e)
                pred = "replace('x','x')"

            # wrap any replace(...) that isn’t already quoted in single quotes
            # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
            pred = re.sub(
                r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
                r'"\1"',
                pred
            )

            # assign and print
            rec['input']["predictions"].append(pred) 
            # print(pred)

        loaded_data.append(rec['input']) 
    print(f"found {null_ctr} nulls")

    return loaded_data

MODEL_TO_PREDS_PATH_MAPPING = {
    "Codestral-22B": "outputs/ada_bal_1008_codestral_22b_2048_async.jsonl",
    "Qwen2.5-32B-Instruct": "outputs/ada_bal_1008_qwen_25_32b_instruct_512_async_ath.jsonl",
    "Qwen2.5Coder-32B-Instruct": "outputs/ada_bal_1008_qwen_25_coder_32b_instruct_512_async.jsonl",
    "QwQ-32B": "outputs/ada_bal_1008_qwen_qwq_32b_8192_async.jsonl",
    "Qwen3-32B (with CoT)": "outputs/ada_bal_1008_qwen3_32b_8192_async.jsonl",
    "Qwen3-32B": "outputs/ada_bal_1008_qwen3_32b_no_reason_8192_async.jsonl",
    "Qwen3-30B-A3B": "outputs/ada_bal_1008_qwen3_30b_a3b_8192_async.jsonl",
    "DeepSeek-R1-Distill-Qwen-32B": "outputs/ada_bal_1008_deepseek_r1_distill_qwen_32b_8192_async.jsonl",
    "o3-mini": "outputs/o3_mini_8192_medium.jsonl",
    "o4-mini": "outputs/o4_mini_8192_medium.jsonl",
    "Gemini 2.5 Flash": "outputs/ada_bal_1008_gemini_flash.jsonl",
    "Claude-3.5-Sonnet": "outputs/prakam_ada_bal_1008_claude_35_sonnet_8192.jsonl",
    "Claude-3.7-Sonnet": "outputs/prakam_ada_bal_1008_claude_37_sonnet_no_think_10000.jsonl",
    "Claude-3.7-Sonnet (Thinking)": "outputs/prakam_ada_bal_1008_claude_37_sonnet_10000.jsonl",
    "GPT-OSS 20B": "outputs/ada_bal_1008_gpt_oss_20b_8192_async_ath.jsonl",
    "GPT-OSS 120B": "outputs/ada_bal_1008_gpt_oss_120b_8192_async.jsonl",
    "GPT-5": "outputs/ada_bal_1008_gpt_5_8192.jsonl",
    "Claude-4 Sonnet": "outputs/prakam_ada_bal_1008_claude_4_sonnet_no_think_8192.jsonl",
    "Claude-4 Sonnet (Thinking)": "outputs/prakam_ada_bal_1008_claude_4_sonnet_8192.jsonl",
    "Qwen3-Coder-30B-A3B-Instruct": "outputs/ada_bal_1008_qwen3_coder_30b_a3b_instruct_2048_async.jsonl",
}


# main
if __name__ == "__main__":
    rel_classifier = RelationshipClassifier()
    all_categories = rel_classifier.get_all_categories()
    simple_categories = ["F-", "F+", "B-", "B+", "CF-", "CF+", "CB-", "CB+"]
    # print(all_categories)
    pred_rel_type_dist = defaultdict(lambda: [])
    invalid_cascades = 0
    
    conf_mat_failure_cascade_len = {i :{j: 0 for j in range(5+1)} for i in range(1, 5+1)}
    conf_mat_success_cascade_len = {i :{j: 0 for j in range(5+1)} for i in range(1, 5+1)}
    conf_mat_per_model_cascade_len = {}

    conf_mat_per_model_simple = {}
    conf_mat_failure_simple = {category: {category: 0 for category in simple_categories+["INVALID"]} for category in simple_categories}
    conf_mat_success_simple = {category: {category: 0 for category in simple_categories+["INVALID"]} for category in simple_categories}
    confusion_matrix_failure = {category: {category: 0 for category in all_categories+["INVALID"]} for category in all_categories}
    confusion_matrix_success = {category: {category: 0 for category in all_categories+["INVALID"]} for category in all_categories}
    
    conf_mat_per_model = {}
    for model, preds_path in MODEL_TO_PREDS_PATH_MAPPING.items():
        conf_mat_per_model_cascade_len[model] = {
            "failure": {i :{j: 0 for j in range(1,5+1)} for i in range(1, 5+1)},
            "success": {i :{j: 0 for j in range(1,5+1)} for i in range(1, 5+1)},
        }
        conf_mat_per_model[model] = {
            "failure": {category: {category: 0 for category in all_categories+["INVALID"]} for category in all_categories},
            "success": {category: {category: 0 for category in all_categories+["INVALID"]} for category in all_categories},
        }
        conf_mat_per_model_simple[model] = {
            "failure": {category: {category: 0 for category in simple_categories+["INVALID"]} for category in simple_categories},
            "success": {category: {category: 0 for category in simple_categories+["INVALID"]} for category in simple_categories},
        }
        preds = load_model_predictions_multi(
            preds_path, 
            extract_method=extract_last_python_block,
        )
        metrics_path = preds_path.replace("outputs/", "metric_values/").replace(".jsonl", ".json")
        metric_values = json.load(open(metrics_path))
        for index, instance in enumerate(preds):
            metric_value = metric_values[index]
            inputs = instance['inputs']
            outputs = instance['outputs']
            pred_cascade = pick_highest_reward_prog_seq(programs=instance['predictions'], inputs=inputs, outputs=outputs)
            # print(pred_cascade)
            prediction = {"original_programs": pred_cascade}
            # _, ground_truth_category = rel_classifier.classify_binary(instance)
            try: _, predicted_category = rel_classifier.classify_binary(prediction)
            except ValueError: # When invalid program cascade is predicted assign INVALID as default.
                predicted_category = "INVALID"
                invalid_cascades += 1
                # exit()

            # print("GT category recomputed", ground_truth_category)
            # print("GT category cached:", instance['bfcc_category'])
            # assert instance["bfcc_category"] == ground_truth_category, 'POTENTIALLY FLAWED BFCC LABELS'
            # print("pred category", predicted_category)
            # print(metric_value)
            pred_rel_type_dist[model].append({
                "instance": instance,
                "metric_value": metric_value,
                "pred_bfcc_category": predicted_category,
                "ground_truth_bfcc_category": instance['bfcc_category'],
            })
            if metric_value['passing'] == 0:
                for i in range(4):
                    gt_label = simple_categories[2*i+1] if instance['bfcc_category'][i] == "1" else simple_categories[2*i]
                    if predicted_category == "INVALID": pred_label = "INVALID"
                    else:
                        pred_label = simple_categories[2*i+1] if predicted_category[i] == "1" else simple_categories[2*i]
                    conf_mat_per_model_simple[model]["failure"][gt_label][pred_label] += 1
                    conf_mat_failure_simple[gt_label][pred_label] += 1
                conf_mat_per_model[model]["failure"][instance['bfcc_category']][predicted_category] += 1
                confusion_matrix_failure[instance['bfcc_category']][predicted_category] += 1
                conf_mat_failure_cascade_len[len(instance['original_programs'])][len(pred_cascade)] += 1
            elif metric_value["passing"] == 1:
                for i in range(4):
                    gt_label = simple_categories[2*i+1] if instance['bfcc_category'][i] == "1" else simple_categories[2*i]
                    if predicted_category == "INVALID": pred_label = "INVALID"
                    else:
                        pred_label = simple_categories[2*i+1] if predicted_category[i] == "1" else simple_categories[2*i]
                    conf_mat_per_model_simple[model]["success"][gt_label][pred_label] += 1
                    conf_mat_success_simple[gt_label][pred_label] += 1
                conf_mat_per_model[model]["success"][instance['bfcc_category']][predicted_category] += 1
                confusion_matrix_success[instance['bfcc_category']][predicted_category] += 1
                conf_mat_success_cascade_len[len(instance['original_programs'])][len(pred_cascade)] += 1
            # exit()
            # print(category)
        # print(model, path)
    # print(confusion_matrix_failure)
    plot_cascade_len_confusion_matrix(conf_mat_failure_cascade_len, normalize=False, save_path="confusion_matrices_clen/pbebenchlite_failure_cases.png")
    plot_cascade_len_confusion_matrix(conf_mat_success_cascade_len, normalize=False, save_path="confusion_matrices_clen/pbebenchlite_success_cases.png")

    plot_simple_confusion_matrix(conf_mat_failure_simple, "confusion_matrices_simple/pbebenchlite_failure_cases.png")
    plot_simple_confusion_matrix(conf_mat_success_simple, "confusion_matrices_simple/pbebenchlite_success_cases.png")
    plot_confusion_matrix(confusion_matrix_failure, "confusion_matrices/pbebenchlite_failure_cases.png")
    plot_confusion_matrix(confusion_matrix_success, "confusion_matrices/pbebenchlite_success_cases.png")
    for model in tqdm(conf_mat_per_model, desc="conf mat per model"):
        plot_confusion_matrix(conf_mat_per_model[model]["failure"], f"confusion_matrices/{model}_pbebenchlite_failure_cases.png")  
        plot_confusion_matrix(conf_mat_per_model[model]["success"], f"confusion_matrices/{model}_pbebenchlite_success_cases.png")    
    # print(confusion_matrix_success)
    # print(len(pred_rel_type_dist))
    # print(invalid_cascades)