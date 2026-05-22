import os
import re
import ast
import sys
import json
import pathlib
import argparse
import numpy as np
import editdistance
from typing import Union
import matplotlib.pyplot as plt
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

PADDING = 2
PADDING2 = 4

def plot_perf_by_param(pass_rate_by_cascade, save_path="plots/perf_by_sampling_budget.png", title="Pass@1 vs Sampling Budget (gpt-oss-120b)", suptitle="Max sequence length: 16384", color="red", xlabel="Sampling Budget", x=None, rotation=None, selected_xticks=None):
    # Extract x and y values
    xticks = list(pass_rate_by_cascade.keys()) 
    x = xticks if x is None else x
    y = [val[0] for val in pass_rate_by_cascade.values()]
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, color=f"tab:{color}")

    # Add asymptote line at max y
    max_y = max(y)
    ax = plt.gca()
    ax.axhline(y=max_y, color="gray", linestyle="--", linewidth=1)

    # Annotate the line with the value
    plt.text(
        x[-1], max_y + 0.01, f"{max_y:.3f}", 
        ha="right", va="bottom", fontsize=10,#, color="gray"
    )
    
    # # Ensure max_y is labeled on y-axis
    # yticks = list(plt.yticks()[0])
    # if max_y not in yticks:
    #     yticks.append(max_y)
    #     yticks = sorted(yticks)
    # plt.yticks(yticks, fontsize=10)
    
    # Style adjustments for academic look
    plt.title(title, fontsize=14, weight="bold", pad=20)
    plt.suptitle(suptitle, fontsize=11, y=0.90)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Pass@1", fontsize=12)
    plt.ylim(0, 1.05)
    if rotation:
        if selected_xticks:
            plt.xticks(selected_xticks, labels=selected_xticks, fontsize=10, rotation=rotation)
        else: plt.xticks(x, labels=xticks, fontsize=10, rotation=rotation)

    else:
        if selected_xticks:
            plt.xticks(selected_xticks, labels=selected_xticks, fontsize=10)
        else: plt.xticks(x, labels=xticks, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Remove grid and make spines minimalist
    ax = plt.gca()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Save high-res figure
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_multi_perfs_by_param(pass_rate_by_cascade, save_path="plots/perf_by_sampling_budget.png", title="Pass@1 vs Sampling Budget (gpt-oss-120b)", suptitle="Max sequence length: 16384", colors={}, xlabel="Sampling Budget", x=None, xticks=None, labels=None):
    # Extract x and y values
    y = {k: [val[0] for val in v] for k,v in pass_rate_by_cascade.items()}
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot
    plt.figure(figsize=(8, 5))
    for key in y:
        # print(x[key], y[key])
        plt.plot(
            x[key], y[key], marker='o', linestyle='-', label=key, 
            linewidth=2, markersize=6, color=f"tab:{colors[key]}"
        )
    plt.legend(loc="lower right")
    
    # Style adjustments for academic look
    plt.title(title, fontsize=14, weight="bold", pad=20)
    plt.suptitle(suptitle, fontsize=11, y=0.90)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Pass@1", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(xticks, labels=labels, fontsize=10, rotation=90)
    plt.yticks(fontsize=10)
    
    # Remove grid and make spines minimalist
    ax = plt.gca()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Save high-res figure
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# main
if __name__ == "__main__":
    pass_rate_by_sampling_budget_gpt_oss_120b = {
        1: (0.25, 64),
        2: (0.5, 64),
        4: (0.6094, 64),
        8: (0.7344, 64),
        12: (0.7812, 64),
        16: (0.8281, 64),
        20: (0.8594, 64),
        24: (0.7969, 64),
        28: (0.8438, 64),
        32: (0.8438, 64),
        64: (0.9062, 64),
    }
    pass_rate_by_max_seq_len_gpt_oss_120b = {
        2048: (0, 64),
        3840: (0.2812, 64),
        5632: (0.625, 64),
        7424: (0.7656, 64),
        9216: (0.8594, 64),
        12800: (0.875, 64),
        14592: (0.875, 64),
        16384: (0.8438, 64),
        32768: (0.9219, 64),
        65536: (0.9062, 64),
    } 
    pass_rate_by_sampling_budget_gpt5 = {
        1: (0.1543, 64),
        2: (0.2416, 64),
        3: (0.2974, 64),
        4: (0.3377, 64),
        5: (0.3694, 64),
        6: (0.3956, 64),
        7: (0.418, 64),
        8: (0.4375, 64),
    }
    pass_rate_by_max_seq_len_gpt5_reasoning_mode = {
        "minimal": [(0,64)],
        "low": [(0,64),(0.0156,64),(0.1719,64),(0.6094,64)],
        "medium": [(0,64), (0.3906,64), (0.8125,64)],
        "high": [(0.6875,64), (0.9219,64), (0.9375,64)],
    }
    plot_perf_by_param(
        pass_rate_by_sampling_budget_gpt_oss_120b,
        save_path="plots/perf_by_sampling_budget_gpt_oss_120b.png",
    )
    plot_perf_by_param(
        pass_rate_by_max_seq_len_gpt_oss_120b, 
        save_path="plots/perf_by_max_seq_len_gpt_oss_120b.png",
        title="Pass@1 vs Max Sequence Length (gpt-oss-120b)",
        suptitle="Sampling budget: 32",
        xlabel="Max Sequence Length",
        # x=[2,3,4,5,6,8,9,10,12,14], 
        x=[2048,3840,5632,7424,9216,12800,14592,16384,32768,65536],
        selected_xticks=[2048,5632,9216,12800,16384,32768,65536],
        rotation=45,
        color="green",
    )
    plot_multi_perfs_by_param(
        pass_rate_by_max_seq_len_gpt5_reasoning_mode,
        save_path="plots/perf_by_max_seq_len_gpt5_reasoning_modes.png",
        title="Pass@1 vs Max Completion Tokens (GPT-5)",
        suptitle="Sampling budget: 1",
        xlabel="Max Completion Tokens",
        x={
            "minimal": [1],
            "low": [2+PADDING, 4+PADDING2, 8+PADDING2, 16+PADDING2],
            "medium": [8+PADDING2, 16+PADDING2, 32],
            "high": [32, 64, 128],
        }, 
        colors={
            "minimal": "blue",
            "low": "green",
            "medium": "orange",
            "high": "red",
        },
        labels=[512,1024,2048,4096,8192,16384,32768,65536],
        xticks=[1, 2+PADDING, 4+PADDING2, 8+PADDING2, 16+PADDING2, 32, 64, 128],
    )
    plot_perf_by_param(
        pass_rate_by_sampling_budget_gpt5, 
        save_path="plots/perf_by_sampling_budget_gpt5.png",
        title="Pass@1 vs Sampling Budget (GPT-5)",
        suptitle="Reasoning (Med), Max Completion Tokens: 65536",
        xlabel="Sampling Budget",
        x=[1,2,3,4,5,6,7,8], 
        color="red",
    )
    pass_rate_by_max_seq_len_gpt5 = {
        512: (0, 64),
        1024: (0, 64),
        2048: (0.0156, 64),
        4096: (0.1719, 64),
        8192: (0.6904, 64),
        16384: (0.8125, 64),
        32768: (0.9219, 64),
        65536: (0.9375, 64),
    }
    plot_perf_by_param(
        pass_rate_by_max_seq_len_gpt5, 
        save_path="plots/perf_by_max_seq_len_gpt5.png",
        title="Pass@1 vs Max Completion Token (GPT-5)",
        suptitle="Sampling budget: 1",
        xlabel="Max Completion Tokens",
        x=[512,1024,2048,4096,8192,16384,32768,65536], 
        color="green",
        rotation=45,
        selected_xticks=[512,4096,8192,16384,32768,65536]
    )