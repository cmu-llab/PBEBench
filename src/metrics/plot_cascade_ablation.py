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

def plot_perf_by_cascade_length(pass_rate_by_cascade, save_path="plots/perf_by_cascade_length.png", title: str="Pass@1 vs Ground Truth Cascade Length", model: str="gpt-oss-120b"):
    # Extract x and y values
    x = list(pass_rate_by_cascade.keys())
    y = [val[0] for val in pass_rate_by_cascade.values()]
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, color="tab:blue")
    
    # Style adjustments for academic look
    plt.title(title, fontsize=14, weight="bold", pad=15)
    plt.xlabel("Ground Truth Cascade Length", fontsize=12)
    plt.ylabel("Pass@1", fontsize=12)
    plt.ylim(0, 1.05)
    if model == "gpt-oss-120b":
        plt.xticks(range(2, 21), fontsize=10)
    elif model == 'gpt-5':
        plt.xticks([20,25,30], fontsize=10)
    plt.yticks(fontsize=10)
    
    # Remove grid and make spines minimalist
    ax = plt.gca()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    if model == "gpt-oss-120b":
        # Annotate select points with dotted guides
        for xi, yi in zip(x, y):
            if xi in [2, 5, 10, 15, 20]:
                # Offset text if close to top
                offset = -0.08 if yi > 0.9 else 0.05
                plt.text(xi, yi + offset, f"{yi:.2f}", ha="center", fontsize=9)
                
                # Add dotted reference lines
                plt.axhline(y=yi, xmax=(xi-2)/18, linestyle="--", color="gray", alpha=0.5, linewidth=1)
                plt.axvline(x=xi, ymax=yi, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    elif model == "gpt-5":
        # Annotate select points with dotted guides
        for xi, yi in zip(x, y):
            if xi in [20, 25, 30]:
                # Offset text if close to top
                offset = -0.08 if yi > 0.9 else 0.05
                plt.text(xi, yi + offset, f"{yi:.2f}", ha="center", fontsize=9)
                
                # Add dotted reference lines
                plt.axhline(y=yi, xmax=(xi-2)/18, linestyle="--", color="gray", alpha=0.5, linewidth=1)
                plt.axvline(x=xi, ymax=yi, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # Save high-res figure
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
# main
if __name__ == "__main__":
    pass_rate_by_cascade_gpt_oss_120b = {2: (1.0, 64), 3: (1.0, 64), 4: (0.9688, 64), 5: (1.0, 64), 6: (0.9688, 64), 7: (0.9688, 64), 8: (0.9688, 64), 9: (0.8438, 64), 10: (0.8438, 64), 11: (0.875, 64), 12: (0.7656, 64), 13: (0.6719, 64), 14: (0.4531, 64), 15: (0.5, 64), 16: (0.3906, 64), 17: (0.2812, 64), 18: (0.1719, 64), 19: (0.0781, 64), 20: (0.0469, 64)}
    pass_rate_by_cascade_gpt5 = {20: (0.3377, 64), 25: (0.1406, 64), 30: (0.0469, 64)}
    plot_perf_by_cascade_length(
        pass_rate_by_cascade_gpt_oss_120b, 
        save_path="plots/perf_by_cascade_length_gpt_oss_120b.png",
        title="Pass@1 vs Ground Truth Cascade Length (gpt-oss-120b)",
        model="gpt-oss-120b",
    )
    plot_perf_by_cascade_length(
        pass_rate_by_cascade_gpt5, 
        save_path="plots/perf_by_cascade_length_gpt5.png",
        title="Pass@1 vs Ground Truth Cascade Length (GPT-5)",
        model="gpt-5"
    )