import os
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

def get_dataset_stats(data):
    return {
        "bfcc_categories": dict(sorted(Counter([rec["bfcc_category"] for rec in data]).items(), key=lambda x: x[0])),
        "cascade_lengths": dict(sorted(Counter([rec['cascade_length'] for rec in data]).items(), key=lambda x: x[0])),
    }

def plot_dataset_distributions_v2(dataset_name,
                                  reln_dist,
                                  cascade_len_dist,
                                  reln_title_suffix="Relation Type Distribution",
                                  cascade_title_suffix="Cascade Length Distribution",
                                  reln_xlabel="Relation Type",
                                  reln_ylabel="Count",
                                  cascade_xlabel="Cascade Length",
                                  cascade_ylabel="Count",
                                  figsize=(7,4),
                                  dpi=300,
                                  reln_color="steelblue",
                                  cascade_color="seagreen"):
    """
    Publication-ready distributions. Guarantees slim bars for 1-3 categories.
    Saves two PNGs to ./plots/.
    """
    os.makedirs("plots", exist_ok=True)

    # Modern clean style (prefer Helvetica, fall back to DejaVu Sans)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out"
    })

    def _draw_bar_plot(ax, labels, values, color, title, xlabel, ylabel, rotate_labels=False):
        n = len(labels)
        # choose slim widths for very small n
        if n <= 1:
            width = 0.25
        elif n == 2:
            width = 0.30
        elif n == 3:
            width = 0.40
        else:
            width = 0.60

        positions = np.arange(n)
        rects = ax.bar(positions, values, width=width, color=color, edgecolor='none')

        # xticks + labels
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90 if rotate_labels else 0, fontsize=9)

        # Prevent the single/few bars from filling the whole axis:
        margin = max(1.0, 0.8)  # keep extra space both sides
        left = positions[0] - margin * 0.8
        right = positions[-1] + margin * 0.8
        ax.set_xlim(left, right)

        # ensure room at top for numeric labels
        maxv = max(values) if len(values) > 0 else 1
        top = maxv * 1.12 if maxv > 0 else 1.0
        ax.set_ylim(0, top)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        # light horizontal gridlines (paper friendly)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

        # annotate bars
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

        return rects

    # ---- Relation type plot ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    reln_keys = list(reln_dist.keys())
    reln_vals = [reln_dist[k] for k in reln_keys]
    _draw_bar_plot(ax,
                   labels=reln_keys,
                   values=reln_vals,
                   color=reln_color,
                   title=f"{dataset_name}: {reln_title_suffix}",
                   xlabel=reln_xlabel,
                   ylabel=reln_ylabel,
                   rotate_labels=(len(reln_keys) > 8))
    plt.tight_layout()
    fig.savefig(f"plots/{dataset_name}_reln_dist.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- Cascade length plot ----
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cascade_keys = sorted(cascade_len_dist.keys())
    cascade_vals = [cascade_len_dist[k] for k in cascade_keys]
    # tick labels as the actual cascade lengths (strings)
    tick_labels = [str(k) for k in cascade_keys]
    _draw_bar_plot(ax,
                   labels=tick_labels,
                   values=cascade_vals,
                   color=cascade_color,
                   title=f"{dataset_name}: {cascade_title_suffix}",
                   xlabel=cascade_xlabel,
                   ylabel=cascade_ylabel,
                   rotate_labels=False)
    plt.tight_layout()
    fig.savefig(f"plots/{dataset_name}_cascade_dist.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# main
if __name__ == "__main__":
    data = {
        "PBEBench-Lite": json.load(open("data/adaptive_balanced_1008_complete_promptsfile.json")),
        "PBEBench-Lite-MoreEg": json.load(open("data/adaptive_balanced_240_more_eg_50_inputs_promptsfile.json")),
        "PBEBench": json.load(open("data/adaptive_balanced_64_hard_50_inputs_2_20_cascade_bbr_promptsfile.json")),
        "PBEBench (25, 30)": json.load(open("data/adaptive_balanced_64_hard_50_inputs_25_cascade_unified_latest_promptsfile.json"))+json.load(open("data/adaptive_balanced_64_hard_50_inputs_30_cascade_unified_latest_promptsfile.json")), 
    }
    for data_name, dataset in data.items():
        print(data_name)
        stats = get_dataset_stats(dataset)
        bfcc_categ_dist = stats['bfcc_categories']
        cascade_len_dist = stats['cascade_lengths']
        print("BFCC Reln Type Dist:", bfcc_categ_dist)
        print("Ground Cascade Length Dist:", cascade_len_dist)       
        print() 
        plot_dataset_distributions_v2(
            dataset_name=data_name,
            reln_dist=bfcc_categ_dist,
            cascade_len_dist=cascade_len_dist,
        )
