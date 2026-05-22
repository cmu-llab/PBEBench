import os
import json
import math
import matplotlib.pyplot as plt

def format_patience(p):
    if p >= 1_000_000:
        return f"{p//1_000_000}M"
    elif p >= 1000:
        return f"{p//1000}k"
    return str(p)

def kl_divergence_uniform(category_dist, epsilon=1e-8):
    keys = list(category_dist.keys())
    n = len(keys)
    U = {k: 1 / n for k in keys}
    total = sum(category_dist.values()) + epsilon * n
    Q = {k: (v + epsilon) / total for k, v in category_dist.items()}

    kl = 0.0
    for k in keys:
        kl += U[k] * math.log(U[k] / Q[k])
    return kl


def plot_efficiency_vs_patience_annotate_kl(data_dir, cascade_lengths, patience_values, save_path=None, cascade_len: int=-1, color: str="red"):
    plt.figure(figsize=(9, 6))

    for cascade_len in cascade_lengths:
        pats, effs, kls = [], [], []
        for patience in patience_values:
            fname = os.path.join(data_dir, f"generation_stats_{cascade_len}_{patience}.json")
            if not os.path.exists(fname):
                continue
            with open(fname, "r") as f:
                stats = json.load(f)
            eff = stats["efficiency"]
            kl = kl_divergence_uniform(stats["category_distribution"])
            pats.append(patience)
            effs.append(eff)
            kls.append(kl)

        if pats:
            plt.plot(pats, effs, marker="o", color=color)
            for x, y, kl in zip(pats, effs, kls):
                kl_label = "KL=0" if abs(kl) < 1e-6 else f"KL={kl:.2f}"
                color = "green" if abs(kl) < 1e-6 else "red"
                weight = "bold" if abs(kl) < 1e-6 else "normal"
                plt.text(x, y, kl_label, fontsize=8, color=color, fontweight=weight, ha="right", va="bottom")

    # X axis formatting
    plt.xticks(patience_values, [format_patience(p) for p in patience_values])
    plt.xlabel("Patience")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency vs Patience for Cascade Length {cascade_len} (KL divergence annotated)")
    # plt.legend()
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


# Example usage
cascade_lengths = [5, 10, 15, 20, 25]
colors = ["crimson", "royalblue", "forestgreen", "darkorange", "gold"]
for cascade_length, color in zip(cascade_lengths, colors):
    patience_values = [100000, 250000, 500000, 750000, 1000000]
    plot_efficiency_vs_patience_annotate_kl("efficiency_ablations", [cascade_length], patience_values, 
                                            cascade_len=cascade_length, color=color,
                                            save_path=f"plots/efficiency_vs_patience_kl_annotated_cascade_{cascade_length}.png")
