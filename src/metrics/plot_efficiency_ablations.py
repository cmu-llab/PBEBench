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


def plot_efficiency_vs_kl(data_dir, cascade_lengths, patience_values, save_path=None):
    plt.figure(figsize=(8, 6))

    for cascade_len in cascade_lengths:
        effs, kls, pats = [], [], []
        for patience in patience_values:
            fname = os.path.join(
                data_dir, f"generation_stats_{cascade_len}_{patience}.json"
            )
            if not os.path.exists(fname):
                continue
            with open(fname, "r") as f:
                stats = json.load(f)
            eff = stats["efficiency"]
            kl = kl_divergence_uniform(stats["category_distribution"])
            effs.append(eff)
            kls.append(kl)
            pats.append(patience)

        if effs:
            plt.plot(effs, kls, marker="o", label=f"Cascade length {cascade_len}")
            # annotate points with compact patience values
            for x, y, p in zip(effs, kls, pats):
                plt.text(x, y, format_patience(p), fontsize=8, ha="right", va="bottom")

    plt.xlabel("Efficiency")
    plt.ylabel("KL Divergence from Uniform")
    plt.title("Efficiency vs Proximity to Balanced Distribution")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


# Example usage
cascade_lengths = [5, 10, 15, 20, 25]
patience_values = [100000, 250000, 500000, 750000, 1000000]
plot_efficiency_vs_kl("efficiency_ablations", cascade_lengths, patience_values, save_path="plots/efficiency_vs_kl.png")
