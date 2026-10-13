"""Plot the three-cohort WSI bootstrap and paired AUC comparisons."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amc', type=Path, required=True)
    parser.add_argument('--tcga', type=Path, required=True)
    parser.add_argument('--dhmc', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    global plt, np, pd, OUTPUT, COHORTS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    OUTPUT = args.output_dir
    COHORTS = [('AMC', args.amc), ('TCGA', args.tcga), ('DHMC', args.dhmc)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    auc_boxplots()
    paired_auc_forest()

MODEL_KEYS = ['resnet50', 'vit16', 'ding', 'alsubaie', 'cellomaps', 'mixscon_ce', 'mixscon_fl']
MODEL_NAMES = ['ResNet-50', 'ViT-16', 'Ding et al.', 'Alsubaie et al.', 'CellOMaps', 'MixSCon (CE)', 'MixSCon (FL)']
COLORS = ['#8CA0B3', '#7D93A8', '#71879D', '#647C92', '#596F85', '#2D6A8A', '#C85C3C']
def auc_boxplots() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.1), constrained_layout=True)
    for axis, (cohort, directory) in zip(axes, COHORTS):
        frame = pd.read_csv(directory / "bootstrap_wsi_metrics.csv")
        values = [
            frame.loc[frame["model"].eq(model), "auc"].dropna().to_numpy()
            for model in MODEL_KEYS
        ]
        boxes = axis.boxplot(
            values,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#444444"},
            capprops={"color": "#444444"},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
        axis.set_title(cohort, loc="left", fontweight="bold")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Macro-AUC")
        axis.set_xticks(range(1, 8), MODEL_NAMES, rotation=18, ha="right")
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Within-WSI patch bootstrap macro-AUC distributions", fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT / "wsi_auc_bootstrap_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def paired_auc_forest() -> None:
    comparators = MODEL_KEYS[:-1]
    names = MODEL_NAMES[:-1]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.0), sharey=True, constrained_layout=True)
    for axis, (cohort, directory) in zip(axes, COHORTS):
        frame = pd.read_csv(directory / "paired_differences_wsi_vs_mixscon_fl.csv")
        frame = frame.loc[frame["metric"].eq("auc")].set_index("comparator")
        medians = np.array([frame.loc[key, "median"] for key in comparators], dtype=float)
        lower = np.array([frame.loc[key, "q025"] for key in comparators], dtype=float)
        upper = np.array([frame.loc[key, "q975"] for key in comparators], dtype=float)
        y = np.arange(len(comparators))
        colors = ["#C85C3C" if lo > 0 or hi < 0 else "#6D7D8A" for lo, hi in zip(lower, upper)]
        for yi, point, lo, hi, color in zip(y, medians, lower, upper, colors):
            axis.errorbar(
                point,
                yi,
                xerr=[[point - lo], [hi - point]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
            )
        axis.axvline(0, color="#222222", linewidth=1, linestyle="--")
        axis.set_title(cohort, fontweight="bold")
        axis.set_xlabel("Paired Δ macro-AUC")
        axis.grid(axis="x", color="#E1E5E9", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    fig.suptitle("MixSCon FL minus comparator under identical bootstrap samples", fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT / "paired_auc_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    main()
