
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import plot_ant_main_visualization as antviz


TOTAL_STEPS = antviz.TOTAL_STEPS
MLES_BUDGET = antviz.MLES_BUDGET
COLORS = antviz.COLORS


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_half_baseline_dir(root: Path) -> Path:
    candidates = []
    for path in (root / "experiment").glob("half_baseline*"):
        summary = path / "baseline_summary.json"
        if not summary.exists():
            continue
        data = load_json(summary)
        save_dirs = " ".join(str(r.get("save_dir", "")) for r in data.get("results", []))
        mean_reward = float(data.get("mean_reward_across_seeds", float("nan")))
        candidates.append((path, save_dirs, mean_reward))

    if not candidates:
        raise FileNotFoundError("No half_baseline*/baseline_summary.json found")
    return min(candidates, key=lambda item: item[2])[0]


def load_half_random_curves(root: Path):
    curve_dir = root / "experiment" / "half_rs2"
    if not curve_dir.exists():
        curve_dir = root / "experiment" / "half_rs"
    curve_paths = sorted((curve_dir / "final").glob("reward_curves*.json"))
    if not curve_paths:
        curve_paths = sorted(curve_dir.glob("reward_curves*.json"))
    if curve_paths:
        return antviz.load_ppo_curves(curve_paths)

    log_path = curve_dir / "half_random.log"
    if not log_path.exists():
        log_path = root / "experiment" / "half_random.log"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    runs = re.findall(r"G18_I10 run\d+.*?mean_reward=([-+]?\d+(?:\.\d+)?)", text)
    if not runs:
        raise ValueError(f"No G18_I10 final run rewards found in {log_path}")

    curves = []
    for value in runs:
        reward = float(value)
        curves.append((np.asarray([0.0, float(TOTAL_STEPS)]), np.asarray([reward, reward])))
    return curves


def find_half_random_dir(root: Path) -> Path:
    preferred = root / "experiment" / "half_rs2"
    if preferred.exists():
        return preferred
    return root / "experiment" / "half_rs"


def find_curves(root: Path):
    final_best = load_json(root / "experiment" / "half_evom" / "final_best_individual.json")
    g0_best = load_json(root / "experiment" / "half_evom" / "gen0_best_individual.json")
    final_label = final_best["best_individual"]["label"]
    g0_label = g0_best["label"]

    baseline_dir = find_half_baseline_dir(root)
    random_dir = find_half_random_dir(root)

    return {
        "Manually Designed Baseline": antviz.load_ppo_curves(baseline_dir.glob("*reward_curves*.json")),
        f"EVOM G0 best ({g0_label})": antviz.load_ppo_curves(
            (root / "experiment" / "half_evom" / "G0best").glob("reward_curves*.json")
        ),
        f"EVOM final best ({final_label})": antviz.load_ppo_curves(
            (root / "experiment" / "half_evom" / "final").glob("reward_curves*.json")
        ),
        "Random search best": load_half_random_curves(root),
        "MLES": [
            antviz.load_mles_curve(path)
            for path in sorted((root / "experiment" / "half_mles").glob("evaluations*.jsonl"))
        ],
    }, final_label, g0_label, baseline_dir, random_dir


def baseline_design():
    return {
        "policy_type": "SB3 MLP 64x64",
        "value_type": "SB3 MLP 64x64",
        "policy_nodes": [
            {"label": "Input 17", "kind": "io"},
            {"label": "Linear 17 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Linear 64 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Mean 6", "kind": "linear"},
        ],
        "value_nodes": [
            {"label": "Input 17", "kind": "io"},
            {"label": "Linear 17 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Linear 64 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Value 1", "kind": "linear"},
        ],
    }


def residual_node(dim=256):
    return {
        "label": f"Residual Block {dim}",
        "kind": "residual",
        "expanded_residual": True,
        "dim": dim,
    }


def evom_g0_design():
    return {
        "policy_type": "LayerNorm MLP + Tanh",
        "value_type": "LayerNorm MLP + Tanh",
        "policy_nodes": [
            {"label": "Input 17", "kind": "io"},
            {"label": "Linear 17 -> 256", "kind": "linear"},
            {"label": "LayerNorm + Tanh", "kind": "normalization"},
            {"label": "Linear 256 -> 256", "kind": "linear"},
            {"label": "LayerNorm + Tanh", "kind": "normalization"},
            {"label": "Mean 6", "kind": "linear"},
        ],
        "value_nodes": [
            {"label": "Input 17", "kind": "io"},
            {"label": "Linear 17 -> 256", "kind": "linear"},
            {"label": "LayerNorm + Tanh", "kind": "normalization"},
            {"label": "Linear 256 -> 256", "kind": "linear"},
            {"label": "LayerNorm + Tanh", "kind": "normalization"},
            {"label": "Value 1", "kind": "linear"},
        ],
    }


def evom_final_design():
    return {
        "policy_type": "LayerNorm MLP + SiLU",
        "value_type": "LayerNorm MLP + SiLU",
        "policy_nodes": [
            {"label": "Input 17", "kind": "io"},
            {"label": "Linear 17 -> 256", "kind": "linear"},
            {"label": "LayerNorm + SiLU", "kind": "normalization"},
            {"label": "Linear 256 -> 256", "kind": "linear"},
            {"label": "LayerNorm + SiLU", "kind": "normalization"},
            {"label": "Linear 256 -> 256", "kind": "linear"},
            {"label": "LayerNorm + SiLU", "kind": "normalization"},
            {"label": "Mean 6", "kind": "linear"},
        ],
        "value_nodes": [
            {"label": "Input 17", "kind": "io"},
            {"label": "Linear 17 -> 256", "kind": "linear"},
            {"label": "LayerNorm + SiLU", "kind": "normalization"},
            {"label": "Linear 256 -> 256", "kind": "linear"},
            {"label": "LayerNorm + SiLU", "kind": "normalization"},
            {"label": "Linear 256 -> 256", "kind": "linear"},
            {"label": "LayerNorm + SiLU", "kind": "normalization"},
            {"label": "Value 1", "kind": "linear"},
        ],
    }


def write_data_txt(
    path: Path,
    generations,
    history_best,
    elite_mean,
    grid,
    aggregated,
    baseline_dir,
    random_dir,
):
    with path.open("w", encoding="utf-8") as f:
        f.write("HalfCheetah main experiment visualization data\n")
        f.write("=" * 52 + "\n\n")

        f.write("[Data sources]\n")
        f.write(f"EVOM: experiment/half_evom\n")
        f.write(f"Manually Designed Baseline: {baseline_dir.relative_to(path.parents[1]) if baseline_dir.is_relative_to(path.parents[1]) else baseline_dir}\n")
        f.write(f"Random search: {random_dir.relative_to(path.parents[1]) if random_dir.is_relative_to(path.parents[1]) else random_dir}/reward_curves*.json\n")
        f.write("MLES: experiment/half_mles/evaluations*.jsonl\n\n")

        f.write("[Outer evolution convergence]\n")
        f.write("generation\thistorical_best\telite_population_mean\n")
        for gen, best, mean in zip(generations, history_best, elite_mean):
            f.write(f"{int(gen)}\t{best:.6f}\t{mean:.6f}\n")

        f.write("\n[Inner evaluation convergence summary]\n")
        f.write("method\truns\tfinal_mean\tfinal_min\tfinal_max\n")
        for method, agg in aggregated.items():
            final_values = agg["sampled"][:, -1]
            f.write(
                f"{method}\t{len(final_values)}\t"
                f"{np.nanmean(final_values):.6f}\t"
                f"{np.nanmin(final_values):.6f}\t"
                f"{np.nanmax(final_values):.6f}\n"
            )

        f.write("\n[Notes]\n")
        f.write("All curves are resampled to 0-5,000,000 env steps.\n")
        f.write("PPO/EVOM curves use reward_curves.json eval rewards and cumulative best.\n")
        f.write("Random search curves use the copied final reward_curves.json artifacts without display scaling.\n")
        f.write(
            f"MLES curves use candidate score cumulative best, limited to "
            f"C0001-C{MLES_BUDGET:04d}, then scaled to 5M steps.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    generations, history_best, elite_mean = antviz.load_outer_evolution(
        root / "experiment" / "half_evom"
    )
    curve_sets, final_label, g0_label, baseline_dir, random_dir = find_curves(root)

    grid = np.linspace(0, TOTAL_STEPS, 201)
    aggregated = {
        method: antviz.aggregate_curves(curves, grid)
        for method, curves in curve_sets.items()
    }

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(18, 10.8), dpi=160)
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[0.88, 1.55],
        hspace=0.11,
        wspace=0.42,
    )

    ax_outer = fig.add_subplot(gs[0, 0:3])
    ax_inner = fig.add_subplot(gs[0, 3:6])
    ax_base = fig.add_subplot(gs[1, 0:2])
    ax_g0 = fig.add_subplot(gs[1, 2:4])
    ax_final = fig.add_subplot(gs[1, 4:6])

    antviz.plot_outer(ax_outer, generations, history_best, elite_mean)
    antviz.plot_inner(ax_inner, grid, aggregated)
    ax_inner.set_ylabel("Historical best reward", labelpad=8)

    antviz.draw_model_panel(ax_base, "Manually Designed Baseline", baseline_design())
    antviz.draw_model_panel(ax_g0, f"EVOM Gen-0 Best ({g0_label})", evom_g0_design())
    antviz.draw_model_panel(ax_final, f"EVOM Final Best ({final_label})", evom_final_design())

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["linear"], edgecolor="#9aa5ad", label="Linear"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["activation"], edgecolor="#9aa5ad", label="Activation"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["normalization"], edgecolor="#9aa5ad", label="Normalization"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["dropout"], edgecolor="#9aa5ad", label="Dropout"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["residual"], edgecolor="#9aa5ad", label="Residual"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["io"], edgecolor="#9aa5ad", label="Input/Output"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle("HalfCheetah-v4 Main Experiment: EVOM vs Baselines", fontsize=20, fontweight="bold")
    fig.subplots_adjust(bottom=0.075, top=0.925, left=0.055, right=0.985, wspace=0.42)

    png_path = outdir / "halfcheetah_main_visualization.png"
    pdf_path = outdir / "halfcheetah_main_visualization.pdf"
    txt_path = outdir / "halfcheetah_main_visualization_data.txt"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    write_data_txt(
        txt_path,
        generations,
        history_best,
        elite_mean,
        grid,
        aggregated,
        baseline_dir,
        random_dir,
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
