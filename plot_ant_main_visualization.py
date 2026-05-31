
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


TOTAL_STEPS = 5_000_000
MLES_BUDGET = 1667
ANT_RANDOM_DIR = "ant_random_20260529_003443"
ANT_RANDOM_LABEL = "G3_I9"


COLORS = {
    "baseline": "#2f6fb0",
    "g0": "#f28e2b",
    "evom": "#59a14f",
    "random": "#b07aa1",
    "mles": "#e15759",
    "outer_best": "#2f6fb0",
    "outer_mean": "#f28e2b",
    "linear": "#bed8f3",
    "activation": "#ffdca8",
    "normalization": "#ccefad",
    "dropout": "#e4a3ed",
    "residual": "#f7b6b2",
    "io": "#d6dde1",
    "other": "#eeeeee",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def natural_generation(path: Path) -> int:
    match = re.search(r"gen(\d+)_summary\.json$", path.name)
    return int(match.group(1)) if match else -1


def load_outer_evolution(evom_dir: Path):
    rows = []
    for path in sorted(evom_dir.glob("gen*_summary.json"), key=natural_generation):
        data = load_json(path)
        gen = int(data["generation"])

        if gen == 0:
            current_best = safe_float(data.get("best_fitness"))
            elite_mean = safe_float(data.get("mean_fitness"))
        else:
            selected = data.get("selected_parents", {})
            offspring = data.get("offspring", {})
            current_best = safe_float(
                selected.get("best_fitness", offspring.get("best_fitness"))
            )
            elite_mean = safe_float(
                selected.get("mean_fitness", offspring.get("mean_fitness"))
            )

        rows.append((gen, current_best, elite_mean))

    generations = np.array([r[0] for r in rows], dtype=float)
    generation_best = np.array([r[1] for r in rows], dtype=float)
    history_best = np.maximum.accumulate(generation_best)
    elite_mean = np.array([r[2] for r in rows], dtype=float)
    return generations, history_best, elite_mean


def reward_curve_from_json(path: Path):
    data = load_json(path)
    section = data.get("eval") or data.get("rollout")
    if not section:
        raise ValueError(f"No eval/rollout curve in {path}")
    steps = np.asarray(section["steps"], dtype=float)
    rewards = np.asarray(section["rewards"], dtype=float)
    order = np.argsort(steps)
    steps = steps[order]
    rewards = rewards[order]
    steps = np.clip(steps, 0, TOTAL_STEPS)
    history_best = np.maximum.accumulate(rewards)
    return steps, history_best


def previous_value_resample(x, y, grid):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)

    order = np.argsort(x)
    x, y = x[order], y[order]
    idx = np.searchsorted(x, grid, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def aggregate_curves(curves, grid):
    sampled = np.vstack([previous_value_resample(x, y, grid) for x, y in curves])
    return {
        "mean": np.nanmean(sampled, axis=0),
        "min": np.nanmin(sampled, axis=0),
        "max": np.nanmax(sampled, axis=0),
        "sampled": sampled,
    }


def load_ppo_curves(paths):
    return [reward_curve_from_json(path) for path in sorted(paths)]


def load_mles_curve(path: Path, budget: int = MLES_BUDGET):
    xs = []
    ys = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            match = re.match(r"C(\d+)$", str(item.get("candidate_id", "")))
            if not match:
                continue
            idx = int(match.group(1))
            if idx > budget:
                continue
            xs.append(idx / budget * TOTAL_STEPS)
            ys.append(float(item["score"]))

    if not xs:
        raise ValueError(f"No MLES records within C0001-C{budget:04d}: {path}")

    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    y = np.asarray(ys, dtype=float)[order]
    return x, np.maximum.accumulate(y)


def find_curves(root: Path):
    evom_summary = load_json(root / "experiment" / "ant_evom" / "final_selection" / "elite_evaluation_summary.json")
    evom_final_label = evom_summary["elites"][0]["label"]
    evom_g0_label = evom_summary["elites"][1]["label"]

    rs_dir = root / "experiment" / ANT_RANDOM_DIR
    rs_summary = load_json(rs_dir / "final_selection" / "elite_evaluation_summary.json")
    rs_best_label = ANT_RANDOM_LABEL

    return {
        "Manually Designed Baseline": load_ppo_curves(
            (root / "experiment" / "ant-PPO" / "baseline").glob("seed_*/reward_curves.json")
        ),
        f"EVOM G0 best ({evom_g0_label})": load_ppo_curves(
            (root / "experiment" / "ant_evom" / "final_selection" / evom_g0_label).glob("run*/reward_curves.json")
        ),
        f"EVOM final best ({evom_final_label})": load_ppo_curves(
            (root / "experiment" / "ant_evom" / "final_selection" / evom_final_label).glob("run*/reward_curves.json")
        ),
        f"Random search best ({rs_best_label})": load_ppo_curves(
            (rs_dir / "final_selection" / rs_best_label).glob("run*/reward_curves.json")
        ),
        "MLES": [
            load_mles_curve(path)
            for path in sorted((root / "experiment" / "ant_mles").glob("evaluations*.jsonl"))
        ],
    }, evom_summary


def parse_design_string(design):
    if isinstance(design, str):
        try:
            return json.loads(design)
        except json.JSONDecodeError:
            return {}
    return design or {}


def split_layer_text(text):
    text = str(text)
    text = text.replace("→", "->").replace("—", "-")
    if "ResidualBlock" in text or "Residual Block" in text:
        return [text]
    parts = re.split(r"\s*(?:->|\+)\s*", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_layer_config(raw_layers, stream_name):
    layers = ["Input 27"]
    if isinstance(raw_layers, (str, int, float)):
        raw_layers = [raw_layers]

    for raw in raw_layers or []:
        if isinstance(raw, (int, float)):
            layers.append(f"Linear {int(raw)}")
            continue
        for part in split_layer_text(raw):
            part = re.sub(r"\s+", " ", part).strip()
            if not part:
                continue
            if part.lower().startswith("input"):
                continue
            layers.append(part)

    if stream_name == "policy_net":
        if not any("8" in layer and "Linear" in layer for layer in layers):
            layers.append("Linear -> 8")
    else:
        if not any("1" in layer and "Linear" in layer for layer in layers):
            layers.append("Linear -> 1")
    return layers


def layer_kind(label):
    lower = label.lower()
    if lower.startswith("input") or "output" in lower or "log_std" in lower:
        return "io"
    if "dropout" in lower:
        return "dropout"
    if "residual" in lower or "skip" in lower or "spectralnorm" in lower:
        return "residual"
    if "norm" in lower or "batchnorm" in lower:
        return "normalization"
    if (
        "linear" in lower
        or "dense" in lower
        or "fc" in lower
        or "hidden layer" in lower
    ):
        return "linear"
    if any(a in lower for a in ["relu", "silu", "gelu", "tanh", "mish", "swish"]):
        return "activation"
    return "other"


def wrap_label(text, width=18):
    text = str(text).replace("nn.Parameter", "Parameter")
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else text


def activation_name(text):
    lower = str(text).lower()
    for name in ["leakyrelu", "relu", "silu", "gelu", "tanh", "mish", "swish"]:
        if name in lower:
            return {
                "leakyrelu": "LeakyReLU",
                "relu": "ReLU",
                "silu": "SiLU",
                "gelu": "GELU",
                "tanh": "Tanh",
                "mish": "Mish",
                "swish": "Swish",
            }[name]
    return ""


def compact_layer_label(text):
    text = str(text).replace("→", "->").replace("nn.Parameter", "Parameter")
    lower = text.lower()

    if lower.startswith("input"):
        match = re.search(r"(\d+)", text)
        return f"Input {match.group(1)}" if match else "Input"

    if "log_std" in lower:
        dim = re.search(r"\((\d+),?\)", text)
        init = re.search(r"(-?\d+(?:\.\d+)?)\)?\)?$", text)
        left = f"log_std {dim.group(1)}" if dim else "log_std"
        right = f"init {init.group(1)}" if init else "learned"
        return f"{left}\n{right}"

    if "residualblock" in lower or "residual block" in lower:
        dim = re.search(r"\((\d+)", text)
        parts = [f"Residual Block {dim.group(1)}" if dim else "Residual Block"]
        extras = []
        if "spectralnorm" in lower:
            extras.append("SpectralNorm")
        act = activation_name(text)
        if act:
            extras.append(act)
        if "dropout" in lower:
            extras.append("Dropout")
        if extras:
            parts.append(" + ".join(extras))
        return "\n".join(parts)

    if "hidden layer" in lower:
        idx = re.search(r"hidden layer\s*(\d+)", lower)
        units = re.search(r"(\d+)\s*units", lower)
        head = "Hidden"
        if idx:
            head += f" {idx.group(1)}"
        if units:
            head += f": {units.group(1)}"
        act = activation_name(text)
        return f"{head}\n{act}" if act else head

    if lower.startswith("output"):
        units = re.search(r"(\d+)\s*unit", lower)
        head = f"Output {units.group(1)}" if units else "Output"
        act = activation_name(text)
        return f"{head}\n{act}" if act else head

    linear = re.search(r"linear\((\d+)\s*,\s*(\d+)\)", lower)
    if linear:
        return f"Linear {linear.group(1)} -> {linear.group(2)}"

    dense = re.search(r"dense:\s*(\d+)", lower)
    if dense:
        return f"Linear {dense.group(1)}"

    norm = re.search(r"(batchnorm1d|layernorm)\((\d+)\)", lower)
    if norm:
        name = "BatchNorm" if "batch" in norm.group(1) else "LayerNorm"
        return f"{name}\n{norm.group(2)}"

    drop = re.search(r"dropout\(([^)]+)\)", lower)
    if drop:
        return f"Dropout\np={drop.group(1)}"

    act = activation_name(text)
    if act and len(text) <= 24:
        return act

    return wrap_label(text, width=18)


def model_type_summary(text, fallback):
    lower = str(text).lower()
    if "sb3" in lower:
        return "SB3 default MLP"
    if "spectral" in lower:
        return "Residual MLP + SpectralNorm"
    if "residual" in lower:
        return "Residual MLP"
    if "multilayer" in lower or "mlp" in lower:
        act = activation_name(text)
        if "linear output" in lower:
            return "MLP value head"
        if act:
            return f"MLP + {act}"
        return "MLP"
    return fallback


def baseline_design():
    return {
        "policy_type": "SB3 default MLP",
        "value_type": "SB3 default MLP",
        "policy_nodes": [
            {"label": "Input 27", "kind": "io"},
            {"label": "Linear 27 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Linear 64 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Mean 8", "kind": "linear"},
        ],
        "value_nodes": [
            {"label": "Input 27", "kind": "io"},
            {"label": "Linear 27 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Linear 64 -> 64", "kind": "linear"},
            {"label": "Tanh", "kind": "activation"},
            {"label": "Value 1", "kind": "linear"},
        ],
    }


def evom_g0_design():
    return {
        "policy_type": "Plain MLP",
        "value_type": "Plain MLP",
        "policy_nodes": [
            {"label": "Input 27", "kind": "io"},
            {"label": "Linear 27 -> 256", "kind": "linear"},
            {"label": "ReLU", "kind": "activation"},
            {"label": "Linear 256 -> 128", "kind": "linear"},
            {"label": "ReLU", "kind": "activation"},
            {"label": "Mean 8 (Tanh)", "kind": "linear"},
        ],
        "value_nodes": [
            {"label": "Input 27", "kind": "io"},
            {"label": "Linear 27 -> 128", "kind": "linear"},
            {"label": "ReLU", "kind": "activation"},
            {"label": "Linear 128 -> 64", "kind": "linear"},
            {"label": "ReLU", "kind": "activation"},
            {"label": "Value 1", "kind": "linear"},
        ],
    }


def evom_final_design():
    shared_stem = [
        {"label": "Input 27", "kind": "io"},
        {"label": "Linear 27 -> 256", "kind": "linear"},
        {"label": "BN + SiLU + Dropout(0.1)", "kind": "normalization"},
        {
            "label": "Residual Block 256",
            "kind": "residual",
            "expanded_residual": True,
            "dim": 256,
        },
        {"label": "Linear 256 -> 128", "kind": "linear"},
        {"label": "BN + SiLU + Dropout(0.1)", "kind": "normalization"},
        {"label": "Linear 128 -> 64", "kind": "linear"},
        {"label": "BN + SiLU + Dropout(0.1)", "kind": "normalization"},
    ]
    return {
        "policy_type": "Residual MLP + SpectralNorm",
        "value_type": "Residual MLP + SpectralNorm",
        "policy_nodes": shared_stem
        + [
            {"label": "Mean 8 (Tanh)", "kind": "linear"},
        ],
        "value_nodes": shared_stem
        + [
            {"label": "Value 1", "kind": "linear"},
        ],
    }


def get_evom_designs(evom_summary):
    final = evom_summary["elites"][0]
    g0 = evom_summary["elites"][1]
    return (
        final["label"],
        parse_design_string(final["network_design"]),
        g0["label"],
        parse_design_string(g0["network_design"]),
    )


def residual_dim(layer):
    if isinstance(layer, dict):
        if layer.get("dim") is not None:
            return int(layer["dim"])
        label = str(layer.get("label", ""))
    else:
        label = str(layer)
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 256


def unpack_layer(layer):
    if isinstance(layer, dict):
        raw_label = layer["label"]
        return {
            "raw_label": raw_label,
            "label": raw_label,
            "kind": layer.get("kind", layer_kind(raw_label)),
            "expanded_residual": bool(layer.get("expanded_residual")),
            "dim": residual_dim(layer),
        }

    raw_label = layer
    return {
        "raw_label": raw_label,
        "label": compact_layer_label(raw_label),
        "kind": layer_kind(raw_label),
        "expanded_residual": False,
        "dim": residual_dim(raw_label),
    }


def draw_box(ax, xcenter, ycenter, width, height, label, kind, fontsize=11.2, zorder=2):
    rect = Rectangle(
        (xcenter - width / 2, ycenter - height / 2),
        width,
        height,
        facecolor=COLORS[kind],
        edgecolor="#9aa5ad",
        linewidth=0.7,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(
        xcenter,
        ycenter,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        linespacing=0.95,
        transform=ax.transAxes,
        zorder=zorder + 1,
    )


def draw_vertical_arrow(ax, x, y0, y1, color="#9aa5ad", linewidth=0.7, scale=7, zorder=1):
    arrow = FancyArrowPatch(
        (x, y0),
        (x, y1),
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=linewidth,
        color=color,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_expanded_residual(ax, xcenter, ycenter, width, height, dim):
    inner_w = width * 0.90
    small_h = min(0.039, height * 0.18)
    arrow_color = "#9aa5ad"

    ys = [
        ycenter + height * 0.30,
        ycenter + height * 0.09,
        ycenter - height * 0.12,
        ycenter - height * 0.34,
    ]
    labels = [
        f"SN Linear {dim} -> {dim}",
        "SiLU",
        f"SN Linear {dim} -> {dim}",
        "Add + SiLU + Dropout",
    ]
    kinds = ["linear", "activation", "linear", "residual"]

    for y0, y1 in zip(ys[:-1], ys[1:]):
        draw_vertical_arrow(ax, xcenter, y0 - small_h / 2, y1 + small_h / 2, scale=7)

    for label, kind, y in zip(labels, kinds, ys):
        draw_box(
            ax,
            xcenter,
            y,
            inner_w,
            small_h,
            label,
            kind,
            fontsize=9.4,
            zorder=3,
        )

    x_skip = xcenter - inner_w / 2 - width * 0.07
    skip_start_y = ys[0] + small_h / 2 + height * 0.115
    add_y = ys[-1]
    ax.add_line(
        Line2D(
            [xcenter, x_skip],
            [skip_start_y, skip_start_y],
            color=arrow_color,
            linewidth=0.85,
            transform=ax.transAxes,
            zorder=4,
        )
    )
    ax.add_line(
        Line2D(
            [x_skip, x_skip],
            [skip_start_y, add_y],
            color=arrow_color,
            linewidth=0.85,
            transform=ax.transAxes,
            zorder=4,
        )
    )
    ax.text(
        x_skip - width * 0.018,
        (skip_start_y + add_y) / 2,
        "skip",
        ha="right",
        va="center",
        fontsize=8.6,
        color=arrow_color,
        transform=ax.transAxes,
        zorder=5,
    )
    skip_arrow = FancyArrowPatch(
        (x_skip, add_y),
        (xcenter - inner_w / 2 - width * 0.018, add_y),
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.85,
        color=arrow_color,
        shrinkA=0,
        shrinkB=0,
        transform=ax.transAxes,
        zorder=5,
    )
    ax.add_patch(skip_arrow)


def node_arrow_anchors(ycenter, height, spec):
    if not spec["expanded_residual"]:
        return ycenter + height / 2, ycenter - height / 2

    small_h = min(0.039, height * 0.18)
    first_y = ycenter + height * 0.30
    add_y = ycenter - height * 0.34
    return first_y + small_h / 2, add_y - small_h / 2


def draw_network_column(ax, xcenter, title, layers, kind_width=0.44):
    ax.text(
        xcenter,
        0.895,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )

    specs = [unpack_layer(layer) for layer in layers]
    n = len(specs)
    y_top = 0.835
    y_bottom = 0.035
    normal_h = 0.066 if n <= 7 else 0.049
    heights = [
        normal_h * 4.20 if spec["expanded_residual"] else normal_h
        for spec in specs
    ]
    span = y_top - y_bottom
    if n > 1:
        gap = (span - sum(heights)) / (n - 1)
        if gap < 0.010:
            scale = span / (sum(heights) + 0.010 * (n - 1))
            heights = [h * scale for h in heights]
            gap = 0.010 * scale
    else:
        gap = 0
    box_w = kind_width

    centers = []
    edge = y_top
    for spec, box_h in zip(specs, heights):
        y = edge - box_h / 2
        edge = edge - box_h - gap
        input_y, output_y = node_arrow_anchors(y, box_h, spec)
        centers.append((xcenter, y, box_h, input_y, output_y))

        if spec["expanded_residual"]:
            draw_expanded_residual(ax, xcenter, y, box_w, box_h, spec["dim"])
        else:
            draw_box(
                ax,
                xcenter,
                y,
                box_w,
                box_h,
                spec["label"],
                spec["kind"],
            )

    for (_, _, _, _, out_y), (_, _, _, in_y, _) in zip(centers[:-1], centers[1:]):
        draw_vertical_arrow(ax, xcenter, out_y, in_y)


def draw_model_panel(ax, panel_title, design):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.99,
        panel_title,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
    )

    policy = design.get("policy_net", {})
    value = design.get("value_net", {})
    if "policy_nodes" in design:
        policy_layers = design["policy_nodes"]
        value_layers = design["value_nodes"]
        policy_type = design.get("policy_type", "Policy design")
        value_type = design.get("value_type", "Value design")
    else:
        policy_layers = normalize_layer_config(policy.get("layer_config", []), "policy_net")
        value_layers = normalize_layer_config(value.get("layer_config", []), "value_net")
        policy_type = model_type_summary(policy.get("type", ""), "Policy design")
        value_type = model_type_summary(value.get("type", ""), "Value design")

    ax.text(
        0.25,
        0.935,
        wrap_label(policy_type, 28),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#666666",
        style="italic",
        transform=ax.transAxes,
    )
    ax.text(
        0.75,
        0.935,
        wrap_label(value_type, 28),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#666666",
        style="italic",
        transform=ax.transAxes,
    )

    draw_network_column(ax, 0.25, "PolicyNet", policy_layers)
    draw_network_column(ax, 0.75, "ValueNet", value_layers)


def plot_outer(ax, generations, history_best, elite_mean):
    ax.plot(
        generations,
        history_best,
        marker="o",
        linewidth=2.2,
        color=COLORS["outer_best"],
        label="Population historical best",
    )
    ax.plot(
        generations,
        elite_mean,
        marker="s",
        linewidth=2.2,
        color=COLORS["outer_mean"],
        label="Elite population mean",
    )
    ax.set_title("Outer Evolution Convergence", fontsize=16, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (100k-step PPO proxy)")
    ax.set_xlim(0, max(generations))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def plot_inner(ax, grid, aggregated):
    entries = [
        ("Manually Designed Baseline", "baseline"),
        ("EVOM G0 best", "g0"),
        ("EVOM final best", "evom"),
        ("Random search best", "random"),
        ("MLES", "mles"),
    ]

    for label_prefix, color_key in entries:
        actual_label = next(k for k in aggregated if k.startswith(label_prefix))
        color = COLORS[color_key]
        agg = aggregated[actual_label]
        ax.fill_between(
            grid / 1_000_000,
            agg["min"],
            agg["max"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            grid / 1_000_000,
            agg["mean"],
            color=color,
            linewidth=2.1,
            label=label_prefix,
        )

    ax.set_title("Inner Evaluation Convergence", fontsize=16, fontweight="bold")
    ax.set_xlabel("Aligned training budget (million env steps)")
    ax.set_ylabel("Historical best evaluation reward")
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def write_data_txt(path: Path, generations, history_best, elite_mean, grid, aggregated):
    with path.open("w", encoding="utf-8") as f:
        f.write("Ant main experiment visualization data\n")
        f.write("=" * 44 + "\n\n")

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

        f.write("\n[Inner aligned grid]\n")
        f.write("All methods are resampled to 0-5,000,000 env steps.\n")
        f.write("PPO curves use reward_curves.json eval rewards and cumulative best.\n")
        f.write(f"Random search uses experiment/{ANT_RANDOM_DIR}/final_selection/{ANT_RANDOM_LABEL}.\n")
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

    generations, history_best, elite_mean = load_outer_evolution(
        root / "experiment" / "ant_evom"
    )
    curve_sets, evom_summary = find_curves(root)
    final_label, _, g0_label, _ = get_evom_designs(evom_summary)

    grid = np.linspace(0, TOTAL_STEPS, 201)
    aggregated = {
        method: aggregate_curves(curves, grid)
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
        wspace=0.28,
    )

    ax_outer = fig.add_subplot(gs[0, 0:3])
    ax_inner = fig.add_subplot(gs[0, 3:6])
    ax_base = fig.add_subplot(gs[1, 0:2])
    ax_g0 = fig.add_subplot(gs[1, 2:4])
    ax_final = fig.add_subplot(gs[1, 4:6])

    plot_outer(ax_outer, generations, history_best, elite_mean)
    plot_inner(ax_inner, grid, aggregated)

    draw_model_panel(ax_base, "Manually Designed Baseline", baseline_design())
    draw_model_panel(ax_g0, f"EVOM Gen-0 Best ({g0_label})", evom_g0_design())
    draw_model_panel(ax_final, f"EVOM Final Best ({final_label})", evom_final_design())

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["linear"], edgecolor="#9aa5ad", label="Linear"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["activation"], edgecolor="#9aa5ad", label="Activation"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["normalization"], edgecolor="#9aa5ad", label="Normalization"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["dropout"], edgecolor="#9aa5ad", label="Dropout"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["residual"], edgecolor="#9aa5ad", label="Residual/Spectral"),
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

    fig.suptitle("Ant-v4 Main Experiment: EVOM vs Baselines", fontsize=20, fontweight="bold")
    fig.subplots_adjust(bottom=0.075, top=0.925, left=0.055, right=0.985)

    png_path = outdir / "ant_main_visualization.png"
    pdf_path = outdir / "ant_main_visualization.pdf"
    txt_path = outdir / "ant_main_visualization_data.txt"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    write_data_txt(txt_path, generations, history_best, elite_mean, grid, aggregated)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
