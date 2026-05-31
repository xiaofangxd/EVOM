from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)

RED = "#d7191c"
BLUE = "#0758d6"
BLACK = "#111111"
LIGHT_BG = "#f7f7f7"


def add_segments(ax, x, y, segments, fontsize=6.7, family="Times New Roman"):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_width = ax.get_window_extent(renderer=renderer).width
    cursor = x

    for text, color, weight in segments:
        artist = ax.text(
            cursor,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontfamily=family,
            fontweight=weight,
            color=color,
        )
        fig.canvas.draw()
        cursor += artist.get_window_extent(renderer=renderer).width / axes_width


def draw_template(output_dir: Path, filename: str, title: str, blocks: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.55, 2.55), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box = FancyBboxPatch(
        (0.04, 0.115),
        0.92,
        0.84,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.15,
        edgecolor="#7f7f7f",
        facecolor=LIGHT_BG,
    )
    ax.add_patch(box)

    x = 0.08
    y = 0.90
    line = 0.052
    gap = 0.020
    family = "Times New Roman"
    normal = 6.65
    small = 6.35

    ax.text(
        x,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontfamily=family,
        fontweight="bold",
        color=BLACK,
    )
    y -= line * 1.25

    def plain(text, color=BLACK, size=normal, weight="normal", indent=0.0):
        nonlocal y
        ax.text(
            x + indent,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=size,
            fontfamily=family,
            fontweight=weight,
            color=color,
        )
        y -= line

    def rich(parts, size=normal):
        nonlocal y
        add_segments(ax, x, y, parts, fontsize=size, family=family)
        y -= line

    for block in blocks:
        kind = block["kind"]
        if kind == "gap":
            y -= gap
        elif kind == "heading":
            plain(block["text"], weight="bold", size=7.25)
        elif kind == "plain":
            plain(
                block["text"],
                color=block.get("color", BLACK),
                size=block.get("size", small),
                weight=block.get("weight", "normal"),
                indent=block.get("indent", 0.0),
            )
        elif kind == "rich":
            rich(block["parts"], size=block.get("size", small))
        else:
            raise ValueError(f"Unknown block kind: {kind}")

    pdf_path = output_dir / f"{filename}.pdf"
    png_path = output_dir / f"{filename}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def common_header() -> list[dict]:
    return [
        {
            "kind": "rich",
            "parts": [
                ("Environment: ", BLACK, "normal"),
                ("{ENV_DESCRIPTION}", RED, "bold"),
            ],
        },
        {
            "kind": "rich",
            "parts": [
                ("State/action info: ", BLACK, "normal"),
                ("{FIXED_STATE_ACTION}", RED, "bold"),
            ],
        },
        {
            "kind": "rich",
            "parts": [
                ("Required output: ", BLACK, "normal"),
                ("JSON design", RED, "bold"),
                (" + ", BLACK, "normal"),
                ("complete Python code", RED, "bold"),
            ],
        },
    ]


def initialization_blocks() -> list[dict]:
    return common_header() + [
        {"kind": "gap"},
        {"kind": "heading", "text": "Initialization prompt"},
        {"kind": "plain", "text": "Design both actor and critic networks.", "indent": 0.0},
        {"kind": "plain", "text": "- encourage innovative PolicyNet / ValueNet", "color": BLUE, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "- continuous-control, high-dimensional observations", "color": BLUE, "weight": "bold", "indent": 0.035},
        {"kind": "gap"},
        {"kind": "heading", "text": "Output constraints"},
        {"kind": "plain", "text": "1. {network_design_json template}", "color": RED, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "2. {PolicyNet, ValueNet code template}", "color": RED, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "3. no training logic or shared parameters", "indent": 0.035},
    ]


def mutation_blocks() -> list[dict]:
    return common_header() + [
        {"kind": "gap"},
        {"kind": "heading", "text": "Mutation prompt"},
        {
            "kind": "rich",
            "parts": [
                ("Current architecture: ", BLACK, "normal"),
                ("{ind.label}", BLUE, "bold"),
            ],
        },
        {"kind": "plain", "text": "- {ind.network_design}", "color": BLUE, "weight": "bold", "indent": 0.035},
        {"kind": "gap"},
        {"kind": "heading", "text": "Mutation task"},
        {"kind": "plain", "text": "Analyze PolicyNet and ValueNet characteristics.", "indent": 0.0},
        {"kind": "plain", "text": "- modify either or both networks", "color": RED, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "- mutated architecture must differ from parent", "color": RED, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "Return JSON design and complete runnable code.", "color": BLUE, "weight": "bold", "indent": 0.035},
    ]


def crossover_blocks() -> list[dict]:
    return common_header() + [
        {"kind": "gap"},
        {"kind": "heading", "text": "Crossover prompt"},
        {"kind": "plain", "text": "Parent 1: {parent1.label}", "color": BLUE, "weight": "bold", "indent": 0.0},
        {"kind": "plain", "text": "- {parent1.network_design}", "color": BLUE, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "Parent 2: {parent2.label}", "color": BLUE, "weight": "bold", "indent": 0.0},
        {"kind": "plain", "text": "- {parent2.network_design}", "color": BLUE, "weight": "bold", "indent": 0.035},
        {"kind": "gap"},
        {"kind": "heading", "text": "Crossover task"},
        {"kind": "plain", "text": "Analyze strengths of both actor-critic pairs.", "indent": 0.0},
        {"kind": "plain", "text": "- fuse complementary architectural features", "color": RED, "weight": "bold", "indent": 0.035},
        {"kind": "plain", "text": "Return one coherent PolicyNet / ValueNet pair.", "color": BLUE, "weight": "bold", "indent": 0.035},
    ]


def draw_all(output_dir: Path) -> None:
    draw_template(output_dir, "prompt_initialization_template", "Initialization Prompt", initialization_blocks())
    draw_template(output_dir, "prompt_mutation_template", "Mutation Prompt", mutation_blocks())
    draw_template(output_dir, "prompt_crossover_template", "Crossover Prompt", crossover_blocks())
    draw_combined_template(output_dir)


def draw_code_line(ax, x: float, y: float, text: str, color: str = BLACK, weight: str = "normal", fontsize: float = 4.35):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontfamily="Times New Roman",
        fontweight=weight,
        color=color,
    )


def draw_two_columns(
    ax,
    left_x: float,
    right_x: float,
    y: float,
    left: tuple[str, str, str],
    right: tuple[str, str, str],
    fontsize: float = 3.95,
) -> None:
    draw_code_line(ax, left_x, y, left[0], color=left[1], weight=left[2], fontsize=fontsize)
    draw_code_line(ax, right_x, y, right[0], color=right[1], weight=right[2], fontsize=fontsize)


def draw_panel(ax, x: float, y_top: float, w: float, h: float, title: str, lines: list):
    panel = FancyBboxPatch(
        (x, y_top - h),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.0,
        edgecolor="#7f7f7f",
        facecolor=LIGHT_BG,
    )
    ax.add_patch(panel)

    y = y_top - 0.021
    ax.text(
        x + 0.025,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        fontfamily="Times New Roman",
        fontweight="bold",
        color=BLACK,
    )
    y -= 0.037
    for item in lines:
        if len(item) == 3 and isinstance(item[0], str):
            text, color, weight = item
            draw_code_line(ax, x + 0.026, y, text, color=color, weight=weight, fontsize=3.95)
        else:
            draw_two_columns(ax, x + 0.026, x + w * 0.52, y, item[0], item[1])
        y -= 0.023


def draw_combined_template(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.55, 6.80), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.992,
        "EVOM Prompt Templates Used in Experiments",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        fontfamily="Times New Roman",
        fontweight="bold",
        color=BLACK,
    )

    line_step = 0.0120

    def panel(title: str, y_top: float, h: float, lines: list[tuple[str, str, str, float]]) -> None:
        box = FancyBboxPatch(
            (0.045, y_top - h),
            0.91,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=1.0,
            edgecolor="#7f7f7f",
            facecolor=LIGHT_BG,
        )
        ax.add_patch(box)
        y = y_top - 0.014
        draw_code_line(ax, 0.065, y, title, weight="bold", fontsize=5.1)
        y -= 0.024
        for text, color, weight, indent in lines:
            draw_code_line(ax, 0.065 + indent, y, text, color=color, weight=weight, fontsize=3.65)
            y -= line_step

    init_lines = [
        ("System: You are an expert in agent architecture design, specializing in actor-critic", BLACK, "bold", 0.0),
        ("neural networks for {TASK_NAME} tasks.", RED, "bold", 0.020),
        ("Output must include: JSON describing both PolicyNet and ValueNet, plus complete", BLACK, "normal", 0.020),
        ("runnable code defining BOTH the PolicyNet and ValueNet classes.", BLACK, "normal", 0.020),
        ("User: Design a neural network policy AND value network for the {TASK_NAME}", BLACK, "bold", 0.0),
        ("environment (PPO actor-critic).", BLACK, "bold", 0.020),
        ("[Environment Description]", BLACK, "bold", 0.0),
        ("{ENV_DESCRIPTION}", RED, "bold", 0.020),
        ("{FIXED_STATE_ACTION}", RED, "bold", 0.020),
        ("[Design Principle]", BLACK, "bold", 0.0),
        ("Encourage innovative network architecture design for BOTH actor (PolicyNet) and critic (ValueNet).", BLACK, "normal", 0.020),
        ("Note: continuous control with high-dimensional observation space.", BLACK, "normal", 0.020),
        ("[Output Requirements]", BLACK, "bold", 0.0),
        ("Please output two parts:", BLACK, "normal", 0.020),
        ("1. Network structure description (JSON format): {network_design_json}", RED, "bold", 0.020),
        ("2. Complete network class code (Python, ready to run):", RED, "bold", 0.020),
        ("{network_code_template}", RED, "bold", 0.045),
        ("{network_code_notes}", RED, "bold", 0.045),
    ]
    mutation_lines = [
        ("System: You are an expert in actor-critic neural network architecture design for continuous control.", BLACK, "bold", 0.0),
        ("Analyze the given PolicyNet and ValueNet pair and create a meaningful mutation.", BLACK, "normal", 0.020),
        ("User: Analyze and mutate the following {TASK_NAME} actor-critic architecture (PolicyNet + ValueNet).", BLACK, "bold", 0.0),
        ("[Environment Description]", BLACK, "bold", 0.0),
        ("{ENV_DESCRIPTION}", RED, "bold", 0.020),
        ("{FIXED_STATE_ACTION}", RED, "bold", 0.020),
        ("[Current Architecture - {ind.label}]", BLUE, "bold", 0.0),
        ("{ind.network_design}", BLUE, "bold", 0.020),
        ("[Mutation Task]", BLACK, "bold", 0.0),
        ("1. Analyze both the PolicyNet and ValueNet of the current architecture and identify their characteristics.", BLACK, "normal", 0.020),
        ("2. Based on actor-critic design for continuous control, enhance or modify", RED, "bold", 0.020),
        ("EITHER OR BOTH networks.", RED, "bold", 0.045),
        ("3. IMPORTANT: the mutated architecture MUST be different from original.", RED, "bold", 0.020),
        ("[Output Requirements]", BLACK, "bold", 0.0),
        ("1. Network structure description (JSON format): {network_design_json}", RED, "bold", 0.020),
        ("2. Complete code defining BOTH classes (Python):", RED, "bold", 0.020),
        ("{network_code_template}", RED, "bold", 0.045),
        ("{network_code_notes}", RED, "bold", 0.045),
    ]
    crossover_lines = [
        ("System: You are an expert in actor-critic neural network architecture design for continuous control.", BLACK, "bold", 0.0),
        ("Analyze both parent architectures (PolicyNet + ValueNet) and create an intelligent fusion.", BLACK, "normal", 0.020),
        ("User: Analyze and crossover the following two {TASK_NAME} actor-critic architectures", BLACK, "bold", 0.0),
        ("(each contains PolicyNet + ValueNet).", BLACK, "bold", 0.020),
        ("[Environment Description]", BLACK, "bold", 0.0),
        ("{ENV_DESCRIPTION}", RED, "bold", 0.020),
        ("{FIXED_STATE_ACTION}", RED, "bold", 0.020),
        ("[Parent 1 - {parent1.label}]", BLUE, "bold", 0.0),
        ("{parent1.network_design}", BLUE, "bold", 0.020),
        ("[Parent 2 - {parent2.label}]", BLUE, "bold", 0.0),
        ("{parent2.network_design}", BLUE, "bold", 0.020),
        ("[Crossover Task]", BLACK, "bold", 0.0),
        ("1. Analyze the strengths of both parents' PolicyNet and ValueNet.", BLACK, "normal", 0.020),
        ("2. Based on actor-critic design for continuous control, intelligently fuse", RED, "bold", 0.020),
        ("architectural features from both parents.", RED, "bold", 0.045),
        ("3. Create a coherent new (PolicyNet, ValueNet) pair that combines complementary strengths.", BLACK, "normal", 0.020),
        ("[Output Requirements]", BLACK, "bold", 0.0),
        ("1. Network structure description (JSON format): {network_design_json}", RED, "bold", 0.020),
        ("2. Complete code defining BOTH classes (Python):", RED, "bold", 0.020),
        ("{network_code_template}", RED, "bold", 0.045),
        ("{network_code_notes}", RED, "bold", 0.045),
    ]

    panel("(a) Initialization Prompt", 0.955, 0.260, init_lines)
    panel("(b) Mutation Prompt", 0.650, 0.275, mutation_lines)
    panel("(c) Crossover Prompt", 0.325, 0.300, crossover_lines)

    pdf_path = output_dir / "prompt_templates_combined.pdf"
    png_path = output_dir / "prompt_templates_combined.png"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.015)
    try:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    except PermissionError:
        pdf_path = output_dir / "prompt_templates_combined_updated.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("evom_overleaf_new") / "figures",
        help="Directory for prompt template pdf/png files",
    )
    args = parser.parse_args()
    draw_all(args.outdir)


if __name__ == "__main__":
    main()
