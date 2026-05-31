from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiment"
OUT = ROOT / "figures" / "current_result_tables.md"
MLES_BUDGET = 1667
ANT_RANDOM_DIR = "ant_random_20260529_003443"
ANT_RANDOM_LABEL = "G3_I9"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f}\u00b1{std:.2f}"


def baseline_result(path: Path) -> tuple[float, float]:
    data = load_json(path)
    rewards = np.asarray(data["rewards"], dtype=float)
    return float(np.mean(rewards)), float(np.std(rewards))


def elite_json_result(path: Path, index: int = 0) -> tuple[float, float]:
    elite = load_json(path)["elites"][index]
    return float(elite["final_mean_reward"]), float(elite["final_std_reward"])


def mles_result(pattern: str) -> tuple[float, float]:
    best_scores = []
    for path in sorted(EXP.glob(pattern)):
        best = -float("inf")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                match = re.match(r"C(\d+)$", str(item.get("candidate_id", "")))
                if not match:
                    continue
                if int(match.group(1)) > MLES_BUDGET:
                    continue
                best = max(best, float(item["score"]))
        if np.isfinite(best):
            best_scores.append(best)
    arr = np.asarray(best_scores, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def log_result(path: Path, label: str | None = None) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"(G\d+_I\d+): final_mean_reward=([-+]?\d+(?:\.\d+)?) "
        r"\(std=([-+]?\d+(?:\.\d+)?)\)",
        text,
    )
    if not matches:
        raise ValueError(f"No final_mean_reward records found in {path}")
    if label:
        for got_label, mean, std in reversed(matches):
            if got_label == label:
                return float(mean), float(std)
        raise ValueError(f"No label {label} in {path}")
    got_label, mean, std = matches[-2] if matches[-1][0].startswith("G0_") and len(matches) >= 2 else matches[-1]
    return float(mean), float(std)


def write_table(f, title: str, headers: list[str], rows: list[list[str]]):
    f.write(f"## {title}\n\n")
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(row) + " |\n")
    f.write("\n")


def main():
    ant_main = {
        "PPO": baseline_result(EXP / "ant-PPO" / "baseline" / "baseline_summary.json"),
        "MLES": mles_result("ant_mles/evaluations*.jsonl"),
        "Random Search": elite_json_result(
            EXP / ANT_RANDOM_DIR / "final_selection" / "elite_evaluation_summary.json",
            index=0,
        ),
        "EVOM": elite_json_result(EXP / "ant_evom" / "final_selection" / "elite_evaluation_summary.json"),
    }
    half_main = {
        "PPO": baseline_result(EXP / "half_baseline" / "baseline_summary.json"),
        "MLES": mles_result("half_mles/evaluations*.jsonl"),
        "Random Search": log_result(EXP / "half_rs2" / "run.log", "G19_I3"),
        "EVOM": log_result(EXP / "half_evom" / "run.log", "G15_I15"),
    }

    ant_ablation = {
        "Mixed EVOM (N=16)": ant_main["EVOM"],
        "Mutation only": log_result(EXP / "ant_mutation_2.log", "G11_I12"),
        "Crossover only": log_result(EXP / "ant_cross3.log", "G17_I4"),
        "Population N=8": elite_json_result(EXP / "ant_20260525_034124" / "final_selection" / "elite_evaluation_summary.json"),
        "Population N=32": log_result(EXP / "ant_pop32_3.log", "G14_I30"),
    }
    half_ablation = {
        "Mixed EVOM (N=16)": half_main["EVOM"],
        "Mutation only": log_result(EXP / "half_mutation.log", "G19_I4"),
        "Crossover only": log_result(EXP / "half_crossover.log", "G3_I12"),
        "Population N=8": log_result(EXP / "half_pop8.log", "G3_I3"),
        "Population N=32": log_result(EXP / "half_pop32.log", "G12_I0"),
    }

    ant_llm = {
        "DeepSeek": ant_main["EVOM"],
        "Claude Opus 4.7": log_result(EXP / "antclaude.log", "G16_I8"),
        "Qwen3.6 Plus": log_result(EXP / "ant_qwenplus.log", "G12_I9"),
    }
    half_llm = {
        "DeepSeek": half_main["EVOM"],
        "Claude Opus 4.7": log_result(EXP / "half_claude.log", "G3_I6"),
        "Qwen3.6 Plus": log_result(EXP / "half_qwenplus.log", "G4_I8"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("# Current Ant/HalfCheetah Result Tables\n\n")
        f.write("Values are final full-budget rewards reported as mean\u00b1std over three runs.\n\n")

        write_table(
            f,
            "Main Comparison",
            ["Method", "Ant-v4", "HalfCheetah-v4"],
            [[name, fmt(*ant_main[name]), fmt(*half_main[name])] for name in ant_main],
        )

        write_table(
            f,
            "EVOM Ablations",
            ["Variant", "Ant-v4", "HalfCheetah-v4"],
            [[name, fmt(*ant_ablation[name]), fmt(*half_ablation[name])] for name in ant_ablation],
        )

        write_table(
            f,
            "LLM Backbone Comparison",
            ["LLM", "Ant-v4", "HalfCheetah-v4"],
            [[name, fmt(*ant_llm[name]), fmt(*half_llm[name])] for name in ant_llm],
        )

    print(OUT)
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
