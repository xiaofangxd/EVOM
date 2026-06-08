# EVOM

<p align="center">
  <b>Agentic Meta-Evolution of Actor-Critic Architectures for Reinforcement Learning</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="RL" src="https://img.shields.io/badge/RL-PPO-green">
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-Ant%20%7C%20HalfCheetah-orange">
  <img alt="License" src="https://img.shields.io/badge/License-TBD-lightgrey">
</p>

EVOM is an LLM-guided meta-evolution framework for discovering executable actor-critic architectures for continuous-control reinforcement learning. The method uses an outer evolutionary loop to search over architecture programs and an inner PPO loop to evaluate each candidate with low-budget training before full-budget retraining of selected elites.

## Overview

| Component | Description |
| --- | --- |
| Search object | Executable `PolicyNet` and `ValueNet` actor-critic architecture programs |
| Design agent | LLM-guided initialization, mutation, and crossover |
| Inner evaluator | PPO training with low-budget fitness estimation |
| Final evaluation | Full-budget PPO retraining of selected elite architectures |
| Tasks | `Ant-v4` and `HalfCheetah-v4` |
| Comparisons | Manual PPO, random search, MLES-style direct program search, ablations, and LLM-backbone variants |

## Installation

```bash
conda create -n evom python=3.11
conda activate evom
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA driver if the default wheel is not suitable.

## API Keys

EVOM reads keys from environment variables. No API key is stored in the source code.

```bash
export DEEPSEEK_API_KEY="your_deepseek_key"
export AIBERM_API_KEY="your_aiberm_key"
export REFLECT_API_KEY="your_reflect_key"
```

| Profile | Provider key |
| --- | --- |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `gpt54mini`, `gemini35flash`, `claudeopus47` | `AIBERM_API_KEY` |
| `qwen36flash`, `qwen36plus` | `REFLECT_API_KEY` |

## Quick Start

Run the main EVOM experiments:

```bash
python evom.py --task ant --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task halfcheetah --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--task` | `ant` or `halfcheetah` |
| `--population-size` | Number of architectures retained per generation |
| `--generations` | Number of outer-loop generations |
| `--offspring-mode` | `mixed`, `mutation_only`, or `crossover_only` |
| `--llm-profile` | LLM backend profile |
| `--gpu-ids` | GPUs assigned to PPO workers |
| `--no-final-selection` | Skip full-budget elite retraining |

## Ablation Examples

```bash
python evom.py --task ant --population-size 16 --generations 20 --offspring-mode mutation_only --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 16 --generations 20 --offspring-mode crossover_only --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 8 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 32 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 16 --generations 20 --llm-profile claudeopus47 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
```

## Baselines

Manual PPO:

```bash
python baseline_parallel.py --task ant --seeds 42 123 456 --save-dir ./results/ant_baseline --timesteps 5000000 --eval-freq 25000 --n-eval-episodes 5 --gpu-parallel 3 --gpu-ids 1 2 3
python baseline_parallel.py --task halfcheetah --seeds 42 123 456 --save-dir ./results/halfcheetah_baseline --timesteps 5000000 --eval-freq 25000 --n-eval-episodes 5 --gpu-parallel 3 --gpu-ids 1 2 3
```

Random search:

```bash
python random_search.py --task ant --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python random_search.py --task halfcheetah --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
```

MLES-style direct program search requires the LLM4AD/MLES source version that contains `llm4ad.method.mles`. Add that source directory to `PYTHONPATH` before running:

```bash
export PYTHONPATH=/path/to/MLES:$PYTHONPATH
python mles_ant.py --total-env-steps 5000000 --pop-size 16 --num-samplers 8 --num-evaluators 8
python mles_halfcheetah.py --total-env-steps 5000000 --pop-size 16 --num-samplers 8 --num-evaluators 8
```

## Outputs

EVOM creates timestamped run directories under `runs/`.

| File | Content |
| --- | --- |
| `population_history.json` | Outer-loop architecture records and fitness values |
| `final_selection/elite_evaluation_summary.json` | Full-budget elite evaluation summary |
| `reward_curves.json` | PPO evaluation curves when available |
| candidate folders | Generated programs, logs, checkpoints, and evaluation artifacts |

## Repository Layout

```text
evom.py
random_search.py
baseline_parallel.py
baseline_ant_v4.py
baseline_halfcheetah.py
train_eval_worker_ant.py
train_eval_worker_halfcheetah.py
eval_elite_worker_ant.py
eval_elite_worker_halfcheetah.py
task_description_ant.py
task_description_halfcheetah.py
mles_ant.py
mles_halfcheetah.py
```

## Citation

If you use this repository, please cite the EVOM paper associated with this code release.
