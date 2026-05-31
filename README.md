# EVOM

EVOM is an LLM-guided meta-evolution framework for discovering actor-critic architectures for continuous-control reinforcement learning. It treats architecture search as a bi-level process: an outer evolutionary loop uses an LLM design agent to initialize, mutate, and crossover executable actor-critic programs, while an inner PPO loop evaluates each candidate with low-budget training and retrains selected elites with the full budget.

The current code supports Ant-v4 and HalfCheetah-v4, with manually designed PPO, random search, MLES-style direct program search, reproduction-operator ablations, population-size sensitivity, and LLM-backbone comparisons.

## Installation

```bash
conda create -n evom python=3.11
conda activate evom
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA driver if the default wheel is not suitable.

## API Keys

EVOM reads API keys from environment variables. No key is stored in the source code.

```bash
export DEEPSEEK_API_KEY="your_deepseek_key"
export AIBERM_API_KEY="your_aiberm_key"
export REFLECT_API_KEY="your_reflect_key"
```

Default EVOM and random-search runs use DeepSeek. The AIBERM key is used by `gpt54mini`, `gemini35flash`, and `claudeopus47`; the REFLECT key is used by `qwen36flash` and `qwen36plus`.

## Main EVOM Runs

```bash
python evom.py --task ant --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task halfcheetah --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
```

## Ablations

```bash
python evom.py --task ant --population-size 16 --generations 20 --offspring-mode mutation_only --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 16 --generations 20 --offspring-mode crossover_only --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 8 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 32 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python evom.py --task ant --population-size 16 --generations 20 --llm-profile claudeopus47 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
```

Available LLM profiles are `deepseek`, `gpt54mini`, `gemini35flash`, `claudeopus47`, `qwen36flash`, and `qwen36plus`.

## Baselines

```bash
python baseline_parallel.py --task ant --seeds 42 123 456 --save-dir ./results/ant_baseline --timesteps 5000000 --eval-freq 25000 --n-eval-episodes 5 --gpu-parallel 3 --gpu-ids 1 2 3
python baseline_parallel.py --task halfcheetah --seeds 42 123 456 --save-dir ./results/halfcheetah_baseline --timesteps 5000000 --eval-freq 25000 --n-eval-episodes 5 --gpu-parallel 3 --gpu-ids 1 2 3
python random_search.py --task ant --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
python random_search.py --task halfcheetah --population-size 16 --generations 20 --cpu-parallel 48 --eval-rounds 3 --final-runs 3 --gpu-ids 1 2 3
```

The MLES comparison requires the LLM4AD/MLES source version that contains `llm4ad.method.mles`. Add that source directory to `PYTHONPATH` before running:

```bash
export PYTHONPATH=/path/to/MLES:$PYTHONPATH
python mles_ant.py --total-env-steps 5000000 --pop-size 16 --num-samplers 8 --num-evaluators 8
python mles_halfcheetah.py --total-env-steps 5000000 --pop-size 16 --num-samplers 8 --num-evaluators 8
```

## Outputs

EVOM writes each run to a timestamped directory under `runs/`. Important files include:

- `population_history.json`: outer-loop candidate records and fitness values
- `final_selection/elite_evaluation_summary.json`: full-budget elite evaluation
- `reward_curves.json`: PPO evaluation curves when available
- generated candidate programs and training logs under each run directory

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
plot_ant_main_visualization.py
plot_halfcheetah_main_visualization.py
plot_prompt_template.py
summarize_current_results.py
```

## Citation

If you use this repository, please cite the EVOM paper associated with this code release.
