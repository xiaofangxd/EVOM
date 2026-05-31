
import os
import sys
import json
import time
import random
import subprocess
import requests
import numpy as np
import argparse
import importlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


TASK_NAME = None
ENV_DESCRIPTION = None
FIXED_STATE_ACTION = None
STATE_DIM = None
ACTION_DIM = None
ACTION_TYPE = None
EVAL_SEEDS = None
MAX_EPISODE_STEPS = None
TRAIN_CONFIG = None
FULL_TRAIN_CONFIG = None






IDEA_API_KEY = ""
IDEA_MODEL = "deepseek-v4-flash"
IDEA_API_URL = "https://api.deepseek.com/chat/completions"
LLM_PROFILE = "deepseek"

LLM_PROFILES = {
    "deepseek": {
        "run_suffix": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-v4-flash",
        "api_url": "https://api.deepseek.com/chat/completions",
    },
    "gpt54mini": {
        "run_suffix": "gpt54mini",
        "api_key_env": "AIBERM_API_KEY",
        "model": "openai/gpt-5.4-mini",
        "api_url": "https://aiberm.com/v1/chat/completions",
    },
    "gemini35flash": {
        "run_suffix": "gemini35flash",
        "api_key_env": "AIBERM_API_KEY",
        "model": "gemini-3.5-flash",
        "api_url": "https://aiberm.com/v1/chat/completions",
    },
    "claudeopus47": {
        "run_suffix": "claudeopus47",
        "api_key_env": "AIBERM_API_KEY",
        "model": "claude-opus-4-7",
        "api_url": "https://aiberm.com/v1/chat/completions",
    },
    "qwen36flash": {
        "run_suffix": "qwen36flash",
        "api_key_env": "REFLECT_API_KEY",
        "model": "qwen3.6-flash",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
    "qwen36plus": {
        "run_suffix": "qwen36plus",
        "api_key_env": "REFLECT_API_KEY",
        "model": "qwen3.6-plus",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
}

LLM_PROFILE_ALIASES = {
    "gpt-5.4-mini": "gpt54mini",
    "openai/gpt-5.4-mini": "gpt54mini",
    "gemini-3.5-flash": "gemini35flash",
    "claude-opus-4-7": "claudeopus47",
    "qwen3.6-flash": "qwen36flash",
    "qwen3.6-plus": "qwen36plus",
}

POPULATION_SIZE = 16
NUM_GENERATIONS = 20
MAX_WORKERS = 64
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_TOKENS = 8192
API_MAX_RETRIES = 3
OFFSPRING_MODE = "mixed"



CPU_PARALLEL = 48
NUM_EVAL_ROUNDS = 3
WORKER_SCRIPT = None


GPU_IDS = [4, 5, 6]
RUN_FINAL_SELECTION = True
ELITE_EVAL_WORKER = None
NUM_FINAL_RUNS = 3
FINAL_PARALLEL = len(GPU_IDS) * 16



TASK_CONFIGS = {
    "ant": {
        "module": "task_description_ant",
        "worker": "train_eval_worker_ant.py",
        "elite_eval_worker": "eval_elite_worker_ant.py",
        "run_prefix": "ant",
    },
    "halfcheetah": {
        "module": "task_description_halfcheetah",
        "worker": "train_eval_worker_halfcheetah.py",
        "elite_eval_worker": "eval_elite_worker_halfcheetah.py",
        "run_prefix": "halfcheetah",
    },
}


def load_task_config(task_name: str):
    global TASK_NAME, ENV_DESCRIPTION, FIXED_STATE_ACTION
    global STATE_DIM, ACTION_DIM, ACTION_TYPE
    global EVAL_SEEDS, MAX_EPISODE_STEPS, TRAIN_CONFIG, FULL_TRAIN_CONFIG
    global WORKER_SCRIPT, ELITE_EVAL_WORKER

    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}")

    config = TASK_CONFIGS[task_name]


    import importlib
    task_module = importlib.import_module(config["module"])

    TASK_NAME = task_module.TASK_NAME
    ENV_DESCRIPTION = task_module.ENV_DESCRIPTION
    FIXED_STATE_ACTION = task_module.FIXED_STATE_ACTION
    STATE_DIM = task_module.STATE_DIM
    ACTION_DIM = task_module.ACTION_DIM
    ACTION_TYPE = task_module.ACTION_TYPE
    EVAL_SEEDS = task_module.EVAL_SEEDS
    MAX_EPISODE_STEPS = task_module.MAX_EPISODE_STEPS
    TRAIN_CONFIG = task_module.TRAIN_CONFIG
    FULL_TRAIN_CONFIG = task_module.FULL_TRAIN_CONFIG


    WORKER_SCRIPT = os.path.join(PROJECT_DIR, config["worker"])
    ELITE_EVAL_WORKER = os.path.join(PROJECT_DIR, config["elite_eval_worker"])

    return config["run_prefix"]


def get_network_code_template():
    return f"""```python
import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, obs_length: int, act_length: int):
        super().__init__()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass


class ValueNet(nn.Module):
    def __init__(self, obs_length: int):
        super().__init__()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass
```"""

NETWORK_CODE_NOTES = """Notes:
- Class names MUST be exactly `PolicyNet` and `ValueNet` (both required).
- PolicyNet.__init__ takes (obs_length, act_length); ValueNet.__init__ takes (obs_length).
- forward signatures must match the template above.
- PolicyNet outputs continuous action means (will be used with Normal distribution).
- Do NOT include any training logic, only the two nn.Module class definitions.
- Do NOT share parameters between the two networks (they will be optimized jointly by PPO)."""

NETWORK_DESIGN_JSON = """{{
  "policy_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }},
  "value_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }}
}}"""



class Logger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, "run.log")
        self._f = open(self.log_path, "w", encoding="utf-8")
        self._call_count = 0

    def log(self, msg: str, also_print: bool = True):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._f.write(line + "\n")
        self._f.flush()
        if also_print:
            print(msg)

    def log_llm_call(self, step: str, system: str, prompt: str, response: str,
                     elapsed: float, thinking_len: int = 0):
        self._call_count += 1
        sep = "=" * 60
        self._f.write(f"\n{sep}\n")
        self._f.write(f"LLM调用 #{self._call_count} | 步骤: {step} | 耗时: {elapsed:.1f}s")
        if thinking_len:
            self._f.write(f" | 思考: {thinking_len}字")
        self._f.write(f" | 输出: {len(response)}字\n")
        self._f.write(f"{sep}\n")
        self._f.write(f"[System Prompt]\n{system}\n\n")
        self._f.write(f"[User Prompt]\n{prompt}\n\n")
        self._f.write(f"[Response]\n{response}\n\n")
        self._f.flush()

    def close(self):
        self._f.close()


LOG: Logger = None
RUN_DIR: str = ""



def configure_llm_profile(profile_name: str) -> dict:
    global IDEA_API_KEY, IDEA_MODEL, IDEA_API_URL, LLM_PROFILE

    canonical_name = LLM_PROFILE_ALIASES.get(profile_name, profile_name)
    if canonical_name not in LLM_PROFILES:
        available = sorted(set(LLM_PROFILES) | set(LLM_PROFILE_ALIASES))
        raise ValueError(f"Unknown LLM profile: {profile_name}. Available: {available}")

    profile = LLM_PROFILES[canonical_name]
    api_key = os.environ.get(profile["api_key_env"], "").strip()
    if not api_key:
        raise ValueError(
            f"LLM profile '{canonical_name}' requires environment variable "
            f"{profile['api_key_env']}. Do not put API keys in source code."
        )

    IDEA_API_KEY = api_key
    IDEA_MODEL = profile["model"]
    IDEA_API_URL = profile["api_url"]
    LLM_PROFILE = profile["run_suffix"]
    return profile


def call_llm(system_prompt: str, user_prompt: str, thinking: bool = False, step: str = "",
             api_key: str = "", model: str = "", api_url: str = "") -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
    }

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            if not thinking:
                t0 = time.time()
                resp = requests.post(api_url, headers=headers, json=data, timeout=300)
                elapsed = time.time() - t0
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    LOG.log(f"      (LLM {elapsed:.1f}s, {len(content)}字)")
                    LOG.log_llm_call(step, system_prompt, user_prompt, content, elapsed)
                    return content
                raise Exception(f"API失败({elapsed:.1f}s): {resp.status_code}, {resp.text[:200]}")

            req_data = dict(data)
            req_data["enable_thinking"] = True
            req_data["stream"] = True
            t0 = time.time()
            thinking_text = ""
            content_text = ""
            phase = "waiting"
            with requests.post(api_url, headers=headers, json=req_data, stream=True, timeout=600) as resp:
                if resp.status_code != 200:
                    raise Exception(f"API失败: {resp.status_code}, {resp.text[:200]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        if not chunk.get("choices"):
                            continue
                        delta = chunk["choices"][0]["delta"]
                        rc = delta.get("reasoning_content", "")
                        cc = delta.get("content", "")
                        if rc:
                            if phase != "thinking":
                                phase = "thinking"
                                print(f"      (thinking中...)", end="", flush=True)
                            thinking_text += rc
                        if cc:
                            if phase != "content":
                                phase = "content"
                                print(f" {time.time()-t0:.0f}s, {len(thinking_text)}字思考)")
                            content_text += cc
                    except json.JSONDecodeError:
                        pass
            elapsed = time.time() - t0
            LOG.log(f"      (LLM+思考 {elapsed:.1f}s, 思考{len(thinking_text)}字, 输出{len(content_text)}字)")
            LOG.log_llm_call(step, system_prompt, user_prompt, content_text, elapsed, len(thinking_text))
            return content_text

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            LOG.log(f"      ⚠ API调用失败(第{attempt}/{API_MAX_RETRIES}次): {type(e).__name__}")
            if attempt >= API_MAX_RETRIES:
                raise Exception(f"API调用{API_MAX_RETRIES}次均失败: {e}")
            wait = 10 * attempt
            LOG.log(f"      等待{wait}秒后重试...")
            time.sleep(wait)


def call_idea_llm(system_prompt: str, user_prompt: str, step: str = "") -> str:
    return call_llm(system_prompt, user_prompt, thinking=False, step=step,
                    api_key=IDEA_API_KEY, model=IDEA_MODEL, api_url=IDEA_API_URL)


def extract_python_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def extract_json_from_response(response: str) -> str:
    start_idx = response.find('{')
    if start_idx == -1:
        return "Network structure description not found"

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(response)):
        char = response[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response[start_idx:i+1]

    return "Network structure description not found"


def _build_cpu_env(extra: dict = None) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTORCH_NUM_THREADS"] = "1"
    if extra:
        env.update(extra)
    return env


def _build_gpu_env(gpu_id: int, extra: dict = None) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTORCH_NUM_THREADS"] = "1"
    if extra:
        env.update(extra)
    return env


def _wrap_with_taskset(cmd: list[str], core_id: int) -> list[str]:
    if PIN_CORES and sys.platform.startswith("linux"):
        return ["taskset", "-c", str(core_id)] + cmd
    return cmd



class Individual:
    def __init__(self, idx: int, gen: int):
        self.idx = idx
        self.gen = gen
        self.network_design: str = ""
        self.network_code: str = ""
        self.fitness: float = 0.0
        self.sim_results: dict = {}
        self.label: str = f"G{gen}_I{idx}"

    def summary(self) -> str:
        sr = self.sim_results
        if "error" in sr:
            return (
                f"个体标识: {self.label}\n"
                f"适应度(mean_reward): {self.fitness:.2f}\n"
                f"评估失败: {sr.get('error', 'unknown')}"
            )
        return (
            f"个体标识: {self.label}\n"
            f"适应度(平均奖励): {self.fitness:.2f}\n"
            f"评估轮次: {sr.get('num_rounds', 'N/A')}\n"
            f"标准差: {sr.get('std_reward', 0.0):.2f}"
        )

    def detailed_summary(self) -> str:
        basic = self.summary()
        design = f"\n\n【网络结构设计】\n{self.network_design}\n"
        return basic + design



def llm_init_individual(idx: int, gen: int) -> Individual:
    ind = Individual(idx, gen)

    network_code_template = get_network_code_template()
    network_design_json = """{{
  "policy_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }},
  "value_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }}
}}"""

    network_code_notes = """Notes:
- Class names MUST be exactly `PolicyNet` and `ValueNet` (both required).
- PolicyNet.__init__ takes (obs_length, act_length); ValueNet.__init__ takes (obs_length).
- forward signatures must match the template above.
- PolicyNet outputs continuous action means (will be used with Normal distribution).
- Do NOT include any training logic, only the two nn.Module class definitions.
- Do NOT share parameters between the two networks (they will be optimized jointly by PPO)."""

    prompt = f"""Design a neural network policy AND value network for the {TASK_NAME} environment (PPO actor-critic).

【Environment Description】
{ENV_DESCRIPTION}

{FIXED_STATE_ACTION}

【Design Principle】
Encourage innovative network architecture design for BOTH actor (PolicyNet) and critic (ValueNet).
Note: This is a continuous control task with high-dimensional observation space.

【Output Requirements】
Please output two parts:

1. Network structure description (JSON format):
{network_design_json}

2. Complete network class code (Python, ready to run):
{network_code_template}

{network_code_notes}"""

    system = (
        f"You are an expert in agent architecture design, specializing in actor-critic neural networks for {TASK_NAME} tasks. "
        "Output must include: 1) JSON describing both PolicyNet and ValueNet  "
        "2) Complete runnable code defining BOTH the PolicyNet and ValueNet classes."
    )

    response = call_idea_llm(system, prompt, step=f"G{gen}_I{idx}-Init")

    ind.network_design = extract_json_from_response(response)
    ind.network_code = extract_python_code(response)

    return ind


def llm_mutate_individual(ind: Individual, gen: int, new_idx: int) -> Individual:
    new_ind = Individual(new_idx, gen)

    network_code_template = get_network_code_template()
    network_design_json = """{{
  "policy_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }},
  "value_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }}
}}"""

    network_code_notes = """Notes:
- Class names MUST be exactly `PolicyNet` and `ValueNet` (both required).
- PolicyNet.__init__ takes (obs_length, act_length); ValueNet.__init__ takes (obs_length).
- forward signatures must match the template above.
- PolicyNet outputs continuous action means (will be used with Normal distribution).
- Do NOT include any training logic, only the two nn.Module class definitions.
- Do NOT share parameters between the two networks (they will be optimized jointly by PPO)."""

    prompt = f"""Analyze and mutate the following {TASK_NAME} actor-critic architecture (PolicyNet + ValueNet).

【Environment Description】
{ENV_DESCRIPTION}

{FIXED_STATE_ACTION}

【Current Architecture - {ind.label}】
{ind.network_design}

**Mutation Task**:
1. Analyze both the PolicyNet and ValueNet of the current architecture and identify their characteristics.
2. Based on your knowledge of actor-critic design for continuous control, enhance or modify EITHER OR BOTH networks.
3. **IMPORTANT**: The mutated architecture MUST be different from the original.

【Output Requirements】
Please output two parts:

1. Network structure description (JSON format):
{network_design_json}

2. Complete code defining BOTH classes (Python):
{network_code_template}

{network_code_notes}"""

    system = (
        "You are an expert in actor-critic neural network architecture design for continuous control. "
        "Analyze the given PolicyNet and ValueNet pair and create a meaningful mutation. "
        "Output must include: 1) JSON describing both networks  "
        "2) Complete runnable code defining BOTH the PolicyNet and ValueNet classes."
    )

    response = call_idea_llm(system, prompt, step=f"G{gen}_I{new_idx}-Mutate(from {ind.label})")

    new_ind.network_design = extract_json_from_response(response)
    new_ind.network_code = extract_python_code(response)

    LOG.log(f"      变异: {ind.label} → G{gen}_I{new_idx}")
    return new_ind


def llm_crossover_single(parent1: Individual, parent2: Individual,
                          gen: int, child_idx: int) -> Individual:

    network_code_template = get_network_code_template()
    network_design_json = """{{
  "policy_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }},
  "value_net": {{
    "type": "...",
    "layer_config": ["..."],
    "param_init": "...",
    "estimated_params": "..."
  }}
}}"""

    network_code_notes = """Notes:
- Class names MUST be exactly `PolicyNet` and `ValueNet` (both required).
- PolicyNet.__init__ takes (obs_length, act_length); ValueNet.__init__ takes (obs_length).
- forward signatures must match the template above.
- PolicyNet outputs continuous action means (will be used with Normal distribution).
- Do NOT include any training logic, only the two nn.Module class definitions.
- Do NOT share parameters between the two networks (they will be optimized jointly by PPO)."""

    prompt = f"""Analyze and crossover the following two {TASK_NAME} actor-critic architectures (each contains PolicyNet + ValueNet).

【Environment Description】
{ENV_DESCRIPTION}

{FIXED_STATE_ACTION}

【Parent 1 - {parent1.label}】
{parent1.network_design}

【Parent 2 - {parent2.label}】
{parent2.network_design}

**Crossover Task**:
1. Analyze the strengths of both parents' PolicyNet and ValueNet.
2. Based on your knowledge of actor-critic design for continuous control, intelligently fuse architectural features from both parents.
3. Create a coherent new (PolicyNet, ValueNet) pair that combines complementary strengths.

【Output Requirements】
Please output two parts:

1. Network structure description (JSON format):
{network_design_json}

2. Complete code defining BOTH classes (Python):
{network_code_template}

{network_code_notes}"""

    system = (
        "You are an expert in actor-critic neural network architecture design for continuous control. "
        "Analyze both parent architectures (PolicyNet + ValueNet) and create an intelligent fusion. "
        "Output must include: 1) JSON describing both networks  "
        "2) Complete runnable code defining BOTH the PolicyNet and ValueNet classes."
    )

    response = call_idea_llm(system, prompt,
                           step=f"Crossover-Child{child_idx}(Parents:{parent1.label}+{parent2.label})")

    child = Individual(child_idx, gen)

    child.network_design = extract_json_from_response(response)
    child.network_code = extract_python_code(response)

    LOG.log(f"      交叉子代: G{gen}_I{child_idx} (父代:{parent1.label}+{parent2.label})")
    return child



def _save_individual_code(ind: Individual) -> str:
    save_dir = os.path.join(RUN_DIR, f"individuals/{ind.label}")
    os.makedirs(save_dir, exist_ok=True)
    code_path = os.path.join(save_dir, "policy_net.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(ind.network_code)
    return code_path


def _train_one_individual(ind: Individual, worker_id: int, round_idx: int) -> dict:
    save_dir = os.path.join(RUN_DIR, f"individuals/{ind.label}")
    os.makedirs(save_dir, exist_ok=True)
    code_path = _save_individual_code(ind)


    import tempfile
    temp_dir = tempfile.mkdtemp(prefix=f"evom_train_{ind.label}_r{round_idx}_")
    result_path = os.path.join(temp_dir, "result.json")
    log_path = os.path.join(temp_dir, "train.log")


    seed_map = {1: 42, 2: 123, 3: 456}
    train_seed = seed_map.get(round_idx, 42 + round_idx * 100)

    env = _build_cpu_env()
    cmd = [
        sys.executable, WORKER_SCRIPT,
        "--code-path", code_path,
        "--result-path", result_path,
        "--timesteps", str(TRAIN_CONFIG["timesteps"]),
        "--max-steps", str(TRAIN_CONFIG["max_steps"]),
        "--seed", str(train_seed),
    ]

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    except Exception as e:
        LOG.log(f"    ✗ {ind.label} R{round_idx} (CPU{worker_id}): 子进程启动失败 - {e}")

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"round": round_idx, "best_reward": -1000.0, "elapsed": time.time() - t0}

    elapsed = time.time() - t0

    if not os.path.exists(result_path):
        LOG.log(f"    ✗ {ind.label} R{round_idx} (CPU{worker_id}): worker 无输出, exit={proc.returncode}")

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"round": round_idx, "best_reward": -1000.0, "elapsed": elapsed}

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)


    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    if result.get("status") != "ok":
        LOG.log(f"    ✗ {ind.label} R{round_idx} (CPU{worker_id}): {result.get('error')}")
        return {"round": round_idx, "best_reward": -1000.0, "elapsed": elapsed}

    best_reward = float(result["best_reward"])
    LOG.log(
        f"    ✓ {ind.label} R{round_idx} (CPU{worker_id}, {elapsed:.0f}s): "
        f"best_reward={best_reward:.2f}"
    )
    return {"round": round_idx, "best_reward": best_reward, "elapsed": elapsed}


def run_population(population: list[Individual]) -> list[Individual]:
    n_tasks = len(population) * NUM_EVAL_ROUNDS
    LOG.log(
        f"  开始训练 {len(population)} 个个体 × {NUM_EVAL_ROUNDS} 轮 = {n_tasks} 个任务 "
        f"(并发数={CPU_PARALLEL}, CPU并行, "
        f"timesteps={TRAIN_CONFIG['timesteps']})"
    )


    tasks = []
    for r in range(1, NUM_EVAL_ROUNDS + 1):
        for ind in population:
            tasks.append((ind, r))

    per_ind_rounds: dict[str, list[dict]] = {ind.label: [] for ind in population}
    overall_t0 = time.time()


    with ThreadPoolExecutor(max_workers=CPU_PARALLEL) as executor:
        futures = {}
        for task_idx, (ind, r) in enumerate(tasks):
            worker_id = task_idx % CPU_PARALLEL
            future = executor.submit(_train_one_individual, ind, worker_id, r)
            futures[future] = (ind, r)

        completed = 0
        for future in as_completed(futures):
            ind, r = futures[future]
            round_data = future.result()
            per_ind_rounds[ind.label].append(round_data)
            completed += 1


            if completed % 10 == 0 or completed == n_tasks:
                LOG.log(f"  进度: {completed}/{n_tasks} 任务完成 ({completed*100//n_tasks}%)")

    LOG.log(f"  全部 {n_tasks} 个评估任务完成, 总耗时 {time.time()-overall_t0:.1f}s")


    for ind in population:
        rounds_data = sorted(per_ind_rounds[ind.label], key=lambda d: d["round"])
        reward_list = [d["best_reward"] for d in rounds_data]
        mean_reward = float(np.mean(reward_list)) if reward_list else -1000.0
        std_reward = float(np.std(reward_list)) if reward_list else 0.0

        ind.fitness = mean_reward


        ind.sim_results = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "fitness": ind.fitness,
        }


        save_dir = os.path.join(RUN_DIR, f"individuals/{ind.label}")
        info = {
            "label": ind.label,
            "gen": ind.gen,
            "idx": ind.idx,
            "network_design": ind.network_design,
            "fitness": ind.fitness,
        }
        with open(os.path.join(save_dir, "individual_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        LOG.log(
            f"    ◆ {ind.label} 聚合: "
            f"fitness={ind.fitness:.2f} (std={std_reward:.2f})"
        )

    return population



def crossover(population: list[Individual], gen: int, num_offspring: int) -> list[Individual]:
    offspring = []

    for i in range(num_offspring):
        if len(population) >= 2:
            parent1, parent2 = random.sample(population, 2)
        else:
            parent1 = parent2 = population[0]

        child = llm_crossover_single(parent1, parent2, gen, i)
        offspring.append(child)

    return offspring


def elitist_selection(parents: list[Individual], offspring: list[Individual],
                      pop_size: int) -> list[Individual]:
    combined = parents + offspring
    combined.sort(key=lambda x: x.fitness, reverse=True)
    return combined[:pop_size]



def evolution_loop(run_prefix: str):
    global LOG, RUN_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(PROJECT_DIR, f"runs/{run_prefix}_{timestamp}")
    os.makedirs(RUN_DIR, exist_ok=True)

    LOG = Logger(RUN_DIR)
    LOG.log("="*80)
    LOG.log(f"{TASK_NAME} 网络结构进化系统")
    LOG.log("="*80)
    LOG.log(f"种群大小: {POPULATION_SIZE}")
    LOG.log(f"进化代数: {NUM_GENERATIONS}")
    LOG.log(f"子代生成模式: {OFFSPRING_MODE}")
    LOG.log(f"LLM模型: {LLM_PROFILE} ({IDEA_MODEL})")
    LOG.log(f"LLM API: {IDEA_API_URL}")
    LOG.log(f"评估方式: PPO 少量训练 (timesteps={TRAIN_CONFIG['timesteps']}, "
            f"CPU并发={CPU_PARALLEL}) — 适应度=平均奖励")
    LOG.log(f"运行目录: {RUN_DIR}")
    LOG.log("="*80)


    LOG.log("\n[第0代] 初始化种群")
    LOG.log(f"  并行生成 {POPULATION_SIZE} 个个体（最多{MAX_WORKERS}个并发）...")

    population = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(llm_init_individual, i, 0) for i in range(POPULATION_SIZE)]
        for future in as_completed(futures):
            ind = future.result()
            population.append(ind)
            LOG.log(f"    ✓ 完成 {ind.label}")

    population.sort(key=lambda x: x.idx)

    LOG.log(f"\n  评估第0代种群...")
    population = run_population(population)

    gen0_summary = {
        "generation": 0,
        "offspring_mode": OFFSPRING_MODE,
        "llm_profile": LLM_PROFILE,
        "llm_model": IDEA_MODEL,
        "llm_api_url": IDEA_API_URL,
        "population_size": len(population),
        "individuals": [
            {
                "label": ind.label,
                "fitness": ind.fitness,
                "network_design": ind.network_design,
            }
            for ind in population
        ],
        "best_fitness": max(ind.fitness for ind in population),
        "mean_fitness": sum(ind.fitness for ind in population) / len(population),
    }
    with open(os.path.join(RUN_DIR, "gen0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(gen0_summary, f, indent=2, ensure_ascii=False)

    best = max(population, key=lambda x: x.fitness)
    LOG.log(f"\n  第0代完成: 最佳={best.label}(fitness={best.fitness:.2f}), "
            f"平均fitness={sum(ind.fitness for ind in population)/len(population):.2f}")


    for gen in range(1, NUM_GENERATIONS + 1):
        LOG.log(f"\n{'='*80}")
        LOG.log(f"[第{gen}代] 开始进化")
        LOG.log(f"{'='*80}")

        mutants = []
        offspring = []

        if OFFSPRING_MODE == "mixed":

            LOG.log(f"\n  步骤1: 变异（并行生成{POPULATION_SIZE//2}个变异个体）")

            selected_for_mutation = random.sample(population, POPULATION_SIZE // 2)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(llm_mutate_individual, parent, gen, i)
                          for i, parent in enumerate(selected_for_mutation)]
                for future in as_completed(futures):
                    mutant = future.result()
                    mutants.append(mutant)
                    LOG.log(f"    ✓ 完成 {mutant.label}")

            mutants.sort(key=lambda x: x.idx)


            LOG.log(f"\n  步骤2: 交叉（并行生成{POPULATION_SIZE//2}个交叉子代）")
            crossover_idx_offset = POPULATION_SIZE // 2

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for i in range(POPULATION_SIZE // 2):
                    if len(population) >= 2:
                        parent1, parent2 = random.sample(population, 2)
                    else:
                        parent1 = parent2 = population[0]
                    child_idx = crossover_idx_offset + i
                    futures.append(executor.submit(llm_crossover_single, parent1, parent2, gen, child_idx))

                for future in as_completed(futures):
                    child = future.result()
                    offspring.append(child)
                    LOG.log(f"    ✓ 完成 {child.label}")

            offspring.sort(key=lambda x: x.idx)

        elif OFFSPRING_MODE == "mutation_only":
            LOG.log(f"\n  步骤1: 变异消融（并行生成{POPULATION_SIZE}个变异个体）")

            selected_for_mutation = random.sample(population, POPULATION_SIZE)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(llm_mutate_individual, parent, gen, i)
                          for i, parent in enumerate(selected_for_mutation)]
                for future in as_completed(futures):
                    mutant = future.result()
                    mutants.append(mutant)
                    LOG.log(f"    ✓ 完成 {mutant.label}")

            mutants.sort(key=lambda x: x.idx)
            LOG.log("\n  步骤2: 跳过交叉（mutation_only 模式）")

        elif OFFSPRING_MODE == "crossover_only":
            LOG.log("\n  步骤1: 跳过变异（crossover_only 模式）")
            LOG.log(f"\n  步骤2: 交叉消融（并行生成{POPULATION_SIZE}个交叉子代）")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for i in range(POPULATION_SIZE):
                    if len(population) >= 2:
                        parent1, parent2 = random.sample(population, 2)
                    else:
                        parent1 = parent2 = population[0]
                    futures.append(executor.submit(llm_crossover_single, parent1, parent2, gen, i))

                for future in as_completed(futures):
                    child = future.result()
                    offspring.append(child)
                    LOG.log(f"    ✓ 完成 {child.label}")

            offspring.sort(key=lambda x: x.idx)

        else:
            raise ValueError(f"Unknown OFFSPRING_MODE: {OFFSPRING_MODE}")


        all_offspring = mutants + offspring
        LOG.log(f"\n  步骤3: 评估全部 {len(all_offspring)} 个子代（{len(mutants)}变异 + {len(offspring)}交叉，并行）")
        all_offspring = run_population(all_offspring)

        best_offspring = max(all_offspring, key=lambda x: x.fitness)
        LOG.log(f"\n  第{gen}代子代: 最佳={best_offspring.label}(fitness={best_offspring.fitness:.2f}), "
                f"平均fitness={sum(ind.fitness for ind in all_offspring)/len(all_offspring):.2f}")


        LOG.log(f"\n  步骤4: 精英选择")
        population = elitist_selection(population, all_offspring, POPULATION_SIZE)

        best_parent = max(population, key=lambda x: x.fitness)
        LOG.log(f"  选择后的父代: 最佳={best_parent.label}(fitness={best_parent.fitness:.2f}), "
                f"平均fitness={sum(ind.fitness for ind in population)/len(population):.2f}")


        gen_summary = {
            "generation": gen,
            "offspring_mode": OFFSPRING_MODE,
            "llm_profile": LLM_PROFILE,
            "llm_model": IDEA_MODEL,
            "llm_api_url": IDEA_API_URL,
            "offspring": {
                "population_size": len(all_offspring),
                "mutation_count": len(mutants),
                "crossover_count": len(offspring),
                "individuals": [
                    {"label": ind.label, "fitness": ind.fitness}
                    for ind in all_offspring
                ],
                "best_fitness": max(ind.fitness for ind in all_offspring),
                "mean_fitness": sum(ind.fitness for ind in all_offspring) / len(all_offspring),
            },
            "selected_parents": {
                "population_size": len(population),
                "individuals": [
                    {"label": ind.label, "fitness": ind.fitness}
                    for ind in population
                ],
                "best_fitness": max(ind.fitness for ind in population),
                "mean_fitness": sum(ind.fitness for ind in population) / len(population),
            },
        }
        with open(os.path.join(RUN_DIR, f"gen{gen}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(gen_summary, f, indent=2, ensure_ascii=False)


    LOG.log(f"\n{'='*80}")
    LOG.log("进化完成！")
    LOG.log(f"{'='*80}")


    final_elite_summary = {
        "generation": "final_elite",
        "description": "Final elite population after all generations",
        "offspring_mode": OFFSPRING_MODE,
        "llm_profile": LLM_PROFILE,
        "llm_model": IDEA_MODEL,
        "llm_api_url": IDEA_API_URL,
        "population_size": len(population),
        "individuals": [
            {"label": ind.label, "fitness": ind.fitness}
            for ind in sorted(population, key=lambda x: x.fitness, reverse=True)
        ],
        "best_fitness": max(ind.fitness for ind in population),
        "mean_fitness": sum(ind.fitness for ind in population) / len(population),
    }
    with open(os.path.join(RUN_DIR, "final_elite_population.json"), "w", encoding="utf-8") as f:
        json.dump(final_elite_summary, f, indent=2, ensure_ascii=False)

    final_best = max(population, key=lambda x: x.fitness)
    LOG.log(f"最终精英种群最佳个体: {final_best.label}")
    LOG.log(f"最终最佳fitness: {final_best.fitness:.2f}")


    final_summary = {
        "offspring_mode": OFFSPRING_MODE,
        "llm_profile": LLM_PROFILE,
        "llm_model": IDEA_MODEL,
        "llm_api_url": IDEA_API_URL,
        "best_individual": {
            "label": final_best.label,
            "fitness": final_best.fitness,
            "network_design": final_best.network_design,
        },
    }
    with open(os.path.join(RUN_DIR, "final_best_individual.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    LOG.log(f"\n结果已保存至: {RUN_DIR}")



    with open(os.path.join(RUN_DIR, "gen0_summary.json"), "r", encoding="utf-8") as f:
        gen0_data = json.load(f)

    gen0_best_label = max(gen0_data["individuals"], key=lambda x: x["fitness"])["label"]


    gen0_best_dir = os.path.join(RUN_DIR, f"individuals/{gen0_best_label}")
    with open(os.path.join(gen0_best_dir, "individual_info.json"), "r", encoding="utf-8") as f:
        gen0_best_info = json.load(f)
    with open(os.path.join(gen0_best_dir, "policy_net.py"), "r", encoding="utf-8") as f:
        gen0_best_code = f.read()

    gen0_best_summary = {
        "offspring_mode": OFFSPRING_MODE,
        "llm_profile": LLM_PROFILE,
        "llm_model": IDEA_MODEL,
        "llm_api_url": IDEA_API_URL,
        "label": gen0_best_label,
        "fitness": gen0_best_info["fitness"],
        "network_design": gen0_best_info["network_design"],
        "network_code": gen0_best_code,
    }
    with open(os.path.join(RUN_DIR, "gen0_best_individual.json"), "w", encoding="utf-8") as f:
        json.dump(gen0_best_summary, f, indent=2, ensure_ascii=False)

    LOG.log(f"\n第一代最佳个体: {gen0_best_label} (fitness={gen0_best_info['fitness']:.2f})")
    LOG.log(f"已保存至: {RUN_DIR}/gen0_best_individual.json")


    if not RUN_FINAL_SELECTION:
        LOG.log(f"\n[跳过] RUN_FINAL_SELECTION=False，不对精英种群做最终筛选。")
        LOG.close()
        return



    class Gen0Individual:
        def __init__(self, label, fitness, network_design, network_code):
            self.label = label
            self.fitness = fitness
            self.network_design = network_design
            self.network_code = network_code

    gen0_best_ind = Gen0Individual(
        gen0_best_label,
        gen0_best_info["fitness"],
        gen0_best_info["network_design"],
        gen0_best_code
    )

    eval_gen0_control = (
        OFFSPRING_MODE == "mixed"
        and LLM_PROFILE == "deepseek"
        and POPULATION_SIZE == 16
        and NUM_GENERATIONS == 20
        and NUM_EVAL_ROUNDS == 3
    )
    if eval_gen0_control and final_best.label != gen0_best_label:
        elite_individuals = [final_best, gen0_best_ind]
    else:
        elite_individuals = [final_best]

    LOG.log(f"\n{'='*80}")
    LOG.log(f"最终精英详细评估：")
    LOG.log(f"  1. 最终代最佳个体 {final_best.label} × {NUM_FINAL_RUNS} 次训练评估")
    if eval_gen0_control:
        if final_best.label == gen0_best_label:
            LOG.log(f"  注意: 最终代最佳个体就是第一代最佳个体 {gen0_best_label}，不重复评估")
        else:
            LOG.log(f"  2. 第一代最佳个体 {gen0_best_label} × {NUM_FINAL_RUNS} 次训练评估（主实验G0对照）")
    else:
        LOG.log(f"  对照/消融/参数敏感性/LLM实验仅评估最终最佳个体，不做G0详细评估")
    LOG.log(f"  总计 {len(elite_individuals) * NUM_FINAL_RUNS} 个任务（每次5个episodes，与baseline相同）")
    LOG.log(f"{'='*80}")

    try:
        summary = run_final_selection(elite_individuals, RUN_DIR, LOG)
        LOG.log(f"\n✓ 精英评估完成")
        LOG.log(f"  详细结果见: {RUN_DIR}/final_selection/elite_evaluation_summary.json")
    except Exception as e:
        LOG.log(f"精英评估失败: {e}")
        import traceback
        LOG.log(traceback.format_exc())

    LOG.close()


def _eval_elite_one(label: str, network_code: str, run_idx: int,
                     gpu_id: int, save_root: str) -> dict:
    save_dir = os.path.join(save_root, label, f"run{run_idx}")
    os.makedirs(save_dir, exist_ok=True)

    code_path = os.path.join(save_dir, "policy_net.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(network_code)

    result_path = os.path.join(save_dir, "eval_result.json")


    seed_map = {0: 42, 1: 123, 2: 456}
    train_seed = seed_map.get(run_idx, 42 + run_idx * 100)

    env = _build_gpu_env(gpu_id)
    cmd = [
        sys.executable, ELITE_EVAL_WORKER,
        "--code-path", code_path,
        "--result-path", result_path,
        "--timesteps", str(FULL_TRAIN_CONFIG["timesteps"]),
        "--max-steps", str(MAX_EPISODE_STEPS),
        "--n-eval-episodes", "5",
        "--eval-freq", "25000",
        "--seed", str(train_seed),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except Exception as e:
        LOG.log(f"    ✗ {label} run{run_idx} (GPU{gpu_id}): 子进程启动失败 - {e}")
        return {"label": label, "run_idx": run_idx, "mean_reward": -1000.0,
                "elapsed": time.time() - t0, "error": f"subprocess launch failed: {e}",
                "save_dir": save_dir}

    elapsed = time.time() - t0

    if not os.path.exists(result_path):
        LOG.log(f"    ✗ {label} run{run_idx} (GPU{gpu_id}): worker 无结果输出 "
                f"(exit={proc.returncode})")
        return {"label": label, "run_idx": run_idx, "mean_reward": -1000.0,
                "elapsed": elapsed, "error": f"no result (exit={proc.returncode})",
                "save_dir": save_dir}

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "ok":
        LOG.log(f"    ✗ {label} run{run_idx} (GPU{gpu_id}): {data.get('error', 'unknown error')}")
        return {"label": label, "run_idx": run_idx, "mean_reward": -1000.0,
                "elapsed": elapsed, "error": data.get("error"), "save_dir": save_dir}

    mean_reward = float(data.get("mean_reward", -1000.0))
    std_reward = float(data.get("std_reward", 0.0))
    LOG.log(f"    ✓ {label} run{run_idx} (GPU{gpu_id}, {elapsed/60:.1f}min): "
            f"mean_reward={mean_reward:.2f}±{std_reward:.2f}")


    return {
        "label": label,
        "run_idx": run_idx,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "elapsed": elapsed,
        "save_dir": save_dir,
        "error": None
    }



def run_final_selection(elite_population: list, run_dir: str, logger):
    final_root = os.path.join(run_dir, "final_selection")
    os.makedirs(final_root, exist_ok=True)

    n_tasks = len(elite_population) * NUM_FINAL_RUNS

    logger.log(f"  使用GPU: {GPU_IDS}")


    tasks = []
    for r in range(NUM_FINAL_RUNS):
        for ind in elite_population:
            tasks.append((ind, r))

    logger.log(f"  并发数 = {FINAL_PARALLEL}, 总任务数 = {n_tasks}, "
               f"每个评估 timesteps={FULL_TRAIN_CONFIG['timesteps']}, 5次episodes")

    overall_t0 = time.time()
    per_label_runs: dict = {ind.label: [] for ind in elite_population}
    gpu_pool = GPU_IDS


    with ThreadPoolExecutor(max_workers=FINAL_PARALLEL) as executor:
        futures = {}
        for task_idx, (ind, r) in enumerate(tasks):
            gpu_id = gpu_pool[task_idx % len(gpu_pool)]
            future = executor.submit(_eval_elite_one, ind.label, ind.network_code,
                                     r, gpu_id, final_root)
            futures[future] = (ind, r)

        completed = 0
        for future in as_completed(futures):
            ind, r = futures[future]
            run_data = future.result()
            per_label_runs[ind.label].append(run_data)
            completed += 1


            if completed % 5 == 0 or completed == n_tasks:
                logger.log(f"  进度: {completed}/{n_tasks} 任务完成 ({completed*100//n_tasks}%)")

    total_elapsed = time.time() - overall_t0
    logger.log(f"  全部 {len(tasks)} 个详细评估完成, 总耗时 {total_elapsed/60:.1f} 分钟")


    aggregated = []
    for ind in elite_population:
        runs = sorted(per_label_runs[ind.label], key=lambda d: d["run_idx"])
        reward_list = [d["mean_reward"] for d in runs]
        mean_reward = float(np.mean(reward_list)) if reward_list else -1000.0
        std_reward = float(np.std(reward_list)) if reward_list else 0.0

        entry = {
            "label": ind.label,
            "evolution_fitness": ind.fitness,
            "final_mean_reward": mean_reward,
            "final_std_reward": std_reward,
            "per_run_mean_rewards": reward_list,
            "network_design": ind.network_design,
            "run_dirs": [d["save_dir"] for d in runs]
        }
        aggregated.append(entry)
        logger.log(
            f"    ◆ {ind.label}: final_mean_reward={mean_reward:.2f} "
            f"(std={std_reward:.2f}) [evolution_fit={ind.fitness:.2f}]"
        )


    aggregated.sort(key=lambda e: e["evolution_fitness"], reverse=True)


    selection_summary = {
        "num_elites": len(elite_population),
        "num_runs_per_elite": NUM_FINAL_RUNS,
        "offspring_mode": OFFSPRING_MODE,
        "llm_profile": LLM_PROFILE,
        "llm_model": IDEA_MODEL,
        "llm_api_url": IDEA_API_URL,
        "total_elapsed_sec": total_elapsed,
        "eval_config": {
            "timesteps": FULL_TRAIN_CONFIG['timesteps'],
            "n_eval_episodes": 5,
            "seeds": [42, 123, 456]
        },
        "elites": aggregated,
    }
    summary_path = os.path.join(final_root, "elite_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2, ensure_ascii=False)
    logger.log(f"  精英评估总览: {summary_path}")

    return selection_summary


def parse_args():
    parser = argparse.ArgumentParser(description="EVOM network architecture search")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument(
        "--offspring-mode",
        type=str,
        default="mixed",
        choices=["mixed", "mutation_only", "crossover_only"],
    )
    parser.add_argument(
        "--llm-profile",
        type=str,
        default="deepseek",
        choices=sorted(set(LLM_PROFILES) | set(LLM_PROFILE_ALIASES)),
    )
    parser.add_argument("--cpu-parallel", type=int, default=48)
    parser.add_argument("--eval-rounds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--final-runs", type=int, default=3)
    parser.add_argument("--final-episodes", type=int, default=None)
    parser.add_argument("--core-offset", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--no-pin-cores", action="store_true")
    parser.add_argument("--no-final-selection", action="store_true")
    return parser.parse_args()


def main():
    global POPULATION_SIZE, NUM_GENERATIONS, OFFSPRING_MODE
    global CPU_PARALLEL, NUM_EVAL_ROUNDS, NUM_FINAL_RUNS
    global MAX_WORKERS, GPU_IDS, FINAL_PARALLEL, RUN_FINAL_SELECTION

    args = parse_args()

    POPULATION_SIZE = args.population_size
    NUM_GENERATIONS = args.generations
    OFFSPRING_MODE = args.offspring_mode
    CPU_PARALLEL = args.cpu_parallel
    if args.gpu_ids is not None:
        GPU_IDS = args.gpu_ids
    FINAL_PARALLEL = len(GPU_IDS) * 16
    NUM_EVAL_ROUNDS = args.eval_rounds
    NUM_FINAL_RUNS = args.final_runs
    MAX_WORKERS = args.max_workers
    RUN_FINAL_SELECTION = not args.no_final_selection
    llm_profile = configure_llm_profile(args.llm_profile)

    print(f"Loading task config: {args.task}")
    run_prefix = load_task_config(args.task)
    if LLM_PROFILE != "deepseek":
        run_prefix = f"{run_prefix}_{LLM_PROFILE}"
    if OFFSPRING_MODE != "mixed":
        run_prefix = f"{run_prefix}_{OFFSPRING_MODE}"

    if args.eval_episodes is not None:
        TRAIN_CONFIG["timesteps"] = args.eval_episodes
    if args.final_episodes is not None:
        FULL_TRAIN_CONFIG["timesteps"] = args.final_episodes

    print(f"Task: {TASK_NAME}")
    print(f"Observation dim: {STATE_DIM}, action dim: {ACTION_DIM}")
    print(f"Population size: {POPULATION_SIZE}, generations: {NUM_GENERATIONS}")
    print(f"Offspring mode: {OFFSPRING_MODE}")
    print(f"LLM profile: {LLM_PROFILE} ({llm_profile['model']})")
    print(f"Evolution evaluation parallelism: CPU={CPU_PARALLEL}")
    print(
        f"Evaluation rounds: {NUM_EVAL_ROUNDS} "
        f"(timesteps per round={TRAIN_CONFIG['timesteps']}, LLM max workers={MAX_WORKERS})"
    )
    if RUN_FINAL_SELECTION:
        print(
            f"Final selection: yes "
            f"(runs={NUM_FINAL_RUNS}, timesteps={FULL_TRAIN_CONFIG['timesteps']}, "
            f"GPU parallel={FINAL_PARALLEL}, GPUs={GPU_IDS})"
        )
    else:
        print("Final selection: no")
    print()

    evolution_loop(run_prefix)


if __name__ == "__main__":
    main()
