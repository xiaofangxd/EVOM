
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests


PROJECT_DIR = Path(__file__).resolve().parent

DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

TASK_DESCRIPTION = """
Design a direct programmatic controller for Gymnasium HalfCheetah-v4.

The controller receives the raw HalfCheetah-v4 observation vector:
- obs shape: (17,)
- action shape: (6,), continuous torques clipped to [-1, 1]
- objective: maximize forward running reward while minimizing control cost

Return only a valid implementation of policy(obs, t, memory).
The policy must be deterministic, fast, and use only numpy/math operations.
It must not use reinforcement learning training, neural networks, torch, files,
network access, multiprocessing, or environment creation.
"""

TEMPLATE_PROGRAM = r'''
import numpy as np

def policy(obs: np.ndarray, t: int, memory: dict) -> np.ndarray:
    """Direct HalfCheetah-v4 controller.

    Args:
        obs: raw observation, shape (17,).
        t: current step in this episode.
        memory: mutable per-episode dict for simple controller state.

    Returns:
        action: np.ndarray, shape (6,), clipped to [-1, 1].
    """
    phase = 0.11 * t
    gait = np.array([
        np.sin(phase),
        -np.sin(phase + 0.6),
        np.sin(phase + 1.2),
        -np.sin(phase + np.pi),
        np.sin(phase + np.pi + 0.6),
        -np.sin(phase + np.pi + 1.2),
    ], dtype=np.float32)
    action = 0.45 * gait
    return np.clip(action, -1.0, 1.0)
'''


def _load_llm4ad():
    try:
        from llm4ad.method.mles import MLES
    except ImportError:
        try:
            from llm4ad.method.mmeoh import MMEoH as MLES
        except ImportError as exc:
            raise SystemExit(
                "Found llm4ad, but could not import MLES/MMEoH. "
                "Run this script from the QingL2000/MLES repository or install that repo on PYTHONPATH."
            ) from exc

    try:
        from llm4ad.tools.llm.llm_api_https_mmeoh import HttpsApi
    except ImportError:
        try:
            from llm4ad.tools.llm.llm_api_https import HttpsApi
        except ImportError as exc:
            raise SystemExit(
                "Found llm4ad, but could not import HttpsApi. "
                "Please check the LLM4AD/MLES installation."
            ) from exc
    return MLES, HttpsApi


class RequestsChatLLM:

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        timeout: int = 300,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        max_retries: int = 3,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    @staticmethod
    def _messages(prompt: Any) -> list[dict[str, str]]:
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt
        if isinstance(prompt, list):
            prompt = "\n".join(str(item) for item in prompt)
        return [{"role": "user", "content": str(prompt)}]

    def draw_sample(self, prompt: Any, *args, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(prompt),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"LLM API returned non-JSON response "
                        f"(status={response.status_code}): {response.text[:1000]}"
                    ) from exc

                if "choices" not in data:
                    raise RuntimeError(
                        f"LLM API returned no choices (status={response.status_code}): "
                        f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                    )

                message = data["choices"][0].get("message", {})
                content = message.get("content")
                if not content:
                    raise RuntimeError(
                        f"LLM API returned empty content: "
                        f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                    )
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 20))

        raise RuntimeError(f"LLM API failed after {self.max_retries} attempts: {last_error}")

    def draw_samples(self, prompts: list[Any], *args, **kwargs) -> list[str]:
        return [self.draw_sample(prompt, *args, **kwargs) for prompt in prompts]


def _make_env(seed: int):
    import gymnasium as gym

    env = gym.make("HalfCheetah-v4")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def _as_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (6,):
        raise ValueError(f"policy must return shape (6,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("policy returned NaN or Inf")
    return np.clip(action, -1.0, 1.0)


def _extract_position(env) -> tuple[float, float, float]:
    data = env.unwrapped.data
    qpos = np.asarray(data.qpos)
    x = float(qpos[0]) if qpos.shape[0] > 0 else 0.0
    z = float(qpos[1]) if qpos.shape[0] > 1 else 0.0
    pitch = float(qpos[2]) if qpos.shape[0] > 2 else 0.0
    return x, z, pitch


def _trajectory_image(trajectories: list[list[tuple[float, float, float]]], title: str, path: Path) -> str:
    if not trajectories:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
    for idx, traj in enumerate(trajectories[:6]):
        arr = np.asarray(traj, dtype=np.float32)
        if arr.size == 0:
            continue
        axes[0].plot(arr[:, 0], arr[:, 1], linewidth=1.2, label=f"ep{idx}")
        axes[1].plot(arr[:, 2], linewidth=1.2)
    axes[0].set_title("x-height trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("torso pitch")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("pitch")
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(path, format="png")
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class HalfCheetahDirectPolicyEvaluation:
    def __init__(
        self,
        save_dir: Path,
        seeds: list[int],
        episodes_per_seed: int,
        max_steps: int,
        max_evaluations: int | None = None,
    ):
        from llm4ad.base import Evaluation

        class _Evaluation(Evaluation):
            def __init__(self, outer: HalfCheetahDirectPolicyEvaluation):
                super().__init__(
                    template_program=TEMPLATE_PROGRAM,
                    task_description=TASK_DESCRIPTION,
                    exec_code=True,
                    safe_evaluate=False,
                )
                self.outer = outer

            def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
                return self.outer.evaluate_program(program_str, callable_func)

        self.save_dir = save_dir
        self.seeds = seeds
        self.episodes_per_seed = episodes_per_seed
        self.max_steps = max_steps
        self.max_evaluations = max_evaluations
        self.counter = 0
        self.completed = 0
        self.budget_lock = threading.Lock()
        self.records_path = save_dir / "evaluations.jsonl"
        self.evaluation = _Evaluation(self)

    def _reserve_candidate_id(self) -> str:
        while True:
            with self.budget_lock:
                if self.max_evaluations is None or self.counter < self.max_evaluations:
                    self.counter += 1
                    return f"C{self.counter:04d}"
                if self.completed >= self.counter:
                    self._write_budget_summary_locked()
                    self._exit_budget_reached()
            time.sleep(0.2)

    def _mark_candidate_completed(self):
        with self.budget_lock:
            self.completed += 1
            should_stop = (
                self.max_evaluations is not None
                and self.counter >= self.max_evaluations
                and self.completed >= self.counter
            )
            if should_stop:
                self._write_budget_summary_locked()
        if should_stop:
            self._exit_budget_reached()

    def _write_budget_summary_locked(self):
        records = _read_records(self.records_path)
        records.sort(key=lambda item: item.get("score", -1000.0), reverse=True)
        summary = {
            "task": "halfcheetah",
            "method": "LLM4AD-MLES direct policy",
            "num_records": len(records),
            "best": records[0] if records else None,
            "stopped_at_candidate_budget": True,
            "max_evaluations": self.max_evaluations,
        }
        (self.save_dir / "mles_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _exit_budget_reached(self):
        print(f"MLES HalfCheetah reached candidate budget: {self.max_evaluations}. Results: {self.save_dir}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def evaluate_program(self, program_str: str, callable_func: callable):
        candidate_id = self._reserve_candidate_id()
        candidate_dir = self.save_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "program.py").write_text(program_str, encoding="utf-8")

        rewards: list[float] = []
        lengths: list[int] = []
        errors: list[str] = []
        trajectories: list[list[tuple[float, float, float]]] = []

        for seed in self.seeds:
            for ep_idx in range(self.episodes_per_seed):
                env_seed = seed + ep_idx * 1000
                env = None
                try:
                    env = _make_env(env_seed)
                    obs, _ = env.reset(seed=env_seed)
                    memory: dict = {}
                    ep_reward = 0.0
                    ep_len = 0
                    trajectory = [_extract_position(env)]

                    for t in range(self.max_steps):
                        action = _as_action(callable_func(obs.copy(), t, memory))
                        obs, reward, terminated, truncated, _ = env.step(action)
                        ep_reward += float(reward)
                        ep_len += 1
                        trajectory.append(_extract_position(env))
                        if terminated or truncated:
                            break

                    rewards.append(ep_reward)
                    lengths.append(ep_len)
                    trajectories.append(trajectory)
                except Exception as exc:
                    rewards.append(-1000.0)
                    lengths.append(0)
                    errors.append(f"seed={env_seed}: {type(exc).__name__}: {exc}")
                finally:
                    if env is not None:
                        env.close()

        score = float(np.mean(rewards)) if rewards else -1000.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        observation = (
            f"candidate={candidate_id}; mean_reward={score:.2f}; "
            f"std={std_reward:.2f}; mean_length={mean_length:.1f}; "
            f"rewards={[round(r, 2) for r in rewards]}; errors={errors[:3]}"
        )
        image_b64 = _trajectory_image(
            trajectories,
            title=f"HalfCheetah-v4 {candidate_id} reward={score:.1f}",
            path=candidate_dir / "behavior.png",
        )
        record = {
            "candidate_id": candidate_id,
            "score": score,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "rewards": rewards,
            "lengths": lengths,
            "errors": errors,
            "program_path": str(candidate_dir / "program.py"),
            "image_path": str(candidate_dir / "behavior.png") if image_b64 else "",
            "observation": observation,
        }
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._mark_candidate_completed()
        return {"score": score, "image": image_b64, "observation": observation}


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description="LLM4AD MLES direct-policy baseline for HalfCheetah-v4")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)



    parser.add_argument("--total-env-steps", type=int, default=5_000_000)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--max-sample-nums", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--selection-num", type=int, default=None)
    parser.add_argument("--operators", default="e1,e2,m1_M,m2_M")
    parser.add_argument("--num-samplers", type=int, default=8)
    parser.add_argument("--num-evaluators", type=int, default=8)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY))
    parser.add_argument("--model", default=DEEPSEEK_MODEL)
    parser.add_argument("--host", default=DEEPSEEK_HOST)
    parser.add_argument("--api-url", default=os.environ.get("DEEPSEEK_API_URL", DEEPSEEK_API_URL))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required. Do not put API keys in source code.")

    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.episodes_per_seed <= 0:
        raise SystemExit("--episodes-per-seed must be positive")
    if args.total_env_steps <= 0:
        raise SystemExit("--total-env-steps must be positive")

    episodes_per_candidate = len(args.seeds) * args.episodes_per_seed
    rollout_steps_per_candidate_upper_bound = episodes_per_candidate * args.max_steps
    max_sample_nums = args.max_sample_nums
    if max_sample_nums is None:
        max_sample_nums = math.ceil(args.total_env_steps / rollout_steps_per_candidate_upper_bound)

    MLES, _ = _load_llm4ad()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir or PROJECT_DIR / "runs" / f"halfcheetah_mles_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    llm = RequestsChatLLM(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        timeout=300,
    )
    evaluator = HalfCheetahDirectPolicyEvaluation(
        save_dir=save_dir,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        max_evaluations=max_sample_nums,
    )

    config_args = vars(args).copy()
    config_args.pop("api_key", None)
    config = config_args | {
        "task": "halfcheetah",
        "method": "LLM4AD-MLES direct policy",
        "computed_max_sample_nums": max_sample_nums,
        "hard_candidate_budget": max_sample_nums,
        "episodes_per_candidate": episodes_per_candidate,
        "rollout_steps_per_candidate_upper_bound": rollout_steps_per_candidate_upper_bound,
        "save_dir": str(save_dir),
    }
    (save_dir / "mles_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mles_kwargs = {
        "llm": llm,
        "evaluation": evaluator.evaluation,
        "profiler": None,
        "max_generations": args.max_generations,
        "max_sample_nums": max_sample_nums,
        "pop_size": args.pop_size,
        "operators": tuple(x.strip() for x in args.operators.split(",") if x.strip()),
        "num_samplers": args.num_samplers,
        "num_evaluators": args.num_evaluators,
        "debug_mode": args.debug,
        "multi_thread_or_process_eval": "thread",
    }
    if args.selection_num is not None:
        mles_kwargs["selection_num"] = args.selection_num

    mles = MLES(**mles_kwargs)
    mles.run()

    records = _read_records(evaluator.records_path)
    records.sort(key=lambda item: item.get("score", -1000.0), reverse=True)
    summary = {
        "task": "halfcheetah",
        "method": "LLM4AD-MLES direct policy",
        "num_records": len(records),
        "best": records[0] if records else None,
    }
    (save_dir / "mles_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"MLES HalfCheetah direct-policy run finished. Results: {save_dir}")


if __name__ == "__main__":
    main()
