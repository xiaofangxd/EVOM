
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ENV_ID = "HalfCheetah-v4"
INITIAL_LOG_STD = 0.0


class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.rollout_rewards = []
        self.rollout_steps = []
        self.eval_rewards = []
        self.eval_steps = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.rollout_rewards.append(float(mean_reward))
            self.rollout_steps.append(int(self.num_timesteps))

    def update_eval_rewards(self, step: int, mean_reward: float):
        self.eval_rewards.append(float(mean_reward))
        self.eval_steps.append(int(step))

    def save_rewards(self):
        data = {
            "rollout": {
                "steps": self.rollout_steps,
                "rewards": self.rollout_rewards,
            },
            "eval": {
                "steps": self.eval_steps,
                "rewards": self.eval_rewards,
            },
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, reward_logger: RewardLoggerCallback = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_logger = reward_logger

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.reward_logger is not None and self.n_calls % self.eval_freq == 0:
            if len(self.evaluations_results) > 0:
                mean_reward = np.mean(self.evaluations_results[-1])
                self.reward_logger.update_eval_rewards(self.num_timesteps, mean_reward)
        return result


class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        policy_net_class,
        value_net_class,
        initial_log_std=INITIAL_LOG_STD,
        **kwargs,
    ):
        self.policy_net_class = policy_net_class
        self.value_net_class = value_net_class
        self.initial_log_std = float(initial_log_std)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        self.policy_net = self.policy_net_class(obs_dim, action_dim)
        self.value_net = self.value_net_class(obs_dim)
        self.latent_dim_pi = action_dim
        self.latent_dim_vf = 1

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        action_dim = self.action_space.shape[0]
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * self.initial_log_std,
            requires_grad=True,
        )
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def forward(self, obs, deterministic=False):
        obs = obs.float()
        mean_actions = self.policy_net(obs)
        values = self.value_net(obs).squeeze(-1)
        distribution = DiagGaussianDistribution(self.action_space.shape[0])
        distribution.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        observation = observation.float()
        mean_actions = self.policy_net(observation)
        if deterministic:
            return mean_actions
        std = torch.exp(self.log_std)
        return mean_actions + torch.randn_like(mean_actions) * std

    def predict_values(self, obs):
        obs = obs.float()
        return self.value_net(obs).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        obs = obs.float()
        mean_actions = self.policy_net(obs)
        values = self.value_net(obs).squeeze(-1)
        distribution = DiagGaussianDistribution(self.action_space.shape[0])
        distribution.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy


def load_network_classes(code_path: str):
    spec = importlib.util.spec_from_file_location("policy_net_module", code_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["policy_net_module"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "PolicyNet"):
        raise AttributeError("Generated code is missing class `PolicyNet`")
    if not hasattr(module, "ValueNet"):
        raise AttributeError("Generated code is missing class `ValueNet`")
    return module.PolicyNet, module.ValueNet


def train_and_evaluate(
    code_path,
    total_timesteps,
    max_steps,
    n_eval_episodes,
    seed=42,
    save_dir=None,
    eval_freq=25000,
):
    torch.set_num_threads(1)
    os.makedirs(save_dir, exist_ok=True)

    PolicyNetCls, ValueNetCls = load_network_classes(code_path)

    env = make_vec_env(ENV_ID, n_envs=1, seed=seed)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    policy_kwargs = {
        "policy_net_class": PolicyNetCls,
        "value_net_class": ValueNetCls,
        "initial_log_std": INITIAL_LOG_STD,
    }

    model = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
        device="cuda",
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        policy_kwargs=policy_kwargs,
    )

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=seed + 1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    reward_logger = RewardLoggerCallback(
        save_path=os.path.join(save_dir, "reward_curves.json"),
        verbose=0,
    )
    eval_callback = CustomEvalCallback(
        eval_env,
        reward_logger=reward_logger,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="halfcheetah_v4_evom",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, eval_callback, checkpoint_callback],
        progress_bar=False,
    )

    reward_logger.save_rewards()
    model.save(os.path.join(save_dir, "final_model"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))

    eval_env.training = False
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    eval_rewards = []
    eval_lengths = []
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            if done[0]:
                break
        eval_rewards.append(float(episode_reward))
        eval_lengths.append(int(episode_length))

    results = {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_length": float(np.mean(eval_lengths)),
        "eval_rewards": eval_rewards,
        "eval_lengths": eval_lengths,
        "hyperparameters": {
            "n_timesteps": total_timesteps,
            "normalize": True,
            "n_envs": 1,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "log_std_init": INITIAL_LOG_STD,
            "net_arch": "custom_llm_generated",
            "activation_fn": "custom",
            "ortho_init": "custom",
            "normalize_reward": False,
        },
        "n_timesteps": total_timesteps,
        "seed": seed,
    }

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    eval_env.close()
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--timesteps", type=int, default=5000000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    save_dir = os.path.dirname(args.result_path)

    try:
        result = train_and_evaluate(
            args.code_path,
            args.timesteps,
            args.max_steps,
            args.n_eval_episodes,
            seed=args.seed,
            save_dir=save_dir,
            eval_freq=args.eval_freq,
        )
        result["status"] = "ok"
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "mean_reward": -1000.0,
        }

    result["elapsed"] = time.time() - t0
    summary_result = {
        "status": result.get("status", "error"),
        "mean_reward": result.get("mean_reward", -1000.0),
        "std_reward": result.get("std_reward", 0.0),
        "elapsed": result["elapsed"],
        "seed": args.seed,
        "save_dir": save_dir,
    }
    if "error" in result:
        summary_result["error"] = result["error"]

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
