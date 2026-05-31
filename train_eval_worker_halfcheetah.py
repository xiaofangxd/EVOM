
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
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ENV_ID = "HalfCheetah-v4"
INITIAL_LOG_STD = 0.0


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


def train_and_evaluate(code_path, total_timesteps, max_steps, seed=42):
    torch.set_num_threads(1)

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
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=total_timesteps)

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=seed + 1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    eval_rewards = []
    for _ in range(3):
        obs = eval_env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_reward += reward[0]
            if done[0]:
                break
        eval_rewards.append(float(ep_reward))

    eval_env.close()
    env.close()

    return {
        "best_reward": float(np.mean(eval_rewards)),
        "eval_rewards": eval_rewards,
        "total_timesteps": total_timesteps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.timesteps is not None:
        total_timesteps = args.timesteps
    elif args.episodes is not None:
        total_timesteps = args.episodes * 1000
    else:
        total_timesteps = 100000

    t0 = time.time()
    try:
        result = train_and_evaluate(
            args.code_path,
            total_timesteps,
            args.max_steps,
            seed=args.seed,
        )
        result["status"] = "ok"
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "best_reward": -1000.0,
        }

    result["elapsed"] = time.time() - t0
    result["seed"] = args.seed
    result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
