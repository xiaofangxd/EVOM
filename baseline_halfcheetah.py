
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ENV_ID = "HalfCheetah-v4"

HYPERPARAMS = {
    "n_timesteps": 5_000_000,
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
    "log_std_init": 0.0,
    "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    "activation_fn": torch.nn.Tanh,
    "ortho_init": True,
    "normalize_reward": False,
}


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


def _serializable_hyperparams():
    return {
        key: str(value) if not isinstance(value, (int, float, bool, str)) else value
        for key, value in HYPERPARAMS.items()
    }


def train_baseline(
    save_dir: str,
    n_timesteps: int = None,
    seed: int = 42,
    eval_freq: int = 25000,
    n_eval_episodes: int = 3,
    device: str = "cuda",
):
    os.makedirs(save_dir, exist_ok=True)

    if n_timesteps is None:
        n_timesteps = HYPERPARAMS["n_timesteps"]

    print("=" * 60)
    print("HalfCheetah-v4 Baseline Training with SB3-default 64x64 Tanh MLP Policy")
    print("=" * 60)
    print(f"Total timesteps: {n_timesteps:,}")
    print(f"Evaluation episodes: {n_eval_episodes}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)

    env = make_vec_env(ENV_ID, n_envs=HYPERPARAMS["n_envs"], seed=seed)
    if HYPERPARAMS["normalize"]:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=HYPERPARAMS["normalize_reward"],
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=HYPERPARAMS["gamma"],
        )

    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=seed + 1)
    if HYPERPARAMS["normalize"]:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=HYPERPARAMS["learning_rate"],
        n_steps=HYPERPARAMS["n_steps"],
        batch_size=HYPERPARAMS["batch_size"],
        n_epochs=HYPERPARAMS["n_epochs"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        clip_range=HYPERPARAMS["clip_range"],
        ent_coef=HYPERPARAMS["ent_coef"],
        vf_coef=HYPERPARAMS["vf_coef"],
        max_grad_norm=HYPERPARAMS["max_grad_norm"],
        policy_kwargs=dict(
            net_arch=HYPERPARAMS["net_arch"],
            activation_fn=HYPERPARAMS["activation_fn"],
            ortho_init=HYPERPARAMS["ortho_init"],
            log_std_init=HYPERPARAMS["log_std_init"],
        ),
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
    )

    reward_logger = RewardLoggerCallback(
        save_path=os.path.join(save_dir, "reward_curves.json"),
        verbose=1,
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
        name_prefix="halfcheetah_v4_baseline",
    )

    model.learn(
        total_timesteps=n_timesteps,
        callback=[reward_logger, eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    reward_logger.save_rewards()
    model.save(os.path.join(save_dir, "final_model"))

    if HYPERPARAMS["normalize"]:
        env.save(os.path.join(save_dir, "vec_normalize.pkl"))
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    eval_env.training = False
    eval_rewards = []
    eval_lengths = []

    for i in range(n_eval_episodes):
        obs = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            if done[0]:
                break
        eval_rewards.append(float(episode_reward))
        eval_lengths.append(int(episode_length))
        print(f"Episode {i + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    results = {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_length": float(np.mean(eval_lengths)),
        "eval_rewards": eval_rewards,
        "eval_lengths": eval_lengths,
        "hyperparameters": _serializable_hyperparams(),
        "n_timesteps": n_timesteps,
        "seed": seed,
    }

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    env.close()
    eval_env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train HalfCheetah-v4 baseline with RL-Zoo PPO hyperparameters"
    )
    parser.add_argument("--save-dir", type=str, default="./results/halfcheetah_v4/baseline")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=25000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    results = train_baseline(
        save_dir=save_dir,
        n_timesteps=args.timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
    )

    print("\nTraining complete.")
    print(f"Final mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Results saved in: {save_dir}")


if __name__ == "__main__":
    main()
