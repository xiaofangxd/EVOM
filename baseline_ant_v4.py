
import os
import argparse
import numpy as np
import torch
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import gymnasium as gym





HYPERPARAMS = {

    'n_timesteps': 5_000_000,
    'normalize': True,
    'n_envs': 1,


    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,


    'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
    'activation_fn': torch.nn.Tanh,
    'ortho_init': True,


    'normalize_reward': False,
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
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.rollout_rewards.append(float(mean_reward))
            self.rollout_steps.append(int(self.num_timesteps))

    def update_eval_rewards(self, step: int, mean_reward: float):
        self.eval_rewards.append(float(mean_reward))
        self.eval_steps.append(int(step))

    def save_rewards(self):
        data = {
            'rollout': {
                'steps': self.rollout_steps,
                'rewards': self.rollout_rewards
            },
            'eval': {
                'steps': self.eval_steps,
                'rewards': self.eval_rewards
            }
        }

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

        if self.verbose > 0:
            print(f"奖励曲线已保存到: {self.save_path}")


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


def train_baseline(
    save_dir: str,
    n_timesteps: int = None,
    seed: int = 42,
    eval_freq: int = 25000,
    n_eval_episodes: int = 3,
    device: str = 'cuda'
):
    os.makedirs(save_dir, exist_ok=True)

    if n_timesteps is None:
        n_timesteps = HYPERPARAMS['n_timesteps']

    print("="*60)
    print("Ant-v4 Baseline Training with Official Hyperparameters")
    print("="*60)
    print(f"Total timesteps: {n_timesteps:,}")
    print(f"Evaluation episodes: {n_eval_episodes}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print("="*60)


    env = make_vec_env(
        'Ant-v4',
        n_envs=HYPERPARAMS['n_envs'],
        seed=seed
    )


    if HYPERPARAMS['normalize']:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=HYPERPARAMS['normalize_reward'],
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=HYPERPARAMS['gamma']
        )


    eval_env = make_vec_env('Ant-v4', n_envs=1, seed=seed + 1)
    if HYPERPARAMS['normalize']:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False
        )


    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=HYPERPARAMS['learning_rate'],
        n_steps=HYPERPARAMS['n_steps'],
        batch_size=HYPERPARAMS['batch_size'],
        n_epochs=HYPERPARAMS['n_epochs'],
        gamma=HYPERPARAMS['gamma'],
        gae_lambda=HYPERPARAMS['gae_lambda'],
        clip_range=HYPERPARAMS['clip_range'],
        ent_coef=HYPERPARAMS['ent_coef'],
        vf_coef=HYPERPARAMS['vf_coef'],
        max_grad_norm=HYPERPARAMS['max_grad_norm'],
        policy_kwargs=dict(
            net_arch=HYPERPARAMS['net_arch'],
            activation_fn=HYPERPARAMS['activation_fn'],
            ortho_init=HYPERPARAMS['ortho_init']
        ),
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=os.path.join(save_dir, 'tensorboard')
    )



    reward_logger = RewardLoggerCallback(
        save_path=os.path.join(save_dir, 'reward_curves.json'),
        verbose=1
    )


    eval_callback = CustomEvalCallback(
        eval_env,
        reward_logger=reward_logger,
        best_model_save_path=os.path.join(save_dir, 'best_model'),
        log_path=os.path.join(save_dir, 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )


    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, 'checkpoints'),
        name_prefix='ant_v4_baseline'
    )


    print("\n开始训练...")
    model.learn(
        total_timesteps=n_timesteps,
        callback=[reward_logger, eval_callback, checkpoint_callback],
        progress_bar=True
    )


    reward_logger.save_rewards()


    final_model_path = os.path.join(save_dir, 'final_model')
    model.save(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")


    if HYPERPARAMS['normalize']:
        env.save(os.path.join(save_dir, 'vec_normalize.pkl'))
        print(f"VecNormalize 统计信息已保存")


    print("\n" + "="*60)
    print("最终评估")
    print("="*60)

    eval_env.training = False
    eval_rewards = []
    eval_lengths = []

    for i in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                break

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        print(f"Episode {i+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    mean_length = np.mean(eval_lengths)

    print("="*60)
    print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"平均长度: {mean_length:.2f}")
    print("="*60)


    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'eval_rewards': [float(r) for r in eval_rewards],
        'eval_lengths': [int(l) for l in eval_lengths],
        'hyperparameters': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                           for k, v in HYPERPARAMS.items()},
        'n_timesteps': n_timesteps,
        'seed': seed
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    env.close()
    eval_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Ant-v4 baseline with official hyperparameters')
    parser.add_argument('--save-dir', type=str, default='./results/ant_v4/baseline',
                       help='Directory to save models and logs')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total timesteps to train (default: 5M)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval-freq', type=int, default=25000,
                       help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=3,
                       help='Number of episodes for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')

    args = parser.parse_args()


    save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')

    results = train_baseline(
        save_dir=save_dir,
        n_timesteps=args.timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device
    )

    print("\n训练完成！")
    print(f"最终平均奖励: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"结果保存在: {save_dir}")


if __name__ == '__main__':
    main()
