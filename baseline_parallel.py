
import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict


GPU_IDS = [4, 5, 6]
GPU_PARALLEL = 3


def _build_gpu_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _train_one_seed(
    task: str,
    seed: int,
    gpu_id: int,
    save_dir: str,
    timesteps: int,
    eval_freq: int,
    n_eval_episodes: int
) -> Dict:
    start_time = time.time()


    if task == "ant":
        script = "baseline_ant_v4.py"
    elif task == "halfcheetah":
        script = "baseline_halfcheetah.py"
    else:
        return {
            "seed": seed,
            "gpu_id": gpu_id,
            "success": False,
            "error": f"未知任务: {task}"
        }


    seed_dir = os.path.join(save_dir, f"seed_{seed}")
    cmd = [
        sys.executable,
        script,
        "--save-dir", save_dir,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--eval-freq", str(eval_freq),
        "--n-eval-episodes", str(n_eval_episodes),
        "--device", "cuda"
    ]


    os.makedirs(seed_dir, exist_ok=True)
    log_path = os.path.join(seed_dir, "training.log")

    print(f"  启动 seed={seed} (GPU{gpu_id})...")

    try:

        env = _build_gpu_env(gpu_id)

        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

        elapsed = time.time() - start_time

        if proc.returncode != 0:
            return {
                "seed": seed,
                "gpu_id": gpu_id,
                "success": False,
                "error": f"训练失败，退出码={proc.returncode}，详见 {log_path}",
                "elapsed": elapsed
            }


        results_path = os.path.join(seed_dir, "results.json")
        if not os.path.exists(results_path):
            return {
                "seed": seed,
                "gpu_id": gpu_id,
                "success": False,
                "error": f"未找到结果文件: {results_path}",
                "elapsed": elapsed
            }

        with open(results_path, "r") as f:
            results = json.load(f)

        print(f"  ✓ seed={seed} (GPU{gpu_id}, {elapsed/60:.1f}min): "
              f"奖励={results['mean_reward']:.2f}±{results['std_reward']:.2f}")

        return {
            "seed": seed,
            "gpu_id": gpu_id,
            "success": True,
            "mean_reward": results["mean_reward"],
            "std_reward": results["std_reward"],
            "elapsed": elapsed,
            "save_dir": seed_dir
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "seed": seed,
            "gpu_id": gpu_id,
            "success": False,
            "error": f"异常: {str(e)}",
            "elapsed": elapsed
        }


def train_baseline_parallel(
    task: str,
    seeds: List[int],
    save_dir: str,
    timesteps: int = 5_000_000,
    eval_freq: int = 25000,
    n_eval_episodes: int = 5,
    gpu_parallel: int = None
):
    os.makedirs(save_dir, exist_ok=True)

    if gpu_parallel is None:
        gpu_parallel = GPU_PARALLEL

    print("="*80)
    print(f"Baseline并行训练 - {task.upper()}")
    print("="*80)
    print(f"任务: {task}")
    print(f"Seeds: {seeds}")
    print(f"训练步数: {timesteps:,}")
    print(f"评估频率: {eval_freq:,}")
    print(f"评估episodes: {n_eval_episodes}")
    print(f"GPU并行数: {gpu_parallel}")
    print(f"使用GPU: {GPU_IDS}")
    print(f"保存目录: {save_dir}")
    print("="*80)


    gpu_pool = GPU_IDS
    results = []

    with ThreadPoolExecutor(max_workers=gpu_parallel) as executor:
        futures = []

        for idx, seed in enumerate(seeds):
            gpu_id = gpu_pool[idx % len(gpu_pool)]
            future = executor.submit(
                _train_one_seed,
                task, seed, gpu_id, save_dir,
                timesteps, eval_freq, n_eval_episodes
            )
            futures.append(future)


        for future in as_completed(futures):
            result = future.result()
            results.append(result)


    print("\n" + "="*80)
    print("训练完成")
    print("="*80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"成功: {len(successful)}/{len(results)}")

    if successful:
        rewards = [r["mean_reward"] for r in successful]
        mean_reward = sum(rewards) / len(rewards)
        print(f"\n所有seeds平均奖励: {mean_reward:.2f}")
        print("\n各seed结果:")
        for r in sorted(successful, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']}: {r['mean_reward']:.2f}±{r['std_reward']:.2f} "
                  f"(GPU{r['gpu_id']}, {r['elapsed']/60:.1f}min)")

    if failed:
        print(f"\n失败的seeds ({len(failed)}):")
        for r in sorted(failed, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']} (GPU{r['gpu_id']}): {r['error']}")


    summary = {
        "task": task,
        "seeds": seeds,
        "timesteps": timesteps,
        "total_runs": len(results),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "results": results
    }

    if successful:
        summary["mean_reward_across_seeds"] = mean_reward
        summary["rewards"] = rewards

    summary_path = os.path.join(save_dir, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n汇总结果已保存到: {summary_path}")
    print("="*80)

    return summary


def main():
    global GPU_IDS, GPU_PARALLEL

    parser = argparse.ArgumentParser(
        description="Baseline并行训练 - 使用与EVOM相同的GPU并行策略"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["ant", "halfcheetah"],
        help="任务名称"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="随机种子列表（默认: 42 123 456）"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="保存目录"
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="训练步数（默认: 5M）"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=25000,
        help="评估频率（默认: 25000）"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5,
        help="评估episodes数（默认: 5）"
    )
    parser.add_argument(
        "--gpu-parallel", type=int, default=None,
        help=f"GPU并行数（默认: {GPU_PARALLEL}）"
    )
    parser.add_argument(
        "--gpu-ids", type=int, nargs="+", default=None,
        help=f"使用的GPU编号列表（默认: {' '.join(map(str, GPU_IDS))}）"
    )

    args = parser.parse_args()

    if args.gpu_ids is not None:
        GPU_IDS = args.gpu_ids
        if args.gpu_parallel is None:
            GPU_PARALLEL = len(GPU_IDS)


    summary = train_baseline_parallel(
        task=args.task,
        seeds=args.seeds,
        save_dir=args.save_dir,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        gpu_parallel=args.gpu_parallel
    )


    if summary["failed_runs"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
