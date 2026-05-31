# Baseline训练使用说明

## 修改内容

### 1. 新增功能
- ✅ **自动保存训练和评估奖励曲线** 到 `reward_curves.json`
- ✅ **统一的结果保存路径** `./results/{task}/baseline/seed_{seed}/`
- ✅ **多seed并行训练脚本**

### 2. 文件结构

修改后的训练结果保存在：
```
results/
├── ant/
│   └── baseline/
│       ├── seed_42/
│       │   ├── reward_curves.json    # 训练和评估奖励曲线
│       │   ├── results.json          # 最终评估结果
│       │   ├── best_model/           # 最佳模型
│       │   ├── final_model.zip       # 最终模型
│       │   ├── checkpoints/          # 训练checkpoint
│       │   ├── tensorboard/          # TensorBoard日志
│       │   └── vec_normalize.pkl     # 环境归一化参数
│       ├── seed_123/
│       └── seed_456/
└── humanoid/
    └── baseline/
        ├── seed_42/
        ├── seed_123/
        └── seed_456/
```

### 3. reward_curves.json 格式

```json
{
  "rollout": {
    "steps": [512, 1024, 1536, ...],
    "rewards": [100.5, 150.2, 200.8, ...]
  },
  "eval": {
    "steps": [10000, 20000, 30000, ...],
    "rewards": [1000.0, 1500.5, 2000.3, ...]
  }
}
```

- `rollout`: 训练过程中的奖励 (rollout/ep_rew_mean)
- `eval`: 评估阶段的奖励 (eval/mean_reward)

## 使用方法

### 方法1: 单个seed训练

#### Ant任务
```bash
# 使用默认seed=42
python baseline_ant.py

# 指定seed
python baseline_ant.py --seed 123

# 自定义保存路径
python baseline_ant.py --seed 456 --save-dir ./my_results/ant/baseline
```

#### Humanoid任务
```bash
# 使用默认seed=42
python baseline_humanoid.py

# 指定seed
python baseline_humanoid.py --seed 123
```

### 方法2: 多seed自动训练（推荐）

#### Ant任务 - 训练3个seeds
```bash
python run_baseline_ant_seeds.py
```

自动运行 seed 42, 123, 456 三次训练，结果保存到 `./results/ant/baseline/`

#### Humanoid任务 - 训练3个seeds
```bash
python run_baseline_humanoid_seeds.py
```

自动运行 seed 42, 123, 456 三次训练，结果保存到 `./results/humanoid/baseline/`

## 读取和分析结果

### Python示例：加载奖励曲线

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 加载单个seed的奖励曲线
with open('./results/ant/baseline/seed_42/reward_curves.json', 'r') as f:
    data = json.load(f)

rollout_steps = data['rollout']['steps']
rollout_rewards = data['rollout']['rewards']
eval_steps = data['eval']['steps']
eval_rewards = data['eval']['rewards']

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(np.array(rollout_steps)/1e6, rollout_rewards, 
         label='Training (rollout)', alpha=0.7)
plt.plot(np.array(eval_steps)/1e6, eval_rewards, 
         label='Evaluation', linewidth=2)
plt.xlabel('Training Steps (M)')
plt.ylabel('Reward')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('reward_curve.png')
```

### Python示例：多seed平均

```python
import json
import numpy as np
import matplotlib.pyplot as plt

seeds = [42, 123, 456]
all_eval_rewards = []

# 加载所有seeds的评估曲线
for seed in seeds:
    path = f'./results/ant/baseline/seed_{seed}/reward_curves.json'
    with open(path, 'r') as f:
        data = json.load(f)
    all_eval_rewards.append(data['eval']['rewards'])

# 计算均值和标准差
eval_steps = data['eval']['steps']  # 假设所有seeds的steps相同
mean_rewards = np.mean(all_eval_rewards, axis=0)
std_rewards = np.std(all_eval_rewards, axis=0)

# 绘制均值±标准差
plt.figure(figsize=(10, 6))
steps_m = np.array(eval_steps) / 1e6
plt.plot(steps_m, mean_rewards, label='Mean', linewidth=2)
plt.fill_between(steps_m, 
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.3, label='±1 std')
plt.xlabel('Training Steps (M)')
plt.ylabel('Evaluation Reward')
plt.title('Ant-v5 Baseline (3 seeds)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('baseline_mean_curve.png')
```

## 参数说明

### baseline_ant.py / baseline_humanoid.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-dir` | `./results/{task}/baseline` | 保存路径 |
| `--timesteps` | 10000000 | 总训练步数 |
| `--seed` | 42 | 随机种子 |
| `--eval-freq` | 10000 (ant) / 25000 (humanoid) | 评估频率 |
| `--n-eval-episodes` | 10 | 每次评估的episode数 |
| `--device` | cuda | 训练设备 |

## 注意事项

1. **训练时间**：
   - Ant: 约4-5小时 (单GPU, 10M steps)
   - Humanoid: 约8-10小时 (单GPU, 10M steps)

2. **磁盘空间**：
   - 每个seed约需要 500MB (包含模型、日志、曲线)
   - 3个seeds约需要 1.5GB

3. **GPU显存**：
   - Ant: 约2GB
   - Humanoid: 约3GB

4. **评估频率**：
   - Ant: 每10K steps评估一次 (约890次评估)
   - Humanoid: 每25K steps评估一次 (约400次评估)

## 与旧版本的兼容性

旧的baseline结果保存在：
- `./baseline_ant_full_seed{42,123,456}/`
- `./baseline_humanoid_full_seed{42,123,456}/`

新版本保存在：
- `./results/ant/baseline/seed_{42,123,456}/`
- `./results/humanoid/baseline/seed_{42,123,456}/`

两者互不影响，可以共存。
