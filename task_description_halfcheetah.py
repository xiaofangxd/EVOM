
TASK_NAME = "HalfCheetah-v4"

ENV_DESCRIPTION = """Environment: HalfCheetah-v4 (MuJoCo)

Observation Space: Box(shape=(17,), float64)
  17-dimensional continuous vector containing:
  - Joint positions for the torso and six actuated joints, excluding absolute x-position
  - Joint velocities for the torso and six actuated joints

Action Space: Box(shape=(6,), float32, low=-1.0, high=1.0)
  6-dimensional continuous action vector:
  - Torques for back thigh, back shin, back foot
  - Torques for front thigh, front shin, front foot
  - Actions are clipped to [-1.0, 1.0]

Task Objective:
  - Make the half-cheetah run forward as fast as possible
  - Reward = forward_reward - control_cost
  - The task normally does not terminate early; episodes truncate at 1000 steps
"""

FIXED_STATE_ACTION = """
[Fixed State Representation and Action Space Processing]
All individuals use the same state representation and action space processing:

State Representation:
- Raw observation space: 17-dimensional vector of positions and velocities
- Preprocessing: Normalized via VecNormalize, same as baseline
- Final input dimension: 17

Action Space:
- Type: Continuous action space, 6 dimensions
- Processing: PolicyNet outputs 6-dimensional continuous values
- Each action controls torque for one HalfCheetah hinge joint
"""

STATE_DIM = 17
ACTION_DIM = 6
ACTION_TYPE = "continuous"

EVAL_SEEDS = [42]
MAX_EPISODE_STEPS = 1000

TRAIN_CONFIG = {
    "timesteps": 100000,
    "max_steps": 1000,
}

FULL_TRAIN_CONFIG = {
    "timesteps": 5000000,
    "max_steps": 1000,
    "eval_freq": 25000,
    "n_eval_episodes": 5,
}
