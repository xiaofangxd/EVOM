

TASK_NAME = "Ant-v4"

ENV_DESCRIPTION = """Environment: Ant-v4 (MuJoCo)

Observation Space: Box(shape=(27,), float32)
  27-dimensional continuous vector containing:
  - Joint positions (8 dimensions): hip and ankle joints for 4 legs
  - Joint velocities (8 dimensions): angular velocities
  - Center of mass position (2 dimensions): x, y coordinates
  - Center of mass velocity (2 dimensions): vx, vy
  - Orientation (4 dimensions): quaternion representation
  - Angular velocity (3 dimensions): rotational velocities

Action Space: Box(shape=(8,), float32, low=-1.0, high=1.0)
  8-dimensional continuous action vector:
  - Each dimension controls torque applied to one joint
  - Actions are clipped to [-1.0, 1.0] range
  - Controls: 4 hip joints + 4 ankle joints (one per leg)

Task Objective:
  - Make the ant robot move forward as fast as possible
  - Reward = forward_velocity - control_cost - contact_cost + healthy_reward
  - Episode terminates if the ant falls (z-position too low)
  - Typical episode length: 1000 steps
"""


FIXED_STATE_ACTION = """
【Fixed State Representation and Action Space Processing】
All individuals use the same state representation and action space processing:

State Representation:
- Raw observation space: 27-dimensional vector (joint positions, velocities, COM position/velocity, orientation, angular velocity)
- Preprocessing: Normalized via VecNormalize (same as baseline)
- Final input dimension: 27

Action Space:
- Type: Continuous action space, 8 dimensions
- Processing: PolicyNet outputs 8-dimensional continuous values, clipped to [-1.0, 1.0]
- Each action controls torque for one joint (4 hips + 4 ankles)
"""


STATE_DIM = 27
ACTION_DIM = 8
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
