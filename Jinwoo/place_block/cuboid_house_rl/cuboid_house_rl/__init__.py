# Cuboid House Construction Agent
# Recurrent PPO with Rule-Based Subtask Controller
# Platform: CraftGround 2.6.15

"""
Project Structure:
    cuboid_house_rl/
    ├── envs/
    │   ├── __init__.py
    │   ├── house_building_env.py    # Main Gymnasium environment wrapper
    │   ├── action_wrapper.py        # MultiDiscrete action conversion
    │   └── observation_builder.py   # Observation preprocessing
    ├── models/
    │   ├── __init__.py
    │   ├── network.py               # 3D CNN + LSTM actor-critic
    │   └── action_dist.py           # Masked multi-discrete distribution
    ├── training/
    │   ├── __init__.py
    │   ├── ppo.py                   # Recurrent PPO algorithm
    │   ├── rollout_buffer.py        # Sequence-aware rollout storage
    │   └── train.py                 # Main training script
    ├── utils/
    │   ├── __init__.py
    │   ├── blueprint.py             # House blueprint generation
    │   ├── completion.py            # Completion ratio tracking
    │   ├── stuck_detector.py        # Stuck detection logic
    │   └── coord_transform.py       # Agent-relative coordinate transforms
    └── config.py                    # All hyperparameters and constants
"""
