"""
Configuration for Cuboid House Construction Agent.
All hyperparameters and constants in one place.
"""
import numpy as np


# ==============================================================================
# WORLD
# ==============================================================================
WORLD_SIZE = 40
GROUND_Y = 0

# Minecraft world Y coordinate of the ground surface block (grass layer).
# CraftGround uses vanilla MC 1.21 default superflat preset:
#   y=-64 bedrock, y=-63 dirt, y=-62 dirt, y=-61 grass (surface)
#   Player stands at y=-60 on the grass.
# Conversion: internal_y = minecraft_y - MINECRAFT_GROUND_Y
#             minecraft_y = internal_y + MINECRAFT_GROUND_Y
MINECRAFT_GROUND_Y = -61

# Block types
AIR = 0
OAK_PLANKS = 1
SOLID = 2  # ground, barrier, bedrock — anything unbreakable

NUM_BLOCK_TYPES = 3  # air, oak_planks, solid

# Barrier walls at edges
BARRIER_POSITIONS = {
    "x_min": 0, "x_max": WORLD_SIZE - 1,
    "z_min": 0, "z_max": WORLD_SIZE - 1,
}

# ==============================================================================
# HOUSE BLUEPRINT
# ==============================================================================
HOUSE_ORIGIN_X = 17
HOUSE_ORIGIN_Z = 17
HOUSE_WIDTH = 7     # x direction
HOUSE_DEPTH = 7     # z direction
HOUSE_HEIGHT = 5    # y=1 (floor) to y=5 (ceiling)

FLOOR_Y = 1
WALL_Y_MIN = 2
WALL_Y_MAX = 4
CEILING_Y = 5

# Door position (south wall, non-corner, vertically adjacent)
DOOR_X = 20
DOOR_Z = HOUSE_ORIGIN_Z  # south face (z=17)
DOOR_Y_BOTTOM = 2
DOOR_Y_TOP = 3

# Expected block counts
FLOOR_BLOCKS = HOUSE_WIDTH * HOUSE_DEPTH  # 49
WALL_BLOCKS = 70  # 3 layers of perimeter minus 2 door blocks
CEILING_BLOCKS = HOUSE_WIDTH * HOUSE_DEPTH  # 49
TOTAL_BLOCKS = FLOOR_BLOCKS + WALL_BLOCKS + CEILING_BLOCKS  # 168

# ==============================================================================
# SPAWN
# ==============================================================================
SPAWN_X = 20
SPAWN_Y = 1  # standing on ground
SPAWN_Z = 15  # 2 blocks south of house edge (door at z=17)
SPAWN_YAW = 0.0  # facing +Z (toward house center)
SPAWN_PITCH = 0.71  # ~41° downward — hits house floor at z≈18, dist 4.0 (within 4.5 reach)

# ==============================================================================
# OBSERVATION
# ==============================================================================
LOCAL_WINDOW_SIZE = 11  # 11x11x11 agent-centered
LOCAL_WINDOW_HALF = LOCAL_WINDOW_SIZE // 2  # 5

# Voxel grid channels
BLOCK_GRID_CHANNELS = NUM_BLOCK_TYPES  # 3 (air, planks, solid one-hot)
VISIBILITY_GRID_CHANNELS = NUM_BLOCK_TYPES  # 3 (air, visible, other)
BLUEPRINT_CHANNELS = 2  # (1) blueprint target, (2) needs placement
STACKED_CHANNELS = BLOCK_GRID_CHANNELS + VISIBILITY_GRID_CHANNELS + BLUEPRINT_CHANNELS  # 8

# Feature sizes
AGENT_STATE_SIZE = 15  # pos(3) + orient(2) + hotbar_slot(9) + has_planks(1)
RAYCAST_INFO_SIZE = 25  # reduced from 29 with fewer block types
INVENTORY_SIZE = 10  # current_slot(9) + has_planks(1)
COMPLETION_SIZE = 4  # floor, wall, ceiling, door ratios
SUBTASK_ID_SIZE = 4  # one-hot: floor, walls, ceiling, done
PREV_ACTION_SIZE = 8  # one per action dimension
TIME_REMAINING_SIZE = 1
STUCK_RATIO_SIZE = 1  # stuck_steps / patience (0.0 ~ 1.0)
TARGET_DIRECTION_SIZE = 3  # delta_yaw(-1~+1), delta_pitch(-1~+1), distance(normalized)

NON_VOXEL_SIZE = (
    AGENT_STATE_SIZE + RAYCAST_INFO_SIZE + INVENTORY_SIZE +
    COMPLETION_SIZE + SUBTASK_ID_SIZE + PREV_ACTION_SIZE +
    TIME_REMAINING_SIZE + STUCK_RATIO_SIZE + TARGET_DIRECTION_SIZE
)  # ~71

# Raycast
RAYCAST_MAX_DISTANCE = 5.0

# ==============================================================================
# ACTION SPACE
# ==============================================================================
# MultiDiscrete([3, 3, 2, 2, 3, 9, 7, 7])
ACTION_DIMS = [3, 3, 2, 2, 3, 9, 7, 7]
NUM_ACTION_DIMS = len(ACTION_DIMS)
TOTAL_ACTION_LOGITS = sum(ACTION_DIMS)  # 36

# Action dimension indices
ACT_FWD_BACK = 0    # 0=back, 1=stop, 2=forward
ACT_LEFT_RIGHT = 1  # 0=left, 1=stop, 2=right
ACT_JUMP = 2        # 0=no, 1=yes
ACT_SNEAK = 3       # 0=no, 1=yes
ACT_INTERACT = 4    # 0=place, 1=nothing, 2=attack (attack masked in survival)
ACT_HOTBAR = 5      # 0-8: select slot
ACT_PITCH = 6       # camera pitch delta
ACT_YAW = 7         # camera yaw delta

# Camera delta mapping (index -> degrees per step)
# Camera delta in degrees per step.
CAMERA_DELTA_MAP = [-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]

# Hotbar: slot 0 has oak planks, rest are empty
PLANKS_SLOT = 0

# Action masking: only planks slot is valid in hotbar
def get_hotbar_mask():
    """Returns mask for hotbar dimension. True = valid."""
    mask = [False] * 9
    mask[PLANKS_SLOT] = True
    return mask

# ==============================================================================
# INITIAL POLICY BIAS
# ==============================================================================
INITIAL_BIAS = np.zeros(TOTAL_ACTION_LOGITS, dtype=np.float32)

# [0] Forward/Back (indices 0-2)
INITIAL_BIAS[0:3] = [-0.7, 0.7, 1.0]        # ~50% fwd, 40% stop, 10% back
# [1] Left/Right (indices 3-5)
INITIAL_BIAS[3:6] = [-0.5, 2.0, -0.5]        # ~73% straight
# [2] Jump (indices 6-7)
INITIAL_BIAS[6:8] = [1.0, -1.0]              # ~88% no jump
# [3] Sneak (indices 8-9)
INITIAL_BIAS[8:10] = [2.0, -2.0]             # ~98% no sneak
# [4] Interact (indices 10-12): place / nothing / attack(masked)
INITIAL_BIAS[10:13] = [1.5, 0.5, -10.0]      # ~73% place, attack masked out
# [5] Hotbar (indices 13-21)
INITIAL_BIAS[13:22] = [2.0, 1.0, 0, 0, 0, 0, 0, 0, 0]
# [6] Pitch (indices 22-28): delta map [-10, -3, -1, 0, +1, +3, +10] degrees
#     Positive = look down. Favor staying near current angle with slight downward bias.
INITIAL_BIAS[22:29] = [-2.0, -1.0, -0.5, 3.0, -0.5, -1.0, -2.0]  # symmetric, no drift
# [7] Yaw (indices 29-35)
INITIAL_BIAS[29:36] = [-2.0, -1.0, -0.5, 3.0, -0.5, -1.0, -2.0]  # ~85% center

# ==============================================================================
# REWARDS
# ==============================================================================
REWARD_CORRECT_PLACEMENT = 5.0
REWARD_INCORRECT_PLACEMENT = -0.5
REWARD_INCORRECT_REMOVAL = -0.5
REWARD_PROGRESS_SCALE = 15.0
REWARD_TIME_PENALTY = -0.005
REWARD_FLOOR_COMPLETE = 15.0
REWARD_WALLS_COMPLETE = 20.0
REWARD_CEILING_COMPLETE = 15.0
REWARD_DOOR_CORRECT = 10.0
REWARD_HOUSE_COMPLETE = 100.0
REWARD_STUCK_PENALTY = -10.0
REWARD_PROXIMITY_PER_BLOCK = -0.01     # penalty per block beyond threshold (linear)
REWARD_STATIONARY_PENALTY = -0.02      # per step when staying at same block too long
STATIONARY_THRESHOLD = 500             # steps at same block before penalty kicks in
REWARD_LOOKING_AT_TARGET = 0.02        # per step when aiming at unbuilt blueprint position
HOUSE_PROXIMITY_THRESHOLD = 5          # manhattan distance threshold from house edge

# ==============================================================================
# EPISODE
# ==============================================================================
# CraftGround: 1 step = 1 tick (0.05s), 3000 steps = 150 seconds
MAX_EPISODE_STEPS = 3000
STUCK_PATIENCE = 1000      # steps without progress before termination
STUCK_MIN_DELTA = 0.001    # minimum completion change to count as progress


# ==============================================================================
# SUBTASK IDs
# ==============================================================================
SUBTASK_FLOOR = 0
SUBTASK_WALLS = 1
SUBTASK_CEILING = 2
SUBTASK_DONE = 3
NUM_SUBTASKS = 4

# ==============================================================================
# NETWORK
# ==============================================================================
CNN_CHANNELS = [32, 64, 128]
CNN_KERNEL_SIZE = 3
CNN_STRIDE_LAST = 2
CNN_OUTPUT_SIZE = 3456  # 3*3*3*128 after 3 conv layers

SHARED_MLP_SIZE = 512
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
ACTOR_MLP_SIZE = 128
CRITIC_MLP_SIZE = 128

# ==============================================================================
# PPO HYPERPARAMETERS
# ==============================================================================
SEQUENCE_LENGTH = 64
GAMMA = 0.995
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4
CLIP_RATIO = 0.2
ENTROPY_COEFF_START = 0.05   # initial entropy (high → explore)
ENTROPY_COEFF_END = 0.005    # final entropy (low → exploit)
ENTROPY_COEFF = ENTROPY_COEFF_START  # default (overridden by annealing)
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 2048
MINI_BATCH_SIZE = 512
UPDATE_EPOCHS = 4

# ==============================================================================
# STAGE 1: GAZE TRAINING (Curriculum)
# ==============================================================================
GAZE_TIME_PENALTY = -0.05           # higher than stage 2 to outweigh proximity
GAZE_REWARD_SUCCESS = 1.0           # reward when looking at current target
GAZE_REWARD_ANGULAR_SCALE = 0.1     # shaping reward for reducing angle to target
GAZE_REWARD_TARGET_ADVANCE = 0.5    # bonus for completing a target
GAZE_MAX_EPISODE_STEPS = 2500       # 20 targets × 100 steps + 500 buffer
GAZE_STEPS_PER_TARGET = 100         # steps budget per target
GAZE_NUM_TARGETS_PER_EPISODE = 20   # targets per episode (subset of 168)
GAZE_GRADUATION_THRESHOLD = 0.80    # success rate for graduation
GAZE_GRADUATION_WINDOW = 10         # episodes for rolling average

# Gaze-specific initial bias: flatter camera distribution
GAZE_INITIAL_BIAS = INITIAL_BIAS.copy()
# Forward/back: equal probability (33% each)
GAZE_INITIAL_BIAS[0:3] = [0.0, 0.0, 0.0]
# Left/right: equal probability (33% each)
GAZE_INITIAL_BIAS[3:6] = [0.0, 0.0, 0.0]
# Pitch: more exploration, less center-heavy
GAZE_INITIAL_BIAS[sum(ACTION_DIMS[:ACT_PITCH]):sum(ACTION_DIMS[:ACT_PITCH+1])] = \
    [-1.0, -0.3, 0.0, 0.5, 0.0, -0.3, -1.0]
# Yaw: more exploration
GAZE_INITIAL_BIAS[sum(ACTION_DIMS[:ACT_YAW]):sum(ACTION_DIMS[:ACT_YAW+1])] = \
    [-1.0, -0.3, 0.0, 0.5, 0.0, -0.3, -1.0]
