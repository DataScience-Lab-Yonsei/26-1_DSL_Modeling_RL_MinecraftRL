"""
Configuration for Cuboid House Construction Agent.

V3: Infinite superflat world, random house size, origin-based coordinates,
    4-slot inventory, jump+place wall building.
"""
import numpy as np


# ==============================================================================
# WORLD (infinite superflat, no size limit)
# ==============================================================================
MINECRAFT_GROUND_Y = -61  # superflat ground surface block Y

# Block types
AIR = 0
OAK_PLANKS = 1
SOLID = 2  # ground, barrier, bedrock

NUM_BLOCK_TYPES = 3

# ==============================================================================
# HOUSE (dynamic — set per episode)
# ==============================================================================
HOUSE_WIDTH_MIN = 5
HOUSE_WIDTH_MAX = 10
HOUSE_DEPTH_MIN = 5
HOUSE_DEPTH_MAX = 10
HOUSE_HEIGHT = 5  # fixed: floor(y=1) + 4 wall layers(y=2,3,4,5)

FLOOR_Y = 1
WALL_Y_MIN = 2
WALL_Y_MAX = 5
WALL_HEIGHT = 4  # y=2,3,4,5

# Door: front wall center, y=2,3 (2 blocks tall)
DOOR_HEIGHT_BOTTOM = 2
DOOR_HEIGHT_TOP = 3

# ==============================================================================
# INVENTORY (4 slots)
# ==============================================================================
SLOT_PLANKS = 0       # slots 0~4: oak_planks (5 stacks = 320)
SLOT_AXE = 5          # slot 5: diamond_axe
SLOT_DOOR = 6         # slot 6: oak_door
SLOT_GLASS = 7        # slots 7~8: glass (2 stacks = 128)
NUM_INVENTORY_SLOTS = 9

INVENTORY_ITEMS = {
    SLOT_PLANKS: "oak_planks",
    SLOT_AXE: "diamond_axe",
    SLOT_DOOR: "oak_door",
    SLOT_GLASS: "glass",
}

# ==============================================================================
# OBSERVATION (V3: 45 dimensions, origin-based)
# ==============================================================================

# --- origin_set flag ---
ORIGIN_SET_SIZE = 1  # 0 or 1

# --- Agent State: pos(3) + orient(4) + hotbar(1) + has_planks(1) + has_axe(1) ---
AGENT_STATE_SIZE = 10  # pos(3) + sin/cos yaw(2) + sin/cos pitch(2) + hotbar(1) + has_planks(1) + has_axe(1)

# --- Raycast Info (origin-based absolute coords) ---
# ray_hit(1) + hit_block_type(3) + hit_pos_absolute(3) + hit_distance(1)
# + face_normal(3) + placement_pos_absolute(3) + placement_valid(1)
# + hit_matches_blueprint(1) + placement_is_correct(1)
RAYCAST_INFO_SIZE = 17

# --- Global Progress ---
PROGRESS_SIZE = 3  # floor_ratio, wall_ratio, ceiling_ratio

# --- Incorrect blocks count ---
INCORRECT_COUNT_SIZE = 1

# --- Time & Stuck ---
TIME_REMAINING_SIZE = 1
STUCK_RATIO_SIZE = 1

# --- Target Direction (relative, sin/cos) ---
# sin(delta_yaw), cos(delta_yaw), sin(delta_pitch), cos(delta_pitch)
TARGET_DIRECTION_SIZE = 4

# --- Target Distance ---
TARGET_DISTANCE_SIZE = 1

# --- Target Absolute Position (origin-based) ---
TARGET_ABSOLUTE_SIZE = 3

# --- House Size ---
HOUSE_SIZE_SIZE = 3  # width, depth, height

# --- Total ---
FLAT_OBS_SIZE = (
    ORIGIN_SET_SIZE +
    AGENT_STATE_SIZE + RAYCAST_INFO_SIZE +
    PROGRESS_SIZE + INCORRECT_COUNT_SIZE +
    TIME_REMAINING_SIZE + STUCK_RATIO_SIZE +
    TARGET_DIRECTION_SIZE + TARGET_DISTANCE_SIZE +
    TARGET_ABSOLUTE_SIZE +
    HOUSE_SIZE_SIZE
)  # 1+10+17+3+1+1+1+4+1+3+3 = 45

# Raycast
RAYCAST_MAX_DISTANCE = 5.0

# ==============================================================================
# ACTION SPACE
# ==============================================================================
# MultiDiscrete([3, 3, 2, 2, 3, 9, 9, 9])
ACTION_DIMS = [3, 3, 2, 2, 3, 9, 9, 9]
NUM_ACTION_DIMS = len(ACTION_DIMS)
TOTAL_ACTION_LOGITS = sum(ACTION_DIMS)  # 40

ACT_FWD_BACK = 0    # 0=back, 1=stop, 2=forward
ACT_LEFT_RIGHT = 1  # 0=left, 1=stop, 2=right
ACT_JUMP = 2        # 0=no, 1=yes
ACT_SNEAK = 3       # 0=no, 1=yes
ACT_INTERACT = 4    # 0=place, 1=nothing, 2=attack
ACT_HOTBAR = 5      # 0-8: select slot
ACT_PITCH = 6       # camera pitch delta
ACT_YAW = 7         # camera yaw delta

CAMERA_DELTA_MAP = [-10.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 10.0]

PLANKS_SLOT = SLOT_PLANKS  # backward compat alias

# ==============================================================================
# INITIAL POLICY BIAS
# ==============================================================================
INITIAL_BIAS = np.zeros(TOTAL_ACTION_LOGITS, dtype=np.float32)
INITIAL_BIAS[0:3] = [-0.7, 0.7, 1.0]
INITIAL_BIAS[3:6] = [-0.5, 2.0, -0.5]
INITIAL_BIAS[6:8] = [1.0, -1.0]
INITIAL_BIAS[8:10] = [2.0, -2.0]
INITIAL_BIAS[10:13] = [1.5, 0.5, -10.0]
INITIAL_BIAS[13:22] = [2.0, 1.0, 0, 0, 0, 0, 0, 0, 0]
# Pitch: [-10, -3, -1, -0.3, 0, +0.3, +1, +3, +10] — center-biased
INITIAL_BIAS[22:31] = [-2.0, -1.0, -0.5, -0.2, 3.0, -0.2, -0.5, -1.0, -2.0]
# Yaw: same
INITIAL_BIAS[31:40] = [-2.0, -1.0, -0.5, -0.2, 3.0, -0.2, -0.5, -1.0, -2.0]

# ==============================================================================
# REWARDS (V3)
# ==============================================================================
# Block placement
REWARD_CORRECT_PLACEMENT = 5.0       # blueprint position (first time only)
REWARD_INCORRECT_PLACEMENT = -0.5    # outside blueprint
REWARD_INCORRECT_BLOCK_REMOVAL = 0.3 # removing a misplaced block (axe)
REWARD_CORRECT_BLOCK_REMOVAL = -2.0  # accidentally removing a correct block

# Completion bonuses
REWARD_STAGE_COMPLETE = 50.0         # floor/wall/ceiling 100%
REWARD_COLUMN_COMPLETE = 3.0         # wall: 4-block column done

# Penalties
REWARD_TIME_PENALTY = -0.005         # per tick
REWARD_PROXIMITY_PER_BLOCK = -0.01   # per block beyond threshold
HOUSE_PROXIMITY_THRESHOLD = 5        # manhattan distance from house edge
REWARD_STUCK_PENALTY = -10.0         # no progress for STUCK_PATIENCE ticks

# ==============================================================================
# EPISODE
# ==============================================================================
MAX_EPISODE_STEPS = 10000  # more steps for random sizes + walls
STUCK_PATIENCE = 1000
STUCK_MIN_DELTA = 0.001

# ==============================================================================
# SUBTASK IDs (used internally, NOT in observation)
# ==============================================================================
SUBTASK_FLOOR = 0
SUBTASK_WALLS = 1
SUBTASK_DOOR = 2
SUBTASK_CEILING = 3
SUBTASK_DONE = 4
NUM_SUBTASKS = 5

# ==============================================================================
# STAGE IDs — aligned with expert (scripted_expert.py)
# ==============================================================================
# 5 stages total, 4 trainable (each has its own network head)
STAGE_FLOOR   = 0
STAGE_WALL    = 1
STAGE_DOOR    = 2
STAGE_CEILING = 3
STAGE_LOOKING = 4  # expert-only, no network head
NUM_STAGES = 4     # trainable stages (floor, wall, door, ceiling)
STAGE_NAMES = ["floor", "wall", "door", "ceiling"]

# Map subtask → trainable stage (DONE maps to ceiling for value bootstrap)
SUBTASK_TO_STAGE = {
    SUBTASK_FLOOR:   STAGE_FLOOR,
    SUBTASK_WALLS:   STAGE_WALL,
    SUBTASK_DOOR:    STAGE_DOOR,
    SUBTASK_CEILING: STAGE_CEILING,
    SUBTASK_DONE:    STAGE_CEILING,
}

# ==============================================================================
# NETWORK (V3: MLP + LSTM, per-stage networks)
# ==============================================================================
SHARED_MLP_SIZE = 256
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
ACTOR_MLP_SIZE = 128
CRITIC_MLP_SIZE = 128

# ==============================================================================
# PPO
# ==============================================================================
SEQUENCE_LENGTH = 64
GAMMA = 0.995
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4
CLIP_RATIO = 0.2
ENTROPY_COEFF_START = 0.05
ENTROPY_COEFF_END = 0.005
ENTROPY_COEFF = ENTROPY_COEFF_START
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 2048
MINI_BATCH_SIZE = 512
UPDATE_EPOCHS = 4

# ==============================================================================
# BEHAVIOUR CLONING
# ==============================================================================
BC_LEARNING_RATE = 1e-3
BC_BATCH_SIZE = 256
BC_EPOCHS = 50
BC_DEMO_DIR = "demos"

# ==============================================================================
# WALL EXPERT TIMING (tune experimentally)
# ==============================================================================
WALL_JUMP_PLACE_DELAY = 6   # ticks after jump before place
WALL_LAND_DELAY = 6          # ticks after place for landing
WALL_FALL_DELAY = 8          # ticks for falling to next column
