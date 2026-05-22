# Cuboid House RL — V3

RL agent that builds cuboid houses in Minecraft using [CraftGround](https://github.com/yhs0602/CraftGround).

## Overview

The agent learns to build a simple house (floor → walls → door → ceiling) on an infinite superflat world.
House size is randomized each episode (width 5~10, depth 5~10, height 5 fixed).
The agent determines where to build by placing its first block, which sets the **origin**.

**Training pipeline:** Scripted Expert → Behaviour Cloning → Hierarchical PPO fine-tune

**Architecture:** Shared MLP + per-stage LSTM/MLP/Head (no CNN). 3 trainable stages (floor, walls, ceiling).
The LOOKING stage (exit house + look at building) is always handled by the expert script — not trained by PPO.

## Quick Start

```bash
# Prerequisites: Python 3.9+, Java 21 (for CraftGround)
pip install -e .

# 1. Test experts visually (no data saved)
python -m cuboid_house_rl.expert.run_floor --preview          # Floor only
python -m cuboid_house_rl.expert.run_wall  --preview          # Full 5-stage build

# 2. Collect expert demos (only 100%-complete episodes saved, accumulates across runs)
python -m cuboid_house_rl.expert.collect_demos --stage all --episodes 20 --preview

# 3. Train with Behaviour Cloning (all stages simultaneously)
python -m cuboid_house_rl.training.train_bc \
    --demo-path demos/demos_all.npz --stage all

# 4. Fine-tune with Hierarchical PPO
python -m cuboid_house_rl.training.train \
    --mode train --resume checkpoints/bc_all.pt

# 5. Evaluate
python -m cuboid_house_rl.training.train \
    --mode eval --resume checkpoints/best.pt
```

## Project Structure

```
cuboid_house_rl/
├── config.py                   # All hyperparameters and constants
├── envs/
│   ├── craftground_adapter.py  # CraftGround API bridge + obs extraction
│   └── house_building_env.py   # Gymnasium environment
├── expert/                     # See expert/README.md for full details
│   ├── movement.py             # Precision movement, camera, raycast utilities
│   ├── floor_expert.py         # Floor placement state machine
│   ├── wall_expert.py          # Wall placement (spiral, jump+place, glass+planks)
│   ├── door_expert.py          # Door installation (break + place)
│   ├── ceiling_expert.py       # Ceiling placement (serpentine from below)
│   ├── scripted_expert.py      # 5-stage dispatcher (floor→wall→door→ceiling→looking)
│   ├── run_floor.py            # Standalone floor expert runner (preview + record)
│   ├── run_wall.py             # Full 5-stage runner (preview + record)
│   ├── collect_demos.py        # Demo collection — stage_id tracking, 100% filter
│   └── verify_demos.py         # Demo quality verification
├── models/
│   ├── action_dist.py          # Masked multi-discrete distribution
│   └── network.py              # Hierarchical Actor-Critic (shared MLP + per-stage heads)
├── training/
│   ├── ppo.py                  # Recurrent PPO algorithm
│   ├── rollout_buffer.py       # Sequence-aware rollout buffer
│   ├── train.py                # Main training loop
│   └── train_bc.py             # Behaviour Cloning trainer
└── utils/
    ├── blueprint.py            # Static blueprint generation
    ├── completion.py           # Stuck detection
    ├── coord_transform.py      # Coordinate transformation utilities
    └── preview_window.py       # cv2 debug preview window (optional)
```

## Design

### World & Environment

- Infinite superflat world (CraftGround, no worldborder)
- `difficulty peaceful` — hostile mobs fully disabled
- `doMobSpawning false`, `fallDamage false`
- House size randomized per episode: width 5~10, depth 5~10, height 5 (fixed)
- Origin set by first block placement; all coordinates are origin-relative
- No world array — uses three block sets: `placed_blocks`, `correct_blocks`, `incorrect_blocks`

### Observation (45 dimensions)

| Component | Dims | Description |
|-----------|------|-------------|
| origin_set | 1 | Whether origin has been established |
| agent_pos | 3 | Origin-relative (x, y, z) |
| agent_orient | 4 | sin(yaw), cos(yaw), sin(pitch), cos(pitch) |
| hotbar_info | 3 | slot/8, has_planks, has_axe |
| raycast | 17 | Hit info with origin-relative coords, face normal, validity |
| progress | 3 | floor/wall/ceiling completion ratios |
| incorrect_count | 1 | Number of misplaced blocks (normalized) |
| time_remaining | 1 | Episode time left |
| stuck_ratio | 1 | Steps without progress / patience |
| target_direction | 4 | sin/cos of delta_yaw and delta_pitch |
| target_distance | 1 | Raw distance to current target |
| target_pos | 3 | Origin-relative target position |
| house_size | 3 | width, depth, height |

### Action Space

`MultiDiscrete([3, 3, 2, 2, 3, 9, 9, 9])` = 40 logits total

| Dim | Size | Meaning |
|-----|------|---------|
| 0 | 3 | Back / Stop / Forward |
| 1 | 3 | Left / Stop / Right |
| 2 | 2 | Jump |
| 3 | 2 | Sneak |
| 4 | 3 | Place / Nothing / Attack |
| 5 | 9 | Hotbar slot (0-8) |
| 6 | 9 | Camera pitch delta (index into CAMERA_DELTA_MAP) |
| 7 | 9 | Camera yaw delta (index into CAMERA_DELTA_MAP) |

Camera delta map (9 steps): `[-10°, -3°, -1°, -0.1°, 0°, +0.1°, +1°, +3°, +10°]`

### Inventory

| Slot | Item | Use |
|------|------|-----|
| 0–4 | oak_planks (64 each) | Floor, wall corners, ceiling, y=5 top row |
| 5 | diamond_axe | Breaking wall blocks for door |
| 6 | oak_door | Door installation |
| 7–8 | glass (64 each) | Wall non-corner blocks (y=2,3,4) |

### Network (Hierarchical)

Single shared MLP encoder with **per-stage actor and critic heads**. Each
construction stage (floor, walls, ceiling) has its own LSTM → MLP → output head,
allowing each stage to specialise for its distinct behaviour. During rollout and
PPO update, each timestep is routed through the active stage's head; gradients
only flow through the shared MLP + that stage's head.

```
Flat obs (45) → Shared MLP(256, 256) ─┬→ Floor:   Actor LSTM(256) → MLP(128) → 40 logits
                                       │           Critic LSTM(256) → MLP(128) → 1 value
                                       ├→ Wall:    Actor LSTM(256) → MLP(128) → 40 logits
                                       │           Critic LSTM(256) → MLP(128) → 1 value
                                       ├→ Door:    Actor LSTM(256) → MLP(128) → 40 logits
                                       │           Critic LSTM(256) → MLP(128) → 1 value
                                       └→ Ceiling: Actor LSTM(256) → MLP(128) → 40 logits
                                                   Critic LSTM(256) → MLP(128) → 1 value
```

**Stage routing:**
- The environment exposes `current_subtask` (FLOOR=0, WALLS=1, DOOR=2, CEILING=3, DONE=4)
- `SUBTASK_TO_STAGE` maps subtask → trainable stage (DONE maps to ceiling for value bootstrap)
- Each timestep's `stage_id` selects which head to use for forward and loss computation
- LSTM hidden states are per-stage; on stage transition the new stage's LSTM starts fresh
- Door completion is tracked via `door_installed` flag (set when oak_door is placed at a door position)

**Expert LOOKING phase:**
- When env reaches `SUBTASK_DONE` (building 100%), `train.py` runs `run_looking_expert()`
- Creates a `CeilingExpert` in finish-sequence mode (door open → walk out → look at house)
- These steps go through the env (Minecraft world effect) but are NOT stored in rollout buffer
- Episode is then treated as terminated (success) for PPO bookkeeping

### Training Curriculum (Hierarchical PPO)

**4 trainable stages + 1 expert-only stage:**

| Stage ID | Stage | Subtask IDs | Behaviour | Trained by |
|----------|-------|-------------|-----------|------------|
| 0 | Floor | 0 (FLOOR) | Sneak + precision placement | BC + PPO |
| 1 | Wall | 1 (WALLS) | Jump + place columns | BC + PPO |
| 2 | Door | 2 (DOOR) | Break wall + place door | BC + PPO |
| 3 | Ceiling | 3 (CEILING) | Look up + place from below | BC + PPO |
| 4 | Looking | 4 (DONE) | Exit house + look at building | Expert only |

**LOOKING stage:** After building completion (ceiling 100%), the training loop
automatically runs the expert's finish sequence (walk out → turn around → look
at house). These timesteps are **not stored in the PPO buffer** and receive no
gradient updates. This allows the agent to perform the visually appealing
looking behaviour without needing reward engineering for it.

```
Step 1: Collect demos → demos/demos_all.npz (with stage_id per transition)
Step 2: BC pretrain  → all stage heads trained simultaneously
Step 3: PPO fine-tune → each stage's head updated only on its own trajectories
         └─ LOOKING phase handled by expert script (not PPO)
```

**Demo collection policy:**
- Only 100%-complete episodes are saved
- Data accumulates across runs — re-running `collect_demos` appends to existing files
- Each transition includes `stage_id` for splitting trajectories by subtask
- To reset: `rm demos/demos_all.npz`

## Expert Strategies

### Floor Expert (State Machine)

Places floor (y=1) in serpentine order. Uses **raycast feedback control** for aiming — compares actual raycast hit position vs target block to determine pitch/yaw corrections, immune to 1-tick observation lag.

**Phase 0 — Origin placement:**
1. Walk randomly 3 ticks, stop
2. Fix yaw to +Z (within 3°)
3. Aim at ground 2 blocks ahead using `raycast_aim_action`:
   - If hit is further than target → increase pitch (look more down)
   - If hit is closer → decrease pitch (look up)
   - If miss → send +3° pitch
4. Place immediately on raycast confirmation (no verify delay)
5. Align to `(ox+0.5, oz-1.5)` as standing position for Row 0

**Phase 1 — Row 0 from ground (y=1):**
- Stand at `(tx+0.5, oz-1.5)` and aim at target ground top face
- **Gaze lock**: once first block confirmed, keep camera fixed for all remaining blocks in the row (same Z = same pitch angle)

**Phase 2 — Jump onto floor:**
- Walk to `oz-1.35`, jump forward to land on floor blocks (y≈2)

**Phase 3 — Rows 1+ from atop floor (y=2):**
- Walk to standing Z per row, strafe X for each block
- **Gaze lock**: re-aim once per row, keep fixed for remaining blocks in that row

### Wall Expert (Jump + Place)

1. Stand on floor block (y=1), feet at y=2
2. Jump → wait `WALL_JUMP_PLACE_DELAY` ticks → place beneath → land on new block (y=3)
3. Repeat for y=3,4,5 (4 layers per column)
4. Move to next column position, fall down
5. Traversal order: south (+X) → east (+Z) → north (−X) → west (−Z)

### Door Expert

Break 2 wall blocks at south wall center with axe, place door.

1. Move to `(door_x+0.5, oz+2.5)` — 2 blocks inside
2. Fix pitch=0° + yaw=π
3. Aim upper block (y=3) → switch to axe → break
4. Aim lower block (y=2) → break
5. Switch to door slot → place door

### Ceiling Expert (Serpentine from Below)

Place ceiling at y=5 by looking UP at +Z face of wall/ceiling blocks. Agent stays on floor (y=2).

1. yaw=π fixed: +x=strafe right, -x=strafe left, +z=backward
2. Aim at wall top `(x, 5, oz)` +Z face → place at `(x, 5, oz+1)`
3. Serpentine: +x row, step back, -x row, repeat
4. Last rows: can't step back → steeper pitch to reach further blocks

### Looking (Finish Sequence)

After ceiling: exit house, turn around, look at house.

1. Strafe to door x, fix pitch=0°
2. Walk to door (z=oz+2), open door
3. Walk out 5 blocks
4. Turn around (yaw=0), look at y=3
5. Wait 10 ticks → DONE

## Tuning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `COARSE_THRESHOLD` | 0.5 m | Switch to sneak below this distance |
| `FINE_THRESHOLD` | 0.05 m | Consider arrived below this distance |
| `fix_yaw_action` tolerance | 2° (default), 1° (travel_yaw) | Yaw precision |
| `WALL_JUMP_PLACE_DELAY` | 6 ticks | Ticks after jump before placing |
| `WALL_HEIGHT` | 4 | Wall blocks per column (y=2,3,4,5) |
| `_WALK_OFF_TICKS` | 1 | Forward ticks before braking |
| `_ARRIVE_DIST` | 0.4 | Column arrival threshold |
| `MAX_EPISODE_STEPS` | 10000 | Episode timeout |

## Running the Experts

`run_floor.py` and `run_wall.py` are standalone scripts for testing/demoing without saving demo files.

```bash
# Floor only
python -m cuboid_house_rl.expert.run_floor --preview

# Floor + Wall (same episode)
python -m cuboid_house_rl.expert.run_wall --preview

# Record with debug panel overlay
python -m cuboid_house_rl.expert.run_floor --preview --record floor.mp4 --record-fps 20
python -m cuboid_house_rl.expert.run_wall  --preview --record wall.mp4  --record-fps 20
```

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 1 | Number of episodes to run |
| `--preview` | off | Open live cv2 debug window |
| `--preview-delay` | 0.0 | Seconds between steps in preview mode |
| `--record FILE` | off | Save video to FILE (`.mp4` or `.avi`) |
| `--record-fps` | 20 | Video frame rate |
| `--port` | 8023 | CraftGround server port |
| `--seed` | 42 | RNG seed |

**Video output:**
- Without `--preview`: 64×64 game frame upscaled to 480×360
- With `--preview`: full canvas including debug panel (850×360)
- `.mp4` uses mp4v codec; `.avi` uses XVID

## Preview Window

Pass `--preview` to `collect_demos`, `run_floor`, or `run_wall` to open a cv2 debug window showing the game view (64×64 upscaled) and expert debug info (state, position, raycast hit, completion bar). Does not affect gameplay timing.

```bash
python -m cuboid_house_rl.expert.collect_demos --stage walls --episodes 1 \
    --preview --preview-delay 0.1
```

Requires `opencv-python`. Set `DISPLAY=:0` on WSL2.

## Requirements

- Python 3.9+
- Java 21 (for CraftGround Minecraft server)
- craftground >= 2.6.15
- torch >= 2.0
- gymnasium >= 0.29
- numpy >= 1.24
- opencv-python (optional, for `--preview`)
- wandb >= 0.16 (optional, for logging)
