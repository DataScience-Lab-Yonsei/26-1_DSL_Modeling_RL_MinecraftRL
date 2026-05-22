# Expert Module

Scripted experts that demonstrate how to build each stage of the house. Used for:
1. **Visual testing** — verify expert logic before collecting data
2. **Demo collection** — save (obs, action) pairs for Behaviour Cloning

---

## Files

| File | Description |
|------|-------------|
| `movement.py` | Low-level helpers: movement, camera control, raycast utilities |
| `floor_expert.py` | Floor placement state machine (serpentine, raycast-based aim) |
| `wall_expert.py` | Wall placement with glass+planks design (spiral, jump+place) |
| `door_expert.py` | Door installation: break wall blocks with axe, place door |
| `ceiling_expert.py` | Ceiling placement (serpentine, aim at +Z face from below) |
| `scripted_expert.py` | Unified dispatcher: floor → wall → door → ceiling → looking |
| `run_floor.py` | Standalone floor expert runner — preview & record only |
| `run_wall.py` | Full 5-stage runner — preview & record only |
| `collect_demos.py` | Demo collection — saves NPZ files with stage_id for BC training |
| `verify_demos.py` | Demo quality verification and statistics |

---

## Quick Start

### 1. Test experts visually (no data saved)

```bash
# Floor only
python -m cuboid_house_rl.expert.run_floor --preview

# Full build: Floor + Wall + Door + Ceiling + Looking
python -m cuboid_house_rl.expert.run_wall --preview

# Slow-motion preview
python -m cuboid_house_rl.expert.run_wall --preview --preview-delay 0.5

# Record video
python -m cuboid_house_rl.expert.run_wall --record house.mp4
```

### 2. Collect demos for BC training

```bash
# All 5 stages in one run (recommended)
python -m cuboid_house_rl.expert.collect_demos --stage all --episodes 10

# Floor demos only
python -m cuboid_house_rl.expert.collect_demos --stage floor --episodes 10 --preview

# With video recording
python -m cuboid_house_rl.expert.collect_demos --stage all --episodes 5 --record demos.mp4
```

**Only 100%-complete episodes are saved.** Incomplete episodes are skipped automatically.

**Runs accumulate.** Re-running appends to the existing NPZ file — no data is lost between runs. To reset, delete the file manually:
```bash
rm demos/demos_all.npz
```

**Stage IDs are tracked per transition.** Each (obs, action) pair has a `stage_id` (0=floor, 1=wall, 2=door, 3=ceiling, 4=looking) for hierarchical PPO training.

**Log format:**
```
[  200] F:15% W:0% C:0% | correct=8 wrong=0 | pos=(22.7,1.0,76.9) | size=7x9
[  400] F:48% W:0% C:0% | correct=26 wrong=0 | pos=(23.5,2.0,78.1) | size=7x9
  Floor done: 100% (steps=580, placed=63)
  Wall done: 100% (steps=4200)
  Door done! (step=4800)
  Ceiling done: 100% (steps=6500)
  Door opened! (step=7200)
  Looking done! (step=7500)
Episode 1: F:100% W:100% C:100% | correct=185 wrong=0 | steps=7500 | size=7x9 | saved ✓
```

### 3. Verify collected demos

```bash
python -m cuboid_house_rl.expert.verify_demos --demo-path demos/demos_all.npz
python -m cuboid_house_rl.expert.verify_demos --demo-path demos/demos_floor.npz --preview
```

---

## collect_demos flags

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | `floor` | `floor` / `walls` / `all` |
| `--episodes` | 20 | Number of episodes to attempt |
| `--max-steps` | `MAX_EPISODE_STEPS` | Step limit per episode |
| `--port` | 8023 | CraftGround server port |
| `--seed` | 42 | RNG seed |
| `--output-dir` | `demos/` | Directory to save NPZ files |
| `--preview` | off | Show live debug window |
| `--preview-delay` | 0.1 | Seconds between steps in preview mode |
| `--record FILE` | off | Save video to FILE (`.mp4` or `.avi`) |
| `--record-fps` | 20 | Video frame rate |

Output files: `demos/demos_floor.npz`, `demos/demos_walls.npz`, `demos/demos_all.npz`

---

## run_floor / run_wall flags

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 1 | Number of episodes to run |
| `--max-steps` | `MAX_EPISODE_STEPS` | Step limit per episode |
| `--port` | 8023 | CraftGround server port |
| `--seed` | 42 | RNG seed |
| `--preview` | off | Show live cv2 debug window |
| `--preview-delay` | 0.0 | Seconds between steps in preview mode |
| `--record FILE` | off | Save video to FILE (`.mp4` or `.avi`) |
| `--record-fps` | 20 | Video frame rate |

---

## Hotbar Layout

| Slot | Item | Count |
|------|------|-------|
| 0–4 | oak_planks | 64 each (320 total) |
| 5 | diamond_axe | 1 |
| 6 | oak_door | 64 |
| 7–8 | glass | 64 each (128 total) |

When a planks slot is empty, the expert automatically switches to the next planks slot.
When a glass slot is empty, switches to the next glass slot.

---

## 5-Stage Build Order

```
Stage 0: Floor   (y=1, serpentine)
Stage 1: Wall    (y=2–5, spiral, glass+planks)
Stage 2: Door    (break + place at south wall center)
Stage 3: Ceiling (y=5, serpentine from below)
Stage 4: Looking (exit house, turn around, look at house)
```

---

## Expert Strategies

### Floor Expert (State Machine)

Places floor blocks (y=1) in serpentine order across a random-sized rectangle (5–10 × 5–10).

**Phase 0 — Origin placement:**
1. Walk randomly 3 ticks, stop
2. Fix yaw to +Z (within 2°)
3. Aim at ground 2 blocks ahead using raycast feedback control
4. Require 3 consecutive raycast confirmations (block hit + pitch within 3° of center)
5. Place on confirmation → origin set
6. Align to `(ox+0.5, oz-1.0)` as standing position for Row 0

**Phase 1 — Row 0 from ground (y=1):**
- Stand at `(tx+0.5, oz-1.0)` and aim at target ground top face
- 3-tick raycast + center confirmation before placing
- If block hit but off-center: `fine_aim_to_center` adjusts pitch only
- Gaze lock: once first block confirmed, keep camera fixed for remaining row

**Phase 2 — Jump onto floor:**
- Walk to `oz-0.7`, jump forward to land on floor blocks (y≈2)

**Phase 3 — Rows 1+ from atop floor (y=2):**
- Serpentine: even rows go +X, odd rows go -X
- Stand at `tz-0.5` (closer to target for steeper pitch)
- Same 3-tick confirmation + center check
- Gaze lock: re-aim once per row, keep fixed for remaining blocks

---

### Wall Expert (Spiral, Jump + Place)

Builds 4-block-high wall columns around the floor perimeter using a counterclockwise spiral.

**Design:** corners + y=5 (top row) use **planks**, all other wall blocks use **glass**.

**Spiral start** determined by depth parity:

| Depth | Last row | Agent ends at | Wall start |
|-------|----------|---------------|------------|
| Odd   | Even row (+X) | East (x=ox+width-1) | East wall first |
| Even  | Odd row (-X)  | West (x=ox)         | West wall first |

**Movement:**
- Navigate to column center: face column direction (tolerance=1°) → walk forward
  - If pitch > 80°: lower pitch first (avoid gimbal lock)
  - Arrival: dist < 0.4 from column center
- Falling off column: fix travel yaw (tolerance=1°) → forward → brake → free-fall
- Floor recovery: if y ≤ 1.5, face column → jump + forward
  - Stuck detection: 3 ticks no movement → force jump
- ±π boundary fix: when dx≈0 or dz≈0, force right-turn to avoid yaw oscillation

**Per-column sequence:**
1. Navigate to column center at floor level (y < 2.5)
2. Select material (glass or planks based on corner/y=5)
3. Fix pitch to +90° (look straight down) within 5° tolerance
4. Jump → wait 6 ticks → place block below
5. Verify: 3 consecutive ticks at same y → y gained ≥ 0.5 = success, else retry
6. Repeat until 4 blocks placed (y ≈ 6)
7. Walk forward in travel direction → fall off → next column

**After all columns:** walk to house center → DONE

---

### Door Expert

Installs a door at the front wall (south, z=oz) center after wall construction.

**Door position:** `(ox + width // 2, y=2-3, oz)`

**Sequence:**
1. **MOVE_TO_DOOR:** Walk from house center to `(door_x+0.5, oz+2.5)`
2. **FACE_DOOR:** Fix pitch to 0° + yaw to π (-Z)
3. **AIM_UPPER:** Aim at `(door_x, 3, oz)` +Z face → switch to axe
4. **BREAK_UPPER:** Attack until block gone (raycast no longer hits target)
5. **AIM_LOWER:** Aim at `(door_x, 2, oz)` +Z face
6. **BREAK_LOWER:** Attack until block gone
7. **PLACE_DOOR:** Switch to door slot, aim at floor, place door
8. **DONE**

---

### Ceiling Expert (Serpentine from Below)

Places ceiling blocks at y=5 by looking UP at the +Z face of wall/ceiling blocks.

**Agent stays on floor (y=2)**, yaw fixed to π (-Z direction):
- +x movement = strafe right
- -x movement = strafe left
- +z movement (back) = backward

**Sequence:**
1. **MOVE_TO_START:** Strafe to `(ox+1.5, oz+2.5)`, fix pitch to 0°
2. **First row:** Aim at wall top block `(x, 5, oz)` +Z face → place → ceiling at `(x, 5, oz+1)`
3. **Serpentine:** Move +x placing each block, then step back (+z), move -x, repeat
4. **Last rows (can't step back):** Stay in place, adjust pitch steeper to reach further blocks
5. **Stuck detection:** If z doesn't change for 3 ticks → switch to steeper aim

**Finish sequence (Looking):**
1. Reset pitch to 0°, strafe to door x
2. Fix yaw to π, walk forward to door (z=oz+2)
3. Open door (interact)
4. Walk out 5 blocks
5. Turn around (yaw=0, +Z), look at house (y=3)
6. Wait 10 ticks → DONE

---

## ScriptedExpert

`scripted_expert.py` chains all 5 stage experts:

```python
expert = ScriptedExpert(stage="all")  # "floor" | "walls" | "all"
expert.reset()
while not expert.is_done():
    action = expert.get_action(env)
    obs, reward, terminated, truncated, info = env.step(action)
    stage_id = expert.current_stage_id  # 0=floor, 1=wall, 2=door, 3=ceiling, 4=looking
```

**Stage IDs for Hierarchical PPO:**

| stage_id | Expert | Description |
|----------|--------|-------------|
| 0 | FloorExpert | Floor placement |
| 1 | WallExpert | Wall construction (glass+planks) |
| 2 | DoorExpert | Door installation |
| 3 | CeilingExpert | Ceiling placement |
| 4 | CeilingExpert (finish) | Exit house + look at house |

---

## Timing Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WALL_JUMP_PLACE_DELAY` | 6 ticks | Ticks after jump before placing |
| `WALL_HEIGHT` | 4 | Blocks per column (y=2,3,4,5) |
| `_WALK_OFF_TICKS` | 1 | Ticks of forward walk before braking |
| `_ARRIVE_DIST` | 0.4 | Distance threshold for column arrival |
| `MAX_EPISODE_STEPS` | 10000 | Episode timeout |

---

## Environment Reset

On `env.reset()`:
1. **Origin-based fill:** Clear previous house using absolute coordinates `(ox, 1, oz)` to `(ox+w, 6, oz+d)`
2. **Random teleport:** `tp @p ~±100 ~ ~±100`
3. **Inventory reset:** Clear + give planks, axe, door, glass
