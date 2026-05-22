"""
Visual debug test — watch the agent in Minecraft.
Shows HUD (coordinates, inventory) so you can see what's happening.

Run: python visual_debug.py
"""

from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
from craftground.initial_environment_config import GameMode, Difficulty, WorldType
try:
    from craftground import ActionSpaceVersion
except ImportError:
    from craftground.environment import ActionSpaceVersion
import numpy as np
import time

config = InitialEnvironmentConfig(
    image_width=640,
    image_height=360,
    gamemode=GameMode.CREATIVE,
    difficulty=Difficulty.PEACEFUL,
    world_type=WorldType.SUPERFLAT,
    hud_hidden=False,   # SHOW HUD so we can see coordinates and inventory
    request_raycast=True,
    initial_extra_commands=[
        "gamerule doDaylightCycle false",
        "time set day",
        "give @s minecraft:cobblestone 64",
        "give @s minecraft:dirt 64",
    ],
)

env = CraftGroundEnvironment(
    initial_env=config,
    port=8023,
    action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    render_action=True,
)

obs, info = env.reset()

def noop():
    return {
        'forward': 0, 'back': 0, 'left': 0, 'right': 0,
        'jump': 0, 'sneak': 0, 'sprint': 0,
        'attack': 0, 'use': 0, 'drop': 0, 'inventory': 0,
        'camera': np.array([0.0, 0.0], dtype=np.float32),
        'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0,
        'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0,
        'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0,
    }

full = obs['full']
spawn_x = int(round(full.x))
spawn_z = int(round(full.z))
print(f"Spawn: x={full.x:.1f} y={full.y:.1f} z={full.z:.1f}")
print(f"Yaw={full.yaw:.1f} Pitch={full.pitch:.1f}")

# Place a MARKER: gold block where we want the agent to build
target_x = spawn_x
target_y = -60
target_z = spawn_z + 5
print(f"\nTarget block position: ({target_x}, {target_y}, {target_z})")
print("Placing gold block marker at target...")

# Send setblock command to mark the target
try:
    env.add_command(f"setblock {target_x} {target_y} {target_z} minecraft:gold_block")
except:
    pass

# Wait for command
for _ in range(5):
    obs, _, _, _, _ = env.step(noop())

full = obs['full']
print(f"\nAfter commands: x={full.x:.1f} y={full.y:.1f} z={full.z:.1f}")
print("Inventory:")
for i, item in enumerate(full.inventory[:9]):
    if item.translation_key != "block.minecraft.air":
        print(f"  Slot {i}: {item.translation_key} x{item.count}")

# Now walk forward toward the target
print("\n=== Walking forward 15 steps ===")
for i in range(15):
    action = noop()
    action['forward'] = 1
    obs, _, _, _, _ = env.step(action)
full = obs['full']
print(f"Position: x={full.x:.1f} y={full.y:.1f} z={full.z:.1f}")

# Look down
print("\n=== Looking down ===")
for i in range(8):
    action = noop()
    action['camera'] = np.array([15.0, 0.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
full = obs['full']
print(f"Pitch: {full.pitch:.1f}")

# Check raycast
rc = full.raycast_result
print(f"\nRaycast type: {rc.type}")
if hasattr(rc, 'target_block'):
    tb = rc.target_block
    print(f"Raycast target_block: x={tb.x} y={tb.y} z={tb.z}")
    print(f"Raycast block name: {tb.translation_key}")
    print(f"If we USE here, new block goes at: ({tb.x}, {tb.y + 1}, {tb.z})")
    print(f"Our target was at: ({target_x}, {target_y}, {target_z})")
    if tb.x == target_x and tb.y + 1 == target_y and tb.z == target_z:
        print(">>> MATCH! Placement would hit the target!")
    else:
        print(f">>> MISMATCH. off by dx={target_x - tb.x} dy={target_y - (tb.y+1)} dz={target_z - tb.z}")

# Try placing
print("\n=== Placing block (USE) ===")
action = noop()
action['use'] = 1
obs, _, _, _, _ = env.step(action)
full = obs['full']
rc = full.raycast_result
if hasattr(rc, 'target_block'):
    tb = rc.target_block
    print(f"After USE - raycast block: x={tb.x} y={tb.y} z={tb.z} name={tb.translation_key}")

# Now let the user observe for a while
print("\n=== Holding for 100 steps so you can observe in Minecraft ===")
print("(The Minecraft window should be visible. Look for the gold block marker.)")
for i in range(100):
    obs, _, _, _, _ = env.step(noop())
    time.sleep(0.05)

env.close()
print("\nDone.")
