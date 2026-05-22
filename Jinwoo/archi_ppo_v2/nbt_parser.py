"""
NBT Blueprint Parser for CraftGround House Builder.

Parses Minecraft .nbt structure files into a 3D blueprint representation
that the RL agent can use as a build target.

Dependencies: nbtlib (pip install nbtlib)
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import os

try:
    import nbtlib
    from nbtlib import nbt
    HAS_NBTLIB = True
except ImportError:
    HAS_NBTLIB = False
    print("WARNING: nbtlib not installed. Install with: pip install nbtlib")


# ──────────────────────────────────────────────────────────────────────
# Block ID mapping: maps Minecraft block names → integer IDs for the
# agent's observation. 0 = air (empty), 1+ = solid blocks.
# ──────────────────────────────────────────────────────────────────────
DEFAULT_BLOCK_MAP = {
    "minecraft:air": 0,
    "minecraft:stone": 1,
    "minecraft:cobblestone": 2,
    "minecraft:oak_planks": 3,
    "minecraft:spruce_planks": 4,
    "minecraft:birch_planks": 5,
    "minecraft:oak_log": 6,
    "minecraft:spruce_log": 7,
    "minecraft:glass": 8,
    "minecraft:glass_pane": 9,
    "minecraft:oak_door": 10,
    "minecraft:oak_stairs": 11,
    "minecraft:cobblestone_stairs": 12,
    "minecraft:oak_slab": 13,
    "minecraft:torch": 14,
    "minecraft:crafting_table": 15,
    "minecraft:furnace": 16,
    "minecraft:chest": 17,
    "minecraft:oak_fence": 18,
    "minecraft:ladder": 19,
    "minecraft:dirt": 20,
    "minecraft:grass_block": 21,
    "minecraft:stone_bricks": 22,
    "minecraft:bricks": 23,
    "minecraft:white_wool": 24,
    "minecraft:oak_trapdoor": 25,
}
NEXT_BLOCK_ID = max(DEFAULT_BLOCK_MAP.values()) + 1


@dataclass
class BlueprintBlock:
    """Single block in the blueprint."""
    x: int
    y: int
    z: int
    block_name: str  # e.g., "minecraft:oak_planks"
    block_state: str  # full state string with properties
    block_id: int  # integer ID for the agent


@dataclass
class Blueprint:
    """Parsed blueprint from an .nbt file."""
    blocks: List[BlueprintBlock] = field(default_factory=list)
    size_x: int = 0
    size_y: int = 0
    size_z: int = 0
    # 3D numpy array: shape (size_x, size_y, size_z), values = block IDs
    grid: Optional[np.ndarray] = None
    # Map from block_name → block_id used in this blueprint
    block_map: Dict[str, int] = field(default_factory=dict)
    # Reverse map: block_id → block_name
    id_to_name: Dict[int, str] = field(default_factory=dict)
    # Position → BlueprintBlock lookup (for block-type checking)
    pos_to_block: Dict[Tuple[int, int, int], 'BlueprintBlock'] = field(default_factory=dict)
    # Build order: blocks sorted by layer (Y), then by distance from center
    build_order: List[BlueprintBlock] = field(default_factory=list)
    # Per-layer block counts
    layer_block_counts: Dict[int, int] = field(default_factory=dict)
    # Origin offset (where to place the structure in the world)
    origin: Tuple[int, int, int] = (0, 0, 0)


def parse_nbt_file(filepath: str, block_map: Optional[Dict[str, int]] = None) -> Blueprint:
    """
    Parse a Minecraft .nbt structure file into a Blueprint.

    Args:
        filepath: Path to the .nbt file.
        block_map: Optional custom block name → ID mapping.
                   If None, uses DEFAULT_BLOCK_MAP and auto-assigns new IDs.

    Returns:
        Blueprint object with all block data and the 3D grid.
    """
    if not HAS_NBTLIB:
        raise ImportError("nbtlib is required. Install with: pip install nbtlib")

    if block_map is None:
        block_map = dict(DEFAULT_BLOCK_MAP)

    next_id = max(block_map.values()) + 1 if block_map else 1

    # Load the NBT file
    nbt_file = nbtlib.load(filepath)

    # Structure NBT format:
    # - size: [x, y, z] list
    # - palette: list of block states
    # - blocks: list of {pos: [x,y,z], state: index_into_palette, nbt: optional}
    root = nbt_file
    if hasattr(nbt_file, 'root'):
        root = nbt_file.root
    # Handle different nbtlib versions
    if isinstance(root, dict) and '' in root:
        root = root['']

    size = [int(s) for s in root["size"]]
    size_x, size_y, size_z = size[0], size[1], size[2]

    # Parse palette
    palette = []
    for entry in root["palette"]:
        block_name = str(entry["Name"])
        properties = {}
        if "Properties" in entry:
            properties = {str(k): str(v) for k, v in entry["Properties"].items()}
        palette.append((block_name, properties))

    # Parse blocks
    blocks = []
    grid = np.zeros((size_x, size_y, size_z), dtype=np.int32)

    for block_entry in root["blocks"]:
        pos = [int(p) for p in block_entry["pos"]]
        state_idx = int(block_entry["state"])

        bx, by, bz = pos[0], pos[1], pos[2]
        block_name, properties = palette[state_idx]

        # Skip air blocks
        if block_name == "minecraft:air" or block_name == "minecraft:structure_void":
            continue

        # Assign block ID
        # Strip properties for the base block name lookup
        base_name = block_name
        if base_name not in block_map:
            block_map[base_name] = next_id
            next_id += 1

        block_id = block_map[base_name]

        # Build state string
        if properties:
            prop_str = ",".join(f"{k}={v}" for k, v in sorted(properties.items()))
            block_state = f"{block_name}[{prop_str}]"
        else:
            block_state = block_name

        block = BlueprintBlock(
            x=bx, y=by, z=bz,
            block_name=block_name,
            block_state=block_state,
            block_id=block_id,
        )
        blocks.append(block)
        grid[bx, by, bz] = block_id

    # Build reverse map
    id_to_name = {v: k for k, v in block_map.items()}

    # Compute build order: layer by layer (bottom up), within each layer
    # sort by distance from center for stable ordering
    cx, cz = size_x / 2.0, size_z / 2.0
    build_order = sorted(
        blocks,
        key=lambda b: (b.y, (b.x - cx) ** 2 + (b.z - cz) ** 2),
    )

    # Layer block counts
    layer_counts = {}
    for b in blocks:
        layer_counts[b.y] = layer_counts.get(b.y, 0) + 1

    # Position → block lookup (for block-type matching in rewards)
    pos_to_block = {(b.x, b.y, b.z): b for b in blocks}

    blueprint = Blueprint(
        blocks=blocks,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        grid=grid,
        block_map=block_map,
        id_to_name=id_to_name,
        pos_to_block=pos_to_block,
        build_order=build_order,
        layer_block_counts=layer_counts,
    )

    return blueprint


def blueprint_to_setblock_commands(
    blueprint: Blueprint,
    origin: Tuple[int, int, int] = (0, -60, 0),
) -> List[str]:
    """
    Convert a blueprint into Minecraft /setblock commands.
    Useful for placing the target structure as a reference.

    Args:
        blueprint: Parsed Blueprint object.
        origin: World coordinates for the blueprint's (0,0,0) corner.

    Returns:
        List of Minecraft setblock command strings.
    """
    ox, oy, oz = origin
    commands = []
    for block in blueprint.build_order:
        wx = ox + block.x
        wy = oy + block.y
        wz = oz + block.z
        commands.append(f"setblock {wx} {wy} {wz} {block.block_state}")
    return commands


def blueprint_to_initial_block_states(
    blueprint: Blueprint,
    origin: Tuple[int, int, int] = (0, -60, 0),
) -> List[dict]:
    """
    Convert blueprint to CraftGround's initialBlockStates format.

    Returns list of dicts with keys: x, y, z, block_state.
    """
    ox, oy, oz = origin
    states = []
    for block in blueprint.blocks:
        states.append({
            "x": ox + block.x,
            "y": oy + block.y,
            "z": oz + block.z,
            "block_state": block.block_state,
        })
    return states


def get_blueprint_stats(blueprint: Blueprint) -> dict:
    """Get summary statistics about a blueprint."""
    block_type_counts = {}
    for b in blueprint.blocks:
        block_type_counts[b.block_name] = block_type_counts.get(b.block_name, 0) + 1

    return {
        "total_blocks": len(blueprint.blocks),
        "dimensions": (blueprint.size_x, blueprint.size_y, blueprint.size_z),
        "num_layers": len(blueprint.layer_block_counts),
        "max_height": max(b.y for b in blueprint.blocks) if blueprint.blocks else 0,
        "block_types": block_type_counts,
        "layer_counts": blueprint.layer_block_counts,
    }


def create_simple_blueprint(
    structure_type: str = "wall_3high",
) -> Blueprint:
    """
    Create simple synthetic blueprints for curriculum training.
    Useful when you don't have .nbt files or want to start simple.

    Args:
        structure_type: One of "single_block", "row", "wall_2high",
                       "wall_3high", "small_room", "cube_3x3x3".

    Returns:
        Blueprint object.
    """
    block_map = dict(DEFAULT_BLOCK_MAP)
    blocks = []
    block_name = "minecraft:cobblestone"
    block_id = block_map[block_name]

    if structure_type == "single_block":
        blocks.append(BlueprintBlock(0, 0, 0, block_name, block_name, block_id))
        sx, sy, sz = 1, 1, 1

    elif structure_type == "row":
        # 2–10 random-length row, random orientation: X-axis or Z-axis
        n = random.randint(2, 10)
        if random.random() < 0.5:
            for i in range(n):
                blocks.append(BlueprintBlock(i, 0, 0, block_name, block_name, block_id))
            sx, sy, sz = n, 1, 1
        else:
            for i in range(n):
                blocks.append(BlueprintBlock(0, 0, i, block_name, block_name, block_id))
            sx, sy, sz = 1, 1, n

    elif structure_type == "row_2":
        # Fixed 2-block row (X-axis) — simplest possible row task
        for i in range(2):
            blocks.append(BlueprintBlock(i, 0, 0, block_name, block_name, block_id))
        sx, sy, sz = 2, 1, 1

    elif structure_type == "pillar_2":
        # 2 blocks stacked vertically — introduces Y-axis stacking
        for y in range(2):
            blocks.append(BlueprintBlock(0, y, 0, block_name, block_name, block_id))
        sx, sy, sz = 1, 2, 1

    elif structure_type == "wall_2high":
        # Kept for backward compatibility, but wall_3high is recommended
        for x in range(5):
            for y in range(2):
                blocks.append(BlueprintBlock(x, y, 0, block_name, block_name, block_id))
        sx, sy, sz = 5, 2, 1

    elif structure_type == "wall_3high":
        # 5 wide × 3 tall wall — height 3 forces real scaffolding
        # because Minecraft jump height (1.25) cannot reach y=2 from ground
        for x in range(5):
            for y in range(3):
                blocks.append(BlueprintBlock(x, y, 0, block_name, block_name, block_id))
        sx, sy, sz = 5, 3, 1

    elif structure_type == "small_room":
        # 5×4×5 hollow room — floor + 4 walls, 4 blocks tall
        # y=0: solid floor (25 blocks)
        # y=1: walls only (16 blocks) — reachable by jumping
        # y=2: walls only (16 blocks) — borderline, needs scaffolding
        # y=3: walls only (16 blocks) — definitely needs scaffolding
        for x in range(5):
            for y in range(4):
                for z in range(5):
                    is_wall = (x == 0 or x == 4 or z == 0 or z == 4)
                    is_floor = (y == 0)
                    if is_wall or is_floor:
                        blocks.append(BlueprintBlock(
                            x, y, z, block_name, block_name, block_id
                        ))
        sx, sy, sz = 5, 4, 5

    elif structure_type == "cube_3x3x3":
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    blocks.append(BlueprintBlock(
                        x, y, z, block_name, block_name, block_id
                    ))
        sx, sy, sz = 3, 3, 3

    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")

    grid = np.zeros((sx, sy, sz), dtype=np.int32)
    for b in blocks:
        grid[b.x, b.y, b.z] = b.block_id

    cx, cz = sx / 2.0, sz / 2.0
    build_order = sorted(
        blocks, key=lambda b: (b.y, (b.x - cx)**2 + (b.z - cz)**2)
    )

    layer_counts = {}
    for b in blocks:
        layer_counts[b.y] = layer_counts.get(b.y, 0) + 1

    return Blueprint(
        blocks=blocks,
        size_x=sx, size_y=sy, size_z=sz,
        grid=grid,
        block_map=block_map,
        id_to_name={v: k for k, v in block_map.items()},
        pos_to_block={(b.x, b.y, b.z): b for b in blocks},
        build_order=build_order,
        layer_block_counts=layer_counts,
    )


def save_blueprint_json(blueprint: Blueprint, filepath: str):
    """Save blueprint metadata to JSON for inspection."""
    data = {
        "size": [blueprint.size_x, blueprint.size_y, blueprint.size_z],
        "total_blocks": len(blueprint.blocks),
        "blocks": [
            {
                "x": b.x, "y": b.y, "z": b.z,
                "name": b.block_name,
                "state": b.block_state,
                "id": b.block_id,
            }
            for b in blueprint.blocks
        ],
        "block_map": blueprint.block_map,
        "layer_counts": blueprint.layer_block_counts,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Demo: create synthetic blueprints
    for stype in ["single_block", "row", "wall_2high", "small_room", "cube_3x3x3"]:
        bp = create_simple_blueprint(stype)
        stats = get_blueprint_stats(bp)
        print(f"\n=== {stype} ===")
        print(f"  Blocks: {stats['total_blocks']}")
        print(f"  Dimensions: {stats['dimensions']}")
        print(f"  Max height: {stats['max_height']}")
        print(f"  Layers: {stats['layer_counts']}")
