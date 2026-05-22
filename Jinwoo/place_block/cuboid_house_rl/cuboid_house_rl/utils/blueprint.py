"""
Blueprint generation for the cuboid house.
Produces a 3D numpy array specifying the target block at every position.
"""
import numpy as np
from cuboid_house_rl.config import (
    WORLD_SIZE, AIR, OAK_PLANKS,
    HOUSE_ORIGIN_X, HOUSE_ORIGIN_Z, HOUSE_WIDTH, HOUSE_DEPTH,
    FLOOR_Y, WALL_Y_MIN, WALL_Y_MAX, CEILING_Y,
    DOOR_X, DOOR_Z, DOOR_Y_BOTTOM, DOOR_Y_TOP,
    FLOOR_BLOCKS, WALL_BLOCKS, CEILING_BLOCKS, TOTAL_BLOCKS,
)


def create_blueprint() -> np.ndarray:
    """
    Create the house blueprint as a 3D array.

    Returns:
        blueprint: np.ndarray of shape (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE)
                   Values are AIR (0) or OAK_PLANKS (1).
                   Only the house region has non-zero values.
    """
    blueprint = np.full(
        (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE), AIR, dtype=np.int8
    )

    x0 = HOUSE_ORIGIN_X
    x1 = HOUSE_ORIGIN_X + HOUSE_WIDTH  # exclusive
    z0 = HOUSE_ORIGIN_Z
    z1 = HOUSE_ORIGIN_Z + HOUSE_DEPTH  # exclusive

    # Floor: y=1, full 7x7 rectangle
    blueprint[x0:x1, FLOOR_Y, z0:z1] = OAK_PLANKS

    # Walls: y=2,3,4, perimeter only (1 block thick)
    for y in range(WALL_Y_MIN, WALL_Y_MAX + 1):
        # South wall (z=z0) and North wall (z=z1-1)
        blueprint[x0:x1, y, z0] = OAK_PLANKS
        blueprint[x0:x1, y, z1 - 1] = OAK_PLANKS
        # West wall (x=x0) and East wall (x=x1-1), excluding corners
        blueprint[x0, y, z0 + 1 : z1 - 1] = OAK_PLANKS
        blueprint[x1 - 1, y, z0 + 1 : z1 - 1] = OAK_PLANKS

    # Ceiling: y=5, full 7x7 rectangle
    blueprint[x0:x1, CEILING_Y, z0:z1] = OAK_PLANKS

    # Door: 2 positions must be AIR (overwrite wall blocks)
    blueprint[DOOR_X, DOOR_Y_BOTTOM, DOOR_Z] = AIR
    blueprint[DOOR_X, DOOR_Y_TOP, DOOR_Z] = AIR

    return blueprint


def get_blueprint_block_positions(blueprint: np.ndarray) -> list:
    """
    Get a list of all positions where blocks should be placed.

    Returns:
        List of (x, y, z) tuples where blueprint == OAK_PLANKS
    """
    positions = np.argwhere(blueprint == OAK_PLANKS)
    return [(int(p[0]), int(p[1]), int(p[2])) for p in positions]


def get_door_positions() -> list:
    """Get the two doorway positions that must remain AIR."""
    return [
        (DOOR_X, DOOR_Y_BOTTOM, DOOR_Z),
        (DOOR_X, DOOR_Y_TOP, DOOR_Z),
    ]


def get_phase_positions(blueprint: np.ndarray) -> dict:
    """
    Separate blueprint positions by construction phase.

    Returns:
        dict with keys 'floor', 'walls', 'ceiling', each containing
        a list of (x, y, z) tuples.
    """
    x0 = HOUSE_ORIGIN_X
    x1 = HOUSE_ORIGIN_X + HOUSE_WIDTH
    z0 = HOUSE_ORIGIN_Z
    z1 = HOUSE_ORIGIN_Z + HOUSE_DEPTH

    floor_positions = []
    wall_positions = []
    ceiling_positions = []

    for x in range(x0, x1):
        for z in range(z0, z1):
            # Floor
            if blueprint[x, FLOOR_Y, z] == OAK_PLANKS:
                floor_positions.append((x, FLOOR_Y, z))
            # Ceiling
            if blueprint[x, CEILING_Y, z] == OAK_PLANKS:
                ceiling_positions.append((x, CEILING_Y, z))

    # Walls
    for y in range(WALL_Y_MIN, WALL_Y_MAX + 1):
        for x in range(x0, x1):
            for z in range(z0, z1):
                if blueprint[x, y, z] == OAK_PLANKS:
                    is_edge_x = (x == x0 or x == x1 - 1)
                    is_edge_z = (z == z0 or z == z1 - 1)
                    if is_edge_x or is_edge_z:
                        wall_positions.append((x, y, z))

    assert len(floor_positions) == FLOOR_BLOCKS, \
        f"Expected {FLOOR_BLOCKS} floor blocks, got {len(floor_positions)}"
    assert len(wall_positions) == WALL_BLOCKS, \
        f"Expected {WALL_BLOCKS} wall blocks, got {len(wall_positions)}"
    assert len(ceiling_positions) == CEILING_BLOCKS, \
        f"Expected {CEILING_BLOCKS} ceiling blocks, got {len(ceiling_positions)}"

    return {
        "floor": floor_positions,
        "walls": wall_positions,
        "ceiling": ceiling_positions,
    }


def validate_blueprint(blueprint: np.ndarray):
    """Sanity check the blueprint."""
    total_planks = np.sum(blueprint == OAK_PLANKS)
    assert total_planks == TOTAL_BLOCKS, \
        f"Expected {TOTAL_BLOCKS} planks, got {total_planks}"

    # Check door positions are air
    for dx, dy, dz in get_door_positions():
        assert blueprint[dx, dy, dz] == AIR, \
            f"Door position ({dx},{dy},{dz}) should be AIR"

    # Check door is not at a corner
    x0 = HOUSE_ORIGIN_X
    x1 = HOUSE_ORIGIN_X + HOUSE_WIDTH - 1
    z0 = HOUSE_ORIGIN_Z
    z1 = HOUSE_ORIGIN_Z + HOUSE_DEPTH - 1
    corners_x = {x0, x1}
    corners_z = {z0, z1}
    assert not (DOOR_X in corners_x and DOOR_Z in corners_z), \
        "Door must not be at a corner"

    print(f"Blueprint valid: {total_planks} blocks, door at "
          f"({DOOR_X},{DOOR_Y_BOTTOM},{DOOR_Z}) and "
          f"({DOOR_X},{DOOR_Y_TOP},{DOOR_Z})")


if __name__ == "__main__":
    bp = create_blueprint()
    validate_blueprint(bp)
    phases = get_phase_positions(bp)
    for phase, positions in phases.items():
        print(f"  {phase}: {len(positions)} blocks")
    print(f"  door: {get_door_positions()}")
