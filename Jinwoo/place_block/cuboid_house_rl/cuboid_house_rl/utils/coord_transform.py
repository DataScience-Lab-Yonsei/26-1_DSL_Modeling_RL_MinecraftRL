"""
Agent-relative coordinate transformations.
Converts world-space positions and normals into the agent's local frame.
"""
import math
import numpy as np


def world_to_agent_relative(
    world_pos: np.ndarray,
    agent_pos: np.ndarray,
    agent_yaw: float,
) -> np.ndarray:
    """
    Transform a world-space position to agent-relative coordinates.

    The agent's local frame:
        +x = agent's right
        +y = up (unchanged)
        +z = agent's forward

    Args:
        world_pos: (3,) array [x, y, z] in world coordinates.
        agent_pos: (3,) array [x, y, z] agent position in world.
        agent_yaw: agent's yaw angle in radians.

    Returns:
        (3,) array [local_x, local_y, local_z] in agent-relative frame.
    """
    # Offset from agent
    dx = world_pos[0] - agent_pos[0]
    dy = world_pos[1] - agent_pos[1]
    dz = world_pos[2] - agent_pos[2]

    # Rotate by negative yaw to go world -> agent frame
    cos_y = math.cos(-agent_yaw)
    sin_y = math.sin(-agent_yaw)

    local_x = cos_y * dx + sin_y * dz    # agent's right
    local_y = dy                           # up is always up
    local_z = -sin_y * dx + cos_y * dz    # agent's forward

    return np.array([local_x, local_y, local_z], dtype=np.float32)


def rotate_normal_to_agent(
    world_normal: np.ndarray,
    agent_yaw: float,
) -> np.ndarray:
    """
    Rotate a world-space face normal into agent-relative frame.

    Args:
        world_normal: (3,) array with +/-1 values [nx, ny, nz].
        agent_yaw: agent's yaw in radians.

    Returns:
        (3,) array [local_nx, local_ny, local_nz] in agent frame.
    """
    cos_y = math.cos(-agent_yaw)
    sin_y = math.sin(-agent_yaw)

    nx, ny, nz = world_normal

    local_nx = cos_y * nx + sin_y * nz
    local_ny = ny
    local_nz = -sin_y * nx + cos_y * nz

    return np.array([local_nx, local_ny, local_nz], dtype=np.float32)


def extract_local_voxel_window(
    world_grid: np.ndarray,
    agent_pos: tuple,
    window_size: int = 11,
    default_value: int = 2,  # SOLID for out-of-bounds
) -> np.ndarray:
    """
    Extract an agent-centered local window from the world grid.

    Args:
        world_grid: 3D array of shape (W, H, D) with block types.
        agent_pos: (x, y, z) integer agent position.
        window_size: size of the local cube (must be odd).
        default_value: value for out-of-bounds positions (SOLID=2).

    Returns:
        3D array of shape (window_size, window_size, window_size).
    """
    half = window_size // 2
    ax, ay, az = int(agent_pos[0]), int(agent_pos[1]), int(agent_pos[2])
    W, H, D = world_grid.shape

    local = np.full(
        (window_size, window_size, window_size),
        default_value,
        dtype=world_grid.dtype,
    )

    # Compute source and destination ranges
    for axis, (center, world_dim) in enumerate([(ax, W), (ay, H), (az, D)]):
        src_start = max(0, center - half)
        src_end = min(world_dim, center + half + 1)
        dst_start = src_start - (center - half)
        dst_end = dst_start + (src_end - src_start)

        if axis == 0:
            x_src = slice(src_start, src_end)
            x_dst = slice(dst_start, dst_end)
        elif axis == 1:
            y_src = slice(src_start, src_end)
            y_dst = slice(dst_start, dst_end)
        else:
            z_src = slice(src_start, src_end)
            z_dst = slice(dst_start, dst_end)

    local[x_dst, y_dst, z_dst] = world_grid[x_src, y_src, z_src]

    return local


def one_hot_3d(grid: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encode a 3D grid.

    Args:
        grid: integer array of shape (W, H, D) with values in [0, num_classes).
        num_classes: number of classes.

    Returns:
        float32 array of shape (W, H, D, num_classes).
    """
    W, H, D = grid.shape
    one_hot = np.zeros((W, H, D, num_classes), dtype=np.float32)
    for c in range(num_classes):
        one_hot[:, :, :, c] = (grid == c).astype(np.float32)
    return one_hot
