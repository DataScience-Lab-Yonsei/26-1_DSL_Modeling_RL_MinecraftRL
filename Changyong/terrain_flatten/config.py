"""
Terrain Flatten 설정.

태스크: 12x12 구역의 지형을 TARGET_Y 이하로 평탄화 (블록 채굴).
기본 월드(default world) + 크리에이티브 모드로 즉시 채굴.
"""
import os
import torch

ROOT     = os.path.dirname(__file__)
DEMO_DIR = os.path.join(ROOT, "demos")
DEMO_PATH = os.path.join(DEMO_DIR, "demos.npz")

# ── 서버 포트 ──────────────────────────────────────────────────────
PORT = 8014   # imitation_learning(8013)과 충돌 방지

# ── 작업 구역 (절대 Minecraft 좌표) ────────────────────────────────
AREA_X   = 0    # 구역 서쪽 끝 X
AREA_Z   = 4    # 구역 북쪽 끝 Z (플레이어 앞)
AREA_W   = 12   # X 방향 너비 (0 ~ 11)
AREA_D   = 12   # Z 방향 깊이 (4 ~ 15)
TARGET_Y = 63   # 이 Y 초과 블록은 모두 제거 (sea level)

# 지형 최대 높이 (reset fill 상한)
TERRAIN_MAX_Y = 70

# ── 액션 스페이스 (Jinwoo v3 참고: MultiDiscrete) ──────────────────
# [3, 3, 2, 2, 3, 9, 9]
ACT_FWD_BACK   = 0   # 0=back  1=stop  2=forward
ACT_LEFT_RIGHT = 1   # 0=left  1=stop  2=right
ACT_JUMP       = 2   # 0=no    1=yes
ACT_SNEAK      = 3   # 0=no    1=yes
ACT_INTERACT   = 4   # 0=use   1=nothing  2=attack
ACT_PITCH      = 5   # 카메라 pitch delta 인덱스
ACT_YAW        = 6   # 카메라 yaw delta 인덱스
NUM_ACT_DIMS   = 7
ACT_DIMS       = [3, 3, 2, 2, 3, 9, 9]

# 카메라 delta (도/틱) – 인덱스 4 = 0 (정지)
CAMERA_DELTA_MAP = [-10.0, -3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 10.0]
CAMERA_NEUTRAL   = 4   # 0°

# ── 관측 벡터 차원 ─────────────────────────────────────────────────
# agent_pos(3) + orient(4) + raycast(9) + progress(1) + time(1) = 18
OBS_DIM = 18

# ── Raycast ───────────────────────────────────────────────────────
RAYCAST_MAX_DIST = 5.0

# ── 에피소드 ──────────────────────────────────────────────────────
MAX_STEPS          = 4000
COLLECT_N_EPISODES = 20

# ── BC 학습 ───────────────────────────────────────────────────────
BC_EPOCHS     = 50
BC_BATCH_SIZE = 64
BC_LR         = 3e-4
BC_VAL_RATIO  = 0.1

# ── 디바이스 ─────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
