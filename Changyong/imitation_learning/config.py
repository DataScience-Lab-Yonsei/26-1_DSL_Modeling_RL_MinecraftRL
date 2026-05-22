"""
Imitation Learning 전체 설정값 모음.
하이퍼파라미터를 바꾸려면 이 파일만 수정하면 됩니다.
"""
import os
import torch

# ── 경로 ──────────────────────────────────────────────────────────
ROOT          = os.path.dirname(__file__)
DEMO_DIR      = os.path.join(ROOT, "demos")
DEMO_PATH     = os.path.join(DEMO_DIR, "demos.npz")
BC_MODEL_PATH = os.path.join(ROOT, "bc_model.pt")
LOG_DIR       = os.path.join(ROOT, "logs")

# ── 환경 ──────────────────────────────────────────────────────────
PORT     = 8013
IMG_SIZE = 84         # BCPolicy(image) 전용

# ── obs 차원 (VectorBuildingEnv 기준) ─────────────────────────────
OBS_DIM = 45          # env.py VectorBuildingEnv.OBS_DIM과 동일하게 유지

# ── 전문가 (ScriptedExpert) 
EXPERT_YAW_TOL      = 12.0   # 방향 허용 오차 (도)
EXPERT_DIST_TOL     = 1.2    # 전진 멈추는 거리 (blocks) — 가까울수록 정확한 조준 가능
EXPERT_TARGET_PITCH = 50.0   # 블록 설치 목표 pitch (양수 = 아래 방향, craftground 기준)
EXPERT_PITCH_TOL    = 8.0    # pitch 허용 오차 (도)
EXPERT_NAV_PITCH    = 20.0   # 이동 중 목표 pitch (살짝 내려봄)
EXPERT_PLACE_STEPS  = 1      # USE_ITEM 연속 횟수

# ── 데이터 수집 ───────────────────────────────────────────────────
COLLECT_N_EPISODES = 30

# ── BC 학습 ───────────────────────────────────────────────────────
BC_N_EPOCHS   = 50
BC_BATCH_SIZE = 64
BC_LR         = 3e-4
BC_VAL_RATIO  = 0.1

# ── PPO 파인튜닝 ──────────────────────────────────────────────────
FINETUNE_TOTAL_TIMESTEPS = 500_000
FINETUNE_LR              = 1e-4
FINETUNE_ENT_COEF        = 0.005

# ── 디바이스 ─────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
