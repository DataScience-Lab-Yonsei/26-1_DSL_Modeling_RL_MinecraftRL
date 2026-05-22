#!/usr/bin/env bash
# run_compare.sh — baseline / ICM / HRL 순서대로 실행하여 wandb project에 비교 로그
#
# 사용법:
#   bash run_compare.sh                  # 기본 3M 스텝, safe 모드
#   bash run_compare.sh 1000000          # 스텝 수 지정
#   bash run_compare.sh 3000000 survival # 스텝 수 + env 모드 지정
#
# wandb 비교: project="mining_compare" 에서 3개 run을 한 화면에 비교

set -e

TOTAL_STEPS=${1:-3000000}
ENV_MODE=${2:-safe}
DB_PATH="$(dirname "$0")/../../Seoyeon/mining_rl_pack/cave_db.json"
WANDB_PROJECT="mining_compare"

echo "========================================"
echo "  Mining Comparison Experiment"
echo "  total_steps : $TOTAL_STEPS"
echo "  env_mode    : $ENV_MODE"
echo "  wandb       : $WANDB_PROJECT"
echo "  db          : $DB_PATH"
echo "========================================"

# ----------------------------------------------------------
# 1) Baseline — Seoyeon real_mining_v2_0324
#    port: 8030
# ----------------------------------------------------------
echo ""
echo "[1/3] Baseline (Seoyeon v2 — cell-based exploration + gating)"
python "$(dirname "$0")/../../Seoyeon/mining_rl_pack/real_mining_v2_0324.py" \
    --mode        train           \
    --env_mode    "$ENV_MODE"     \
    --db          "$DB_PATH"      \
    --total_steps "$TOTAL_STEPS"  \
    --base_port   8030            \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run   "baseline_v2"

echo "[1/3] Baseline 완료"

# ----------------------------------------------------------
# 2) ICM — Changyong mining_icm_rl
#    port: 8040
# ----------------------------------------------------------
echo ""
echo "[2/3] ICM (Intrinsic Curiosity Module)"
python "$(dirname "$0")/../mining_icm/mining_icm_rl.py" \
    --mode        train           \
    --env_mode    "$ENV_MODE"     \
    --db          "$DB_PATH"      \
    --total_steps "$TOTAL_STEPS"  \
    --base_port   8040            \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run   "icm"

echo "[2/3] ICM 완료"

# ----------------------------------------------------------
# 3) HRL — Manager-Worker depth stages
#    port: 8050
# ----------------------------------------------------------
echo ""
echo "[3/3] HRL (Manager-Worker Depth Stages)"
python "$(dirname "$0")/hrl_mining.py" \
    --mode        train           \
    --env_mode    "$ENV_MODE"     \
    --db          "$DB_PATH"      \
    --total_steps "$TOTAL_STEPS"  \
    --base_port   8050            \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run   "hrl"

echo "[3/3] HRL 완료"

echo ""
echo "========================================"
echo "  전체 완료! wandb project: $WANDB_PROJECT"
echo "========================================"
