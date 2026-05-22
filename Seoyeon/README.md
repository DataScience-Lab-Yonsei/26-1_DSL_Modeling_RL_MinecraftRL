# Seoyeon — 건축 RL + 광물 채굴 RL

Minecraft craftground 환경에서 건축과 채굴 태스크를 단계적으로 접근한 실험 모음.

---

## 실험 목록

| 파일/폴더 | 태스크 | 알고리즘 | 특이사항 |
|---|---|---|---|
| `house_rl_v1_0314.py` | 집 전체 건축 (바닥/벽/지붕/문/가구) | PPO (커스텀 CNN) | 마일스톤 보너스, 단계별 보상 |
| `hollow_box_rl_0314.py` | 속이 뚫린 직육면체 건축 | PPO | 7×7 바닥 + 외벽 3단 + 천장 |
| `line_build_rl_0317.py` | 일직선 블록 10개 배치 | PPO | hollow_box로의 커리큘럼 전이 전 단계 |
| `creative_house/` | 집 건축 (Creative 모드) | PPO | raycast 기반 정밀 배치, 3단계 파인튜닝 |
| `mining_rl_pack/real_mining_v1_0318.py` | 지하 광물 채굴 | PPO | 3-Layer 보상: Y-shaping + λ 탐험 + sparse 채굴 |
| `mining_rl_pack/real_mining_v2_0324.py` | 지하 광물 채굴 (개선) | PPO | Hierarchical 3-Phase + Multiplicative Gating |

---

## 폴더 구조

```
Seoyeon/
├── house_rl_v1_0314.py         # 집 건축 RL (full house)
├── hollow_box_rl_0314.py       # 속이 빈 직육면체 건축 RL
├── line_build_rl_0317.py       # 일직선 빌딩 RL (hollow_box 전이 전용)
├── lidar_wrapper.py            # Lidar 관측 래퍼 (실험용)
├── requirements.txt            # 의존성 목록
├── creative_house/             # Creative 모드 집 건축
│   ├── train.py                # 3단계 훈련 (creative → safe → survival)
│   ├── house_building_wrapper.py  # 환경 래퍼
│   ├── raycast_tracker.py      # raycast 기반 배치 추적
│   ├── mode_config.py          # 모드 설정
│   └── files/                  # 디버그 스크립트
└── mining_rl_pack/
    ├── real_mining_v1_0318.py  # 채굴 RL v1 (3-Layer)
    ├── real_mining_v2_0324.py  # 채굴 RL v2 (3-Phase Gating)
    └── cave_seed_scanner_v1.py # 동굴 시드 스캐너 (CaveSpawnWrapper)
```

---

## 건축 RL

### house_rl_v1_0314.py

집 전체 건축 (바닥 → 벽 → 지붕 → 문/조명/가구 → 완성).

- **관측**: 이미지 + 커스텀 CNN (`NatureCNN` 기반)
- **구조**: `plain_small` 세계, superflat 스폰
- **보상**: 블록 종류별 차등 설치 보상 + 마일스톤 보너스 (바닥/벽/지붕 단계) + 집 완성 보너스
- **모드**: `creative` / `safe` / `survival` 선택 가능

```bash
python house_rl_v1_0314.py --mode train --env_mode creative --total_steps 1000000
python house_rl_v1_0314.py --mode eval --env_mode survival --resume checkpoints/.../best_model
```

### hollow_box_rl_0314.py

속이 빈 직육면체 (7×7 바닥 + 외벽 3단 + 7×7 천장).

- `house_rl_v1_0314.py`와 동일한 obs/action space → **모델 전이 가능**
- 내부(hollow) 채우기 시 강한 패널티로 "빈 공간 유지" 강제

```bash
python hollow_box_rl_0314.py --mode train
python hollow_box_rl_0314.py --mode train --resume <이전 모델>.zip  # 전이 학습
```

### line_build_rl_0317.py

z축 방향 일직선 블록 10개 배치 — `hollow_box`로의 커리큘럼 전이 전 단계.

- `hollow_box`와 동일한 obs shape → **직접 파인튜닝 가능**
- Potential-based shaping: 일직선 연장에 비례한 연속 보상
- 10개 완성 시 완성 보너스

```bash
python line_build_rl_0317.py --mode train
# 완료 후 hollow_box로 전이:
python hollow_box_rl_0314.py --mode train --resume line_build_model.zip
```

### creative_house/

Creative 모드에서 학습하여 safe → survival 순서로 파인튜닝하는 3단계 파이프라인.

- raycast 기반 정밀 블록 배치 추적 (`raycast_tracker.py`)
- `house_building_wrapper.py`: craftground API 브릿지

```bash
cd creative_house

# Phase 1: creative (빠른 사전학습)
python train.py --env_mode creative --total_steps 1000000

# Phase 2: safe 파인튜닝
python train.py --env_mode safe --total_steps 1000000 --resume checkpoints/<ts>/best/best_model

# Phase 3: survival 최종 훈련
python train.py --env_mode survival --total_steps 2000000 --resume checkpoints/<ts>/best/best_model

# 평가
python train.py --mode eval --env_mode survival --resume checkpoints/<ts>/best/best_model
```

---

## 광물 채굴 RL

cave_seed_scanner_v1.py (`CaveSpawnWrapper`)로 동굴 근처 시드를 미리 스캔해 에이전트를 동굴 근처에 스폰.

### real_mining_v1_0318.py — 3-Layer 보상

| Layer | 보상 | 비고 |
|---|---|---|
| Layer 1 | Y-level shaping (지표 → Y=−58 유도) | 항상 활성 |
| Layer 2 | 새 위치 방문 보너스 (λ 감쇠) | 학습 진행에 따라 0으로 수렴 |
| Layer 3 | 인벤토리 채굴 델타 (Diamond +100 등) | sparse |

### real_mining_v2_0324.py — Hierarchical 3-Phase Multiplicative Gating

상위 Phase 달성도가 하위 Phase 보상의 가중치(gate)로 작용하는 계층적 보상.

```
depth_gate   = sigmoid((64 − y − 30) / 10)   # 깊을수록 ~1.0
explore_gate = min(1.0, deep_cells / 50)      # Y<0 탐험 셀 수 기준
mine_gate    = depth_gate × explore_gate
```

| Phase | 내용 | 게이트 |
|---|---|---|
| Phase 1 (Descend) | Y-shaping, Y-delta, 마일스톤 | 없음 (항상) |
| Phase 2 (Explore) | 새 셀 방문 + 블록 파괴 | depth_gate |
| Phase 3 (Mine) | 광물 조준 + 인벤토리 획득 | depth × explore |

광물 보상: Diamond +20, Gold +6, Iron +4, Redstone/Lapis +3, Copper +2, Coal +1

```bash
cd mining_rl_pack

# 훈련
python real_mining_v2_0324.py --mode train --db cave_db.json --total_steps 3000000

# 평가
python real_mining_v2_0324.py --mode eval --resume checkpoints/mining_safe_XXXX/best/best_model
```

---

## 의존성

```
craftground
stable-baselines3
torch
gymnasium
wandb
numpy
opencv-python
```
