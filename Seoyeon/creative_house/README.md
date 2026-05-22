# CraftGround 집 건축 강화학습

CraftGround(Minecraft 1.21 기반 RL 환경)에서 에이전트가 밀폐된 집을 짓고 생존하는 법을 학습합니다.  
PPO + 커리큘럼 학습(creative → safe → survival) 구조로 설계되었습니다.

---

## 파일 구성

```
craftground_house_v3/
├── mode_config.py            # 세 모드의 모든 설정 (단일 진실 소스)
├── raycast_tracker.py        # 블록 설치 감지 + 구조물 완성도 평가
├── house_building_wrapper.py # Gymnasium 래퍼 + 보상 함수 + 환경 생성
├── train.py                  # PPO 훈련 스크립트 + 실시간 영상 콜백
├── debug_raycast.py          # HitResult 실제 필드명 확인 도구 (훈련 전 필수)
├── requirements.txt
└── README.md
```

| 파일 | 핵심 클래스/함수 |
|------|-----------------|
| `mode_config.py` | `ModeConfig`, `MODES`, `INITIAL_INVENTORY_CMDS` |
| `raycast_tracker.py` | `RaycastTracker` |
| `house_building_wrapper.py` | `HouseBuildingWrapper`, `make_house_env()` |
| `train.py` | `HouseCNNExtractor`, `RenderCallback`, `HouseProgressCallback`, `train()`, `evaluate()` |

---

## 환경 요구사항

```bash
# Python 3.11 권장 (CraftGround 공식 지원 버전)
conda create -n craftground python=3.11
conda activate craftground

# Java 21 (Minecraft JVM 구동에 필수)
conda install conda-forge::openjdk=21

# 디스플레이 관련 (Linux 서버)
sudo apt-get install -y xvfb libgl1-mesa-dev libegl1-mesa-dev libglew-dev \
    libglu1-mesa-dev xorg-dev libglfw3-dev

# Python 패키지
pip install -r requirements.txt
pip install pygame   # 실시간 영상 창 (선택, 없어도 훈련은 가능)
```

---

## 빠른 실행 순서

```
[1] debug_raycast.py    ← HitResult 필드명 확인 (반드시 먼저)
         ↓ raycast_tracker.py 3개 헬퍼 수정
[2] train.py creative   ← Phase 1 사전학습  (1M steps)
         ↓ --resume
[3] train.py safe       ← Phase 2 파인튜닝  (1M steps)
         ↓ --resume
[4] train.py survival   ← Phase 3 최종 훈련 (2M steps)
         ↓ --resume
[5] train.py eval       ← 평가
```

---

## 상세 실행 순서

### Step 0 — 설치 확인

```bash
python -c "import craftground; print('craftground OK')"
python -c "from stable_baselines3 import PPO; print('SB3 OK')"
```

---

### Step 1 — HitResult 필드명 확인 (필수)

CraftGround의 proto 메시지 필드명은 버전마다 달라질 수 있습니다.  
**훈련 전 반드시 실행해서 실제 필드명을 확인하세요.**

```bash
# 로컬 (GUI 있음)
python debug_raycast.py --port 8023

# 서버 (headless)
xvfb-run -a python debug_raycast.py --port 8023
```

출력 예시:
```
HitResult 전체 필드:
  type                           = 1
  block_pos                      = x: 10  y: 4  z: 8
  block_state                    = minecraft:oak_planks
```

출력을 보고 `raycast_tracker.py`의 세 헬퍼 함수를 실제 필드명으로 수정하세요:

```python
# raycast_tracker.py 수정 위치

def _hit_type(hit) -> str:
    raw = getattr(hit, "type", None)        # ← 필드명이 다르면 수정

def _hit_pos(hit) -> Optional[tuple]:
    bp = getattr(hit, "block_pos", None)    # ← block_pos가 없으면 수정

def _hit_state(hit) -> str:
    for attr in ("block_state", "block_id", "translation_key"):  # ← 순서 조정
        ...
```

---

### Step 1-B — no_op() 인덱스 검증 (권장)

`build_action()` 안에 JUMP/SNEAK/ATTACK/USE/HOTBAR 인덱스가 추론값으로 표기되어 있습니다.  
**블록 설치가 안 된다면 이 스크립트를 먼저 실행하세요.**

```bash
python debug_action.py --port 8030
```

각 인덱스를 3~5스텝 적용하고 Enter를 누르면 다음으로 넘어갑니다.  
Minecraft 창에서 직접 행동을 확인하고, `house_building_wrapper.py`의 `build_action()` 추론값을 수정하세요.

| 행동 | 현재 추론 인덱스 | 확인 방법 |
|------|:------:|----------|
| JUMP   | `act[1]=1`  | 플레이어가 위로 점프 |
| ATTACK | `act[6]=1`  | 앞 블록 파괴 |
| USE    | `act[7]=1`  | 바닥에 블록 설치 ← **핵심** |
| SNEAK  | `act[8]=1`  | 플레이어 웅크리기 |
| HOTBAR | `act[10]=N` | 핫바 슬롯 이동 |

---

### Step 2 — Phase 1: creative 사전학습

인벤토리 무한, 비행 가능, 즉시 파괴. 생존 압박 전혀 없음.  
에이전트가 블록 배치 행동 시퀀스 자체를 빠르게 학습합니다.

```bash
# 로컬 (GUI 있음) — Minecraft 창 + pygame 영상 창이 동시에 뜸
python train.py \
    --env_mode creative \
    --total_steps 1_000_000

# 서버 (headless)
xvfb-run -a python train.py \
    --env_mode creative \
    --total_steps 1_000_000 \
    --no_render
```

완료 후 저장 경로:
```
checkpoints/creative_YYYYMMDD_HHMMSS/
├── best/
│   └── best_model.zip      ← Phase 2 --resume 에 사용
└── house_ppo_creative_final.zip
```

---

### Step 3 — Phase 2: safe 파인튜닝

Peaceful 서바이벌. 인벤토리 소모가 생기고 재료 관리 전략이 필요해집니다.  
Phase 1 체크포인트를 `--resume`으로 이어받습니다.

```bash
python train.py \
    --env_mode safe \
    --total_steps 1_000_000 \
    --resume checkpoints/creative_<timestamp>/best/best_model
```

---

### Step 4 — Phase 3: survival 최종 훈련

Normal 난이도, 낮밤 사이클, 몬스터 스폰. 집을 짓고 첫 밤을 버텨야 합니다.

```bash
python train.py \
    --env_mode survival \
    --total_steps 2_000_000 \
    --resume checkpoints/safe_<timestamp>/best/best_model
```

---

### Step 5 — 평가

```bash
python train.py \
    --mode eval \
    --env_mode survival \
    --n_eval_episodes 10 \
    --resume checkpoints/survival_<timestamp>/best/best_model
```

출력 예시:
```
  Ep  1: R=+142.3  steps=8431   blocks=87  enclosed=True   night=True
  Ep  2: R=+98.7   steps=12000  blocks=61  enclosed=False  night=False
  ...
평균: 118.4 ± 24.1
```

---

### TensorBoard 모니터링

훈련 중 별도 터미널에서 실행:

```bash
tensorboard --logdir logs/
```

기록되는 지표:

| 키 | 설명 |
|----|------|
| `house/ms_placed_5` ~ `ms_placed_100` | 블록 개수 달성률 |
| `house/ms_has_wall` / `ms_has_roof` | 벽/지붕 형성률 |
| `house/ms_has_door` / `ms_has_light` / `ms_has_furniture` | 각 요소 설치율 |
| `house/enclosed_rate` | 밀폐 완성률 |
| `house/avg_blocks` | 에피소드 평균 블록 수 |

---

## 실시간 영상 창 (RenderCallback)

`pip install pygame` 후 훈련하면 자동으로 별도 창이 열립니다.

```
┌──────────────────────────────────────────────┐
│        에이전트 1인칭 시점 (POV × 4배)        │
│                                               │
├──────────────────────────────────────────────┤
│ Step: 12,345   Ep reward: +18.3              │
│ Blocks:  47   Floor:12  Wall:24  Roof:11     │
│ Door:✓   Light:✓   Furn:✗   ENCLOSED: no    │
│ Mode: CREATIVE                                │
└──────────────────────────────────────────────┘
```

| 상황 | 동작 |
|------|------|
| 로컬 GUI 환경 + pygame 설치됨 | pygame 창 자동으로 열림 |
| headless 서버 (`DISPLAY` 미설정) | 경고 출력 후 자동 비활성화 |
| pygame 미설치 | 경고 출력 후 자동 비활성화 |
| `--no_render` 플래그 | 강제 비활성화 |

> CraftGround 자체 Minecraft 창도 별도로 열립니다 (`render_action=True`).  
> 두 창을 나란히 놓으면 게임 화면과 obs 이미지를 동시에 볼 수 있습니다.

---

## train.py 전체 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `train` | `train` / `eval` |
| `--env_mode` | `creative` | `creative` / `safe` / `survival` |
| `--total_steps` | `1_000_000` | 학습 총 스텝 수 |
| `--n_envs` | `1` | 병렬 환경 수 (각각 JVM 실행, RAM 주의) |
| `--base_port` | `8023` | CraftGround 첫 번째 포트 |
| `--max_episode_steps` | `0` | `0`이면 모드 기본값 사용 |
| `--log_dir` | `logs/` | TensorBoard 로그 저장 경로 |
| `--save_dir` | `checkpoints/` | 체크포인트 저장 경로 |
| `--resume` | `None` | 파인튜닝/평가용 모델 경로 |
| `--seed` | `42` | 월드 시드 |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |
| `--n_eval_episodes` | `5` | 평가 에피소드 수 |
| `--no_render` | `False` | pygame 영상 창 비활성화 |

---

## MDP 설계

### 모드 비교

| 항목 | creative | safe | survival |
|------|:--------:|:----:|:--------:|
| gamemode | creative | survival | survival |
| difficulty | — | peaceful | normal |
| 인벤토리 | **무한** | 유한 | 유한 |
| 체력 감소 | ❌ | ❌ | ✅ |
| 허기 감소 | ❌ | ❌ | ✅ |
| 사망 | ❌ | ❌ | ✅ |
| 몬스터 | ❌ | ❌ | ✅ |
| 낮밤 사이클 | ❌ 항상 낮 | ❌ 항상 낮 | ✅ |
| 즉시 파괴 | ✅ | ❌ | ❌ |
| 비행 | ✅ | ❌ | ❌ |
| max_episode_steps | 6,000 | 12,000 | 12,000 |
| PPO γ | 0.99 | 0.995 | 0.995 |
| PPO lr | 3e-4 | 1e-4 | 5e-5 |
| PPO ent_coef | 0.02 | 0.01 | 0.005 |

### 상태 공간 S (부분 관찰 POMDP)

```
observation_space = Dict({
    "image"    : Box(0, 255, (64, 114, 3), uint8)
                   에이전트 1인칭 POV (114×64 RGB)

    "state"    : Box(-1, 1, (8,), float32)
                   [health/20, food/20, x/256, y/256, z/256,
                    sin(yaw), cos(yaw), pitch/90]

    "raycast"  : Box(-1, 1, (8,), float32)
                   [is_block, is_building, is_door, is_light, is_furniture,
                    rel_x/10, rel_y/10, rel_z/10]
                   현재 바라보는 블록 정보.
                   CraftGround에 voxels 필드가 없어 한 블록씩만 관찰 가능.

    "structure": Box(0, 1, (6,), float32)
                   [floor/50, wall/50, roof/30,
                    has_door, has_light, has_furniture]
                   누적 구조물 완성도 요약.
})
```

### 행동 공간 A

`Discrete(22)` — ActionWrapper 변환

| idx | 행동 | idx | 행동 |
|:---:|------|:---:|------|
| 0 | NO_OP | 9 | CAMERA_LEFT |
| 1 | FORWARD | 10 | CAMERA_RIGHT |
| 2 | BACKWARD | 11 | CAMERA_UP |
| 3 | LEFT | 12 | CAMERA_DOWN |
| 4 | RIGHT | 13~21 | HOTBAR_1~9 |
| 5 | JUMP | | |
| 6 | SNEAK | | |
| 7 | ATTACK (파괴) | | |
| **8** | **USE (설치 ★)** | | |

블록 하나 설치하는 최소 행동 시퀀스:
```
HOTBAR_N → CAMERA_*(바닥/벽 조준) → SNEAK → USE
```

### 보상 함수 R

```
R(s, a) = r_alive + r_block + r_milestone + r_enclosed
        + r_health + r_food + r_death + r_night    ← survival 모드만 활성
```

| 이벤트 | 보상 | 유형 | 모드 |
|--------|:----:|:----:|:----:|
| 매 스텝 생존 | +0.01 | 연속 | 전체 |
| 건축 블록 설치 | +0.30 | 즉시 | 전체 |
| 문 설치 | +2.00 | 즉시 | 전체 |
| 조명 설치 | +1.50 | 즉시 | 전체 |
| 가구 설치 | +2.50 | 즉시 | 전체 |
| 블록 5개 달성 | +1.0 | 일회성 | 전체 |
| 블록 20개 달성 | +2.0 | 일회성 | 전체 |
| 블록 50개 달성 | +4.0 | 일회성 | 전체 |
| 블록 100개 달성 | +6.0 | 일회성 | 전체 |
| 벽 형성 (wall ≥ 4) | +2.0 | 일회성 | 전체 |
| 지붕 형성 (roof ≥ 4) | +3.0 | 일회성 | 전체 |
| 문 첫 설치 | +2.0 | 일회성 | 전체 |
| 조명 첫 설치 | +1.5 | 일회성 | 전체 |
| 가구 첫 설치 | +2.0 | 일회성 | 전체 |
| **밀폐 공간 완성** | **+10.0** | 일회성 | **전체** |
| 밤 1회 생존 | +5.0 | 반복 | survival |
| 체력 1hp 감소 | −0.50 | 즉시 | survival |
| 허기 1단위 감소 | −0.10 | 즉시 | survival |
| 사망 | −20.0 | 종단 | survival |
| 클리핑 | [−10, +10] | per step | — |

**밀폐 판정 조건**: `floor ≥ 4 AND wall ≥ 8 AND roof ≥ 4 AND door ≥ 1`

### 전이 함수 T

- 결정론적: `USE` → 블록 설치, `ATTACK` → 블록 파괴
- 확률론적 (survival): 몬스터 스폰 위치/타이밍, 허기 소모 속도
- 설치 실패 조건: 인벤토리 없음 / 허공 조준 / 이미 블록 있음 / 플레이어 충돌

### 에피소드 종료

```python
terminated = (is_dead == True)           # 사망
truncated  = (step >= max_episode_steps) # 시간 초과
# 밀폐 완성 후에도 에피소드 끝내지 않음 → 집 안 생존 행동까지 학습
```

---

## 모델 아키텍처

```
Observation Dict
│
├── image (64×114×3, uint8)
│     └─ /255 → Conv2d(3→32, k=5, s=2)  → (32,30,55)
│                  └─ Conv2d(32→64, k=5, s=2) → (64,13,26)
│                       └─ Conv2d(64→64, k=3, s=1) → (64,11,24)
│                            └─ Flatten → 16,896
│
├── state (8,)    ─┐
├── raycast (8,)  ─┤ cat(22) → Linear(128) → ReLU → Linear(64) → ReLU → 64
└── structure (6,)─┘
                                                     │
                               cat([16896, 64]=16960) → Linear(256) → ReLU
                                                              │
                                                        features (256)
                                                              │
                                            ┌─────────────────┴──────────────────┐
                                       pi [128,128]                         vf [128,128]
```

---

## 블록 분류

| 카테고리 | 블록 종류 | 설치 보상 |
|----------|----------|:---------:|
| building | oak_planks, cobblestone, oak_slab, stone, oak_log, glass_pane | +0.30 |
| door | oak_door, spruce_door | +2.00 |
| light | torch, wall_torch, lantern | +1.50 |
| furniture | crafting_table, furnace, white_bed, red_bed | +2.50 |

---

## 주의사항

**포트 충돌**: 각 환경이 독립 JVM을 실행합니다. `n_envs=2`이면 포트 8023, 8024가 동시에 열립니다. 포트가 이미 사용 중이면 `--base_port`를 변경하세요.

**RAM**: 환경당 JVM이 약 2~4GB RAM을 사용합니다. 처음에는 반드시 `n_envs=1`로 시작하세요.

**debug_raycast.py 필수**: HitResult 필드명은 CraftGround 버전에 따라 다를 수 있습니다. 훈련 전 반드시 확인 후 `raycast_tracker.py`의 `_hit_type`, `_hit_pos`, `_hit_state` 세 함수를 수정하세요.

**creative 모드 raycast**: creative 모드에서도 `raycast_result`가 동일하게 동작하는지 `debug_raycast.py`로 별도 확인을 권장합니다. 동작 방식이 다르면 블록 설치 감지가 작동하지 않습니다.
