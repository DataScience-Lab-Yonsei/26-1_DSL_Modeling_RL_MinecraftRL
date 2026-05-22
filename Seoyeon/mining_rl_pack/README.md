# CraftGround 지하 광물 채굴 강화학습

PPO 기반 에이전트가 Minecraft 지하를 탐색하며 광물(다이아몬드 등)을 채굴하는 프로젝트.
두 개의 파일로 구성: 동굴 스폰 DB 구축 → 실제 RL 훈련.

---

## 파일 구성

```
cave_seed_scanner_v1.py   동굴 근처 스폰 좌표 DB 사전 구축
real_mining_v2_0324.py    PPO 훈련 / 평가 메인 코드 (Hierarchical 3-Phase 보상)
cave_db.json              scan 후 생성되는 스폰 DB (자동 생성)
```

---

## 빠른 시작

```bash
# 1단계: 동굴 DB 구축 (최초 1회, 백그라운드 실행 권장)
python cave_seed_scanner_v1.py --mode scan --n_seeds 200 --port 8040

# 2단계: 훈련
python real_mining_v2_0324.py --mode train --db cave_db.json --total_steps 3000000

# 3단계: 평가
python real_mining_v2_0324.py --mode eval \
    --resume checkpoints/mining_safe_XXXX/best/best_model \
    --db cave_db.json
```

---

## 전체 흐름

```
[1] cave_seed_scanner_v1.py
    └─ 시드별로 여러 (x, y, z) 후보 탐색
       └─ 낙하량(fall_score) + 이미지 어두움(dark_score) → cave score 계산
          └─ score >= 0.45 인 좌표만 cave_db.json 에 저장

[2] real_mining_v2_0324.py  훈련 시작
    └─ make_env()
       ├─ CraftGround (WorldType.DEFAULT) 서버 시작
       ├─ INVENTORY_CMDS 실행 (clear → 다이아 곡괭이 지급 → 인챈트 → 야간 투시)
       └─ CaveSpawnWrapper: 에피소드마다 cave_db 에서 스폰 좌표 샘플링
          └─ cmds[-1] = "tp @p x y+3 z"  in-place 패치 → reset() 재실행

    └─ Hierarchical 3-Phase 보상 (Multiplicative Gating)
       ├─ Phase 1 (Descend): 게이트 없음 — Y-shaping + delta + 깊이 마일스톤
       ├─ Phase 2 (Explore): depth_gate 게이팅 — 탐험 보너스 + 블록 파괴
       └─ Phase 3 (Mine):    depth_gate × explore_gate 게이팅 — 광물 조준/채굴

    └─ PPO 업데이트 (n_steps=1024 마다)
```

---

## 보상 구조 상세 (Hierarchical 3-Phase)

상위 Phase 달성도가 하위 Phase 보상의 가중치(gate)로 작용.
sigmoid 전환으로 부드러운 커리큘럼 형성.

### 항상 활성 (Phase 무관)

| 구분 | 내용 | 값 |
|---|---|---|
| Step penalty | 매 스텝 | -0.02 |
| 체력 피해 | 피해량 × K | -0.2 × Δhp |
| 사망 | 에피소드 종료 시 | -30.0 |

### Phase 1: Descend (게이트 없음)

| 구분 | 내용 | 값 |
|---|---|---|
| Y-shaping | 깊이 진행도 비례 | 0 ~ +0.15 |
| Y-delta | 이전 최저 Y보다 깊이 갈 때 | +0.5 × ΔY |
| 깊이 마일스톤 | Y 임계값 도달 (1회) | +2 ~ +25 |

### Phase 2: Explore (depth_gate 게이팅)

| 구분 | 내용 | 값 |
|---|---|---|
| 탐험 보너스 | 새 2-block 셀 방문 | depth_gate × λ × 0.5 × depth_weight |
| 블록 파괴 | 아무 블록이든 파괴 | depth_gate × 0.15 |

### Phase 3: Mine (depth_gate × explore_gate 게이팅)

| 구분 | 내용 | 값 |
|---|---|---|
| 광물 조준 | raycast 조준 시 | mine_gate × 0.3 × ore_value |
| 광물 조준+공격 | 조준 + 공격 시 | mine_gate × 1.0 × ore_value |
| 광물 채굴 | 인벤토리 델타 (log 압축) | mine_gate × reward × log(1+gained) |

### 게이트 함수

```
depth_gate(y)  = sigmoid((64 - y - 30) / 10)
  Y=64(지표) → ~0.05,  Y=34 → 0.50,  Y=0 → ~0.97,  Y=-58 → ~1.00

explore_gate(n) = min(1.0, deep_visited_count / 50)
  0셀 → 0.0,  50셀(Y<0 탐험) → 1.0  (선형 클램프)

mine_gate = depth_gate × explore_gate
```

**깊이 마일스톤 (1회성):**

| Y 임계값 | 보상 | 의미 |
|---|---|---|
| Y ≤ 40 | +2.0 | 지하 진입 |
| Y ≤ 0 | +5.0 | Y=0 도달 |
| Y ≤ -20 | +10.0 | 깊은 지하 |
| Y ≤ -40 | +15.0 | 딥슬레이트 층 |
| Y ≤ -58 | +25.0 | 다이아몬드 피크 |

**광물별 보상 (개당, log 압축 적용):**

| 광물 | 기본 보상 | Fortune III 최대 드롭 | 최대 보상 (log 압축) |
|---|---|---|---|
| Diamond | 20.0 | 4개 | 20 × log(5) ≈ 32.2 |
| Raw Gold | 6.0 | 1개 | 6 × log(2) ≈ 4.2 |
| Emerald | 5.0 | 4개 | 5 × log(5) ≈ 8.0 |
| Raw Iron | 4.0 | 1개 | 4 × log(2) ≈ 2.8 |
| Redstone | 3.0 | 9개 | 3 × log(10) ≈ 6.9 |
| Lapis | 3.0 | 28개 | 3 × log(29) ≈ 10.1 |
| Raw Copper | 2.0 | 1개 | 2 × log(2) ≈ 1.4 |
| Coal | 1.0 | 8개 | 1 × log(9) ≈ 2.2 |

> 보상 클리핑: (-30.0, +60.0)

λ (탐험 가중치): 1.0 → 0.1 선형 감쇠 (`LAMBDA_DECAY_STEPS = 1,500,000` per-env 스텝 기준)

---

## 액션 공간 (14개)

| 인덱스 | 이름 | 설명 |
|---|---|---|
| 0 | NO_OP | 대기 |
| 1–4 | FORWARD / BACKWARD / LEFT / RIGHT | 이동 |
| 5 | JUMP | 점프 |
| 6 | ATTACK | 제자리 블록 파기 |
| 7–10 | CAMERA_LEFT/RIGHT/UP/DOWN | 시선 회전 (8°/스텝) |
| 11 | ATTACK_FORWARD | 매크로: 전진 2틱 → 파기 → 전진 수거 |
| 12 | ATTACK_DOWN | 매크로: 시선 아래 16° → 파기 → 낙하 수거 → 시선 복원 |
| 13 | STAIRCASE_DOWN | 매크로: 전진 3틱 → 시선 아래 32° → 파기 → 낙하 수거 → 시선 복원 |

### 채굴 매크로 4-Phase 구조

모든 ATTACK 계열 매크로는 아래 4단계를 순차 실행합니다:

```
Phase 1: 위치/시선 잡기
  ├─ ATTACK:         변화 없음
  ├─ ATTACK_FORWARD: 전진 2틱
  ├─ ATTACK_DOWN:    시선 아래 +8° × 2틱 = +16°
  └─ STAIRCASE_DOWN: 전진 3틱 → 시선 아래 +32° (1틱)

Phase 2: Attack 유지 (8틱)
  └─ attack=True 유지하여 블록 완파

Phase 3: 드롭 아이템 수거
  ├─ 전방 채굴 (ATTACK, ATTACK_FORWARD): 전진 4틱 → 드롭 위로 이동
  └─ 하향 채굴 (ATTACK_DOWN, STAIRCASE_DOWN): 대기 5틱 → 자연 낙하로 수거

Phase 4: 시선 복원
  ├─ ATTACK / ATTACK_FORWARD: 스킵 (시선 변화 없었음)
  ├─ ATTACK_DOWN:    -16° (원래 높이로 복원)
  └─ STAIRCASE_DOWN: -32° (원래 높이로 복원)
```

---

## 관찰 공간

```
image  (64, 114, 3)  uint8    1인칭 시점 이미지 (리사이즈)
state  (12,)         float32  아래 벡터
```

| 인덱스 | 내용 | 범위 |
|---|---|---|
| [0] | Y 위치 (÷64, clip) | -2.0 ~ 2.0 |
| [1] | 목표 깊이 진행도 | 0.0 ~ 1.0 |
| [2] | 체력 (÷20) | 0.0 ~ 1.0 |
| [3] | 배고픔 (÷20) | 0.0 ~ 1.0 |
| [4] | 방향 sin(yaw) | -1.0 ~ 1.0 |
| [5] | 방향 cos(yaw) | -1.0 ~ 1.0 |
| [6] | 피치 (÷90) | -1.0 ~ 1.0 |
| [7] | λ (탐험 가중치) | 0.1 ~ 1.0 |
| [8] | 광물 조준 여부 (raycast) | 0.0 / 1.0 |
| [9] | 조준 광물 가치 (÷100) | 0.0 ~ 1.0 |
| [10] | depth_gate 값 | 0.0 ~ 1.0 |
| [11] | explore_gate 값 | 0.0 ~ 1.0 |

---

## 신경망 구조 (MiningCNNExtractor)

```
Image (64×114×3)                State (12-dim)
       │                              │
   Conv2d(3→32, k=5, s=2) + ReLU     Linear(12→64) + ReLU
   Conv2d(32→64, k=5, s=2) + ReLU    Linear(64→64) + ReLU
   Conv2d(64→64, k=3, s=1) + ReLU         │
   Flatten → 16896-dim                64-dim
       │                              │
       └──────── concat ──────────────┘
                    │
           Linear(16960→256) + ReLU
                    │
              256-dim features
                    │
          ┌─────────┴─────────┐
     Policy MLP           Value MLP
     [128, 128]           [128, 128]
          │                    │
     14 actions            V(s)
```

---

## 에이전트 초기 장비

에피소드 시작 시 자동 지급 (`INVENTORY_CMDS`):

```
clear @p                           이전 에피소드 아이템 초기화 (필수)
diamond_pickaxe × 5               다이아몬드 곡괭이
  └─ Efficiency V / Fortune III / Unbreaking III  인챈트
night_vision 영구 효과            야간 투시 (파티클 숨김)
```

> 횃불/음식은 현재 비활성화 (코드에서 주석 처리됨)

> 사망 시 doImmediateRespawn=true 로 자동 리스폰 + 인벤토리 복구 커맨드 자동 전송

장비 변경 시 `real_mining_v2_0324.py` 상단 `INVENTORY_CMDS` 튜플을 수정.
`clear @p`는 반드시 첫 번째 줄에 유지할 것 (인벤토리 잔존 방지).

---

## 조정 가능한 주요 파라미터

### cave_seed_scanner_v1.py

| 상수 | 기본값 | 설명 |
|---|---|---|
| `SCAN_Y_LEVELS` | `[-35, -45, -55]` | 스캔할 Y 깊이 목록 |
| `SCAN_XZ_RADIUS` | `150` | 스폰 기준 탐색 반경 (블록) |
| `CAVE_SCORE_THRESHOLD` | `0.45` | 동굴 판정 최소 점수 (낮추면 DB 항목 증가) |
| `POSITIONS_PER_SEED` | `8` | 시드당 스캔 위치 수 (높이면 정확도↑, 속도↓) |
| `MAX_ENTRIES_PER_SEED` | `3` | 시드당 DB 저장 최대 항목 수 |
| `SETTLE_TICKS` | `6` | tp 후 물리 안정화 대기 tick |

**CLI 옵션:**

```
--port          포트 번호             (기본: 8040)
--n_seeds       스캔할 시드 수        (기본: 200,  권장: 300~500)
--seed_start    시드 시작 번호        (기본: 0, 이어서 구축 시 활용)
--pos_per_seed  시드당 스캔 위치 수   (기본: 8)
--db            출력 DB 경로          (기본: cave_db.json)
--mode          scan / verify         (기본: scan)
```

---

### real_mining_v2_0324.py — 보상 상수

| 상수 | 기본값 | 설명 |
|---|---|---|
| `Y_TARGET` | `-58` | 목표 깊이 (다이아몬드 피크, 1.18+) |
| `Y_SURFACE` | `64` | 지표면 기준 Y |
| `LAYER1_MAX` | `0.15` | Y-shaping 최대 보상/스텝 |
| `Y_DELTA_BONUS` | `0.5` | Y 최저점 갱신 시 보상 계수 |
| `LAYER2_BASE` | `0.50` | 탐험 보너스 기본값 |
| `EXPLORE_CELL_SIZE` | `2` | 탐험 셀 크기 (2-block 단위) |
| `ORE_AIM_BONUS` | `0.3` | 광물 조준 보상 |
| `ORE_AIM_ATTACK_BONUS` | `1.0` | 광물 조준 + 공격 보상 |
| `BLOCK_BREAK_BONUS` | `0.15` | 블록 파괴 보상 |
| `LAMBDA_DECAY_STEPS` | `1,500,000` | λ 1.0→0.1 감쇠 완료 스텝 수 (per-env) |
| `STEP_PENALTY` | `-0.02` | 스텝당 패널티 |
| `HEALTH_LOSS_K` | `-0.2` | 체력 피해 패널티 계수 |
| `DEATH_PENALTY` | `-30.0` | 사망 패널티 |
| `REWARD_CLIP` | `(-30, 60)` | 보상 클리핑 범위 |
| `DEPTH_GATE_CENTER` | `30.0` | depth_gate sigmoid 중심 (Y≈34에서 gate=0.5) |
| `DEPTH_GATE_SCALE` | `10.0` | depth_gate sigmoid 전환 폭 (~20블록) |
| `EXPLORE_GATE_THRESHOLD` | `50` | explore_gate 최대 셀 수 (Y<0 탐험) |

### real_mining_v2_0324.py — 하이퍼파라미터 (HP)

| 파라미터 | safe | survival | 설명 |
|---|---|---|---|
| `learning_rate` | 3e-4 | 1e-4 | 학습률 |
| `n_steps` | 1024 | 512 | 롤아웃 길이 |
| `batch_size` | 128 | 64 | 미니배치 크기 |
| `n_epochs` | 10 | 10 | 에포크 수 |
| `gamma` | 0.995 | 0.995 | 감가율 |
| `gae_lambda` | 0.95 | 0.95 | GAE λ |
| `clip_range` | 0.2 | 0.2 | PPO 클리핑 범위 |
| `ent_coef` | 0.03 | 0.01 | 엔트로피 계수 (탐험 장려) |
| `vf_coef` | 0.5 | 0.5 | Value function 계수 |
| `max_grad_norm` | 0.5 | 0.5 | 그래디언트 클리핑 |

**CLI 옵션:**

```
--mode          train / eval
--env_mode      safe / survival              (기본: safe)
--db            cave_db.json 경로            (기본: cave_db.json)
--total_steps   총 훈련 스텝 수              (기본: 100,000)
--n_envs        병렬 환경 수                 (기본: 1)
--base_port     시작 포트                    (기본: 8030)
--max_steps     에피소드 최대 스텝 수        (기본: 0=모드 기본값)
--log_dir       로그 디렉토리               (기본: logs)
--save_dir      체크포인트 디렉토리         (기본: checkpoints)
--resume        재개할 모델 경로
--seed          랜덤 시드                    (기본: 42)
--device        cuda / cpu / auto            (기본: auto)
--n_eval_episodes  평가 에피소드 수          (기본: 5, eval 모드)
--wandb_project WandB 프로젝트명             (기본: mining_rl)
--wandb_run     WandB 실행 이름
```

---

## 게임 모드

| 모드 | 난이도 | 몹 스폰 | 에피소드 길이 | 용도 |
|---|---|---|---|---|
| `safe` | peaceful | 없음 | 6,000 스텝 | 채굴 행동 학습에 집중 |
| `survival` | normal | 있음 | 12,000 스텝 | 몹 회피 포함 실전 학습 |

---

## 주의사항

- `cave_seed_scanner_v1.py`와 `real_mining_v2_0324.py`는 **서로 다른 포트**를 사용해야 함
  (스캐너 `--port 8040`, 훈련 `--base_port 8030`)
- `n_envs > 1` 시 포트가 `base_port + i`로 자동 할당됨
- `WorldType.DEFAULT`는 실제 지형을 생성하므로 첫 서버 시작 시 청크 생성 시간이 걸릴 수 있음
- WandB 사용 시 `pip install wandb` 필요 (미설치 시 자동으로 비활성화)
- 횃불/음식은 현재 코드에서 주석 처리됨 — 필요시 `INVENTORY_CMDS`에서 주석 해제
- 사망 시 자동 리스폰 후 인벤토리 복구 커맨드가 자동 전송됨
