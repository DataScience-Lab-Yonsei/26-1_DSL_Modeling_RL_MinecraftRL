# Mining RL 비교 실험

Minecraft 광물 채굴 태스크에서 세 가지 RL 접근법을 비교한다.

| 파일 | 알고리즘 | 탐험 방식 |
|---|---|---|
| `../../Seoyeon/mining_rl_pack/.py` | PPO | 방문 셀 추적 + λ 감쇠 |
| `../mining_icm/mining_icm_rl.py` | PPO real_mining_v2_0324+ ICM | 예측 오류 기반 novelty |
| `hrl_mining.py` | PPO (HRL) | Manager가 깊이 서브골 지정 |

---

## 실행

```bash
# 3개 순서대로 실행 (baseline → ICM → HRL)
bash run_compare.sh

# 스텝 수 / 모드 지정
bash run_compare.sh 1000000 safe

# 개별 실행
python hrl_mining.py --mode train --total_steps 3000000
```

wandb project `mining_compare` 에서 3개 run을 한 화면에 비교할 수 있다.

---

## 공통 환경 설정

### 게임 설정 (safe 모드)

- Difficulty: Peaceful (몹 없음)
- 날씨/시간 고정 (맑음, 낮)
- 즉시 리스폰 활성화
- 에피소드 최대 스텝: **6000** (survival 모드: 12000)

### 초기 인벤토리

```
diamond_pickaxe × 5  (Efficiency V, Fortune III, Unbreaking III)
night_vision (영구)
```

### 액션 공간 — Discrete(14)

| idx | 이름 | 설명 |
|---|---|---|
| 0 | NO_OP | 아무것도 안 함 |
| 1 | FORWARD | 전진 |
| 2 | BACKWARD | 후진 |
| 3 | LEFT | 왼쪽 이동 |
| 4 | RIGHT | 오른쪽 이동 |
| 5 | JUMP | 점프 |
| 6 | ATTACK | 블록 파기 |
| 7 | CAMERA_LEFT | 카메라 좌 8° |
| 8 | CAMERA_RIGHT | 카메라 우 8° |
| 9 | CAMERA_UP | 카메라 위 8° |
| 10 | CAMERA_DOWN | 카메라 아래 8° |
| 11 | ATTACK_FORWARD | 전진 2틱 + 파기 8틱 + 픽업 |
| 12 | ATTACK_DOWN | 아래보기 2틱 + 파기 8틱 + 대기 |
| 13 | STAIRCASE_DOWN | 전진 3틱 + 아래보기 + 파기 (계단식 하강) |

---

## 1. Baseline — `real_mining_v2_0324.py`

Seoyeon 작성. 방문 셀 집합 + λ 감쇠로 탐험 보상을 직접 설계한다.

### Observation Space

```
Dict({
  "image": Box(0, 255, shape=(64, 114, 3), dtype=uint8)
  "state": Box(-2, 2,  shape=(10,),        dtype=float32)
})
```

**state 벡터 (10-dim)**

| idx | 값 | 범위 |
|---|---|---|
| 0 | y / 64.0 | [-2, 2] |
| 1 | depth_progress = (64 − y) / 122 | [0, 1] |
| 2 | health / 20 | [0, 1] |
| 3 | food / 20 | [0, 1] |
| 4 | sin(yaw) | [-1, 1] |
| 5 | cos(yaw) | [-1, 1] |
| 6 | pitch / 90 | [-1, 1] |
| 7 | λ (탐험 가중치, 1.0→0.1 감쇠) | [0.1, 1] |
| 8 | is_ore (raycast로 광물 조준 중) | {0, 1} |
| 9 | ore_value_norm (조준 광물 가치 / 100) | [0, 1] |

### Reward System — Hierarchical 3-Phase (Multiplicative Gating)

보상이 3단계로 나뉘며, 상위 단계 달성도가 하위 단계 보상의 가중치(gate)로 작용한다.

```
depth_gate   = sigmoid((64 − y − 30) / 10)   # 깊을수록 ~1.0
explore_gate = min(1.0, deep_cells / 50)      # Y<0 탐험 셀 수 기준
mine_gate    = depth_gate × explore_gate
```

#### Phase 1 — Descend (항상 활성)

| 보상 항목 | 값 | 조건 |
|---|---|---|
| Step penalty | −0.02 | 매 스텝 |
| Health loss | −0.2 × damage | 피해 받을 때 |
| Death | −30.0 | 사망 시 |
| Y-level shaping | +0.15 × depth_progress | 매 스텝 |
| Y-delta bonus | +0.5 × (prev_min_y − y) | 최저 Y 갱신 시 |
| Milestone Y=40 | +2.0 | 1회 |
| Milestone Y=0 | +5.0 | 1회 |
| Milestone Y=−20 | +10.0 | 1회 |
| Milestone Y=−40 | +15.0 | 1회 |
| Milestone Y=−58 | +25.0 | 1회 |

#### Phase 2 — Explore (× depth_gate)

| 보상 항목 | 값 | 조건 |
|---|---|---|
| 새 셀 탐험 | depth_gate × λ × 0.5 × depth_weight | 2블록 단위 셀 최초 방문 |
| 블록 파괴 | depth_gate × 0.15 | 인벤토리 총량 증가 시 |

λ는 1.5M 스텝 동안 1.0 → 0.1 선형 감쇠 (초반 탐험 강조 → 후반 채굴 집중)

#### Phase 3 — Mine (× mine_gate)

| 보상 항목 | 값 |
|---|---|
| 광물 조준 | mine_gate × 0.3 × ore_value |
| 광물 조준 + 공격 | mine_gate × 1.0 × ore_value |
| 인벤토리 획득 | mine_gate × reward × log(1 + gained) |

**광물별 기본 보상 (log 적용 전)**

| 광물 | 보상 |
|---|---|
| Diamond | 20.0 |
| Emerald | 5.0 |
| Gold | 6.0 |
| Iron | 4.0 |
| Redstone | 3.0 |
| Lapis | 3.0 |
| Copper | 2.0 |
| Coal | 1.0 |

Reward clip: `[−30, 60]`

---

## 2. ICM — `mining_icm_rl.py`

Changyong 작성. Phase 2 탐험 보상을 방문 셀 대신 **ICM (Intrinsic Curiosity Module)** 으로 대체한다.

### Observation Space

```
Dict({
  "image": Box(0, 255, shape=(64, 114, 3), dtype=uint8)
  "state": Box(-2, 2,  shape=(10,),        dtype=float32)
})
```

**state 벡터 (10-dim)** — baseline과 거의 동일, state[7]만 다름

| idx | 값 | baseline과 차이 |
|---|---|---|
| 0–6 | 동일 | — |
| 7 | icm_int_r (직전 스텝 내재 보상) | baseline은 λ |
| 8–9 | 동일 | — |

### ICM 구조 (Pathak et al. 2017)

```
PhiEncoder   : (image, state) → φ(s)          [256-dim embedding]
ForwardModel : (φ(s), action) → φ̂(s')         [다음 상태 예측]
InverseModel : (φ(s), φ(s')) → action_logits  [액션 예측 → 표현 학습]

내재 보상  r_int = η × ‖ForwardModel(φ(s), a) − φ(s')‖²
학습 손실       = β × forward_loss + (1−β) × inverse_loss
```

하이퍼파라미터: `η=0.01, β=0.2, lr=3e-4, buffer=8000, update_freq=512`

### Reward System

#### Phase 1 — Descend (baseline과 동일)

Step penalty, health loss, death, Y-level shaping, Y-delta, milestones 모두 동일.

#### Phase 2 — ICM 내재 보상 (baseline의 셀 추적 대체)

| 보상 항목 | 값 |
|---|---|
| ICM 내재 보상 | icm_scale × r_int (clip: 2.0) |
| 블록 파괴 | +0.15 |

> baseline 차이: 사람이 "새 위치 = 탐험"으로 정의하는 대신, 네트워크가 예측 못하는 상태 전이를 자동으로 탐험으로 간주한다.

#### Phase 3 — Mine (게이팅 없이 항상 활성)

baseline Phase 3와 동일하지만 mine_gate 없이 항상 적용:

| 보상 항목 | 값 |
|---|---|
| 광물 조준 | 0.3 × ore_value |
| 광물 조준 + 공격 | 1.0 × ore_value |
| 인벤토리 획득 | reward × log(1 + gained) |

Reward clip: `[−30, 60]`

---

## 3. HRL — `hrl_mining.py`

Changyong 작성. Rule-based Manager가 깊이 스테이지 서브골을 순서대로 지정하고, Worker PPO가 이를 달성하도록 학습한다.

### 구조

```
Manager (rule-based, 학습 없음)
  DEPTH_STAGES = [Y=40, Y=0, Y=−20, Y=−40, Y=−58]
  에이전트가 서브골 Y ± 5블록 내 도달 → 다음 스테이지
  800스텝 타임아웃 → 강제 다음 스테이지 (−5 패널티)

Worker (PPO, 학습)
  현재 서브골 정보(delta_y, stage_norm)를 obs에 포함하여 학습
```

### Observation Space

```
Dict({
  "image": Box(0, 255, shape=(64, 114, 3), dtype=uint8)
  "state": Box(-2, 2,  shape=(12,),        dtype=float32)   ← 12-dim
})
```

**state 벡터 (12-dim)** — baseline 대비 2개 추가

| idx | 값 | baseline/ICM과 차이 |
|---|---|---|
| 0–6 | 동일 | — |
| 7 | 0.0 (예약) | baseline=λ, ICM=icm_r |
| 8–9 | 동일 | — |
| **10** | **(y − target_y) / 122** | HRL 추가: 서브골까지 y 거리 |
| **11** | **stage_idx / 4** | HRL 추가: 현재 스테이지 진행도 |

### Reward System

#### 항상 활성

Step penalty, health loss, death, Y-delta, depth milestones는 baseline과 동일.

#### Phase 1 — 서브골 방향 Y shaping (baseline과 핵심 차이)

```
target_y       = DEPTH_STAGES[stage_idx]       # 현재 서브골
dist           = max(0, y − target_y)
max_dist       = Y_SURFACE − target_y
local_progress = 1 − dist / max_dist

reward += 0.15 × local_progress
```

> baseline 차이: baseline은 항상 Y=−58을 향해 shaping하지만, HRL은 현재 스테이지 목표(예: Y=40)를 향해 shaping하여 중간 목표에 대한 보상 밀도가 높다.

#### HRL Manager 이벤트

| 이벤트 | 보상 |
|---|---|
| 스테이지 도달 (y ≤ target + 5) | +20.0 |
| 타임아웃 (800스텝 초과) | −5.0 |

#### Phase 2 — 블록 파괴 (항상)

| 보상 항목 | 값 |
|---|---|
| 블록 파괴 | +0.15 |

#### Phase 3 — 채굴 보상 (mine_gate 게이팅)

```
stage_idx 0, 1  (Y > 0)    → mine_gate = 0.00
stage_idx 2     (Y ≤ −20)  → mine_gate = 0.33
stage_idx 3     (Y ≤ −40)  → mine_gate = 0.67
stage_idx 4     (Y ≤ −58)  → mine_gate = 1.00
```

| 보상 항목 | 값 |
|---|---|
| 광물 조준 | mine_gate × 0.3 × ore_value |
| 광물 조준 + 공격 | mine_gate × 1.0 × ore_value |
| 인벤토리 획득 | mine_gate × reward × log(1 + gained) |

Reward clip: `[−30, 60]`

---

## 세 알고리즘 한눈 비교

### Observation

| | Baseline | ICM | HRL |
|---|---|---|---|
| image shape | (64, 114, 3) | (64, 114, 3) | (64, 114, 3) |
| state dim | 10 | 10 | **12** |
| state[7] | λ (탐험 가중치) | icm_int_r | 0 |
| state[10] | — | — | delta_y to subgoal |
| state[11] | — | — | stage_idx norm |

### Reward

| | Baseline | ICM | HRL |
|---|---|---|---|
| Y shaping 기준 | 전역 Y=−58 고정 | 전역 Y=−58 고정 | **현재 스테이지 Y** |
| 탐험 보상 | 방문 셀 + λ 감쇠 | ICM 내재 보상 | 없음 (Manager 대체) |
| 채굴 게이트 | depth × explore | 없음 (항상 활성) | **stage_idx 기반** |
| 서브골 도달 보너스 | 없음 | 없음 | **+20.0** |

### 학습 구조

| | Baseline | ICM | HRL |
|---|---|---|---|
| 학습 정책 수 | 1 (PPO) | 1 (PPO) | 1 (PPO Worker) |
| 별도 네트워크 | 없음 | PhiEncoder, Forward, Inverse | 없음 |
| Manager | 없음 | 없음 | Rule-based (고정) |
| 탐험 설계 | 사람 (λ schedule) | 네트워크 (자동) | 사람 (stage schedule) |

### PPO 하이퍼파라미터 (safe 모드, 3개 공통)

| 파라미터 | 값 |
|---|---|
| learning_rate | 3e-4 |
| n_steps | 1024 |
| batch_size | 128 |
| n_epochs | 10 |
| gamma | 0.995 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.03 |

---

## wandb 로깅 항목

세 알고리즘 모두 project `mining_compare` 에 아래 메트릭을 기록한다.

| 메트릭 | 설명 |
|---|---|
| `ore/diamond` | 에피소드 평균 다이아몬드 획득량 |
| `ore/iron` | 에피소드 평균 철 획득량 |
| `ore/gold` | 에피소드 평균 금 획득량 |
| `ore/redstone` | 에피소드 평균 레드스톤 획득량 |
| `ore/lapis` | 에피소드 평균 청금석 획득량 |
| `ore/coal` | 에피소드 평균 석탄 획득량 |
| `ore/copper` | 에피소드 평균 구리 획득량 |
| `ore/emerald` | 에피소드 평균 에메랄드 획득량 |
| `mining/mean_ep_reward` | 에피소드 평균 총 보상 |
| `mining/mean_ep_length` | 에피소드 평균 길이 |
| `mining/avg_y_level` | 최근 400스텝 평균 Y 위치 |
| `mining/avg_stage_idx` | 최근 400스텝 평균 스테이지 *(HRL만)* |
| `mining/avg_icm_r` | 최근 400스텝 평균 내재 보상 *(ICM만)* |
