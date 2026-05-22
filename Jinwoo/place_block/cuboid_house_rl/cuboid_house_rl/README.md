# Cuboid House Construction Agent

CraftGround (Minecraft 1.21) 위에서 **Recurrent PPO**로 oak plank cuboid house를 짓는 강화학습 에이전트.

---

## 태스크 설명

- **환경:** Minecraft 1.21 Superflat, Survival 모드 (낙하 데미지 OFF, 포만감 OFF)
- **목표:** 7x7x5 oak plank 집 건축 (바닥, 벽, 천장 + 문)
- **총 블록:** 168개 (문 위치 2칸은 air)
- **좌표:** 집 원점 (17, 1, 17), 문 위치 (20, 2-3, 17)
- **지면:** MC y=-61 (grass), 에이전트 발 y=-60

---

## 커리큘럼 학습

2단계 커리큘럼으로 학습:

### Stage 1: Gaze Training (시선 학습)

바닥 블록 위치를 **보는 법**을 먼저 학습합니다. place/attack은 마스킹.

- **목표:** floor 블록 49개(7×7)의 아래 블록(y=0) 윗면을 레이캐스트로 봄
- **자유 방식:** 순서 상관없이 아무 미설치 블록이나 봐도 됨
- **3번 확인:** 같은 블록을 3번 봐야 "설치" 처리 → world에 반영 → observation 업데이트
- **target_direction:** 항상 가장 가까운 미설치 블록 방향을 알려줌
- **졸업 조건:** 최근 20 에피소드 평균 성공률 ≥ 80%
- **에피소드:** 최대 2500 스텝

**Stage 1 보상:**

| 이름 | 값 | 조건 |
|------|-----|------|
| `gaze_success` | +1.0 | 미설치 블록 위치를 레이캐스트로 볼 때마다 |
| `proximity_reward` | +0.00~+0.05 | 가장 가까운 미설치 블록과 거리 ≤ 5 |
| `distance_penalty` | -0.01×(d-5)²/step | 가장 가까운 미설치 블록과 거리 > 5 (제곱 패널티) |
| `angular_shaping` | -0.05~+0.10 | 가장 가까운 타겟 방향으로 카메라 이동 시 (변화량 기반) |
| `angular_penalty` | -0.01×ang/step | 타겟과의 각도에 비례한 지속 패널티 (ang=3→-0.03/step) |
| `time_penalty` | -0.05/step | 매 스텝 |

### Stage 2: Building (건축)

Stage 1의 가중치를 이어받아 실제 블록 배치를 학습합니다.

```bash
# Stage 1: 시선 학습
python -m cuboid_house_rl.training.train \
    --stage 1 --num-envs 1 \
    --total-timesteps 200000

# Stage 2: Stage 1 가중치로 건축 학습
python -m cuboid_house_rl.training.train \
    --stage 2 --num-envs 1 \
    --stage1-checkpoint checkpoints/stage1_graduated.pt \
    --total-timesteps 5000000
```

---

## 모델 구조

```
Observation                          Network                         Output
+------------------+
| Voxel Grids      |-- 3D CNN -- flatten(3456)--+
| (11,11,11,8)     |                            |
+------------------+                            +-- SharedMLP(512)--+
+------------------+                            |                   |
| Flat Features    |----------------------------+                   |
| (71 floats)      |                                                |
+------------------+                        +------------------------+
                                            |
                                  +---------+----------+
                                  |                    |
                            Actor LSTM(256)      Critic LSTM(256)
                                  |                    |
                            MLP(128)              MLP(128)
                                  |                    |
                            8 Action Heads        Value Head
                            (MultiDiscrete)       (scalar)
```

- **3D CNN:** 3층 Conv3d (32->64->128), 입력 11x11x11x8, 출력 3456
- **LSTM:** Actor/Critic 분리 (gradient conflict 방지), 레이어 수 CLI로 조정 가능
- **총 파라미터:** ~3.7M (1층 LSTM), ~4.8M (2층 LSTM)

---

## Observation

### Voxel Grids (11x11x11x8)

에이전트 중심 11블록 범위 로컬 뷰:

| 채널 | 내용 |
|------|------|
| 0-2 | 블록 타입 one-hot (air, planks, solid) |
| 3-5 | 가시성 one-hot (air, visible, non-visible) |
| 6 | 블루프린트 타겟 (여기에 블록 놓아야 함) |
| 7 | 미배치 (블루프린트=planks AND 현재=air) |

Stage 1에서 gaze 3번 성공 → `world[x,y,z] = OAK_PLANKS` → 채널 7이 0으로 바뀜 (미설치 블록 감소).

### Flat Features (71 floats)

| 구성요소 | 크기 | 내용 |
|---------|------|------|
| agent_state | 15 | 위치(3), yaw/pitch(2), 핫바 one-hot(9), has_planks(1) |
| raycast_info | 25 | 히트 여부, 블록 타입, 상대 위치, 거리, 법선, 배치 위치, 유효성, 블루프린트 매칭 |
| inventory | 10 | 현재 슬롯 one-hot(9), has_planks(1) |
| completion | 4 | 바닥/벽/천장/문 완성도 (0.0~1.0) |
| subtask_id | 4 | 현재 서브태스크 one-hot |
| prev_action | 8 | 이전 행동 |
| time_remaining | 1 | 남은 시간 비율 |
| stuck_ratio | 1 | stuck 진행도 (0.0=방금 진전, 1.0=페널티 직전) |
| target_direction | 3 | 가장 가까운 미배치 블록 방향: delta_yaw(-1~+1), delta_pitch(-1~+1), 거리(normalized) |

---

## Action Space

`MultiDiscrete([3, 3, 2, 2, 3, 7, 7, 2])`

| Dim | 이름 | 값 | 초기 bias |
|-----|------|-----|-----------|
| 0 | 전진/후진 | 0=후진, 1=정지, 2=전진 | Stage1: 균등(33%), Stage2: fwd=52% |
| 1 | 좌/우 이동 | 0=좌, 1=정지, 2=우 | Stage1: 균등(33%), Stage2: stop=73% |
| 2 | 달리기 | 0=안함, 1=달리기 | no=88%, sprint=12% |
| 3 | 점프 | 0=안함, 1=점프 | no=88%, jump=12% |
| 4 | 상호작용 | 0=place, 1=noop, 2=attack (masked) | place=73%, noop=27% |
| 5 | 카메라 pitch | [-10, -3, -1, 0, 1, 3, 10]° | center=90%, symmetric |
| 6 | 카메라 yaw | [-10, -3, -1, 0, 1, 3, 10]° | center=90%, symmetric |
| 7 | 핫바 | 0-8 (masked: planks 슬롯만 유효) | slot0=52%, slot1=19% |

**Stage 1:** interact는 noop만 허용 (place/attack 마스킹). 이동 bias는 균등(33%/33%/33%), 카메라는 탐색 중심.

---

## Stage 2 보상 구조

| 이름 | 값 | 조건 |
|------|-----|------|
| `correct_placement` | +5.0 | 블루프린트 위치에 첫 배치 |
| `incorrect_placement` | -0.5 | 블루프린트에 없는 위치 |
| `incorrect_removal` | -0.5 | 있어야 할 블록 제거 |
| `progress` | delta x 15.0 | 매 스텝 completion 변화량 |
| `floor_complete` | +15.0 | 마일스톤 |
| `walls_complete` | +20.0 | 마일스톤 |
| `ceiling_complete` | +15.0 | 마일스톤 |
| `door_correct` | +10.0 | 마일스톤 |
| `house_complete` | +100.0 | 168블록 + 문 완성 |
| `time_penalty` | -0.005/step | 매 스텝 |
| `proximity_penalty` | -0.01 x (dist-5)/step | 집 테두리에서 맨해튼 거리 > 5블록 (선형, 거리 비례) |
| `looking_at_target` | +0.02/step | 미배치 블루프린트 위치를 바라볼 때 |
| `stationary_penalty` | -0.02/step | 같은 블록에 500스텝 이상 머물 때 |
| `stuck_penalty` | -10.0 | 1000스텝 미배치 시 |

에피소드 종료 조건: 성공(전체 완성) / 타임아웃(3000스텝) / Stuck(1000스텝 무진전)

---

## 실행 방법

### 학습

```bash
# Stage 1: 시선 학습 (엔트로피 annealing 권장)
python -m cuboid_house_rl.training.train \
    --stage 1 --num-envs 1 \
    --entropy-start 0.05 --entropy-end 0.005 \
    --eval-interval 999999999 \
    --total-timesteps 300000

# Stage 2: 건축 학습 (Stage 1 가중치 이어받기)
python -m cuboid_house_rl.training.train \
    --stage 2 --num-envs 1 \
    --stage1-checkpoint checkpoints/stage1_graduated.pt \
    --total-timesteps 5000000

# Stage 2: 처음부터 학습 (커리큘럼 없이)
python -m cuboid_house_rl.training.train \
    --num-envs 1 \
    --total-timesteps 5000000

# WandB 로깅 (이름 + 메모 포함)
python -m cuboid_house_rl.training.train \
    --mode train \
    --wandb-project cuboid-house \
    --wandb-run-name "v3-spawn-center" \
    --wandb-notes "spawn center, stationary penalty, linear proximity"

# 체크포인트에서 재개
python -m cuboid_house_rl.training.train \
    --mode train \
    --resume checkpoints/latest.pt
```

### 미리보기 (실시간 렌더링)

```bash
python -m cuboid_house_rl.training.train \
    --mode preview \
    --resume checkpoints/best.pt
```

### 영상 녹화

```bash
python -m cuboid_house_rl.training.train \
    --mode record \
    --resume checkpoints/best.pt \
    --record-path videos/ \
    --record-episodes 5
```

### 평가

```bash
python -m cuboid_house_rl.training.train \
    --mode eval \
    --resume checkpoints/best.pt \
    --eval-episodes 20
```

---

## 주요 CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--stage` | `2` | 1=시선 학습, 2=건축 |
| `--stage1-checkpoint` | `None` | Stage 1 가중치 경로 (Stage 2에서 사용) |
| `--mode` | `train` | train / eval / preview / record |
| `--num-envs` | `4` | 병렬 환경 수 (Stage 1은 1 권장) |
| `--total-timesteps` | `5000000` | 총 학습 스텝 |
| `--lr` | `3e-4` | 학습률 |
| `--gamma` | `0.995` | 할인율 |
| `--gae-lambda` | `0.95` | GAE lambda |
| `--clip-ratio` | `0.2` | PPO 클리핑 |
| `--entropy-start` | `0.05` | 엔트로피 초기값 (annealing) |
| `--entropy-end` | `0.005` | 엔트로피 최종값 |
| `--entropy-coeff` | `None` | 고정 엔트로피 (annealing 비활성화) |
| `--sequence-length` | `64` | LSTM BPTT 시퀀스 길이 |
| `--lstm-layers` | `1` | LSTM 레이어 수 |
| `--batch-size` | `2048` | 롤아웃 버퍼 크기 |
| `--mini-batch-size` | `512` | 미니배치 크기 |
| `--update-epochs` | `4` | PPO 업데이트 에폭 |
| `--eval-interval` | `100000` | 평가 주기 (999999999로 비활성화) |
| `--save-interval` | `50000` | 체크포인트 저장 주기 |
| `--no-wandb` | `false` | WandB 비활성화 |
| `--wandb-project` | `cuboid-house` | WandB 프로젝트 이름 |
| `--wandb-run-name` | `None` | WandB run 이름 (미지정 시 자동 생성) |
| `--wandb-notes` | `None` | WandB run 메모 |
| `--wandb-resume` | `None` | WandB resume 모드 (must/allow/never) |
| `--resume` | `None` | 체크포인트 경로 |

---

## 주요 기능

- **2단계 커리큘럼:** Stage 1(시선) → Stage 2(건축)으로 점진적 학습
- **자유 방식 Gaze Training:** 순서 없이 아무 미설치 블록 3번 봐야 설치 처리
- **Entropy Annealing:** 초반 높은 엔트로피(0.05)로 탐색, 후반 낮은 엔트로피(0.005)로 수렴
- **Blueprint Observation:** voxel grid에 블루프린트/미배치 채널 포함, 에이전트가 어디에 놓아야 하는지 직접 관찰
- **LSTM Hidden State 관리:** 에피소드 경계 + truncation에서 자동 리셋
- **Stuck Detection:** 1000스텝 동안 진행 없으면 에피소드 종료
- **Stationary Penalty:** 같은 블록에 500스텝 이상 머물면 페널티 → 이동/점프 학습 유도
- **Proximity Penalty:** 집에서 멀어질수록 제곱 페널티 → 집 근처 유지 (Stage 1), 선형 (Stage 2)
- **Angular Penalty:** 타겟 방향과의 각도에 비례한 지속 패널티 → 엉뚱한 곳 보기 방지
- **Looking at Target:** 미배치 블루프린트 위치를 바라볼 때 보상 → 시선 제어 학습
- **Freeze Detection:** CraftGround 프리징 자동 감지 + 재초기화 (최대 5회 재시도)
- **Action Masking:** attack, 불필요한 핫바 슬롯 마스킹 (Stage 1에서는 place/attack도 마스킹)
- **Auto Ground Detection:** MC 1.21 superflat 지면 Y좌표 자동 설정 (y=-61)
- **Preview/Record:** cv2 실시간 렌더링, MP4 녹화 지원

---

## 프로젝트 구조

```
cuboid_house_rl/
├── config.py                    # 모든 하이퍼파라미터 및 상수
├── envs/
│   ├── house_building_env.py    # Gymnasium 환경 (Stage 2)
│   ├── gaze_training_env.py     # Gaze 학습 환경 (Stage 1)
│   └── craftground_adapter.py   # CraftGround 연동 (좌표 변환, 명령)
├── models/
│   ├── action_dist.py           # Masked multi-discrete distribution
│   └── network.py               # 3D CNN + LSTM actor-critic
├── training/
│   ├── rollout_buffer.py        # 시퀀스 기반 롤아웃 버퍼
│   ├── ppo.py                   # Recurrent PPO
│   └── train.py                 # 메인 학습/평가/녹화 스크립트
└── utils/
    ├── blueprint.py             # 집 블루프린트 생성/검증
    ├── completion.py            # 완성도 추적 + stuck 감지
    └── coord_transform.py       # 에이전트 상대 좌표 변환
```
