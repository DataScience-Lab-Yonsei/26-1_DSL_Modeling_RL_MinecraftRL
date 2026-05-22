# Changyong — Minecraft RL 실험

craftground 환경 기반으로 PPO, HRL, Imitation Learning, ICM 등 다양한 접근법을 탐구한 실험 모음.

---

## 실험 목록

| 파일/폴더 | 태스크 | 알고리즘 | 특이사항 |
|---|---|---|---|
| `survival_rl.py` | 생존 (체력·허기 유지 + 적 처치) | PPO (CnnPolicy) | 84×84 RGB, 13 이산 액션 |
| `building_rl.py` | 3×3 oak_planks 배치 | PPO (CnnPolicy) | raycast shaping |
| `building_hrl.py` | 3×3 oak_planks 배치 | HRL (Rule-based Manager + PPO Worker) | Manager가 가장 가까운 빈 subgoal 선택 |
| `imitation_learning/` | 3×3 oak_planks 배치 | Behavioral Cloning | 41차원 벡터 obs, 전문가 데모 → BC 학습 |
| `mining_icm/` | Minecraft 광물 채굴 | PPO + ICM | Pathak et al. 2017 내재 호기심 보상 |
| `mining_compare/` | 광물 채굴 3방식 비교 | PPO Baseline / ICM / HRL | wandb project: `mining_compare` |
| `terrain_flatten/` | 12×12 지형 평탄화 | Imitation Learning (BC) | MultiDiscrete 액션, 18차원 벡터 obs |

---

## 폴더 구조

```
Changyong/
├── survival_rl.py          # 생존 RL
├── building_rl.py          # 건축 RL (Flat PPO)
├── building_hrl.py         # 건축 HRL (Manager-Worker)
├── place_blocks.py         # 블록 배치 동작 테스트
├── imitation_learning/     # BC 학습 파이프라인
│   ├── collect_demos.py    # 전문가 데모 수집
│   ├── bc_train.py         # BC 학습
│   ├── bc_eval.py          # BC 평가
│   ├── env.py              # VectorBuildingEnv (41-dim obs)
│   ├── expert.py           # Rule-based 전문가
│   ├── model.py            # MLPPolicy
│   └── config.py
├── mining_icm/             # ICM 탐험 모듈
│   ├── mining_icm_rl.py    # ICM 기반 채굴 RL
│   ├── icm.py              # PhiEncoder / ForwardModel / InverseModel
│   └── cave_seed_scanner_v1.py
├── mining_compare/         # 3가지 채굴 접근법 비교
│   ├── hrl_mining.py       # HRL Manager-Worker 채굴
│   ├── run_compare.sh      # 3개 순서 실행 스크립트
│   ├── cave_seed_scanner_v1.py
│   └── README.md           # 상세 비교 문서 (관측/보상/구조 diff)
└── terrain_flatten/        # 지형 평탄화 (BC)
    ├── env.py              # FlattenEnv (18-dim obs)
    ├── expert.py           # Rule-based 전문가
    ├── collect_demos.py    # 데모 수집
    └── config.py
```

---

## 실행 방법

### 생존 RL

```bash
python survival_rl.py train 2000000
python survival_rl.py eval survival_ppo_model
```

### 건축 RL (Flat PPO)

```bash
python building_rl.py train 1000000
python building_rl.py eval building_rl_model
```

### 건축 HRL

```bash
python building_hrl.py train 1000000
python building_hrl.py eval building_hrl_model
```

### Behavioral Cloning (건축)

```bash
# 1. 전문가 데모 수집
cd imitation_learning && python collect_demos.py

# 2. BC 학습
python bc_train.py 50 64

# 3. 평가
python bc_eval.py
```

### ICM 광물 채굴

```bash
cd mining_icm && python mining_icm_rl.py --mode train --total_steps 3000000
```

### 채굴 3방식 비교 (Baseline / ICM / HRL)

```bash
cd mining_compare && bash run_compare.sh
# wandb project: mining_compare 에서 3개 run 비교 가능
```

자세한 알고리즘 설명·보상 설계는 [mining_compare/README.md](mining_compare/README.md) 참조.

### 지형 평탄화 데모 수집

```bash
cd terrain_flatten && python collect_demos.py
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
