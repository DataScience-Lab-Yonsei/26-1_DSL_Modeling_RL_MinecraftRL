# 26-1 DSL RL — Minecraft 강화학습 프로젝트

**CraftGround** 기반 Minecraft 환경에서 다양한 태스크(생존, 건축, 채굴)를 강화학습으로 해결하는 팀 프로젝트.

---

## 팀원 및 담당 태스크

| 팀원 | 태스크 | 알고리즘 | 폴더 |
|---|---|---|---|
| **Hyeon** | 빨간 블록 접근 (시각 기반) | PPO (CnnPolicy, FrameStack) | [Hyeon/](Hyeon/) |
| **Hyunjin** | 생존 + 전투 (Survive & Hunt) | PPO (MlpPolicy, Raycast Vector) | [Hyunjin/](Hyunjin/) |
| **Seoyeon** | 집 건축 / 지하 광물 채굴 | PPO (CnnPolicy + 커스텀 보상) | [Seoyeon/](Seoyeon/) |
| **Jinwoo** | NBT 기반 건축 / Cuboid House | PPO + BC (MLP / Hierarchical LSTM) | [Jinwoo/](Jinwoo/) |
| **Changyong** | 생존 / 건축 HRL / ICM 채굴 / IL | PPO + HRL + ICM + BC | [Changyong/](Changyong/) |

---

## 태스크 개요

### 생존 (Survival)
- **Hyeon**: Red Wool 블록을 찾아 접근하는 시각 기반 태스크. RGB 임계값으로 빨간색 검출, 7개 이산 액션.
- **Hyunjin**: Hostile mob과 전투하며 생존. Raycast vector obs만 사용, 6단계 커리큘럼 학습.
- **Changyong**: 체력·허기를 유지하며 적을 처치하는 생존 RL.

### 건축 (Building)
- **Seoyeon**: 일직선 → 속빈 상자 → 전체 집 순서의 커리큘럼. Creative 모드 사전학습 후 Survival 파인튜닝.
- **Jinwoo**:
  - `archi_ppo_v2`: NBT 파일 청사진 기반, MLP-only PPO.
  - `cuboid_house_rl_v3`: Scripted Expert → BC → Hierarchical PPO 풀 파이프라인. 랜덤 집 크기.
- **Changyong**: 건축 Flat PPO → HRL (Rule-based Manager + PPO Worker) → Behavioral Cloning 순서로 실험.

### 채굴 (Mining)
- **Seoyeon**: Y-level shaping + λ-decay 탐험 보너스 + sparse 광물 보상의 3-Layer 구조.
- **Changyong**:
  - `mining_icm`: ICM (Intrinsic Curiosity Module, Pathak et al. 2017) 기반 자동 탐험.
  - `mining_compare`: Baseline (셀 추적) / ICM / HRL 세 방식을 동일 wandb project에서 비교.

---

## 공유 환경

### Hyunjin/survive_and_hunt_environment.py

`SurviveAndHuntEnvironment` — 생존+전투 핵심 환경. Hyunjin 폴더가 이 스크립트에 의존

주요 구성:
- `StageConfig`: 스폰할 mob 종류/수, 난이도 설정
- `RewardConfig`: 생존/전투/이동 관련 보상 파라미터 전체
- 관측: 이미지(선택) + raycast vector
- Lidar 기반 적 탐지 및 거리/방향 추적

---

## 폴더 구조

```
26-1_DSL_RL/
├── Hyeon/
│   ├── redblock.py                   # Red Block 접근 태스크
│   └── README.md
│
├── Hyunjin/
│   ├── survive_and_hunt_environment.py          # 공유 생존+전투 환경
│   ├── train_craftground_raycast_curriculum.py  # 메인 학습
│   ├── run_next_training_select_best.py
│   ├── monitor_raycast_training_speed.py
│   ├── monitor_training_terminal.sh
│   ├── rebuild_overlay_preview.py
│   ├── README_raycast_curriculum.md
│   ├── README_directory.md
│   └── readme.md
│
├── Seoyeon/
│   ├── house_rl_v1_0314.py           # 집 건축 (full house)
│   ├── hollow_box_rl_0314.py         # 속빈 상자 건축
│   ├── line_build_rl_0317.py         # 일직선 빌딩
│   ├── lidar_wrapper.py              # Lidar 래퍼
│   ├── creative_house/               # Creative 모드 건축
│   ├── mining_rl_pack/               # 광물 채굴 RL (v1, v2)
│   └── README.md
│
├── Jinwoo/
│   ├── archi_ppo_v2/                 # NBT 기반 MLP PPO
│   │   └── README.md
│   ├── cuboid_house_rl_v3/           # Expert → BC → Hierarchical PPO
│   │   └── cuboid_house_rl/
│   │       └── README.md            # 상세 설계 문서
│   ├── blueprint_datasets/           # 청사진 데이터셋 도구
│   └── place_block/                  # 블록 배치 패키지
│
└── Changyong/
    ├── survival_rl.py                # 생존 RL
    ├── building_rl.py                # 건축 Flat PPO
    ├── building_hrl.py               # 건축 HRL
    ├── place_blocks.py               # 블록 배치 테스트
    ├── imitation_learning/           # Behavioral Cloning
    ├── mining_icm/                   # ICM 채굴 RL
    ├── mining_compare/               # 채굴 3방식 비교
    ├── terrain_flatten/              # 지형 평탄화
    └── README.md
```

---

## 공통 의존성

```
craftground          # Minecraft 환경 (yhs0602/CraftGround)
stable-baselines3    # PPO 구현
torch
gymnasium
wandb
numpy
opencv-python
```

Java 21 필요 (CraftGround Minecraft 서버 실행용).
