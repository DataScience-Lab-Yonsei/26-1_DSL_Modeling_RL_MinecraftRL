# archi_ppo_v2 — NBT 기반 건축 PPO (MLP)

NBT 파일로 정의된 집 청사진을 보고 블록을 배치하는 건축 강화학습. LSTM 없이 MLP-only 네트워크 사용.

---

## 태스크 설명

- **환경**: `HouseBuildingWrapper` — craftground superflat 세계에서 NBT 청사진의 블록 위치를 모두 채우는 것이 목표
- **Stage 0**: 블록 종류 무관, 정확한 위치에 아무 블록이든 배치 (dirt 사용)
- **Stage 1+**: 전체 건축 (블록 종류 포함)
- **청사진**: NBT 파일로 로드하거나 `create_simple_blueprint()`로 생성

---

## 관측 / 액션

### 관측
- 청사진, 에이전트 위치/방향, raycast 정보, 진행도 등 벡터 obs (이미지 없음)

### 액션 공간 — Discrete(16)

| idx | 이름 |
|---|---|
| 0 | NO_OP |
| 1 | FORWARD |
| 2 | BACKWARD |
| 3 | STRAFE_LEFT |
| 4 | STRAFE_RIGHT |
| 5 | JUMP |
| 6 | LOOK_UP |
| 7 | LOOK_DOWN |
| 8 | LOOK_LEFT |
| 9 | LOOK_RIGHT |
| 10 | PLACE_BLOCK |
| 11 | BREAK_BLOCK |
| 12 | JUMP_AND_PLACE |
| 13–15 | HOTBAR_1~3 |

Camera step: pitch 10°, yaw 15°

---

## 네트워크 구조

```
Flat obs → BuilderNetwork (MLP-only, no LSTM)
         → Actor / Critic heads
```

LSTM이 없는 대신 deeper feed-forward trunk를 사용 (`ppo_network.BuilderNetwork`).

---

## 실행 방법

```bash
# 커리큘럼 학습
python ppo_train.py --curriculum

# 특정 NBT 파일로 학습
python ppo_train.py --nbt path/to/house.nbt

# 체크포인트 이어 학습
python ppo_train.py --resume checkpoints_ppo/latest.pt

# 평가 시각화
python visual_debug.py
```

---

## 파일 구조

```
archi_ppo_v2/
├── ppo_train.py     # 학습 엔트리포인트 (MLP-only PPO)
├── building_env.py  # HouseBuildingWrapper 환경
├── nbt_parser.py    # NBT 파일 파서 + Blueprint 자료구조
├── ppo_network.py   # BuilderNetwork (MLP-only)
├── evaluate.py      # 평가 스크립트
└── visual_debug.py  # 시각 디버그
```

> cuboid_house_rl_v3와 차이: LSTM이 없고, 블록 종류가 단순(dirt)하며, NBT 기반 청사진을 직접 사용.
