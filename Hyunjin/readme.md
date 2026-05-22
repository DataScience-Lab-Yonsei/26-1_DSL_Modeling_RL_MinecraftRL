# Hyunjin — Survive and Hunt (Raycast Curriculum RL)

Minecraft craftground 환경에서 적대 mob을 처치하며 생존하는 에이전트. 이미지 없이 **raycast/vector 관측**만으로 커리큘럼 학습.

환경 파일: [`survive_and_hunt_environment.py`](survive_and_hunt_environment.py)

---

## 태스크 설명

- **환경**: `SurviveAndHuntEnvironment` — 적대 mob(husk, zombie, skeleton, spider)과 전투하며 생존
- **목표**: 최대한 오래 생존하면서 적을 처치 (kiting + combat)
- **관측**: 이미지 없이 raycast 기반 vector만 사용 (`RaycastVectorObsWrapper`)
- **정책**: MLP-based PPO (CnnPolicy가 아닌 MlpPolicy)

---

## 커리큘럼 학습 — 6 라운드

| 라운드 | 이름 | 보상 프로필 | 목적 |
|---|---|---|---|
| r01 | survival_bootstrap | survival_kite | 기본 생존 + 이동 학습 |
| r02 | hitfocus_recovery | hitfocus_raycast | 조준 및 공격 집중 |
| r03 | duel_finishconvert | finishconvert_raycast | 1:1 전투 마무리 |
| r04 | kite_pair_finish | (확장) | 2인 kite + 처치 |
| r05 | clear_easy_powerfinish | (확장) | 다수 처치 |
| r06 | generalize_combined | (확장) | 전체 일반화 |

---

## 산출물

각 라운드/에포크별 자동 저장:

| 파일 | 설명 |
|---|---|
| `epoch_XXX.zip` | 에포크 체크포인트 |
| `epoch_XXX_eval_summary.json` | 에포크 평가 결과 |
| `gifs/*.gif` | 에포크 평가 GIF |
| `round_metrics.png` | 라운드별 지표 플롯 |
| `global_epoch_metrics.csv` | 전체 에포크 지표 |
| `global_summary.json` | 전체 요약 |

저장 경로: `Hyunjin/artifacts/raycast_hunt/<run_name>/`

---

## 실행 방법

```bash
# 기본 학습
python train_craftground_raycast_curriculum.py \
  --run-name my_run \
  --num-envs 1 \
  --steps-per-epoch 2048 \
  --eval-episodes 2 \
  --seed 42

# smoke 테스트 (빠른 동작 확인)
python train_craftground_raycast_curriculum.py \
  --run-name smoke \
  --timesteps-scale 0.1 \
  --steps-per-epoch 1024 \
  --eval-episodes 1

# 체크포인트 이어 학습
python train_craftground_raycast_curriculum.py \
  --run-name continue_run \
  --init-checkpoint artifacts/raycast_hunt/<run_name>/epoch_XXX.zip
```

---

## 모니터링

```bash
# 터미널 상주형 모니터 (최신 run 자동 선택)
bash monitor_training_terminal.sh

# 특정 run 모니터
bash monitor_training_terminal.sh <run_name>

# Python 기반 스냅샷 출력
python monitor_raycast_training_speed.py

# 다중 후보 학습 후 최고 모델 선택
python run_next_training_select_best.py
```

---

## 파일 구조

```
Hyunjin/
├── train_craftground_raycast_curriculum.py   # 메인 학습 엔트리포인트
├── run_next_training_select_best.py          # 다중 후보 학습 + 최고 모델 선택
├── monitor_raycast_training_speed.py         # 학습 진행 스냅샷 출력
├── monitor_training_terminal.sh              # 터미널 상주형 모니터
├── rebuild_overlay_preview.py                # GIF/오버레이 미리보기 재생성
├── README_raycast_curriculum.md              # 커리큘럼 상세 설명
└── README_directory.md                       # 디렉토리 가이드
```

의존: `survive_and_hunt_environment.py` (같은 폴더)
