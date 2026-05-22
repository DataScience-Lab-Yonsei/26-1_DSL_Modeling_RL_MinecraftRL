# CraftGround Raycast Curriculum Trainer

`train_craftground_raycast_curriculum.py`는 `SurviveAndHuntEnvironment`를 raycast/vector 중심으로 학습하도록 새로 구성한 엔트리포인트입니다.

## 핵심 특징

- 라운드 분해 커리큘럼:
  - `r01_survival_bootstrap`
  - `r02_hitfocus_recovery`
  - `r03_duel_finishconvert`
  - `r04_kite_pair_finish`
  - `r05_clear_easy_powerfinish`
  - `r06_generalize_combined`
- 정책 입력: 이미지가 아닌 `vector` 관측만 사용 (`RaycastVectorObsWrapper`)
- 에포크 단위 저장:
  - 매 에포크 체크포인트(`epoch_XXX.zip`)
  - 매 에포크 평가 JSON(`epoch_XXX_eval_summary.json`)
  - 매 에포크 평가 GIF(`gifs/*.gif`)
  - 라운드별 지표 플롯(`round_metrics.png`)
- 전역 집계:
  - `global_epoch_metrics.csv`
  - `global_summary.json`

## 실행 예시

```bash
/home/hj/dsl/modeling/venv/bin/python /home/hj/dsl/modeling/Hyunjin/train_craftground_raycast_curriculum.py \
  --run-name s84_raycast_newline \
  --num-envs 1 \
  --steps-per-epoch 2048 \
  --eval-episodes 2 \
  --seed 84
```

빠른 smoke 실행:

```bash
/home/hj/dsl/modeling/venv/bin/python /home/hj/dsl/modeling/Hyunjin/train_craftground_raycast_curriculum.py \
  --run-name smoke_raycast \
  --timesteps-scale 0.1 \
  --steps-per-epoch 1024 \
  --eval-episodes 1
```

기존 체크포인트 이어학습:

```bash
/home/hj/dsl/modeling/venv/bin/python /home/hj/dsl/modeling/Hyunjin/train_craftground_raycast_curriculum.py \
  --run-name continue_from_old \
  --init-checkpoint /mnt/e/RL_pjt/Hyunjin/artifacts/survive_and_hunt/branch_hunter_hitfocus_finetune_50k_lowent/stage3_combined_easy.zip
```

## 산출물 위치

기본 저장 경로:

- 모델/평가/GIF: `Hyunjin/artifacts/raycast_hunt/<run_name>/`
- 텐서보드: `Hyunjin/runs/raycast_hunt/`

## 참고 경로

기존 실험 참고 루트는 기본값으로 아래를 기록합니다.

- `/mnt/e/RL_pjt/Hyunjin`

(`run_meta.json`의 `legacy_root`, `legacy_root_exists`에 반영됨)
