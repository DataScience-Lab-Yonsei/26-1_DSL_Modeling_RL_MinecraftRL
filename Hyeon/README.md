# Hyeon — Red Block Approach Task

Minecraft craftground 환경에서 빨간 블록(red wool)을 찾아 접근하는 시각 기반 강화학습 실험.

---

## 태스크 설명

- **환경**: 플랫한 아레나(`smooth_stone` 바닥), 주변 환경 초기화 후 red wool 블록 랜덤 배치
- **목표**: 에이전트가 빨간 블록 방향으로 이동/회전하여 화면 중앙에 블록이 충분히 들어오면 성공
- **관측**: 84×84 RGB 이미지 (VecFrameStack 8 스택)
- **성공 조건**: `center_red_ratio ≥ 0.06` OR `red_ratio ≥ 0.045`

---

## 액션 공간 — Discrete(7)

| idx | 이름 | 설명 |
|---|---|---|
| 0 | forward | 전진 |
| 1 | turn_left_small | yaw 좌 (작게) |
| 2 | turn_right_small | yaw 우 (작게) |
| 3 | turn_left_big | yaw 좌 (크게) |
| 4 | turn_right_big | yaw 우 (크게) |
| 5 | look_up_small | pitch 위 |
| 6 | look_down_small | pitch 아래 |

---

## 보상 설계

| 항목 | 값 | 설명 |
|---|---|---|
| 스텝 패널티 | −0.03 | 매 스텝 |
| red_ratio 증가 | ×10.0 | 화면 전체 빨간 비율 증가 |
| red_ratio 감소 | ×1.5 | 감소 페널티 |
| center_red 증가 | ×50.0 | 중앙 빨간 비율 증가 (강조) |
| center_red 감소 | ×3.0 | 감소 페널티 |
| 현재 red_ratio 보너스 | ×1.0 | 매 스텝 |
| 현재 center_red 보너스 | ×8.0 | 매 스텝 |
| 탐색 장려 | +0.05 | 블록이 안 보일 때 회전 액션 |
| 블록 보이면 전진 | +0.08 ~ +0.10 | red_ratio / center_red 임계치 초과 시 |
| 성공 | +20.0 | 성공 조건 달성 |
| 타임아웃 | −3.0 | max_steps 초과 |

빨간색 검출: `R ≥ 150, G ≤ 95, B ≤ 95` (RGB 임계값)

---

## 알고리즘 및 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 알고리즘 | PPO (CnnPolicy) |
| 프레임 스택 | 8 |
| total_timesteps | 200,000 |
| learning_rate | 2e-5 (이어 학습: 5e-6) |
| n_steps | 2048 |
| batch_size | 64 |
| clip_range | 0.05 |
| ent_coef | 0.005 |
| 에피소드 max_steps | 150 |
| 타겟 거리 | 8~10 블록 |

---

## 실행 방법

```python
# redblock.py 내부 실행부 (if __name__ == "__main__")에서 선택
train()           # 초기 학습
train_continue()  # 이어 학습 (기존 모델 로드)
test()            # 평가
random_test()     # 랜덤 정책 테스트
```

wandb project: `craftground-rl`

---

## 파일 구조

```
Hyeon/
└── redblock.py    # 환경 + 학습 + 평가 전체 포함
```
