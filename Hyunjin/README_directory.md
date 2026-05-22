# Hyunjin Directory Guide

이 디렉토리는 raycast 기반 CraftGround 강화학습 실행/모니터링 중심으로 정리되어 있습니다.

## 핵심 실행 파일

- `train_craftground_raycast_curriculum.py`: 메인 학습 엔트리포인트
- `run_next_training_select_best.py`: 다중 후보 학습 후 최고 모델 선택 실행기
- `monitor_raycast_training_speed.py`: 학습 진행 상황 스냅샷(JSON/TEXT) 출력
- `monitor_training_terminal.sh`: 터미널 상주형 모니터 (pretty JSON)

## 주요 디렉토리

- `artifacts/raycast_hunt/`: 체크포인트, GIF, 에포크 평가 결과
- `runs/raycast_hunt/`: TensorBoard 이벤트 로그
- `archive/`: 과거 실험 코드/문서 보관
- `CraftGround/`: CraftGround 원본/연동 코드

## 빠른 사용

```bash
# 학습 모니터 (최신 run 자동 선택)
/home/hj/dsl/modeling/Hyunjin/monitor_training_terminal.sh

# 특정 run 모니터
/home/hj/dsl/modeling/Hyunjin/monitor_training_terminal.sh s84_169_20260325_continue
```
