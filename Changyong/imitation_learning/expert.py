"""
ScriptedExpert — 규칙 기반 집짓기 전문가.

상태 머신:
  ORIENT → APPROACH → AIM_DOWN → PLACE → LOOK_UP → (반복)

Jinwoo v3 참고 개선 사항:
  - AIM_DOWN: raycast가 정확히 subgoal 아래 돌 블록을 맞출 때만 PLACE
  - 강제 배치(pitch>=80) 제거: 잘못된 위치 배치 방지
  - AIM_DOWN 타임아웃: 재조준 불가 시 뒤로 이동 후 재시도
  - LOOK_UP: PLACE 후 시야 복원 (다음 이동 준비)
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from building_hrl import HierarchicalBuildingEnv, ACTIONS
from config import (
    EXPERT_YAW_TOL,
    EXPERT_DIST_TOL,
    EXPERT_PLACE_STEPS,
    EXPERT_NAV_PITCH,
)

_NO_OP      = ACTIONS.index("NO_OP")
_FORWARD    = ACTIONS.index("FORWARD")
_BACKWARD   = ACTIONS.index("BACKWARD")
_TURN_LEFT  = ACTIONS.index("TURN_LEFT")
_TURN_RIGHT = ACTIONS.index("TURN_RIGHT")
_LOOK_UP    = ACTIONS.index("LOOK_UP")
_LOOK_DOWN  = ACTIONS.index("LOOK_DOWN")
_USE_ITEM   = ACTIONS.index("USE_ITEM")

# AIM_DOWN에서 정확한 블록을 못 찾으면 뒤로 빠졌다 재시도
_AIM_MAX_STEPS   = 60   # 이 스텝 이상 조준 실패 → 뒤로 이동
_BACKUP_STEPS    = 8    # 뒤로 이동할 스텝 수
# LOOK_UP 목표: 이 pitch 이하가 되면 이동 준비 완료
_NAV_PITCH_THR   = EXPERT_NAV_PITCH   # 20.0도


class ScriptedExpert:
    """
    env.player_pos, env.player_yaw, env.player_pitch, env.current_subgoal 을
    읽어 매 스텝 action 하나를 반환합니다.

    VectorBuildingEnv (player_pitch, raycast_can_place 속성 보유) 와 함께 사용.
    """

    def __init__(self, env):
        self.env = env
        self._phase      = "ORIENT"
        self._phase_step = 0

    def reset(self):
        self._phase      = "ORIENT"
        self._phase_step = 0

    # ── 퍼블릭 API ─────────────────────────────────────────────────
    def get_action(self) -> int:
        sg = self.env.current_subgoal
        if sg is None:
            return _NO_OP

        px, _, pz = self.env.player_pos
        gx, _, gz = sg
        dx, dz    = gx - px, gz - pz
        dist       = math.sqrt(dx ** 2 + dz ** 2)

        bearing  = math.degrees(math.atan2(-dx, dz))
        yaw_diff = (bearing - self.env.player_yaw + 180) % 360 - 180

        pitch = getattr(self.env, "player_pitch", 0.0)

        return self._step(dist, yaw_diff, pitch)

    # ── 상태 머신 ──────────────────────────────────────────────────
    def _step(self, dist: float, yaw_diff: float, pitch: float) -> int:

        # ── ORIENT: yaw 정렬 ─────────────────────────────────────
        if self._phase == "ORIENT":
            if abs(yaw_diff) <= EXPERT_YAW_TOL:
                self._transition("APPROACH")
                return _FORWARD
            return _TURN_LEFT if yaw_diff > 0 else _TURN_RIGHT

        # ── APPROACH: 전진 ───────────────────────────────────────
        if self._phase == "APPROACH":
            if abs(yaw_diff) > EXPERT_YAW_TOL * 2:
                self._transition("ORIENT")
                return _TURN_LEFT if yaw_diff > 0 else _TURN_RIGHT
            if dist > EXPERT_DIST_TOL:
                return _FORWARD
            self._transition("AIM_DOWN")
            return _LOOK_DOWN

        # ── AIM_DOWN: 정확히 subgoal 아래 돌 블록을 조준할 때만 배치 ──
        if self._phase == "AIM_DOWN":
            self._phase_step += 1

            can_place = getattr(self.env, "raycast_can_place", False)
            if can_place:
                self._transition("PLACE")
                return _USE_ITEM

            # 타임아웃: 뒤로 이동 후 재시도
            if self._phase_step > _AIM_MAX_STEPS:
                self._transition("BACKUP")
                return _BACKWARD

            return _LOOK_DOWN

        # ── BACKUP: 뒤로 물러났다가 ORIENT 재시도 ──────────────────
        if self._phase == "BACKUP":
            self._phase_step += 1
            if self._phase_step < _BACKUP_STEPS:
                return _BACKWARD
            self._transition("LOOK_UP")
            return _LOOK_UP

        # ── PLACE: 블록 설치 ─────────────────────────────────────
        if self._phase == "PLACE":
            self._phase_step += 1
            if self._phase_step < EXPERT_PLACE_STEPS:
                return _USE_ITEM
            self._transition("LOOK_UP")
            return _LOOK_UP

        # ── LOOK_UP: 이동 전 시야 복원 ──────────────────────────────
        if self._phase == "LOOK_UP":
            if pitch <= _NAV_PITCH_THR:
                self._transition("ORIENT")
                return _NO_OP
            return _LOOK_UP

        return _NO_OP

    def _transition(self, next_phase: str):
        self._phase      = next_phase
        self._phase_step = 0
