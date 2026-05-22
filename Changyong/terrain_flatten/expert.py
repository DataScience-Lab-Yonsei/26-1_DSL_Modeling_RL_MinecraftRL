"""
FlattenExpert — 12x12 지형 평탄화 전문가 (단순 pitch 스윕 방식).

전략:
  1. 열(tx, tz) 서펜타인 순회
  2. 각 열마다:
     a. (tx+0.5, tz-1.5) 로 이동  ← stone 접근로 위
     b. yaw=0 (+Z 방향) 정렬
     c. pitch를 위(-60°)→아래(+85°)로 스윕
        → raycast가 (tx, Y>TARGET_Y, tz) 에 맞으면 ATTACK
     d. 스윕 완료 후 다음 열

핵심 수정 (이전 버전 대비):
  - 복잡한 candidate_y 제거 → 단순 pitch 스윕
  - 석재 접근로: env.py에서 az-4까지 연장, tz-1.5도 solid 위
  - Jinwoo v3 raycast 피드백은 yaw 정렬에만 사용
"""
import math
import numpy as np

from config import (
    AREA_X, AREA_Z, AREA_W, AREA_D, TARGET_Y,
    CAMERA_DELTA_MAP, CAMERA_NEUTRAL,
    ACT_FWD_BACK, ACT_LEFT_RIGHT, ACT_JUMP, ACT_SNEAK,
    ACT_INTERACT, ACT_PITCH, ACT_YAW,
    NUM_ACT_DIMS,
)

# 이동 정밀도
COARSE_THR = 0.5
FINE_THR   = 0.08

# Yaw 정렬 허용 오차
YAW_TOL = math.radians(3.0)

# Pitch 스윕 범위 (라디안)
SWEEP_UP_TARGET   = math.radians(-55.0)   # 올려다보는 최대
SWEEP_DOWN_TARGET = math.radians(85.0)    # 내려다보는 최대

# 공격 후 대기 틱 (크리에이티브라도 shovel swing 필요)
BREAK_WAIT = 10

# 디버그 출력 간격 (스텝)
DEBUG_INTERVAL = 200


def _noop() -> np.ndarray:
    a = np.zeros(NUM_ACT_DIMS, dtype=np.int64)
    a[ACT_FWD_BACK]   = 1
    a[ACT_LEFT_RIGHT] = 1
    a[ACT_INTERACT]   = 1
    a[ACT_PITCH]      = CAMERA_NEUTRAL
    a[ACT_YAW]        = CAMERA_NEUTRAL
    return a


def _attack() -> np.ndarray:
    a = _noop()
    a[ACT_INTERACT] = 2   # attack
    return a


def _error_to_idx(error_deg: float) -> int:
    """각도 오차 → CAMERA_DELTA_MAP 인덱스."""
    if abs(error_deg) < 0.5:
        return CAMERA_NEUTRAL
    best, best_score = CAMERA_NEUTRAL, abs(error_deg)
    for i, d in enumerate(CAMERA_DELTA_MAP):
        remaining = abs(error_deg - d)
        if (error_deg > 0 and d > error_deg * 1.2) or \
           (error_deg < 0 and d < error_deg * 1.2):
            remaining += 5.0
        if remaining < best_score:
            best, best_score = i, remaining
    return best


def _fix_yaw(agent_yaw: float, target_yaw: float = 0.0):
    """yaw 정렬. (action, fixed) 반환."""
    err = target_yaw - agent_yaw
    err = (err + math.pi) % (2 * math.pi) - math.pi
    a = _noop()
    if abs(err) < YAW_TOL:
        return a, True
    a[ACT_YAW] = _error_to_idx(-math.degrees(err))
    return a, False


def _move_x(agent_x: float, target_x: float):
    """X축 이동 (yaw=0 가정). (action, arrived) 반환."""
    dx = target_x - agent_x
    a  = _noop()
    if abs(dx) <= FINE_THR:
        return a, True
    a[ACT_LEFT_RIGHT] = 0 if dx > 0 else 2
    if abs(dx) <= COARSE_THR:
        a[ACT_SNEAK] = 1
    return a, False


def _move_z(agent_z: float, target_z: float):
    """Z축 이동 (yaw=0 가정). (action, arrived) 반환."""
    dz = target_z - agent_z
    a  = _noop()
    if abs(dz) <= FINE_THR:
        return a, True
    a[ACT_FWD_BACK] = 2 if dz > 0 else 0
    if abs(dz) <= COARSE_THR:
        a[ACT_SNEAK] = 1
    return a, False


class FlattenExpert:
    """
    12x12 지형 평탄화 전문가.

    env는 FlattenEnv 인스턴스여야 하며 다음 속성을 가져야 함:
      agent_x, agent_z, agent_yaw, agent_pitch (라디안)
      _cg_obs, _parse_raycast()
    """

    S_MOVE_TO_COL  = "move_to_col"
    S_FIX_YAW      = "fix_yaw"
    S_SWEEP_UP     = "sweep_up"
    S_SWEEP_DOWN   = "sweep_down"
    S_WAIT_BREAK   = "wait_break"
    S_NEXT_COL     = "next_col"
    S_DONE         = "done"

    def __init__(self):
        self._columns  = self._build_column_list()
        self._col_idx  = 0
        self._state    = self.S_MOVE_TO_COL
        self._wait_cnt = 0
        self._total_steps = 0

    def reset(self):
        self._col_idx  = 0
        self._state    = self.S_MOVE_TO_COL
        self._wait_cnt = 0
        self._total_steps = 0

    def is_done(self) -> bool:
        return self._state == self.S_DONE

    @staticmethod
    def _build_column_list():
        """서펜타인 순서로 (tx, tz) 목록 생성."""
        cols = []
        for dz in range(AREA_D):
            z  = AREA_Z + dz
            xs = range(AREA_W) if dz % 2 == 0 else range(AREA_W - 1, -1, -1)
            for dx in xs:
                cols.append((AREA_X + dx, z))
        return cols

    # ── 메인 API ────────────────────────────────────────────────
    def get_action(self, env) -> np.ndarray:
        self._total_steps += 1

        if self._state == self.S_DONE or self._col_idx >= len(self._columns):
            self._state = self.S_DONE
            return _noop()

        tx, tz = self._columns[self._col_idx]

        # 디버그 출력
        if self._total_steps % DEBUG_INTERVAL == 1:
            print(f"  [expert] step={self._total_steps} state={self._state} "
                  f"col={self._col_idx}/{len(self._columns)} "
                  f"target=({tx},{tz}) "
                  f"pos=({env.agent_x:.1f},{env.agent_y:.1f},{env.agent_z:.1f}) "
                  f"pitch={math.degrees(env.agent_pitch):.1f}°")

        if self._state == self.S_MOVE_TO_COL:
            return self._do_move(env, tx, tz)
        if self._state == self.S_FIX_YAW:
            return self._do_fix_yaw(env)
        if self._state == self.S_SWEEP_UP:
            return self._do_sweep_up(env, tx, tz)
        if self._state == self.S_SWEEP_DOWN:
            return self._do_sweep_down(env, tx, tz)
        if self._state == self.S_WAIT_BREAK:
            return self._do_wait_break(env)
        if self._state == self.S_NEXT_COL:
            return self._do_next_col()
        return _noop()

    # ── S_MOVE_TO_COL ───────────────────────────────────────────
    def _do_move(self, env, tx, tz) -> np.ndarray:
        target_x = tx + 0.5
        target_z = float(tz) - 1.5   # 열 바로 앞 (stone 접근로 위)

        # yaw=0 먼저 정렬
        a, yaw_ok = _fix_yaw(env.agent_yaw, 0.0)
        if not yaw_ok:
            return a

        # X 이동 우선, 그 다음 Z
        a, x_ok = _move_x(env.agent_x, target_x)
        if not x_ok:
            return a

        a, z_ok = _move_z(env.agent_z, target_z)
        if not z_ok:
            return a

        # 도착 → 스윕 준비
        self._state = self.S_FIX_YAW
        return _noop()

    # ── S_FIX_YAW ───────────────────────────────────────────────
    def _do_fix_yaw(self, env) -> np.ndarray:
        a, fixed = _fix_yaw(env.agent_yaw, 0.0)
        if fixed:
            self._state = self.S_SWEEP_UP
        return a

    # ── S_SWEEP_UP: pitch를 올려다보는 쪽으로 ─────────────────
    def _do_sweep_up(self, env, tx, tz) -> np.ndarray:
        pitch = env.agent_pitch   # radians, positive=down
        if pitch <= SWEEP_UP_TARGET:
            self._state = self.S_SWEEP_DOWN
            return _noop()
        # 위로 보기: -3°/tick
        a = _noop()
        a[ACT_PITCH] = 1   # CAMERA_DELTA_MAP[1] = -3.0°
        return a

    # ── S_SWEEP_DOWN: 아래로 스윕하며 공격 ──────────────────────
    def _do_sweep_down(self, env, tx, tz) -> np.ndarray:
        # 현재 raycast 확인
        hit = None
        if env._cg_obs is not None:
            hit = env._parse_raycast(env._cg_obs)

        if hit is not None:
            hx, hy, hz = hit["position"]
            # 목표 열이고 TARGET_Y 초과면 공격
            if hx == tx and hz == tz and hy > TARGET_Y:
                print(f"  [expert] ATTACK ({hx},{hy},{hz}) pitch={math.degrees(env.agent_pitch):.1f}°")
                self._wait_cnt = BREAK_WAIT
                self._state    = self.S_WAIT_BREAK
                return _attack()

        # 스윕 완료?
        pitch = env.agent_pitch
        if pitch >= SWEEP_DOWN_TARGET:
            self._state = self.S_NEXT_COL
            return _noop()

        # 아래로 계속 스윕: +3°/tick
        a = _noop()
        a[ACT_PITCH] = 7   # CAMERA_DELTA_MAP[7] = +3.0°
        return a

    # ── S_WAIT_BREAK: 공격 유지 ─────────────────────────────────
    def _do_wait_break(self, env) -> np.ndarray:
        self._wait_cnt -= 1
        if self._wait_cnt > 0:
            return _attack()
        # 같은 열에 더 있을 수 있으니 다시 스윕 (pitch 유지, DOWN만 계속)
        self._state = self.S_SWEEP_DOWN
        return _noop()

    # ── S_NEXT_COL ──────────────────────────────────────────────
    def _do_next_col(self) -> np.ndarray:
        self._col_idx += 1
        if self._col_idx >= len(self._columns):
            self._state = self.S_DONE
            print(f"  [expert] 모든 열 완료! total_steps={self._total_steps}")
        else:
            self._state = self.S_MOVE_TO_COL
        return _noop()
