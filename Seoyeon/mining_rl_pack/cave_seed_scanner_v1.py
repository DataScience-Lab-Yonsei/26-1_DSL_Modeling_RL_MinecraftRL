"""
cave_seed_scanner.py — 동굴 스폰 시드 DB 사전 구축

사용법:
  # 1) DB 구축 (최초 1회 오프라인 실행)
  python cave_seed_scanner.py --mode scan --n_seeds 200 --port 8040

  # 2) DB 확인
  python cave_seed_scanner.py --mode verify --db cave_db.json --port 8040

  # 3) 훈련 코드에서 사용
  from cave_seed_scanner import CaveSpawnWrapper
  env = CaveSpawnWrapper(make_env(port, ...), db_path="cave_db.json")

핵심 아이디어:
  InitialEnvironmentConfig의 initial_extra_commands는 Python 리스트로 저장되며,
  CraftGround는 reset()마다 이 리스트를 재실행한다.
  리스트를 in-place로 패치([-1] 슬롯 교체)하면 서버 재시작 없이
  매 reset마다 다른 좌표로 tp 가능.

동굴 감지 방식:
  tp 후 몇 tick 대기 → Y-낙하 감지(공간 있음) + 이미지 어두움(지하) 조합.
  score = 0.6 * fall_score + 0.4 * dark_score  (0~1, 높을수록 동굴 가능성 높음)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from craftground import InitialEnvironmentConfig, make
from craftground.initial_environment_config import WorldType
from craftground.environment.action_space import no_op_v2, ActionSpaceVersion


# =================================================================
# 1. 스캔 설정
# =================================================================

# 스캔할 Y 레벨: iron(-24), 중간층(-35), diamond(-50, -58) 순서로 타겟팅
SCAN_Y_LEVELS = [-35, -45, -55]

# x, z 탐색 반경 (스폰 지점 ±SCAN_RADIUS 내 랜덤 좌표)
SCAN_XZ_RADIUS = 150

# 동굴 판정 임계값: 이 이상이면 DB에 저장
CAVE_SCORE_THRESHOLD = 0.45

# 낙하 감지: tp 후 이만큼 이상 떨어지면 오픈 공간으로 판정
FALL_THRESHOLD_BLOCKS = 0.8

# 이미지 밝기: 이 값 미만이면 어둠 속(지하/동굴)으로 판정
DARK_THRESHOLD = 55
FULLY_DARK_THRESHOLD = 20  # 완전히 어두우면 (0점 — solid rock 가능성)

# reset 후 물리 시뮬레이션을 위해 대기할 tick 수
SETTLE_TICKS = 6

# 씨드 하나당 스캔할 (x, y, z) 후보 수
POSITIONS_PER_SEED = 8

# DB에 시드당 최대 저장 항목 수 (score 높은 순)
MAX_ENTRIES_PER_SEED = 3

# 환경 이미지 크기 (스캔용 — 작게 설정해서 속도 향상)
SCAN_IMG_W = 64
SCAN_IMG_H = 64


# =================================================================
# 2. 스캔 환경 — 서버 재시작 없이 좌표 변경
# =================================================================

# tp 커맨드가 들어갈 리스트 인덱스 (TP_SLOT 번째 원소를 패치)
_TP_SLOT = -1

# 기본 커맨드 (tp 명령 제외, tp는 리스트 맨 끝에 동적으로 추가)
_BASE_SCAN_COMMANDS = [
    "gamemode survival @p",
    "difficulty peaceful",
    "gamerule doMobSpawning false",
    "gamerule fallDamage false",       # 낙사 방지
    "gamerule doWeatherCycle false",
    "gamerule doDaylightCycle false",
    "time set 18000",                  # 어두운 시간대 → 이미지 밝기 기준 일관성
    "give @p minecraft:iron_pickaxe 1",
]


class _CaveScanEnv:
    """
    하나의 Minecraft 서버(make())를 유지하면서
    reset()마다 다른 (x, y, z)로 tp할 수 있는 스캔 전용 환경.

    핵심 트릭:
      _cmds 리스트의 _TP_SLOT(-1)을 in-place 수정 후 reset() 호출.
      CraftGround는 reset() 시 initial_extra_commands 리스트를 재실행하므로
      서버 재시작 없이 tp 좌표 변경 가능.
    """

    def __init__(self, port: int, seed: int):
        self.port = port
        self.seed = seed
        self._env = None
        self._cmds: list[str] | None = None  # make() 후 config 내부 리스트 참조

    def _boot(self, x: int, y: int, z: int) -> None:
        """최초 1회 서버 시작. tp 커맨드를 리스트 맨 끝에 포함."""
        self._cmds = _BASE_SCAN_COMMANDS + [f"tp @p {x} {y + 3} {z}"]
        cfg = InitialEnvironmentConfig(
            image_width=SCAN_IMG_W,
            image_height=SCAN_IMG_H,
            seed=str(self.seed),
            world_type=WorldType.DEFAULT,
            render_distance=4,
            simulation_distance=4,
            hud_hidden=True,
            request_raycast=False,  # 스캔 시엔 raycast 불필요
            initial_extra_commands=self._cmds,
        )
        self._env = make(
            initial_env_config=cfg,
            port=self.port,
            verbose=False,
            verbose_gradle=False,
            render_action=False,
            action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        )

    def _set_tp(self, x: int, y: int, z: int) -> None:
        """
        서버 재시작 없이 다음 reset()에서 tp할 좌표를 변경.
        y+3에서 시작해서 낙하 여부를 감지한다.
        """
        assert self._cmds is not None
        self._cmds[_TP_SLOT] = f"tp @p {x} {y + 3} {z}"

    def scan_position(self, x: int, y: int, z: int) -> dict:
        """
        (x, y, z) 좌표 근처가 동굴인지 스캔.

        반환:
            {
                "x": int, "y": int, "z": int,
                "actual_y": float,     # 낙하 후 실제 y
                "brightness": float,   # 이미지 평균 밝기
                "fall_score": float,   # 0~1 (낙하량 기반)
                "dark_score":  float,  # 0~1 (이미지 어두움 기반)
                "score":       float,  # 최종 동굴 점수 (0~1)
            }
        """
        if self._env is None:
            self._boot(x, y, z)
        else:
            self._set_tp(x, y, z)

        # reset: tp 포함 initial_extra_commands 재실행
        try:
            raw, _ = self._env.reset()
        except Exception as e:
            print(f"    ⚠️  reset 실패 (seed={self.seed}, pos=({x},{y},{z})): {e}")
            return self._null_result(x, y, z)

        # settle: 물리 시뮬레이션(낙하) 대기
        noop = no_op_v2()
        for _ in range(SETTLE_TICKS):
            try:
                raw, *_ = self._env.step(noop)
            except Exception:
                break

        return self._compute_score(raw, x, y, z)

    def _compute_score(self, raw, x: int, y: int, z: int) -> dict:
        actual_y = _scalar(raw, "y", default=float(y))
        img = _extract_image(raw)

        # ── 낙하 점수 ──────────────────────────────────────────────
        # tp는 y+3에서 시작. 실제 y가 낮을수록 더 많이 떨어진 것.
        expected_start = y + 3
        fall_amount = expected_start - actual_y  # 양수면 낙하
        fall_score = float(np.clip(fall_amount / 3.0, 0.0, 1.0))

        # ── 이미지 어두움 점수 ──────────────────────────────────────
        brightness = 0.0
        dark_score = 0.0
        if img is not None:
            brightness = float(img.mean())
            if brightness < FULLY_DARK_THRESHOLD:
                # 완전히 어두우면 solid rock 가능성 → 감점
                dark_score = 0.2
            elif brightness < DARK_THRESHOLD:
                # 적절히 어두움 → 지하 동굴
                dark_score = (DARK_THRESHOLD - brightness) / DARK_THRESHOLD
                dark_score = float(np.clip(dark_score, 0.0, 1.0))
            else:
                dark_score = 0.0

        # ── 최종 점수 ──────────────────────────────────────────────
        score = 0.6 * fall_score + 0.4 * dark_score

        return {
            "x": x, "y": y, "z": z,
            "actual_y": round(actual_y, 2),
            "brightness": round(brightness, 2),
            "fall_score": round(fall_score, 3),
            "dark_score":  round(dark_score, 3),
            "score":       round(score, 3),
        }

    @staticmethod
    def _null_result(x, y, z) -> dict:
        return {"x": x, "y": y, "z": z, "actual_y": float(y),
                "brightness": 0.0, "fall_score": 0.0, "dark_score": 0.0, "score": 0.0}

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# =================================================================
# 3. 헬퍼 — hollow_box_rl_0314.py 와 동일 패턴
# =================================================================

def _scalar(obs, key: str, default: float = 0.0) -> float:
    if isinstance(obs, dict):
        full = obs.get("full", obs)
    else:
        full = obs
    if isinstance(full, dict):
        return float(full.get(key, default))
    return float(getattr(full, key, default))


def _extract_image(obs) -> np.ndarray | None:
    """hollow_box_rl_0314.py의 _extract_image와 동일 로직."""
    if isinstance(obs, np.ndarray):
        img = obs
    elif isinstance(obs, dict):
        img = obs.get("pov")
        if img is None:
            img = obs.get("rgb")
        if img is not None:
            img = np.asarray(img, dtype=np.uint8)
        else:
            full = obs.get("full")
            if full is not None and isinstance(getattr(full, "image", None), bytes):
                try:
                    arr = np.frombuffer(full.image, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    return None
            else:
                return None
    elif isinstance(getattr(obs, "image", None), bytes):
        try:
            arr = np.frombuffer(obs.image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    else:
        return None

    if img is not None and img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img.astype(np.uint8) if img is not None else None


# =================================================================
# 4. 시드 DB 구축 메인 함수
# =================================================================

def build_seed_database(
    port: int,
    n_seeds: int = 200,
    positions_per_seed: int = POSITIONS_PER_SEED,
    output_path: str = "cave_db.json",
    seed_start: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """
    여러 시드를 스캔해서 동굴 근처 스폰 좌표 DB를 구축.

    Args:
        port:               CraftGround 포트
        n_seeds:            스캔할 시드 수
        positions_per_seed: 시드당 스캔할 (x, y, z) 후보 수
        output_path:        결과 JSON 경로
        seed_start:         시드 시작 번호 (기존 DB 이어서 구축 시 사용)
        verbose:            진행 상황 출력

    Returns:
        DB 항목 리스트 [{"seed": int, "x": int, "y": int, "z": int, "score": float}, ...]

    예상 소요 시간:
        시드 하나당 약 30~60초 (포지션 8개 × ~5초/reset)
        200시드 → 약 1.5~3시간
        백그라운드에서 한 번만 실행하면 이후 훈련에 계속 재사용 가능.
    """
    database: list[dict] = []
    output_path = Path(output_path)

    print(f"\n[CaveScanner] 시드 DB 구축 시작")
    print(f"  시드 범위: {seed_start} ~ {seed_start + n_seeds - 1}")
    print(f"  시드당 후보: {positions_per_seed}개 × Y레벨 {SCAN_Y_LEVELS}")
    print(f"  동굴 임계값: score >= {CAVE_SCORE_THRESHOLD}")
    print(f"  출력 경로: {output_path}\n")

    total_seeds = n_seeds
    found_total = 0

    for seed_idx, seed in enumerate(range(seed_start, seed_start + n_seeds)):
        t0 = time.time()
        scan_env = _CaveScanEnv(port=port, seed=seed)
        seed_candidates: list[dict] = []

        try:
            for pos_idx in range(positions_per_seed):
                # 랜덤 (x, z) + 여러 Y 레벨 조합 생성
                x = random.randint(-SCAN_XZ_RADIUS, SCAN_XZ_RADIUS)
                z = random.randint(-SCAN_XZ_RADIUS, SCAN_XZ_RADIUS)
                y = random.choice(SCAN_Y_LEVELS)

                result = scan_env.scan_position(x, y, z)

                if verbose:
                    bar = "█" * int(result["score"] * 10)
                    status = "★ CAVE" if result["score"] >= CAVE_SCORE_THRESHOLD else "  ----"
                    print(f"  seed={seed:4d} [{pos_idx+1}/{positions_per_seed}] "
                          f"({x:5d},{y:4d},{z:5d}) "
                          f"score={result['score']:.3f} {bar:<10} {status} "
                          f"(fall={result['fall_score']:.2f}, dark={result['dark_score']:.2f}, "
                          f"bright={result['brightness']:.1f})")

                if result["score"] >= CAVE_SCORE_THRESHOLD:
                    seed_candidates.append({
                        "seed": seed,
                        "x": result["x"],
                        "y": result["y"],
                        "z": result["z"],
                        "score": result["score"],
                    })

        except KeyboardInterrupt:
            print("\n[CaveScanner] 중단됨 — 현재까지 결과 저장 중...")
            scan_env.close()
            _save_db(database, output_path)
            return database
        finally:
            scan_env.close()

        # score 높은 순 정렬 후 시드당 최대 MAX_ENTRIES_PER_SEED개만 저장
        seed_candidates.sort(key=lambda e: -e["score"])
        best = seed_candidates[:MAX_ENTRIES_PER_SEED]
        database.extend(best)
        found_total += len(best)

        elapsed = time.time() - t0
        progress = (seed_idx + 1) / total_seeds * 100
        print(f"\n  ── seed {seed} 완료: {len(best)}개 동굴 발견 "
              f"({elapsed:.1f}s) | 진행: {progress:.1f}% | 누적: {found_total}개\n")

        # 50시드마다 중간 저장 (crash 대비)
        if (seed_idx + 1) % 50 == 0:
            _save_db(database, output_path)
            print(f"  [중간저장] {output_path} ({len(database)}개 항목)")

    _save_db(database, output_path)
    print(f"\n[CaveScanner] 완료! 총 {len(database)}개 동굴 스폰 포인트 저장 → {output_path}")
    return database


def _save_db(database: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(database, f, indent=2)


def load_db(path: str) -> list[dict]:
    with open(path) as f:
        db = json.load(f)
    print(f"[CaveSpawnWrapper] DB 로드: {len(db)}개 항목 ← {path}")
    return db


# =================================================================
# 5. 훈련용 Wrapper — DB를 이용해 에피소드마다 동굴 근처 스폰
# =================================================================

class CaveSpawnWrapper(gym.Wrapper):
    """
    기존 mining 환경을 감싸서 에피소드 시작 시 동굴 근처에 스폰시키는 Wrapper.

    사용 예:
        from cave_seed_scanner import CaveSpawnWrapper

        def make_mining_env(port, ...):
            env = make(
                initial_env_config=InitialEnvironmentConfig(
                    seed=str(seed),
                    world_type=WorldType.DEFAULT,
                    initial_extra_commands=MINING_CMDS,  # ← mutable list여야 함
                    ...
                ),
                port=port, ...
            )
            return env

        # DB 로드 후 래핑
        cave_env = CaveSpawnWrapper(
            env=make_mining_env(port=8030),
            db_path="cave_db.json",
            warmup_episodes=50,  # 처음엔 랜덤 스폰, 이후 동굴 스폰
        )
        obs, info = cave_env.reset()  # 자동으로 동굴 근처 스폰

    주의:
        make() 시 initial_extra_commands에 tp 명령 슬롯을 미리 포함시켜야 함.
        이 Wrapper는 _TP_SLOT(-1, 맨 끝)에 있는 커맨드를 패치한다.

        즉, make_mining_env의 initial_extra_commands 마지막에
        "tp @p 0 0 0" 같은 placeholder tp 명령을 넣어두면 된다.
        (실제 좌표는 reset() 직전에 이 Wrapper가 교체함)
    """

    def __init__(
        self,
        env,
        db_path: str,
        cmds_list: list[str],
        warmup_episodes: int = 0,
        score_weighted_sampling: bool = True,
        fallback_y: int = -45,
    ):
        """
        Args:
            env:                     CraftGround 환경 (make() 결과)
            db_path:                 cave_db.json 경로
            cmds_list:               make() 시 initial_extra_commands에 넘긴 **같은 리스트 객체**.
                                     이 리스트의 _TP_SLOT 항목을 패치한다.
            warmup_episodes:         처음 N 에피소드는 동굴 DB 대신 랜덤 스폰 (디버그용)
            score_weighted_sampling: True면 score에 비례해서 샘플링 (높은 점수 우대)
            fallback_y:              DB 비어있을 때 사용할 기본 Y
        """
        super().__init__(env)
        self.db = load_db(db_path)
        self._cmds = cmds_list
        self.warmup_episodes = warmup_episodes
        self.score_weighted = score_weighted_sampling
        self.fallback_y = fallback_y

        self._episode_count = 0

        # score 기반 샘플링 가중치 사전 계산
        if self.db and score_weighted_sampling:
            scores = np.array([e["score"] for e in self.db], dtype=np.float32)
            self._weights = scores / scores.sum()
        else:
            self._weights = None

        # DB 통계 출력
        if self.db:
            scores = [e["score"] for e in self.db]
            seeds = len({e["seed"] for e in self.db})
            print(f"  시드 수: {seeds}개")
            print(f"  평균 동굴 점수: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print(f"  Y 분포: {[e['y'] for e in self.db[:5]]}...")
        else:
            print("  ⚠️  DB가 비어있습니다. fallback_y={fallback_y}로 스폰합니다.")

    def _sample_spawn(self) -> tuple[int, int, int]:
        """DB에서 스폰 위치 샘플링."""
        if not self.db:
            x = random.randint(-SCAN_XZ_RADIUS, SCAN_XZ_RADIUS)
            z = random.randint(-SCAN_XZ_RADIUS, SCAN_XZ_RADIUS)
            return x, self.fallback_y, z

        if self._weights is not None:
            idx = np.random.choice(len(self.db), p=self._weights)
        else:
            idx = random.randrange(len(self.db))

        entry = self.db[idx]
        return entry["x"], entry["y"], entry["z"]

    def reset(self, **kwargs):
        """
        에피소드 리셋.
        warmup 이후엔 DB에서 동굴 근처 좌표를 골라 tp 커맨드를 패치하고 reset.
        """
        self._episode_count += 1

        if self._episode_count > self.warmup_episodes and self.db:
            x, y, z = self._sample_spawn()
            # tp 커맨드 in-place 패치 (y+3 위에서 시작해 낙하로 동굴 진입 유도)
            self._cmds[_TP_SLOT] = f"tp @p {x} {y + 3} {z}"
            spawn_type = "cave_db"
        else:
            # warmup: 기본 좌표 유지 (혹은 넘겨받은 kwargs 사용)
            spawn_type = "warmup"
            x, y, z = 0, self.fallback_y, 0

        raw, info = self.env.reset(**kwargs)
        info["spawn_type"] = spawn_type
        info["target_spawn"] = (x, y, z)
        return raw, info


# =================================================================
# 6. make_mining_env 참고 예시 (실제 mining 코드에서 복붙해서 사용)
# =================================================================

def make_mining_env_with_cave_spawn(
    port: int,
    seed: int,
    db_path: str,
    max_steps: int = 5000,
) -> CaveSpawnWrapper:
    """
    CaveSpawnWrapper가 적용된 mining 환경 생성 예시.

    포인트: initial_extra_commands를 반드시 mutable list로 만들고
            마지막 원소에 tp placeholder를 포함시켜야 함.
    """
    # ── mutable list (중요: tuple이 아닌 list여야 패치 가능) ──────────
    cmds: list[str] = [
        "gamemode survival @p",
        "difficulty peaceful",
        "gamerule doMobSpawning false",
        "gamerule fallDamage false",
        "gamerule doWeatherCycle false",
        "gamerule doDaylightCycle false",
        "gamerule doImmediateRespawn true",
        "time set 18000",
        # 광물 채굴을 위한 장비
        "give @p minecraft:iron_pickaxe 1",
        "enchant @p efficiency 5",
        "give @p minecraft:torch 64",
        # ── tp placeholder: CaveSpawnWrapper가 이 슬롯을 패치함 ──
        "tp @p 0 -45 0",   # 기본값 — wrapper가 덮어씀
    ]

    env = make(
        initial_env_config=InitialEnvironmentConfig(
            image_width=114,
            image_height=64,
            seed=str(seed),
            world_type=WorldType.DEFAULT,
            render_distance=6,
            simulation_distance=6,
            hud_hidden=False,
            request_raycast=True,
            initial_extra_commands=cmds,  # list 객체 전달
        ),
        port=port,
        verbose=False,
        verbose_gradle=False,
        render_action=False,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    return CaveSpawnWrapper(
        env=env,
        db_path=db_path,
        cmds_list=cmds,     # ← 위의 cmds와 동일한 리스트 객체!
        warmup_episodes=20,
        score_weighted_sampling=True,
    )


# =================================================================
# 7. DB 검증 (scan 완료 후 실제 스폰 확인용)
# =================================================================

def verify_database(
    port: int,
    db_path: str,
    n_samples: int = 10,
) -> None:
    """
    DB에서 랜덤 샘플링해서 실제로 동굴 근처에 스폰되는지 재확인.
    scan 후 한 번 실행해서 결과 품질 검증에 사용.
    """
    db = load_db(db_path)
    if not db:
        print("DB가 비어있습니다.")
        return

    samples = random.sample(db, min(n_samples, len(db)))
    print(f"\n[Verify] {len(samples)}개 항목 검증 중...\n")

    results = []
    for entry in samples:
        scan_env = _CaveScanEnv(port=port, seed=entry["seed"])
        try:
            result = scan_env.scan_position(entry["x"], entry["y"], entry["z"])
            ok = "OK " if result["score"] >= CAVE_SCORE_THRESHOLD else "FAIL"
            print(f"  [{ok}] seed={entry['seed']:4d} "
                  f"({entry['x']:5d},{entry['y']:4d},{entry['z']:5d}) "
                  f"original={entry['score']:.3f} → verify={result['score']:.3f}")
            results.append(result["score"])
        finally:
            scan_env.close()

    if results:
        print(f"\n검증 결과: 평균 score={np.mean(results):.3f} "
              f"| >= 임계값: {sum(1 for s in results if s >= CAVE_SCORE_THRESHOLD)}/{len(results)}")


# =================================================================
# 8. CLI 진입점
# =================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CraftGround 동굴 시드 DB 구축 (방법 1)")
    p.add_argument("--mode", choices=["scan", "verify"], default="scan",
                   help="scan: DB 구축 | verify: 기존 DB 검증")
    p.add_argument("--port",      type=int, default=8040,  help="CraftGround 포트")
    p.add_argument("--n_seeds",   type=int, default=200,   help="스캔할 시드 수")
    p.add_argument("--seed_start",type=int, default=0,     help="시드 시작 번호")
    p.add_argument("--pos_per_seed", type=int, default=POSITIONS_PER_SEED,
                   help="시드당 스캔 포지션 수")
    p.add_argument("--db",        default="cave_db.json",  help="DB 경로")
    p.add_argument("--n_verify",  type=int, default=10,    help="검증 샘플 수")
    p.add_argument("--quiet",     action="store_true",     help="진행상황 출력 억제")
    args = p.parse_args()

    random.seed(0)
    np.random.seed(0)

    if args.mode == "scan":
        build_seed_database(
            port=args.port,
            n_seeds=args.n_seeds,
            positions_per_seed=args.pos_per_seed,
            output_path=args.db,
            seed_start=args.seed_start,
            verbose=not args.quiet,
        )
    elif args.mode == "verify":
        verify_database(
            port=args.port,
            db_path=args.db,
            n_samples=args.n_verify,
        )
