"""
전문가 데이터 수집.

사용법:
    python collect_demos.py [에피소드 수]
    ex) python collect_demos.py 50
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from building_hrl import make_env, TARGET_POSITIONS
from config import PORT, DEMO_DIR, DEMO_PATH, COLLECT_N_EPISODES
from env import VectorBuildingEnv
from expert import ScriptedExpert


def _make_vector_env(seed: str = "42") -> VectorBuildingEnv:
    """craftground 기본 env → VectorBuildingEnv 래핑"""
    base = make_env(port=PORT, seed=seed, hud_hidden=False)()
    # make_env는 Monitor(HierarchicalBuildingEnv(craftground_env)) 반환
    # Monitor를 걷어내고 craftground_env만 꺼내서 VectorBuildingEnv로 재래핑
    inner = base
    while hasattr(inner, "env"):
        inner = inner.env
        if hasattr(inner, "observation_space") and not hasattr(inner, "env"):
            break
    # inner = 최하위 craftground env
    return VectorBuildingEnv(inner)


def collect_episode(env: VectorBuildingEnv, expert: ScriptedExpert):
    obs, _ = env.reset()
    expert.reset()

    observations, actions = [], []
    done = False
    while not done:
        action = expert.get_action()
        observations.append(obs.copy())
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return observations, actions, info.get("placed", 0)


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else COLLECT_N_EPISODES
    os.makedirs(DEMO_DIR, exist_ok=True)
    print(f"[collect] {n_episodes}개 에피소드 수집 → {DEMO_PATH}")

    env    = _make_vector_env()
    expert = ScriptedExpert(env)

    all_obs, all_actions = [], []
    success = 0

    for ep in range(n_episodes):
        t0 = time.time()
        obs_list, act_list, placed = collect_episode(env, expert)
        all_obs.extend(obs_list)
        all_actions.extend(act_list)

        ok = placed >= len(TARGET_POSITIONS)
        success += int(ok)
        tag = "SUCCESS" if ok else f"placed={placed}"
        print(f"  EP {ep+1:3d}/{n_episodes} | steps={len(act_list):4d} | {tag} | {time.time()-t0:.1f}s")

    env.close()

    np.savez_compressed(
        DEMO_PATH,
        observations = np.array(all_obs,     dtype=np.float32),  # (N, 45)
        actions      = np.array(all_actions, dtype=np.int64),    # (N,)
    )
    print(f"\n[collect] 총 {len(all_actions):,} 스텝 저장 | 성공: {success}/{n_episodes}")


if __name__ == "__main__":
    main()
