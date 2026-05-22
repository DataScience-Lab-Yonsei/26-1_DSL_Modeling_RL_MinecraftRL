# lidar_wrapper.py (직접 생성)
import numpy as np
import gymnasium as gym

class LidarWrapper(gym.Wrapper):
    def __init__(self, env, num_rays=16, max_distance=10):
        super().__init__(env)
        self.num_rays = num_rays
        self.max_distance = max_distance

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # info['full'] 혹은 obs['full']에서 라이다 데이터를 추출하거나
        # 환경에서 제공하는 raycast API를 호출하는 로직이 들어가는 곳입니다.
        # 여기서는 라이브러리 내부 구조를 모르므로, 기본 obs에 lidar를 추가하는 뼈대를 제공합니다.
        obs["lidar"] = self._get_lidar_data(info)
        return obs, reward, terminated, truncated, info

    def _get_lidar_data(self, info):
        # 실제 CraftGround의 info['full']에서 데이터를 가져오는 임시 로직
        # 실제 구현은 깃허브의 원본 소스를 참조해야 합니다.
        return np.random.rand(self.num_rays) * self.max_distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["lidar"] = self._get_lidar_data(info)
        return obs, info