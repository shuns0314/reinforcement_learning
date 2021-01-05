import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces
from gym.spaces.box import Box
from baselines.common import atari_wrappers

from baselines.common.vec_env.subproc_vec_env import SubprocVec

import cv2

cv2.ocl.setUseOpenCL(False)


class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space - spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


class WrapPytorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPytorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env = atari_wrappers.NoopResetEnv(env, noop_max=30)
        env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)
        env = atari_wrappers.EpisodicLifeEnv(env)
        env = WrapFrame(env)
        env = WrapPytorch(env)
        return env

    return _thunk
