
import gymnasium as gym
import numpy as np
import torch
import argparse
from collections import deque

from gymnasium.wrappers.frame_stack import LazyFrames

# LEARNING_FROM = "Scratch" # "BehaviorCloning" or "MARWIL" or "Scratch"
FRAME_STACK_NUM = 2
FRAME_SKIP_NUM = 2
SCENARIO = "austria_competition"
# austria_competition_collisionStop
# austria_competition
# circle_cw_competition_collisionStop

from racecar_gym.env import RaceEnv
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from gymnasium.wrappers.frame_stack import FrameStack
from ray.rllib.policy.sample_batch import SampleBatch
    
class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84, FRAME_STACK_NUM), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(DiscreteActions))
        
class DummyEnvContinuous(gym.Env):
    def __init__(self):
        super(DummyEnvContinuous, self).__init__()
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, FRAME_STACK_NUM), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84, FRAME_STACK_NUM), dtype=np.float32)
        self.action_space = gym.spaces.box.Box(low=-1, high= 1., shape=(2,), dtype=np.float32)

def env_creator_discrete(env_config):    
    env = RaceEnv(
        scenario=SCENARIO,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=True if SCENARIO == "austria_competition" else False,
        motor_scale = 0,
        steering_scale = 0,
    )
    env = PreprocessEnv(env)
    env = DiscreteActionWrapper(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    env = FrameSkip(env, frame_skip=FRAME_SKIP_NUM)
    env = FrameStack(env, FRAME_STACK_NUM)
    env = PostprocessEnv(env)
    return env

def env_creator_continuous(env_config):
    env = RaceEnv(
        scenario=SCENARIO,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=True if SCENARIO == "austria_competition" else False,
    )
    env = PreprocessEnv(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    env = FrameSkip(env, frame_skip=FRAME_SKIP_NUM)
    env = FrameStack(env, FRAME_STACK_NUM)
    env = PostprocessEnv(env)
    return env
    
def env_creator_dummy_env(env_config):
    return DummyEnv()

def env_creator_dummy_env_contiguous(env_config):
    return DummyEnvContinuous()

# Old discrete actions
# DiscreteActions = [ [0, 0], [1, 0], [1, 1], [1, -1], [0, 1], [0, -1]  ] 
# 煞車, 煞車左, 煞車右, 直走, 直左, 直右, 平左, 平右
DiscreteActions = [ [-1, 0], [-1, 1], [-1, -1], [1, 0], [1, 1], [1, -1],  [0, 1], [0, -1] ] 
DiscreteActionsMap = {tuple(action): index for index, action in enumerate(DiscreteActions)}

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(len(DiscreteActions))
    
    def action(self, action):
        return DiscreteActions[action]

class FrameSkip(gym.Wrapper):
    def __init__(self, env, frame_skip=1):
        super(FrameSkip, self).__init__(env)
        self.frame_skip = frame_skip
        
    def step(self, action):
        """Take a step in the environment, skipping frames."""
        total_reward = 0.0
        terminated, truncated = False, False
        info = {}

        for _ in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info

class PreprocessEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessEnv, self).__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[1], obs_shape[2], obs_shape[0])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape)

    def observation(self, obs):
        return np.transpose(obs, (1, 2, 0))  
    
class PostprocessEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(PostprocessEnv, self).__init__(env)
        self.count = 0
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=.0, high=1., shape=(obs_shape[1], obs_shape[2], obs_shape[0]), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[1], obs_shape[2], obs_shape[0]), dtype=np.uint8)
                
    def preprocess_obs(self, obs):
        if isinstance(obs, LazyFrames):
            obs = np.array(obs._frames).squeeze(-1).transpose((1,2,0))
        return obs / 255.0
    
    def observation(self, obs):
        return self.preprocess_obs(obs)
    
    # with Expert model support
    # def step(self, action):
    #     observation, reward, terminated, truncated, info = self.env.step(action)
    #     observation = self.preprocess_obs(observation)
        
    #     addition_reward = 0

    #     if EXPERT_MODEL is not None:
    #         expert_action_dist = np.array(DiscreteActions[EXPERT_MODEL.compute_single_action(observation)], dtype=np.float32)
    #         addition_reward -= 0.1 * np.sqrt(np.sum((expert_action_dist - action) ** 2) )
    #         # addition_reward -= 0.05 * int( ( np.sqrt(np.sum((expert_action_dist - action) ** 2) )) > 1 )
                        
    #     reward += addition_reward
                
    #     return observation, reward, terminated, truncated, info

class StackFrames:
    def __init__(self, stack_num):
        self.frames = deque(maxlen=stack_num)
        
    def __call__(self, observation=None):
        self.frames.append(observation)
        while (len(self.frames) < self.frames.maxlen):
            self.frames.append(observation)
        return torch.from_numpy(np.array(list(self.frames)))

class StackFramesNP:
    def __init__(self, stack_num):
        self.stack_num = stack_num
        self.frames = deque(maxlen=stack_num)

    def __call__(self, observation=None):
        self.frames.append(observation)
        while (len(self.frames) < self.frames.maxlen):
            self.frames.append(observation)
        return list(self.frames)
    
    
from ray.tune.registry import register_env
from ray.rllib.algorithms.marwil import MARWILConfig
import os

register_env("RacecarEnvDiscrete", env_creator_discrete)
register_env("RacecarEnvContinuous", env_creator_continuous)
register_env("DummyEnvDiscrete", env_creator_dummy_env)
register_env("DummyEnvContinuous", env_creator_dummy_env_contiguous)

# marwil_config = (
#     MARWILConfig()
#     .environment(env="DummyEnvDiscrete") 
#     .framework("torch")
# )  
# marwil_algo = marwil_config.build()
# marwil_algo.restore(os.path.abspath('./log/ExpertModel/checkpoint_0500'))
EXPERT_MODEL = None
    