import pygame
import numpy as np
import pandas as pd
import os, json
from racecar_gym.env import RaceEnv
import gymnasium as gym

from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.frame_stack import LazyFrames

FRAME_STACK_NUM = 1

def env_creator():
    env = RaceEnv(
        scenario="austria_competition_collisionStop",
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False
    )
    env = PreprocessEnv(env)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, FRAME_STACK_NUM) # For dynamic.
    env = PostprocessEnv(env) # Fix framestack
    return env

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        # Keep dynamic, Straight backward, Left turn, Striaght forward, Right turn 
        self.actions = [ [-1., 0.], [1., -1.], [1., 0.], [1., 1.] ] 
    
    def action(self, action):
        return self.actions[action]
    
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
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[1], obs_shape[1], obs_shape[0]))
        
    def observation(self, obs):
        if isinstance(obs, LazyFrames):
            obs = np.array(obs._frames).squeeze(0)
        return obs 

def convert_state_to_json_compatible(state):
    state_json_compatible = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            state_json_compatible[key] = value.tolist()
        else:
            state_json_compatible[key] = value
    return state_json_compatible

ACTION_DICT = {
    "UP": [1, 0],       # 向前加速
    "DOWN": [-1, 0],     # 剎車
    "LEFT": [0, -1],    # 向左轉
    "RIGHT": [0, 1],    # 向右轉
    "STOP": [0, 0]      # 保持當前速度
}

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Racecar Manual Control")

clock = pygame.time.Clock()
fps = 60

env = env_creator()
obs, *other = env.reset()
done = False

data = []
file_name = "racecar_data.jsonl"

if os.path.exists(file_name):
    os.remove(file_name)

timestamp = 0

with open(file_name, 'a') as f:
    while not done:
        timestamp += 1

        img = env.render()
        surf = pygame.surfarray.make_surface(img.transpose((1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        action = np.array([0, 0], dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action += np.array(ACTION_DICT["UP"], dtype=np.float32)
        if keys[pygame.K_DOWN]:
            action += np.array(ACTION_DICT["DOWN"], dtype=np.float32)
        if keys[pygame.K_LEFT]:
            action += np.array(ACTION_DICT["LEFT"], dtype=np.float32)
        if keys[pygame.K_RIGHT]:
            action += np.array(ACTION_DICT["RIGHT"], dtype=np.float32)

        obs, reward, done, truncated, state = env.step(action)

        state = convert_state_to_json_compatible(state)

        data_entry = {
            "timestamp": timestamp,
            "obs": obs.tolist(),         
            "action": action.tolist(),   
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "state": state               
        }

        f.write(json.dumps(data_entry) + '\n')

        clock.tick(fps)

        if state['lap'] >= 2:
            break

pygame.quit()
print("數據已保存到", file_name)
