import argparse, json, requests, os
import torch
import torch.nn.functional as F
import numpy as np

from racecar_gym.env import RaceEnv
from models.RacecarGymModel import (RacecarCNN, RacecarResnet)
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from gymnasium.wrappers.frame_stack import FrameStack
from utils import *

os.environ["RAY_memory_monitor_refresh_ms"] = "0"

def connect(agent, url: str = 'http://localhost:5000'):
    stackFrames = StackFrames(FRAME_STACK_NUM)
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        
        obs = torch.from_numpy(np.array(obs).astype(np.float32))
        obs = torch.mean(obs, dim=0, keepdim=True) # grayscale
        obs = F.interpolate(obs.unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False)
        obs = obs.squeeze(0).permute(1, 2, 0)
        obs = stackFrames(obs).squeeze(3).permute((1,2,0))
        obs = obs / 255.0 # normalization
                
        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.compute_single_action(obs)  # Replace with actual action
        action_to_take = np.array(DiscreteActions[action_to_take]) # Discrete mode
        
        # Send an action and receive new observation, reward, and done status
        for _ in range(FRAME_SKIP_NUM):
            response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


# python client.py --path=/home/louis/Workspace/NYCU/Classes/113_1/Reinforcement_learning/RL_FINAL_PROJECT_313551153_施儒宏/src/log/11-19-19-41-32_MARWIL_STK8_skip1/checkpoint_0500 --url http://localhost:5001
# python client.py --path=/home/louis/Workspace/NYCU/Classes/113_1/Reinforcement_learning/RL_FINAL_PROJECT_313551153_施儒宏/src/log/11-19-19-41-32_MARWIL_STK8_skip1/checkpoint_0500 --url https://competition3.cgi.lab.nycu.edu.tw/???
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    parser.add_argument('--path', type=str, required=True, help="Path to checkpoint dir")
    args = parser.parse_args()
        
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_model("RacecarCNN", RacecarCNN)
    ModelCatalog.register_custom_model("RacecarResnet", RacecarResnet)
    
    from ray.rllib.algorithms import (PPO, PPOConfig, AlgorithmConfig, Algorithm)
    config = (
        PPOConfig()
        .environment("DummyEnvDiscrete") # DummyEnvDiscrete, DummyEnvContinuous
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .training( 
            # model= {
            #     "custom_model": "RacecarCNN",
            #     "custom_model_config":{
            #         "framestack_num" : FRAME_STACK_NUM
            #     },
            # },
        )
    )
    
    # 加载 checkpoint
    checkpoint_path = os.path.abspath(args.path)
    agent = PPO(config=config)
    agent.restore(checkpoint_path)
    
    connect(agent, url=args.url)
