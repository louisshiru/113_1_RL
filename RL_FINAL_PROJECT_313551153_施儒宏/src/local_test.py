import argparse, os, sys
import torch
import torch.nn.functional as F
import numpy as np
import math

from racecar_gym.env import RaceEnv
from utils import *

os.environ["RAY_memory_monitor_refresh_ms"] = "0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument('--scene', type=str, required=True, help="Eval scene") # austria_competition, circle_cw_competition_collisionStop
    parser.add_argument('--episodes', default=1, type=int, required=False, help="Eval episodes")
    parser.add_argument('--recording', default=True, type=bool, required=False, help="Record Video or not")
    args = parser.parse_args()
    
    from ray.rllib.algorithms import (PPO, PPOConfig, AlgorithmConfig, Algorithm)
    config = (
        PPOConfig()
        .environment("DummyEnvContinuous") # DummyEnvDiscrete, DummyEnvContinuous
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .training( 
            # model={
            #     "dim": 84, 
            #     "use_lstm": True,
            # }
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

    import matplotlib.pyplot as plt
    import cv2
    
    fps = 60
    noise_scales = [[0.001, 0.01]]
    # noise_scales = [[0., 0.], [0.001, 0.01], [0.01, 0.1], [0.05, 0.5], [0.1, 1], ] # 0x, origin, 10x, 50x, 100x
    
    for scales in noise_scales:
        # austria_competition
        # circle_cw_competition_collisionStop     
        env = RaceEnv(
            scenario=args.scene,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False,
            motor_scale=scales[0],
            steering_scale=scales[1]
        )
        
        total_result = 0.
        for episode in range(args.episodes):
            lap = 1
            last_progress = 0
            stackFrames = StackFrames(FRAME_STACK_NUM)
            obs_, state = env.reset()
            done = False
            # print(f"Recording episode {episode + 1}")

            output_file = f"./results/{scales[0]}_{scales[1]}_observations_episode_{episode + 1:03d}.mp4"
            video_writer = None  
            while not done:

                obs = torch.mean(torch.from_numpy(obs_), dim=0, keepdim=True) # grayscale
                obs = F.interpolate(obs.unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False)
                obs = obs.squeeze(0).permute(1, 2, 0)
                obs = stackFrames(obs).squeeze(3).permute((1,2,0))
                obs = obs / 255.0 # normalization

                action = agent.compute_single_action(obs)  
                # action = np.array(DiscreteActions[action]) # Discrete mode

                obs_ = obs_.transpose(1,2,0)
                if len(obs_.shape) == 3 and obs_.shape[-1] == 3:  
                    frame = (obs_).astype(np.uint8)  
                else: 
                    frame = (obs_.squeeze()).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  

                if video_writer is None and args.recording:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(
                        output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
                    )

                for _ in range(FRAME_SKIP_NUM):
                    obs_, reward, done, truncated, state = env.step(action)

                print(f"[{lap}: {state['progress']:.5f} {state['checkpoint']}] {action} ")    

                if args.recording:
                    video_writer.write(frame)
                    
                if math.fabs(state['progress'] - last_progress) > 0.8:
                    lap += 1
                    
                last_progress = state['progress']
                    

            result = state['progress'] + lap - 1
            total_result += result

            print(f"Result = {result:.5f}")
            if video_writer and args.recording:
                video_writer.release()
                print(f" Video for episode {episode + 1} saved to {output_file}")
            
            sys.stderr.flush()
            sys.stdout.flush()
        
        del env

        print(f"Avg result: {(total_result / args.episodes):.5f}, noise_scale={scales}")
