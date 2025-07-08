from td3_agent_CarRacing import CarRacingTD3Agent
from td3_reward_function_design import CarRacingTD3Agent as RewardShapingAgent

import os

directory = "log/CarRacing/reward_test2_off_reward"

# 收集符合條件的檔案
filtered_files = []

for file_name in os.listdir(directory):
    if file_name.startswith("model_") and file_name.endswith(".pth"):
        try:
            _, steps, reward = file_name.split("_")
            reward = int(reward.split(".")[0])
            if reward > 800:
                filtered_files.append(file_name)
        except ValueError:
            continue

if __name__ == "__main__":
    config = {
        "gpu": True,
        "training_steps": 1e8,  # Unused
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32, 
        "warmup_steps": 1000,
        "total_episode": 3000,
        "lra": 4.5e-5,
        "lrc": 4.5e-5,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/CarRacing/evaluation/',
        "update_freq": 2,
		"eval_interval": 1,
		"eval_episode": 1,
    }
    # agent = CarRacingTD3Agent(config)
    agent = RewardShapingAgent(config)
    """
    Record video:
      - set self.env = gym.make('CarRacing-v2', render_mode="human") in environment_wrapper
      - render_mode -> rgb_array
      - self.env = gym.make('CarRacing-v2', render_mode="rgb_array")
      - uncomment something in base_agent.py / record_evaluate
    """

    # print(len(filtered_files))
    # for f in filtered_files:
    #   print(f"test_on_{f}")
    #   agent.record_evaluate(load_path=f"./log/CarRacing/reward_test2_off_reward/{f}")
  
    agent.record_evaluate(load_path=f"./log/CarRacing/reward_test2_off_reward/model_340982_922.pth")
    # agent.record_evaluate(load_path="./log/CarRacing/reward_test1_off_reward/model_207750_929.pth")
    # agent.record_evaluate(load_path="./log/CarRacing/action_noise_injection/model_184375_919.pth")
    # agent.record_evaluate(load_path="./log/CarRacing/td3_test/model_169597_910.pth")
    