import ray, os, sys, argparse, json
import ray.rllib
import ray.rllib.algorithms
from ray.rllib.algorithms import PPOConfig, BCConfig, PPO, DQNConfig
from ray.rllib.algorithms.marwil import MARWIL, MARWILConfig
import gc

from utils import *      
from datetime import datetime

PPO_TIME_S = datetime.now().strftime("%m-%d-%H-%M-%S")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_episodes', type=int, default=300, help="number of total training episodes")
    parser.add_argument('--total_steps', type=int, default=2e6, help="number of total training episodes")
    parser.add_argument('--freq_save', type=int, default=10, help="number of total training episodes")
    parser.add_argument('--checkpoint_root', type=str, default=f"./log/{PPO_TIME_S}_STK{FRAME_STACK_NUM}_skip{FRAME_SKIP_NUM}_{SCENARIO}")
    parser.add_argument('--checkpoint_from', type=str, default=f"")
    # parser.add_argument('--checkpoint_from', type=str, default=f"./log/12-08-13-26-39_STK8_skip1_austria_competition_collisionStop/checkpoint_0001813146")
    return parser.parse_args()

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_memory_usage_threshold"] = "0.95"

# os.environ["RAY_local_gc_interval_s"]                   = "120"
# os.environ["RAY_global_gc_min_interval_s"]              = "120"
# os.environ["RAY_raylet_check_gc_period_milliseconds"]   = "100"
# os.environ["RAY_high_plasma_storage_usage"]             = "0.7"

if __name__ == "__main__":

    args = get_args()
    
    os.makedirs(args.checkpoint_root, exist_ok=True)
    with open(f"{args.checkpoint_root}/ppo_parser.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        
    # ray.init(local_mode=True, dashboard_host="0.0.0.0")
    
    ppo_config = (
        PPOConfig()
        .environment("RacecarEnvDiscrete")
        .framework("torch")
        .env_runners(
            num_env_runners=4, 
            batch_mode="complete_episodes", 
        )
        .resources(num_gpus=1)
        .training(
            # train_batch_size   = 4000,
            # entropy_coeff      = 0.01, 
            # gamma              = 0.95, 
            # lambda_            = 0.99, 
            lr                 = 3e-4,
            # lr_schedule        = [[0, 1e-6], [100000, 1e-5]],
            train_batch_size   = 8192,
            num_epochs         = 10,
            # num_epochs         = 30,
            clip_param         = 0.2,  
            # use_kl_loss        = False,
            # vf_clip_param        = 4
            minibatch_size     = 1024,
            # use_critic = False,
            # use_gae    = False,
            # model={
            #     "dim": 84, 
            #     "use_lstm": True,
            # }

        )
    )
    
    ppo_algo = ppo_config.build()
    if args.checkpoint_from != "":
        ppo_algo.restore(os.path.abspath(args.checkpoint_from))
    
    total_steps = 0
    
    # while total_steps < args.total_steps:
    for episode in range(1, args.total_episodes+1):
        result = ppo_algo.train()
        total_steps += result['env_runners']['episodes_timesteps_total']
        print("============================================")
        print(f"Episode [{episode:3d}]")
        print(f"episode_reward_mean: {result['env_runners']['episode_reward_mean']}")
        print(f"episode_len_mean   : {result['env_runners']['episode_len_mean']}")
        print("============================================")
        if (episode % args.freq_save) == 0:
            save_path = f"{args.checkpoint_root}/checkpoint_{episode:010d}"
            result = ppo_algo.save(save_path)
            print(f"Checkpoint saved at: {save_path}")
            gc.collect()
        (print("\n"), sys.stdout.flush(), sys.stderr.flush())
        
        
    print("Training complete.")

    ray.shutdown()
        
    
