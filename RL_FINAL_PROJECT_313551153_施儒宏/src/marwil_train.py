import os, sys, gc

import ray, ray.data
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.marwil import MARWIL, MARWILConfig
from datetime import datetime
import pyarrow as pa

from utils import *

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_memory_usage_threshold"] = "0.95"

os.environ["RAY_local_gc_interval_s"]                   = "120"
os.environ["RAY_global_gc_min_interval_s"]              = "120"
os.environ["RAY_raylet_check_gc_period_milliseconds"]   = "100"
os.environ["RAY_high_plasma_storage_usage"]             = "0.7"

PPO_TIME_S = datetime.now().strftime("%m-%d-%H-%M-%S")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_episodes', type=int, default=500, help="number of total training episodes")
    parser.add_argument('--freq_save', type=int, default=10, help="number of total training episodes")
    parser.add_argument('--checkpoint_root', type=str, default=f"./log/{PPO_TIME_S}_MARWIL_STK{FRAME_STACK_NUM}_skip{FRAME_SKIP_NUM}")
    
    return parser.parse_args()

def convert_to_sample_batch(batch):
    return SampleBatch({SampleBatch.OBS: batch['obs'].transpose(0, 2, 3, 1) / 255.0, SampleBatch.ACTIONS: batch['action'], "advantages": batch['reward']})

def preprocess_row(row):
    if FRAME_STACK_NUM == 1:
        row['obs'] = np.array(row['obs']).transpose(2,0,1)
    else:
        row['obs'] = np.array(row['obs']).squeeze(-1)
        
    row['action'] = DiscreteActionsMap[tuple(row['action'])] # Discrete actions
    gc.collect()
    return row

if __name__ == "__main__":
    
    args = get_args()
    
    # 1. Read data
    # Note: 單一檔案太大會 OOM, 要分割檔案
    ds = ray.data.read_json(os.path.abspath(f"../imitation_data/STK{FRAME_STACK_NUM}_skip{FRAME_SKIP_NUM}"))
    ds = ds.map(preprocess_row)
    ds = ds.materialize()

    config = (
        MARWILConfig()
        .environment("RacecarEnvDiscrete")
        .resources(num_gpus=1, num_cpus_per_worker=1, num_learner_workers=1)
        .env_runners(num_env_runners=1, num_rollout_workers=1)
        .framework("torch")
        .training(
            beta = 1,
            model = {
                
            }
        )
    )   
    marwil_algo = config.build()
    policy = marwil_algo.get_policy()
    
    batch_size=512
    iter_batches = ds.iter_torch_batches(batch_size=batch_size, collate_fn=convert_to_sample_batch, local_shuffle_buffer_size=batch_size)
    for episode in range(1, args.total_episodes + 1):
        for batch in iter_batches:
            results = policy.learn_on_batch(batch)
            total_loss = results['learner_stats']['total_loss']
        print(f"Training avg loss ", total_loss)        
        if (episode % args.freq_save) == 0:
            save_path = f"{args.checkpoint_root}/checkpoint_{episode:04d}"
            result = marwil_algo.save(save_path)
            gc.collect()
            print(f"Checkpoint saved at: {save_path}")
        sys.stdout.flush()
        sys.stderr.flush()
             
    print("Training complete.")

    ray.shutdown()