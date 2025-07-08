import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack
from gym.vector import AsyncVectorEnv
import sys, os

class DualingNetworkDQN(nn.Module):
    def __init__(self, num_classes=4, num_stacks=4, init_weights=True):
        super(DualingNetworkDQN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(num_stacks, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True),
                                        )
        
        self.adval = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes + 1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x / 255)
        x = x.view(x.size(0), -1)
        x = self.adval(x)
        advantage, value = x[:, :-1], x[:, -1].unsqueeze(1)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

class AtariPR_DnDQNAgent(DQNBaseAgent):    
    def __init__(self, config):
        
        super(AtariPR_DnDQNAgent, self).__init__(config)
        ### TODO ###
        
        self.config = config
        num_envs = config.get('num_envs', 4)
        num_stacks = config.get('num_stacks', 4)
        
        def make_env(env_id, num_stacks):
            def _init():
                env = gym.make(env_id, render_mode="rgb_array")
                env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
                env = FrameStack(env, num_stacks)
                return env
            return _init
        
        # initialize env
        self.envs = AsyncVectorEnv([make_env(config["env_id"], num_stacks)  for _ in range(num_envs)])
        
        ### TODO ###
        # initialize test_env
        self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
        self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
        self.test_env = FrameStack(self.test_env, num_stacks)

        # initialize behavior network and target network
        self.behavior_net = DualingNetworkDQN(self.envs.single_action_space.n, num_stacks)
        self.behavior_net.to(self.device)
        self.target_net = DualingNetworkDQN(self.envs.single_action_space.n, num_stacks)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
        
    def decide_agent_actions(self, observations, epsilon=0.0, action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection
        if random.random() < epsilon:
            actions = action_space.sample()
        else:
            observations = torch.Tensor(np.array(observations)).to(self.device)
            observations = observations.unsqueeze(0) if observations.ndim < 4 else observations
            q_values = self.behavior_net(observations)
            actions = q_values.argmax(dim = 1).cpu().numpy()
        
        return actions if isinstance(actions, int) or actions.shape[0] != 1 else actions[0]

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
        
        ### TODO ###
        # calculate the loss and update the behavior network
        # 1. get Q(s,a) from behavior net
        # 2. get max_a Q(s',a) from target net
        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
        # 4. calculate loss between Q(s,a) and Q_target
        # 5. update behavior net

        q_value = self.behavior_net(state).gather(1, action.long())
        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)

            # if episode terminates at next_state, then q_target = reward
            q_target = reward + self.gamma * q_next * (1 - done)

        criterion = torch.nn.functional.mse_loss
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('Enduro/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    # Overwrite
    # 考慮到每個 environment 的 episode 結束時間不同, 這裡需要全部重寫
    # 為了畫 tensorboard, 不允許每個 episode 結束就開始下一個 episode.
    def train(self):
        episode_idx = 0
        while self.total_time_step <= self.training_steps:
            episode_idx += 1
            
            observations, infos = self.envs.reset()
            episode_rewards = np.zeros(self.envs.num_envs)
            episode_lens    = np.zeros(self.envs.num_envs)
            deads           = np.zeros(self.envs.num_envs, dtype=bool)
            
            while True:
                if self.total_time_step < self.warmup_steps:
                    actions = self.decide_agent_actions(observations, 1.0, self.envs.action_space)
                else:
                    actions = self.decide_agent_actions(observations, self.epsilon, self.envs.action_space)
                    self.epsilon_decay()
            
                next_observations, rewards, terminates, truncates, infos = self.envs.step(actions)
                
                if self.total_time_step >= self.warmup_steps:
                    self.update()
                
                # Check liveness
                if np.any(terminates) or np.any(truncates):
                    logic = np.logical_or(np.array(terminates), np.array(truncates))
                    deads = np.logical_or(deads, logic)
               
                undeads = np.where(deads == False)[0]
                for idx in undeads:
                    # Ignore dead env.
                    self.replay_buffer.append(observations[idx], [actions[idx]], [rewards[idx]], next_observations[idx], [int(terminates[idx])])   
                    episode_rewards[idx] += rewards[idx]
                    episode_lens[idx] += 1
                    
                if np.all(deads):
                    self.writer.add_scalar('Train/Episode Reward (avg.)', int(np.mean(episode_rewards)), self.total_time_step)
                    self.writer.add_scalar('Train/Episode Len (avg.)', int(np.mean(episode_lens)), self.total_time_step)
                    print(f"[{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward (avg.): {int(np.mean(episode_rewards))}  episode len (avg.): {int(np.mean(episode_lens))}  epsilon: {self.epsilon}")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    break
                    
                observations = next_observations
                self.total_time_step += len(undeads)
                
            if episode_idx % self.eval_interval == 0:
                # save model checkpoint
                avg_score = self.evaluate()
                self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
                self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
        
if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 4e8,
		"gamma": 0.99,
		"batch_size": 64,
		"eps_min": 0.1,
		"warmup_steps": 80000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 200000,
		"logdir": 'log/Enduro/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 500,
        "eval_episode": 5,
		"env_id": 'ALE/Enduro-v5',
        "num_envs": 4,
        "num_stacks": 6,
	}
    agent = AtariPR_DnDQNAgent(config)
    args = agent.args_parse()
    
    if args.mode == "train":
        agent.train()
    else:
        if args.load_path != "":
            agent.record_evaluate(args.load_path, seed=args.seed)
        else:
            print("Eval mode: Missing load_path !")