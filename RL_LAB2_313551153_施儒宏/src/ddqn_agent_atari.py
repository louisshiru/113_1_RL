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

class AtariDDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super(AtariDDQNAgent, self).__init__(config)
        ### TODO ###
        # initialize env

        self.env = gym.make(config["env_id"], render_mode="rgb_array")
        self.env = atari_preprocessing.AtariPreprocessing(self.env, screen_size=84, grayscale_obs=True, frame_skip=1)
        self.env = FrameStack(self.env, 4)
  
        ### TODO ###
        # initialize test_env
        self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
        self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
        self.test_env = FrameStack(self.test_env, 4)
  
        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(self.env.action_space.n)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(self.env.action_space.n)
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
            next_action = self.behavior_net(next_state).argmax(1).unsqueeze(1)

            q_next = self.target_net(next_state).gather(1, next_action)

            # # if episode terminates at next_state, then q_target = reward
            q_target = reward + self.gamma * q_next * (1 - done)
        
        criterion = torch.nn.functional.mse_loss
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('DDQN/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"logdir": 'log/DDQN/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
		"env_id": 'ALE/MsPacman-v5',
	}
    agent = AtariDDQNAgent(config)
    args = agent.args_parse()
    
    if args.mode == "train":
        agent.train()
    else:
        if args.load_path != "":
            agent.record_evaluate(args.load_path, seed=args.seed)
        else:
            print("Eval mode: Missing load_path !")
