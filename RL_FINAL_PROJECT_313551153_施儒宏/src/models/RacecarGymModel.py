from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
import torch.nn as nn
import torch
import torch.nn.functional as F

class RacecarCNN(TorchRLModule, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs):
        nn.Module.__init__(self)
        frame_num = model_config['custom_model_config']['framestack_num']
        # Define custom network layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_num, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), 
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
        )
        _ = torch.zeros(1, frame_num, obs_space.shape[1], obs_space.shape[1]) 
        conv_out_size = self._get_conv_output(_)
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, num_outputs)

        self.value_branch = nn.Linear(256, 1)
    
    def _get_conv_output(self, x):
        # Pass the dummy input through the conv layers to get the output size
        x = self.cnn(x)
        return int(torch.flatten(x, 1).size(1))
     
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].squeeze(-1)
        # Assume observation is an image with shape [3, 128, 128]
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        self._value = self.value_branch(x)
        output = self.fc2(x)
        return output, state

    def value_function(self):
        return self._value.squeeze(-1)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RacecarResnet(TorchRLModule, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs):
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        # self.layer2 = self._make_layer(16, 32, 2, stride=1)

        _ = torch.zeros(1, 1, obs_space.shape[0], obs_space.shape[0]) 
        conv_out_size = self._get_conv_output(_)

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        self.value_branch = nn.Linear(128, 1)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _get_conv_output(self, x):
        # Pass the dummy input through the conv layers to get the output size
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        # x = self.layer2(x)
        return int(torch.flatten(x, 1).size(1))
        
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0,3,1,2)
        # Assume observation is an image with shape [3, 128, 128]
        x = x.float() / 255.0
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        # x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        self._value = self.value_branch(x)
        output = self.fc2(x)
        return output, state

    def value_function(self):
        return self._value.squeeze(-1)