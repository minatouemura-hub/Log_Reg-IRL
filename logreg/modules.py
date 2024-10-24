import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def negative_log_likelihood(outputs, labels, model, lambda_reg):
    Nb = (labels == 0).sum().item()
    Ne = (labels == 1).sum().item()
    if Nb == 0 or Ne == 0:
        return None
    baseline_ll = -(1 / Nb) * torch.sum(torch.log(torch.sigmoid(-outputs[labels == 0])))
    expert_ll = -(1 / Ne) * torch.sum(torch.log(torch.sigmoid(outputs[labels == 1])))

    # L2 正則化項の計算
    regularization = 0
    for wx in model.parameters():
        regularization += torch.sum(wx**2)  # ここ若干怪しい
    regularization = (lambda_reg / 2) * regularization
    return baseline_ll + expert_ll + regularization


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        data = [state, action, reward, next_state, done]
        self.buffer.append(data)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.float32)

        return state, action, reward, next_state, done

    def reset(self):
        self.buffer.clear()


class Density_Ratio_Net(nn.Module):
    def __init__(self, state_space, device):
        super().__init__()
        self.state_space = state_space
        self.device = device
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1
        ).to(self.device)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        ).to(self.device)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        ).to(self.device)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3456, 512).to(self.device)
        self.fc2 = nn.Linear(512, 1).to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)  # フラット化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Reward_Net(nn.Module):
    def __init__(self, state_space, device):
        super().__init__()
        self.state_space = state_space
        self.device = device
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1
        ).to(device)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        ).to(device)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        ).to(device)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3456, 512).to(self.device)

        self.fc2 = nn.Linear(512, 1).to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)  # フラット化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class State_Value_Net(nn.Module):
    def __init__(self, state_space, device):
        super().__init__()
        self.state_space = state_space
        self.device = device
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1
        ).to(self.device)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        ).to(self.device)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        ).to(self.device)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3456, 512).to(self.device)
        self.fc2 = nn.Linear(512, 1).to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)  # フラット化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Preference_Net(nn.Module):
    def __init__(self, action_num, state_space, device):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 32).to(device)
        self.fc2 = nn.Linear(32, action_num).to(device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_prob = torch.softmax(self.fc2(x), dim=1)
        return action_prob
