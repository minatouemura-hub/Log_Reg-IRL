import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# この行は、Pythonのインポートシステムがモジュールを検索するディレクトリのリストに、現在のスクリプトの一つ上のディレクトリを追加するためのものです。具体的には、以下のような処理を行っています。
from logreg.modules import (  # noqa: E402
    Density_Ratio_Net,
    Preference_Net,
    ReplayBuffer,
    Reward_Net,
    State_Value_Net,
)

"""
Irl_Netについて
input：state,next_stateが入力
output：π(next_state|state)という状態遷移確率が出力

IRLによるベルマン方程式を用いることで出力が実現される。
"""


class Irl_Net(nn.Module):
    def __init__(self, device, state_space, action_num, lambda_reg=0.1, gamma=0.1):
        super().__init__()
        self.action_num = action_num
        self.state_space = state_space
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.device = device
        self.dens_net = Density_Ratio_Net(device=self.device, state_space=self.state_space).to(
            device=self.device
        )
        self.reward_net = Reward_Net(device=self.device, state_space=self.state_space).to(
            self.device
        )
        self.state_value_net = State_Value_Net(device=self.device, state_space=self.state_space).to(
            self.device
        )

    def forward(self, state, next_state):
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        dens = self.dens_net(state)
        q_hat = self.reward_net(state)
        v_x = self.state_value_net(state)
        v_y = self.state_value_net(next_state)

        return dens, q_hat, v_x, v_y


class Q_net(nn.Module):
    def __init__(self, action_num, device):
        super().__init__()
        self.action_num = action_num
        self.device = device

        self.pref_net = Preference_Net(action_num=self.action_num).to(self.device)
        self.state_value_net = State_Value_Net().to(self.device)

    def forward(self, state):
        state_value = self.state_value_net(state)

        state_action_denc = self.pref_net(state)
        average_func = state_action_denc - torch.mean(state_action_denc, dim=1, keepdim=True)

        Q = state_value + average_func
        return Q


class DQN_agent(nn.Module):
    def __init__(self, action_num, device):
        super().__init__()
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = action_num
        self.buffer_size = 10000
        self.batch_size = 32
        self.device = device

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = Q_net(action_num=self.action_size, device=self.device).to(self.device)
        self.target_net = deepcopy(self.qnet)
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), self.lr)

    def get_action(self, state):
        state = state.to(self.device)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state).detach().cpu().numpy()
            return np.argmax(qs[0])

    def update_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) <= self.batch_size:
            return 0
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()
        target_q = (
            self.target_net(
                torch.tensor(next_states, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
            .detach()
            .max(1)[0]
        )
        # td_targetを対象とした目的関数の構築
        target = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(
            1
        ) + self.gamma * target_q * (
            1 - torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        )
        # target_size:[32,2]
        q = self.qnet(torch.tensor(states, dtype=torch.float32, device=self.device))
        loss = F.mse_loss(target, q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.clone().detach().item()
