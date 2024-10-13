import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from develop.data import make_test_dataset  # noqa
from logreg.models import Irl_Net  # noqa


class Map_StateValue:
    def __init__(self, weight_path: str, usr_id: str, TimeThreshold: int):
        super().__init__()
        self.weight_path = weight_path
        self.usr_id = usr_id
        self.TimeThreshold = TimeThreshold
        if torch.backends.mps.is_available:
            self.device = "mps"
        else:
            self.device = "cpu"

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weight_path, weights_only=True))

    def excute(self, state_path):
        state_space = self.calculate_state_space
        self.model = Irl_Net(device=self.device, state_space=state_space, action_num=4)
        self.load_model()
        self.state_trans = np.load(state_path)

        self.model.eval()
        self.state_values = []
        prev_state_reward = np.array(1, dtype=np.float32)
        with torch.no_grad():
            for state in self.state_trans:
                state, next_state, sources = make_test_dataset(state)  # noqa
                state = F.pad(state, (0, 0, 0, 505 - state.shape[0]))
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                state_value = self.model.reward_net(state)
                state_value = state_value.squeeze(0).squeeze(0)
                state_value = state_value.cpu().detach().numpy()
                diff_state_value = state_value - prev_state_reward
                self.state_values.append(diff_state_value)
                prev_state_reward = state_value
        self.map_state_value()

    def map_state_value(self):
        timestep = list(range(len(self.state_trans)))
        plt.figure(figsize=(10, 6))
        plt.plot(
            timestep[self.TimeThreshold :],  # noqa
            self.state_values[self.TimeThreshold :],  # noqa
            linestyle="-",
            color="b",  # noqa
        )
        plt.title(f"State Value over time of {self.usr_id}")
        plt.xlabel("Time Step")
        plt.ylabel("State Value")
        plt.grid(True)
        plt.savefig(f"mapping/figure/state_value_{self.usr_id}.png")

    def calculate_state_space(self):
        state = self.state_trans[0]

        if isinstance(state, np.ndarray):
            if state.ndim == 1:  # 1次元配列の場合
                return state.shape[0]
            elif state.ndim == 2:  # 2次元配列の場合
                return state.shape[0] * state.shape[1]
        else:
            raise ValueError("Unexpected type or structure for state column")
