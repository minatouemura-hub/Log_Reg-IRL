import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa
import torch
import torch.nn.functional as F  # noqa
from scipy.stats import linregress

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from develop.data import make_test_dataset  # noqa
from logreg.models import Irl_Net  # noqa


class Map_StateValue:
    def __init__(
        self,
        weight_path: str,
        pickel_path: str,
        usr_id: str,
        TimeThreshold: int,
        DiffThreshold: float,
    ):
        super().__init__()
        self.weight_path = weight_path
        self.pickel_path = pickel_path
        self.usr_id = usr_id
        self.TimeThreshold = TimeThreshold
        self.DiffThreshold = DiffThreshold
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
        over_threshold_data = pd.DataFrame({"IND": [0], "StateValue": [0]})
        ind = 0
        with torch.no_grad():
            for state in self.state_trans:
                state, next_state, sources = make_test_dataset(state)  # noqa
                state = F.pad(state, (0, 0, 0, 79 - state.shape[0]))
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                state_value = self.model.reward_net(state)
                state_value = state_value.squeeze(0).squeeze(0)
                state_value = state_value.cpu().detach().numpy()
                diff_state_value = state_value - prev_state_reward
                if np.abs(diff_state_value) >= np.abs(self.DiffThreshold) and (
                    self.TimeThreshold < ind
                ):
                    new_row = pd.DataFrame(
                        {"IND": ind, "StateValue": diff_state_value}, index=[0]
                    )  # noqa
                    over_threshold_data = pd.concat(
                        [over_threshold_data, new_row], ignore_index=True
                    )
                self.state_values.append(diff_state_value)
                # self.state_values.append(state_value)
                prev_state_reward = state_value
                ind += 1
        # self.map_correlation()
        self.map_diff_state_value()
        over_threshold_data = over_threshold_data.iloc[1:]  # noqa
        over_threshold_data.to_csv(
            f"/Users/uemuraminato/Desktop/IRL/mapping/Ind/{self.usr_id}.csv", index=False
        )

    def map_correlation(self):
        state_values = np.array(self.state_values)
        state_values = state_values[self.TimeThreshold :]  # noqa

        # x: SV_{t-1}, y: SV_{t}
        x = state_values[: len(state_values) - 1]
        y = state_values[1:]

        # 相関係数の計算
        correlation = np.corrcoef(x, y)[0, 1]
        print(f"相関係数: {correlation}")

        # 回帰直線の計算
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # 散布図の作成
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, marker="o", label="Data points")

        # 回帰直線の描画
        plt.plot(x, slope * x + intercept, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")

        # グラフの設定
        plt.title(f"Correlation of difference (SV_t - SV_t-1), r={correlation:.2f}")
        plt.xlabel("DiffStateValue_t-1")
        plt.ylabel("DiffStateValue_t")
        plt.grid(True)
        plt.legend()
        plt.show()

    def map_diff_state_value(self):
        self.load_chase_index()
        timestep = np.array(list(range(len(self.state_trans))))
        state_values = np.array(self.state_values)
        plt.figure(figsize=(10, 6))
        plt.plot(
            timestep[self.TimeThreshold :][self.chase_index],  # noqa
            state_values[self.TimeThreshold :][self.chase_index],  # noqa
            linestyle="-",
            color="b",  # noqa
        )
        plt.title(f"State Value of over 0.6_cos_similality:{self.usr_id}")
        plt.xlabel("Time Step")
        plt.ylabel("State Value")
        plt.grid(True)
        plt.savefig(f"/Users/uemuraminato/Desktop/IRL/mapping/figure/state_value_{self.usr_id}.png")

    def calculate_state_space(self):
        state = self.state_trans[0]

        if isinstance(state, np.ndarray):
            if state.ndim == 1:  # 1次元配列の場合
                return state.shape[0]
            elif state.ndim == 2:  # 2次元配列の場合
                return state.shape[0] * state.shape[1]
        else:
            raise ValueError("Unexpected type or structure for state column")

    def load_chase_index(self, target=0):
        with open(self.pickel_path, "rb") as f:
            loaded_group = pickle.load(f)
        self.chase_index = sorted(loaded_group[target], reverse=False)
