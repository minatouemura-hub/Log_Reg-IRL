import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm


def load_dataset(path, pbar_desc="Loading dataset"):
    dataset = torch.load(path)
    with tqdm(total=len(dataset), desc=pbar_desc) as pbar:
        for _ in range(len(dataset)):
            pbar.update(1)
    return dataset


def choose_expert_data(directory_path: str, num: int):
    files = os.listdir(directory_path)
    files = extract_numbers_from_strings(files)
    sampled_number = random.sample(files, num)
    return sampled_number


def Gender_Rate(d_score, s_score, target_gender: str = "F"):
    if target_gender == "F":
        gr = (d_score - s_score) / d_score  # どの程度，その性別らしいか
    else:
        gr = (s_score - d_score) / d_score  # 男性の場合は-1倍することで同じスペクトラムで表現
    return gr


def plot_GenderRate(num: int, two_score_dict: dict, target_gender: str):
    d_scores = np.array(two_score_dict["D_expert_score"])
    s_scores = np.array(two_score_dict["S_expert_score"])
    if target_gender == "F":
        gr = (
            d_scores - s_scores
        ) / d_scores  # 変化量*-1 自身の性別らしさからどの程度変化しているのか
        label = sum(1 for x in gr if x > 0)
    if target_gender == "M":
        gr = (s_scores - d_scores) / d_scores  # 変化量
        label = sum(1 for x in gr if x < 0)
    accuracy = float(label / num)
    # ヒストグラムのプロット
    sns.histplot(gr, bins=8, binrange=(-1, 1), kde=False, color="gray", edgecolor="black")
    # 軸ラベルとタイトル
    plt.xlabel("Value Range")  # X軸のラベル
    plt.ylabel("Frequency")  # Y軸のラベル
    plt.title("Histogram of List Data")  # タイトル
    # グラフの表示
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"./plot/{target_gender}/gr_plot.png")
    plt.close()
    return accuracy


def extract_numbers_from_strings(strings):
    numbers = []
    for string in strings:
        # 正規表現で数字を検索
        match = re.search(r"\d+", string)
        if match:
            numbers.append(int(match.group()))  # 数字をリストに追加
    return numbers


def certificate_diff(score_dict: dict, target: str, how: str = "mannehitneyu"):
    if how == "ttest":
        stat, p_value = ttest_ind(
            score_dict["D_expert_score"], score_dict["S_expert_score"], equal_var=False
        )
        # 有意差の判定 (p値 < 0.05を有意水準とする例)

    elif how == "mannehitneyu":
        stat, p_value = mannwhitneyu(
            score_dict["D_expert_score"], score_dict["S_expert_score"], alternative="two-sided"
        )
    if target == "M":
        other = "F"
    else:
        other = "M"
    summary_data = {
        "Metric": ["Mean", "Variance", "Statistic", "P-Value", "Conclusion", "How"],
        target: [
            round(np.mean(score_dict["D_expert_score"]), 2),
            round(np.var(score_dict["D_expert_score"], ddof=1), 2),
            stat,
            p_value,
            "Significant" if p_value < 0.05 else "Not Significant",
            how,
        ],
        other: [
            round(np.mean(score_dict["S_expert_score"]), 2),
            round(np.var(score_dict["S_expert_score"], ddof=1), 2),
            None,
            None,
            None,
            None,
        ],
    }

    # DataFrameに変換
    results_df = pd.DataFrame(summary_data)

    # CSV形式で保存
    results_df.to_csv(f"./table/{how}_results.csv", index=False)


def check_memory(device):
    if torch.backends.mps.is_available():
        mem_info = torch.mps.current_allocated_memory()
        print(f"Current allocated memory on MPS: {mem_info}")
    else:
        mem_info = torch.cuda.memory_allocated(device)
        print(f"Current allocated memory on CUDA: {mem_info}")


def stable_softmax(x):
    shift_x = x - torch.max(x)
    exps = torch.exp(shift_x)
    softmax = exps / torch.sum(exps)
    return softmax


def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)), dtype=np.float32)


def sys_resampling(N, log_w):
    # ログ重みを線形領域に戻す
    w = np.exp(log_w)
    # 重みを正規化する（合計が1になるように）
    w /= np.sum(w)

    # 系統リサンプリング用のインデックスを生成
    positions = (np.arange(N) + np.random.uniform()) / N
    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(w)

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


# 状況の設定のみ
class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array([[0, 0, 0, 1.0], [0, None, 0, -1.0], [0, 0, 0, 0]])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    # 環境の状態遷移を表す関数
    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state

        self.agent_state = next_state
        return next_state, reward, done
