import numpy as np
import torch
from tqdm import tqdm


def load_dataset(path, pbar_desc="Loading dataset"):
    dataset = torch.load(path)
    with tqdm(total=len(dataset), desc=pbar_desc) as pbar:
        for _ in range(len(dataset)):
            pbar.update(1)
    return dataset


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
