import os
import random

import numpy as np
import pandas as pd
import torch


def make_irl_dataset(BasePath, expert_path, max_num):
    expert_trans = np.load(expert_path)
    # ディレクトリ内の全ての.npyファイルを取得し、特定のファイルを除外
    baseline_files = load_npy_files(BasePath, exclude=expert_path, max_num=max_num)
    baseline_trans = pad_npy_files(baseline_files)

    # データの8割をトレーニング、残りをテスト
    expert_train_len = int(len(expert_trans) * 0.8)
    baseline_train_len = int(len(baseline_trans) * 0.8)

    expert_train = expert_trans[:expert_train_len]
    expert_test = expert_trans[expert_train_len:]
    baseline_train = baseline_trans[:baseline_train_len]
    baseline_test = baseline_trans[baseline_train_len:]

    # 次の状態を計算
    expert_next_train = np.roll(expert_train, shift=-1, axis=0)
    expert_next_train[-1] = np.zeros_like(expert_train[-1])
    expert_next_test = np.roll(expert_test, shift=-1, axis=0)
    expert_next_test[-1] = np.zeros_like(expert_test[-1])

    baseline_next_train = np.roll(baseline_train, shift=-1, axis=0)
    baseline_next_train[-1] = np.zeros_like(baseline_train[-1])
    baseline_next_test = np.roll(baseline_test, shift=-1, axis=0)
    baseline_next_test[-1] = np.zeros_like(baseline_test[-1])

    # データセットの作成
    expert_train_dataset = [
        (state, next_state, 1) for state, next_state in zip(expert_train, expert_next_train)
    ]
    expert_test_dataset = [
        (state, next_state, 1) for state, next_state in zip(expert_test, expert_next_test)
    ]

    baseline_train_dataset = [
        (state, next_state, -1) for state, next_state in zip(baseline_train, baseline_next_train)
    ]
    baseline_test_dataset = [
        (state, next_state, -1) for state, next_state in zip(baseline_test, baseline_next_test)
    ]

    # トレーニングとテストデータを分けて返す
    train_dataset = pd.DataFrame(
        expert_train_dataset + baseline_train_dataset, columns=["state", "next_state", "source"]
    )
    test_dataset = pd.DataFrame(
        expert_test_dataset + baseline_test_dataset, columns=["state", "next_state", "source"]
    )

    return train_dataset, test_dataset


# def make_irl_dataset(BasePath, expert_path, max_num):
#     expert_trans = np.load(expert_path)
#     # ディレクトリ内の全ての.npyファイルを取得し、特定のファイルを除外
#     baseline_files = load_npy_files(BasePath, exclude=expert_path, max_num=max_num)
#     baseline_trans = pad_npy_files(baseline_files)
#     expert_next_state = np.roll(expert_trans, shift=-1, axis=0)
#     expert_next_state[-1] = np.zeros_like(expert_trans[-1])
#     baseline_next_state = np.roll(baseline_trans, shift=-1, axis=0)
#     baseline_next_state[-1] = np.zeros_like(baseline_trans[-1])

#     expert_pairs = [(expert_trans[i], expert_next_state[i]) for i in range(len(expert_trans))]
#     baseline_pairs = [
#         (baseline_trans[i], baseline_next_state[i]) for i in range(len(baseline_trans))
#     ]
#     expert_dataset = [(state, next_state, 1) for state, next_state in expert_pairs]
#     baseline_dataset = [(state, next_state, -1) for state, next_state in baseline_pairs]
#     comb_dataset = expert_dataset + baseline_dataset

#     return pd.DataFrame(comb_dataset, columns=["state", "next_state", "source"])


def load_npy_files(directory, exclude: str, max_num: int = 100):
    npy_files = []
    count = 0
    filenames = os.listdir(directory)
    random.shuffle(filenames)
    # ディレクトリ内のファイルをチェック
    for filename in filenames:
        # .npyファイルかつ指定された除外条件に一致しない場合
        if filename.endswith(".npy") and exclude not in filename:
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            if count >= max_num:
                break
            npy_files.append(data)
            count += 1
    return npy_files


def pad_npy_files(npy_files):
    # 各ファイルの形状を取得
    shapes = [data.shape for data in npy_files]

    # 最大の形状を計算
    max_shape = np.max(shapes, axis=0)  # それぞれの軸で最大値を取得

    # パディングしたファイルを格納するリスト
    padded_files = []

    for data in npy_files:
        # パディングするサイズを計算
        padding = [(0, max_shape[i] - data.shape[i]) for i in range(len(max_shape))]

        # パディング実行（0で埋める）
        padded_data = np.pad(data, padding, mode="constant")
        padded_files += list(padded_data)

    return padded_files


def make_test_dataset(state):
    next_state = np.roll(state, shift=-1, axis=0)
    next_state[-1] = np.zeros_like(state[-1])

    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    sources = torch.zeros(len(state), dtype=torch.int32)

    return state, next_state, sources
