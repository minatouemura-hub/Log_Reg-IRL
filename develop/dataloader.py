import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from develop.data import make_irl_dataset  # noqa: E402


def custom_collate_fun(batch):
    states = [item[0] for item in batch]
    next_states = [item[1] for item in batch]
    sources = [item[2] for item in batch]

    states_padded = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
    next_states_padded = torch.nn.utils.rnn.pad_sequence(next_states, batch_first=True)
    sources_tensor = torch.tensor(sources)
    return states_padded, next_states_padded, sources_tensor


def create_balanced_sampler(dataset):
    # ラベルを取得
    labels = dataset.data["source"]

    # クラスごとのサンプル数を計算して逆数をクラス重みとして設定
    class_counts = labels.value_counts()
    class_weights = 1.0 / class_counts

    # 各サンプルの重みを設定
    sample_weights = labels.map(class_weights).values

    # サンプラーの作成
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


class BookDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.data = dataset
        self.source_counts = self.calculate_source_counts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.data.iloc[idx]["state"]
        next_state = self.data.iloc[idx]["next_state"]
        source = self.data.iloc[idx]["source"]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        source_tensor = torch.tensor(source, dtype=torch.int64)
        # データの形状が200になるように修正
        state_tensor = self.pad_to_shape(state_tensor, (200, 200, 50))
        next_state_tensor = self.pad_to_shape(next_state_tensor, (200, 200, 50))

        source_tensor = torch.tensor(source, dtype=torch.int64)
        return state_tensor, next_state_tensor, source_tensor

    @property
    def calculate_source_counts(self):
        source_counts = self.data["source"].value_counts()
        return source_counts

    @property
    def get_source_ratio(self):
        return np.float32(self.source_counts[-1] / self.source_counts[1])

    @property
    def StateSize(self):
        state = self.data.loc[0, "state"]
        print(state.ndim)
        return state.shape

    @property
    def calculate_state_space(self):
        # 最初の行のstateが2次元配列か1次元配列か確認
        state = self.data.loc[0, "state"]

        if isinstance(state, np.ndarray):
            if state.ndim == 1:  # 1次元配列の場合
                return state.shape[0]
            elif state.ndim == 2:  # 2次元配列の場合
                return state.shape[0] * state.shape[1]
        else:
            raise ValueError("Unexpected type or structure for state column")

    def pad_to_shape(
        self, tensor: torch.Tensor, target_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        # 各次元のパディングサイズを計算
        padding = []
        for current, target in zip(tensor.shape[::-1], target_shape[::-1]):
            pad_size = target - current
            padding.extend([0, pad_size])  # (前, 後) の順でパディング
        return F.pad(tensor, padding, "constant", 0)


def main():
    expert_id = "109636"
    base_url = "/Users/uemuraminato/Desktop/book_script/vec_data/vec_{}_book_com.npy"
    BathPath = "/Users/uemuraminato/Desktop/book_script/vec_data"
    expert_path = base_url.format(expert_id)
    combined_dataset = make_irl_dataset(expert_path=expert_path, BasePath=BathPath)
    book_dataset = BookDataset(combined_dataset)
    state1 = book_dataset.calculate_state_space
    print(state1)
    torch.save(book_dataset, f"dataset/{expert_id}_dataset.pt")


if __name__ == "__main__":
    main()
