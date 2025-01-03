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
    if d_score == 0:
        if target_gender == "F":
            return None
        else:
            return None
    if target_gender == "F":
        gr = (d_score - s_score) / d_score  # 女性らしさ
    else:
        gr = (s_score - d_score) / d_score  # 男性らしさ
    return gr


def plot_GenderRate(num: int, two_score_dict: dict, target_gender: str):
    d_scores = np.array(two_score_dict["d_score"])
    s_scores = np.array(two_score_dict["s_score"])

    gr = np.array([d - s for d, s in zip(d_scores, s_scores)])

    # ラベル計算
    label = sum(1 for x in gr if x > 0) if target_gender == "F" else sum(1 for x in gr if x < 0)
    accuracy = float(label / num)
    # ヒストグラムのプロット
    bins = np.linspace(start=-3, stop=3, num=9)
    sns.histplot(gr, bins=bins, kde=False, color="gray", edgecolor="black")
    # 軸ラベルとタイトル
    plt.xlabel("Value Range")  # X軸のラベル

    plt.ylabel("Frequency")
    max_y = plt.gca().get_ylim()[1]  # y軸の最大値を取得
    plt.yticks(np.arange(0, max_y + 1, 5))

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
    """
    統計検定を行い、結果をCSVとして保存する関数。
    欠損値がある場合は事前に削除する。
    """
    # 欠損値の削除
    df = pd.DataFrame(score_dict).dropna()
    cleaned_score_dict = {
        "D_expert_score": df["D_expert_score"].tolist(),
        "S_expert_score": df["S_expert_score"].tolist(),
    }

    # 検定の実行
    if how == "ttest":
        stat, p_value = ttest_ind(
            cleaned_score_dict["D_expert_score"],
            cleaned_score_dict["S_expert_score"],
            equal_var=False,
        )
    elif how == "mannehitneyu":
        stat, p_value = mannwhitneyu(
            cleaned_score_dict["D_expert_score"],
            cleaned_score_dict["S_expert_score"],
            alternative="two-sided",
        )
    else:
        raise ValueError("Invalid method. Use 'ttest' or 'mannehitneyu'.")

    # 結果のまとめ
    summary_data = {
        "Metric": ["Mean", "Variance", "Statistic", "P-Value", "Conclusion", "How"],
        "D_expert_score": [
            round(np.mean(cleaned_score_dict["D_expert_score"]), 2),
            round(np.var(cleaned_score_dict["D_expert_score"], ddof=1), 2),
            stat,
            p_value,
            "Significant" if p_value < 0.05 else "Not Significant",
            how,
        ],
        "S_expert_score": [
            round(np.mean(cleaned_score_dict["S_expert_score"]), 2),
            round(np.var(cleaned_score_dict["S_expert_score"], ddof=1), 2),
            None,
            None,
            None,
            None,
        ],
    }

    # DataFrameに変換しCSVに保存
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv(f"./table/{how}_results.csv", index=False)

    print("統計検定が完了し、結果が保存されました。")
