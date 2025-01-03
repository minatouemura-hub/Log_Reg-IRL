import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

user_F_M = pd.DataFrame(columns=["Born_Gender", "Femininity", "Masculinity"])
for target in ["F", "M"]:
    # JSONファイルを開く
    with open(
        f"/Users/uemuraminato/Desktop/IRL/plot/{target}_gr_list.json", "r", encoding="utf-8"
    ) as json_file:
        data = json.load(json_file)
    data = pd.DataFrame(data)
    data = data.dropna(subset=["d_score"])
    # データのキーからリストを取得
    d_scores_raw = np.array(data["d_score"])
    s_scores_raw = np.array(data["s_score"])

    # 正規化
    d_scores_normalized = (d_scores_raw - d_scores_raw.min()) / (
        d_scores_raw.max() - d_scores_raw.min()
    )
    s_scores_normalized = (s_scores_raw - s_scores_raw.min()) / (
        s_scores_raw.max() - s_scores_raw.min()
    )

    if target == "F":
        data["diff"] = [
            femininity - masculinity for femininity, masculinity in zip(d_scores_raw, s_scores_raw)
        ]
    else:
        data["diff"] = [
            femininity - masculinity for masculinity, femininity in zip(d_scores_raw, s_scores_raw)
        ]
    with open(
        f"/Users/uemuraminato/Desktop/IRL/plot/{target}_gr_list.json", "w", encoding="utf-8"
    ) as json_file:
        json.dump(data.to_dict(orient="records"), json_file, ensure_ascii=False, indent=4)
    # データを整形 (meltで縦長に変換)
    melted_data = data.melt(
        value_vars=["d_score", "s_score"], var_name="Score Type", value_name="Score"
    )
    print("--差分の基礎統計--\n")
    print(data["diff"].describe())

    # 箱ヒゲ図の作成
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Score Type", y="Score", data=melted_data, palette="Set3")
    plt.title(f"Boxplot for {target} Gender")
    plt.xlabel("Score Type")
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.close()

    # 決定木によるfemilialityとmasculityの検証
    # 性別ごとのデータを作成し、縦方向に結合
    if target == "F":
        # 女性データの作成
        user_F = pd.DataFrame(
            {
                "Born_Gender": ["Female"] * len(data),  # 女性
                "Masculinity": data["s_score"],
                "Femininity": data["d_score"],
            }
        )
        user_F_M = pd.concat([user_F_M, user_F], axis=0, ignore_index=True)
    else:
        # 男性データの作成
        user_M = pd.DataFrame(
            {
                "Born_Gender": ["Male"] * len(data),  # 男性
                "Masculinity": data["d_score"],
                "Femininity": data["s_score"],
            }
        )
        user_F_M = pd.concat([user_F_M, user_M], axis=0, ignore_index=True)
# Masculinity - Femininity の計算
# user_F_M["Predicted_Gender"] = (user_F_M["Masculinity"] - user_F_M["Femininity"] < 0).astype(int)

# # Born_Gender と一致しているか確認
# correct_predictions = (user_F_M["Born_Gender"] == user_F_M["Predicted_Gender"]).sum()
# total_predictions = len(user_F_M)

# # 一致率を計算
# accuracy = correct_predictions / total_predictions


plt.figure(figsize=(8, 6))
palette = ["#E41A1C", "#377EB8"]
sns.histplot(
    x=user_F_M["Masculinity"] - user_F_M["Femininity"],
    hue=user_F_M["Born_Gender"],
    kde=True,
    palette=palette,
    bins=20,
)
plt.title("Distribution of (Masculinity - Femininity) by Gender")
plt.xlabel("Masculinity - Femininity")
plt.ylabel("Count")
plt.savefig("/Users/uemuraminato/Desktop/IRL/score_plot/Dist_Masculinity_Femininity.png")
plt.show()
