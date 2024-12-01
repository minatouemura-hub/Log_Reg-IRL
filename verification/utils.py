import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression  # noqa
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor, plot_tree


class MakeDataSet:
    def __init__(self, json_file_path: str, label: float, SexDataset: pd.DataFrame):
        try:
            with open(json_file_path, "r") as f:
                self.SexRates = pd.DataFrame(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load JSON file: {json_file_path}. Error: {e}")
        self.label = label  # =-1: Male , = 1:Female
        self.SexDataset = SexDataset
        self.bath_dir = "/Users/uemuraminato/Desktop/book_script/vec/vec_gen/"

    def make_data(self):
        for _, row in self.SexRates.iterrows():
            target_id = row["user_id"]  # "expert_id"を取得
            sr = row["gr"]
            # データを探してくる
            # word2vecの結果でもいいかな
            np_vec_path = self.search_ExpertPath(search_id=str(target_id))
            np_vec = np.load(np_vec_path)
            cos_mean, cos_var, count = self.calc_cos_data(dataset=np_vec)
            target_data = pd.DataFrame(
                [[self.label, sr, cos_mean, cos_var, count]],
                columns=["BSex", "SSex", "CosMean", "CosVar", "Count"],
            )
            self.SexDataset = pd.concat([self.SexDataset, target_data], ignore_index=True)
        return self.SexDataset

    def calc_cos_data(self, dataset):
        similality_matrix = cosine_similarity(dataset)
        triu_indices = np.tril_indices_from(
            similality_matrix, k=-1
        )  # k = -1 は対角上の要素以外という意味
        similalities = similality_matrix[triu_indices]
        count_above_threshold = np.sum(similalities >= 0.8)
        return similalities.mean(), np.var(similalities), count_above_threshold

    def search_ExpertPath(self, search_id):
        abs_path = None
        for root, dir, files in os.walk(self.bath_dir):
            for file in files:
                if search_id in file:
                    relative_path = os.path.join(root, file)
                    abs_path = os.path.abspath(relative_path)
                    return abs_path
        if abs_path is None:
            raise ValueError(f"No {search_id} files")


class Verify:
    def __init__(self, json_files: list):
        self.json_files = json_files

    def execute(self):
        linear_results = self.Count_verify()
        decision_tree_results = self.DecisionTree_compare()

        # 結果を表示
        print("Linear Regression Results:")
        for feature, metrics in linear_results.items():
            print(f"Feature: {feature}")
            print(f"  Coefficient: {metrics['Coefficient']}")
            print(f"  Intercept: {metrics['Intercept']}")
            print(f"  R²: {metrics['R²']}")
            print(f"  t-value: {metrics['t-value']}")
            print(f"  p-value: {metrics['p-value']}\n")

        print("Decision Tree Results:")
        for feature, metrics in decision_tree_results.items():
            print(f"Feature: {feature}")
            print(f"  R²: {metrics['R²']}")
            print(f"  Mean Squared Error: {metrics['MSE']}\n")

    def Load_Dataset(self):
        S_dataset = pd.DataFrame()
        # ファイル名に基づいてラベルを設定
        for json_file_path in self.json_files:
            if "M" in json_file_path:
                label = np.random.uniform(-1, 0)
            elif "F" in json_file_path:
                label = np.random.uniform(0, 1)
            else:
                raise ValueError(f"File name does not contain 'M' or 'F': {json_file_path}")
            makedata = MakeDataSet(json_file_path=json_file_path, label=label, SexDataset=S_dataset)
            S_dataset = makedata.make_data()

        return pd.DataFrame(S_dataset)

    def Count_verify(self):
        dataset = self.Load_Dataset()
        dataset = dataset.dropna()

        target = "CosVar"
        features = ["SSex", "BSex"]
        results = {}
        for feature in features:
            # 説明変数と目的変数
            X = dataset[[feature]]  # 説明変数（単回帰では1列）
            y = dataset[target]  # 目的変数

            # 単回帰モデルの構築
            model = LinearRegression()
            model.fit(X, y)

            # 決定係数（R²）を計算
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            # t値とp値の計算
            n = len(y)  # サンプルサイズ
            p = X.shape[1]  # 説明変数の数（単回帰では1）
            residual_sum_of_squares = np.sum((y - y_pred) ** 2)
            variance_estimate = residual_sum_of_squares / (n - p - 1)

            # 標準誤差の計算
            X_with_intercept = np.hstack([np.ones((n, 1)), X])  # 定数項を含める
            cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * variance_estimate
            standard_error = np.sqrt(np.diag(cov_matrix))[1]  # 説明変数の標準誤差

            # t値とp値の計算
            t_value = model.coef_[0] / standard_error
            p_value = (1 - t.cdf(np.abs(t_value), df=n - p - 1)) * 2

            # 結果を保存
            results[feature] = {
                "Coefficient": model.coef_[0],
                "Intercept": model.intercept_,
                "R²": r2,
                "t-value": t_value,
                "p-value": p_value,
            }
        return results

    def DecisionTree_compare(self):
        dataset = self.Load_Dataset()
        dataset = dataset.dropna()

        target = "CosVar"
        features = ["SSex", "BSex"]
        results = {}
        for feature in features:
            X = dataset[[feature]]
            y = dataset[target]

            model = DecisionTreeRegressor(random_state=42, max_depth=4)
            model.fit(X, y)

            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # 決定木を可視化
            plt.figure(figsize=(12, 8))
            plot_tree(model, feature_names=list(X.columns), filled=True, rounded=True)
            plt.title(f"Decision Tree for {feature}")
            plt.show()
            results[feature] = {
                "R²": r2,
                "MSE": mse,
            }
        return results


json_files = [
    "/Users/uemuraminato/Desktop/IRL/plot/M_gr_list.json",
    "/Users/uemuraminato/Desktop/IRL/plot/F_gr_list.json",
]
ver = Verify(json_files=json_files)
S_dataset = ver.execute()
