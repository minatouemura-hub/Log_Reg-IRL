import json
import math
import os
import re
import string

import ipadic
import matplotlib.pyplot as plt
import MeCab
import neologdn
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.linear_model import LinearRegression  # noqa
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor, plot_tree
from stopwordsiso import stopwords
from wordcloud import WordCloud


class MakeNPDataSet:
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
            # 配列全体のサイズ
            total_length = np_vec.shape[0]

            start_index = int(total_length * 0.8)
            # 検証(未知)データを対象に予測
            np_vec = np_vec[start_index:]

            cos_mean, cos_var, count = self.calc_cos_data(dataset=np_vec)
            target_data = pd.DataFrame(
                [[target_id, self.label, sr, cos_mean, cos_var, count]],
                columns=["user_id", "BSex", "SSex", "CosMean", "CosVar", "Count"],
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


class Preprocess:
    def __init__(
        self,
        target_id: int,
        additional_stopwords: dict = {
            "年",
            "賞",
            "人",
            "くん",
            "さん",
            "なか",
            "ぶり",
            "版",
            "作品",
            "化",
            "的",
            "三",
            "二",
            "一",
            "十",
            "受賞",
            "作家",
            "著者",
            "中",
            "者",
            "回",
            "章",
            "日",
            "作",
            "シリーズ",
            "文庫",
            "本",
            "年月",
            "冊",
            "はず",
        },
    ):
        self.target_id = str(target_id)
        self.bath_dir = "/Users/uemuraminato/Desktop/book_script/data"
        self.stopwords = stopwords(["ja", "en"])
        digits_and_alphabets = (
            set(string.digits) | set(string.ascii_lowercase) | set(string.ascii_uppercase)
        )
        self.stopwords.update(additional_stopwords)
        self.stopwords.update(digits_and_alphabets)
        self.mecab = MeCab.Tagger(ipadic.MECAB_ARGS)

    def execute(self):
        target_csv_path = self.search_ExpertPath(self.target_id)
        target_data = pd.read_csv(target_csv_path).fillna("")
        target_data = self.preproceed(target_data)
        return target_data

    def preproceed(self, target_data: pd.DataFrame):
        # 数字を正規表現によって削除
        target_docs = target_data["Intro"].apply(lambda x: re.sub(r"\d+", "", x))
        # 各記事のテキストに対してわかちがき
        docs = target_docs.apply(lambda text: self.wakachi(text))

        # 正規化
        normalized_docs = [[neologdn.normalize(word) for word in doc] for doc in docs]
        # ストップワードを削除
        filtered_docs = [
            [word for word in doc if word not in self.stopwords] for doc in normalized_docs
        ]

        return filtered_docs

    def wakachi(self, text: str):
        parsed = self.mecab.parse(text)
        words = []
        # わかちがきの結果のみを出力（名詞限定）
        for line in parsed.split("\n"):
            if line == "EOS" or line == "":
                continue
            parts = line.split("\t")
            surface_form = parts[0]
            feature_parts = parts[1].split(",")
            part_of_speech = feature_parts[0]
            # 名詞のみ抽出
            if part_of_speech == "名詞":
                base_form = feature_parts[6] if feature_parts[6] != "*" else surface_form
                words.append(base_form)
        return words

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


class StaticalVerify:
    def __init__(self, json_files: list):
        self.json_files = json_files
        self.wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=400,
            font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        )

    def execute(self):
        dataset = self.Load_Dataset()
        dataset = dataset.dropna()
        print(dataset.columns)
        print("---決定木による比較を開始します---")
        decision_tree_results = self.DecisionTree_compare(dataset=dataset)

        print("Decision Tree Results:")
        for feature, metrics in decision_tree_results.items():
            print(f"Feature: {feature}")
            print(f"  R²: {metrics['R²']}")
            print(f"  Mean Squared Error: {metrics['MSE']}\n")

        print("--LDAによるトピック分析を開始します--")

        # 最大SR値を持つトップ3ユーザーのトピック分析
        top_10_rows = dataset.nlargest(10, "SSex")  # SSex列のトップ3行を取得

        # 空のリストを用意して、すべてのユーザーの結果を蓄積
        combined_filtered_texts = []

        for rank, (_, row) in enumerate(top_10_rows.iterrows(), start=1):
            user_id = row["user_id"]
            preprocess = Preprocess(target_id=int(user_id))
            filtered_texts = preprocess.execute()

            # 結果をリストに追加
            combined_filtered_texts.extend(filtered_texts)

        # すべてのデータをまとめてトピック分析
        self.lda_plot(combined_filtered_texts, target="Top_Combined")
        bottom_10_rows = dataset.nsmallest(10, "SSex")  # SSex列の最小3行を取得
        combined_filtered_texts_bottom = []

        for rank, (_, row) in enumerate(bottom_10_rows.iterrows(), start=1):
            user_id = row["user_id"]
            preprocess = Preprocess(target_id=int(user_id))
            filtered_texts = preprocess.execute()

            # 結果をリストに追加
            combined_filtered_texts_bottom.extend(filtered_texts)

        # すべてのデータをまとめてトピック分析（下位10名）
        self.lda_plot(combined_filtered_texts_bottom, target="Bottom_Combined")

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
            makedata = MakeNPDataSet(
                json_file_path=json_file_path, label=label, SexDataset=S_dataset
            )
            S_dataset = makedata.make_data()

        return pd.DataFrame(S_dataset)

    # def Count_verify(self):
    #     dataset = self.Load_Dataset()
    #     dataset = dataset.dropna()

    #     target = "CosVar"
    #     features = ["SSex", "BSex"]
    #     results = {}
    #     for feature in features:
    #         # 説明変数と目的変数
    #         X = dataset[[feature]]  # 説明変数（単回帰では1列）
    #         y = dataset[target]  # 目的変数

    #         # 単回帰モデルの構築
    #         model = LinearRegression()
    #         model.fit(X, y)

    #         # 決定係数（R²）を計算
    #         y_pred = model.predict(X)
    #         r2 = r2_score(y, y_pred)

    #         # t値とp値の計算
    #         n = len(y)  # サンプルサイズ
    #         p = X.shape[1]  # 説明変数の数（単回帰では1）
    #         residual_sum_of_squares = np.sum((y - y_pred) ** 2)
    #         variance_estimate = residual_sum_of_squares / (n - p - 1)

    #         # 標準誤差の計算
    #         X_with_intercept = np.hstack([np.ones((n, 1)), X])  # 定数項を含める
    #         cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * variance_estimate
    #         standard_error = np.sqrt(np.diag(cov_matrix))[1]  # 説明変数の標準誤差

    #         # t値とp値の計算
    #         t_value = model.coef_[0] / standard_error
    #         p_value = (1 - stats.cdf(np.abs(t_value), df=n - p - 1)) * 2

    #         # 結果を保存
    #         results[feature] = {
    #             "Coefficient": model.coef_[0],
    #             "Intercept": model.intercept_,
    #             "R²": r2,
    #             "t-value": t_value,
    #             "p-value": p_value,
    #         }
    #     return results

    def DecisionTree_compare(self, dataset):

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
            if feature == "SSex":
                plt.savefig("verification/plt/decison_tree_reg.png")
            plt.close()
            results[feature] = {
                "R²": r2,
                "MSE": mse,
            }
        return results

    def lda_plot(self, fileterd_texts: list, how: str = "count", target: str = "Max"):
        # dicはリストの中にリスト形式を求める
        dic = Dictionary(fileterd_texts)
        dic.filter_extremes(no_below=3, no_above=0.8)
        corpus = [dic.doc2bow(text) for text in fileterd_texts]
        if how == "tf-idf":
            tfidf = TfidfModel(corpus)
            corpus = tfidf[corpus]
        elif how != "count":
            print("No such vectorize form")
        # Perplexityを指標としたtopic数の決定
        start_num_topic = 2
        end_num_topic = 8
        lowest_perplexity = float("inf")
        highest_coherence = float("-inf")
        perplexity_scores = {}
        coherence_scores = {}
        for num_topic in range(start_num_topic, end_num_topic + 1, 1):
            target_lda = LdaModel(corpus, num_topics=num_topic, id2word=dic)
            perplexity = target_lda.log_perplexity(corpus)
            perplexity_scores[num_topic] = perplexity

            # Coherenceの計算
            coherence_model = CoherenceModel(
                model=target_lda, texts=fileterd_texts, dictionary=dic, coherence="c_v"
            )
            coherence = coherence_model.get_coherence()
            coherence_scores[num_topic] = coherence

            # 最適なトピック数の記録
            if coherence > highest_coherence or (
                coherence == highest_coherence and perplexity < lowest_perplexity
            ):
                lda = target_lda
                # best_num_topic = num_topic
                lowest_perplexity = perplexity
                highest_coherence = coherence

        NUM_WORDS_FOR_WORD_CLOUD = 50

        fig, axs = plt.subplots(ncols=2, nrows=math.ceil(lda.num_topics / 2), figsize=(16, 20))
        axs = axs.flatten()

        for i, t in enumerate(range(lda.num_topics)):

            x = dict(lda.show_topic(t, NUM_WORDS_FOR_WORD_CLOUD))
            im = self.wordcloud.generate_from_frequencies(x)

            axs[i].imshow(im.recolor(colormap="Paired_r", random_state=244), alpha=0.98)
            axs[i].axis("off")
            axs[i].set_title("Topic " + str(t))

        fig.savefig(f"verification/plt/{target}_topic_word.png")
        fig.show()
