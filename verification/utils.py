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
import seaborn as sns
import statsmodels.api as sm  # noqa
import torch.nn.functional as F
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from scipy.stats import ttest_ind  # noqa
from sklearn.ensemble import RandomForestRegressor  # noqa
from sklearn.linear_model import LinearRegression, Ridge  # noqa
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (  # noqa
    KFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler  # noqa
from sklearn.tree import DecisionTreeRegressor, plot_tree  # noqa
from statsmodels.stats.anova import anova_lm  # noqa
from stopwordsiso import stopwords
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
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
        # self.scaler = StandardScaler()

    def make_data(self):
        # self.scaler.fit(self.SexRates[["gr"]])
        # self.SexRates["gr_scaled"] = self.scaler.transform(self.SexRates[["gr"]])
        for _, row in self.SexRates.iterrows():
            target_id = row["user_id"]  # "expert_id"を取得
            sr = row["diff"]
            # データを探してくる
            # word2vecの結果でもいいかな
            np_vec_path = search_ExpertPath(search_id=str(target_id))
            np_vec = np.load(np_vec_path, allow_pickle=False)
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
            # self.SexDataset = self.SexDataset[self.SexDataset["CosVar"] <= 0.20]
        # self.SexDataset = self.SexDataset[np.abs(self.SexDataset["SSex"]) <= 1]
        return self.SexDataset

    def calc_cos_data(self, dataset):
        similality_matrix = cosine_similarity(dataset)
        triu_indices = np.tril_indices_from(
            similality_matrix, k=-1
        )  # k = -1 は対角上の要素以外という意味
        similalities = similality_matrix[triu_indices]
        count_above_threshold = np.sum(similalities >= 0.8)
        return similalities.mean(), np.std(similalities), count_above_threshold


def search_ExpertPath(
    search_id, base_dir: str = "/Users/uemuraminato/Desktop/book_script/vec/vec_gen/"
):
    abs_path = None
    for root, dir, files in os.walk(base_dir):
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
            "PART",
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
            "巻",
            "収録",
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
    def __init__(self, json_files: list, lda_num: int = 5):
        self.json_files = json_files
        self.wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=400,
            font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        )
        self.lda_num = lda_num

    def execute(self):
        if os.path.exists("./verification/sentiment_score.csv"):
            dataset = pd.read_csv("./verification/sentiment_score.csv")
        else:
            dataset = self.Load_Dataset()
            dataset = dataset.dropna(subset=["SSex"])
            dataset = self.emotion_score_dataset(dataset=dataset)
        dataset["emotional"] = dataset["avg_negative"] + dataset["avg_positive"]
        # self.hist_plt(dataset)
        # print("---決定木による比較を開始します---")
        # optimal_min_samples = self.DecisionTree_MSE_Plot(dataset=dataset)
        # print(optimal_min_samples)
        # result = self.DecisionTree_compare(dataset=dataset, optimal_min_samples=optimal_min_samples)
        # print("Decision Tree Results:")
        # for _, metrics in result.iterrows():
        #     print(f"Feature: {metrics['Feature']}")
        #     print(f"Avg Train R² Mean: {metrics['Avg Train R²']}")
        #     print(f"Avg Test R² Mean:{metrics['Avg Test R²']}")
        #     print(f"Train  Mean Squared Error: {metrics['Avg Train MSE']}\n")

        print("--LDAによるトピック比較分析を開始します--")
        # ランダムな女性ユーザーの処理
        random_10_female = dataset[dataset["BSex"] == 1].sample(n=self.lda_num, random_state=42)
        # トップ10ユーザーの処理
        top_10_rows = dataset.nlargest(self.lda_num, "SSex")
        # ボトム10ユーザーの処理
        bottom_10_rows = dataset.nsmallest(self.lda_num, "SSex")
        # ランダムな男性ユーザーの処理
        random_10_male = dataset[dataset["BSex"] == -1].sample(n=self.lda_num, random_state=42)

        # 各群のデータを保存するリスト
        top_combined_texts = []
        bottom_combined_texts = []
        random_female_texts = []
        random_male_texts = []

        # トップ10ユーザーの処理
        for _, row in top_10_rows.iterrows():
            user_id = row["user_id"]
            preprocess = Preprocess(target_id=int(user_id))
            filtered_texts = preprocess.execute()
            top_combined_texts.extend(filtered_texts)

        # ボトム10ユーザーの処理
        for _, row in bottom_10_rows.iterrows():
            user_id = row["user_id"]
            preprocess = Preprocess(target_id=int(user_id))
            filtered_texts = preprocess.execute()
            bottom_combined_texts.extend(filtered_texts)

        # ランダムな女性ユーザーの処理
        for _, row in random_10_female.iterrows():
            user_id = row["user_id"]
            preprocess = Preprocess(target_id=int(user_id))
            filtered_texts = preprocess.execute()
            random_female_texts.extend(filtered_texts)

        # ランダムな男性ユーザーの処理
        for _, row in random_10_male.iterrows():
            user_id = row["user_id"]
            preprocess = Preprocess(target_id=int(user_id))
            filtered_texts = preprocess.execute()
            random_male_texts.extend(filtered_texts)

        # 各群で個別にトピック分析
        self.lda_plot(top_combined_texts, target="Top_10_Users")
        self.lda_plot(bottom_combined_texts, target="Bottom_10_Users")
        self.lda_plot(random_female_texts, target="Random_10_Female_Users")
        self.lda_plot(random_male_texts, target="Random_10_Male_Users")

    def hist_plt(self, dataset: pd.DataFrame):  # noqa
        for target in [
            "CosVar",
            "avg_neutral",
        ]:
            for gender in ["BSex", "SSex"]:
                fig = plt.figure(figsize=(5, 5))  # noqa
                plt.scatter(gender, target, data=dataset, color="Black", alpha=0.8)
                plt.title(f"{target}_{gender}")
                plt.show()
                plt.close()

    def Load_Dataset(self):
        S_dataset = pd.DataFrame()
        # ファイル名に基づいてラベルを設定
        for json_file_path in self.json_files:
            if "M" in json_file_path:
                label = -1
            elif "F" in json_file_path:
                label = 1
            else:
                raise ValueError(f"File name does not contain 'M' or 'F': {json_file_path}")
            makedata = MakeNPDataSet(
                json_file_path=json_file_path, label=label, SexDataset=S_dataset
            )
            S_dataset = makedata.make_data()

        return pd.DataFrame(S_dataset)

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
        start_num_topic = 3
        end_num_topic = 10
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
        if lda.num_topics % 2 != 0:
            axs[-1].axis("off")

        fig.savefig(f"verification/plt/{target}_topic_word.png")

    def DecisionTree_compare(self, dataset: pd.DataFrame, optimal_min_samples: dict, k: int = 10):
        target = "avg_neutral"
        features = ["SSex", "BSex"]
        results = []

        for feature in features:
            X = dataset[[feature]].values
            y = dataset[target].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )

            train_r2_scores = []
            test_r2_scores = []
            train_mse_scores = []
            test_mse_scores = []

            optimal_sample = optimal_min_samples[feature]

            # モデルの訓練
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_split=optimal_sample,
                random_state=42,
            )
            model.fit(X_train, y_train)

            # 訓練データの評価
            y_train_pred = model.predict(X_train)
            train_r2_scores.append(r2_score(y_train, y_train_pred))
            train_mse_scores.append(mean_squared_error(y_train, y_train_pred))

            # テストデータの評価
            y_test_pred = model.predict(X_test)
            test_r2_scores.append(r2_score(y_test, y_test_pred))
            test_mse_scores.append(mean_squared_error(y_test, y_test_pred))

            # 過学習のチェック
            avg_train_r2 = np.mean(train_r2_scores)
            avg_test_r2 = np.mean(test_r2_scores)
            overfitting = (
                avg_train_r2 - avg_test_r2 > 0.1
            )  # 訓練とテストの R² スコア差が 0.1 以上なら過学習

            # 結果を保存
            results.append(
                {
                    "Feature": feature,
                    "Avg Train R²": avg_train_r2,
                    "Avg Test R²": avg_test_r2,
                    "Avg Train MSE": np.mean(train_mse_scores),
                    "Avg Test MSE": np.mean(test_mse_scores),
                    "Overfitting": overfitting,
                }
            )
        # 結果をデータフレームに変換して返す
        return pd.DataFrame(results)

    # 使わんかも
    def DecisionTree_MSE_Plot(
        self, dataset: pd.DataFrame, k: int = 7, min_samples=list(range(5, 10))
    ):
        target = "avg_neutral"
        features = ["SSex", "BSex"]  # SSex のみをモデル検証
        optimal_min_samples = {}
        plot_data = []
        for feature in features:
            X = dataset[[feature]].values
            y = dataset[target].values
            kf = RepeatedKFold(n_repeats=2, random_state=42)

            for min_sample in min_samples:
                mse_scores = []
                r2_scores = []
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        max_depth=3,
                        min_samples_split=min_sample,
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    mse_scores.append(mean_squared_error(y_test, y_pred))
                    r2_scores.append(r2_score(y_test, y_pred))  # MSEを計算
                plot_data.append(
                    {
                        "min_sample": min_sample,
                        "MSE_mean": np.mean(mse_scores),
                        "MSE_std": np.std(mse_scores),
                        "R2_mean": np.mean(r2_scores),
                        "R2_std": np.std(r2_scores),
                    }
                )
            # データフレーム化
            plot_df = pd.DataFrame(plot_data)
            plot_df["score"] = plot_df["R2_mean"] + plot_df["R2_std"]

            # 最適な深さを取得 (scoreが最小の深さ)
            optimal_min_sample = plot_df.loc[plot_df["R2_std"].idxmin(), "min_sample"]
            optimal_min_samples[feature] = optimal_min_sample  # 最適深さを辞書に保存

            # プロット設定
            sns.set_theme(style="ticks", font_scale=1.2)
            plt.figure(figsize=(12, 6), dpi=150)

            # sns.lineplot で平均値をプロット
            sns.lineplot(
                data=plot_df,
                x="min_sample",
                y="MSE_mean",
                marker="o",
                label="MSE Mean",
            )

            # エラーバーを手動で追加
            plt.errorbar(
                plot_df["min_sample"],
                plot_df["MSE_mean"],
                yerr=plot_df["MSE_std"],  # 標準偏差をエラーバーとして使用
                fmt="o",
                capsize=5,  # エラーバーのキャップサイズ
                color="blue",
            )

            # 軸とタイトル設定
            plt.xlabel("Min_Sample")
            plt.ylabel("MSE_Mean")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"verification/plt/{feature}_mse_mean_and_error.png")
            plt.close()
        return optimal_min_samples

    def emotion_score_dataset(self, dataset: pd.DataFrame):
        # 事前学習済みの日本語感情分析モデルとそのトークナイザをロード
        model = AutoModelForSequenceClassification.from_pretrained(
            "christian-phu/bert-finetuned-japanese-sentiment"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "christian-phu/bert-finetuned-japanese-sentiment", model_max_lentgh=512
        )
        # 感情分析のためのパイプラインを設定
        nlp = pipeline(  # noqa
            "sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True
        )
        base_dir = "/Users/uemuraminato/Desktop/book_script/data/"
        sentiment_data = []
        # 全ユーザーを対象にあらすじデータをベクトル化+感情分析を行い，感情スコアの平均と分散を出す
        for user_id in tqdm(dataset["user_id"].unique(), desc="Emotion Analysis Processing"):
            path = search_ExpertPath(user_id, base_dir=base_dir)
            target_df = pd.read_csv(path)
            target_df = target_df.dropna(subset=["Intro"])
            start_ind = int(len(target_df) * 0.8)
            target_df = target_df[start_ind:]
            sentiment_list = []
            for intro in target_df["Intro"]:
                input = tokenizer(
                    intro, padding=True, truncation=True, return_tensors="pt", max_length=512
                )
                output = model(**input)
                logits = output.logits
                probs = F.softmax(logits, dim=-1).detach().numpy()[0]

                # スコアをリストに保存
                sentiment_list.append(
                    {"positive": probs[0], "neutral": probs[1], "negative": probs[2]}
                )

            # 平均スコアを計算
            if sentiment_list:
                df_sentiment = pd.DataFrame(sentiment_list)
                avg_positive = df_sentiment["positive"].mean()
                avg_neutral = df_sentiment["neutral"].mean()
                avg_negative = df_sentiment["negative"].mean()
            else:
                avg_positive = avg_neutral = avg_negative = 0

            # 辞書形式で保存
            sentiment_data.append(
                {
                    "user_id": user_id,
                    "avg_positive": avg_positive,
                    "avg_neutral": avg_neutral,
                    "avg_negative": avg_negative,
                }
            )
        # 感情スコアをデータフレーム化
        sentiment_df = pd.DataFrame(sentiment_data)
        # 元の dataset と結合
        result_dataset = dataset.merge(sentiment_df, on="user_id", how="left")
        result_dataset.to_csv("./verification/sentiment_score.csv")
        return result_dataset
