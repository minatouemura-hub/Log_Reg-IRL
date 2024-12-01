import os
import random  # noqa:F401
import sys
import traceback  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F  # noqa
from torch.nn import Module
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from develop.data import make_irl_dataset  # noqa: E402
from develop.dataloader import (  # noqa: F401 E402
    BookDataset,
    create_balanced_sampler,
    custom_collate_fun,
)
from logreg.models import Irl_Net  # noqa: E402
from logreg.modules import negative_log_likelihood  # noqa: E402


class Train_Irl_model(Module):
    def __init__(
        self, num_epoch, expert_id, group: str = "F", gamma=0.95, lambda_reg=0.05, lr=0.01
    ):
        super().__init__()
        if torch.backends.mps.is_available:
            self.device = "mps"
        else:
            self.device = "cpu"
        self.num_epoch = num_epoch
        self.expert_id = expert_id
        self.group = group
        self.action_num = 4  # arbitary
        self.gamma = gamma
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.epoch_losses = []
        self.bath_dir = "/Users/uemuraminato/Desktop/book_script/vec/preproceed/"

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

    def train(self):
        # 設定&データの処理
        expert_path = self.search_ExpertPath(self.expert_id)
        BathPath = f"/Users/uemuraminato/Desktop/book_script/vec/preproceed/{self.group}/"
        train_data, test_data = make_irl_dataset(
            expert_path=expert_path, BasePath=BathPath, max_num=20
        )
        train_dataset = BookDataset(train_data)
        self.test_dataset = BookDataset(test_data)
        self.state_space = train_dataset.calculate_state_space
        self.be_ratio = torch.tensor(train_dataset.get_source_ratio, dtype=torch.float32).to(
            self.device
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            sampler=create_balanced_sampler(train_dataset),
            collate_fn=custom_collate_fun,
        )
        # trainの設定
        patiance = 5
        best_loss = float("inf")
        epochs_no_improve = 0

        # モデルの初期化
        self.irl_model = Irl_Net(
            device=self.device,
            action_num=self.action_num,
            state_space=self.state_space,
        ).to(self.device)

        self.dens_optim = Adagrad(
            self.irl_model.dens_net.parameters(),
            lr=self.lr,
        )
        self.r_optim = Adagrad(self.irl_model.reward_net.parameters(), lr=self.lr)
        self.v_optim = Adagrad(self.irl_model.state_value_net.parameters(), lr=self.lr)

        # trainの開始
        self.irl_model.train()
        with tqdm(total=self.num_epoch, desc="Epoch", leave=False) as pbar:
            tqdm.write(f"----Training on {self.device}----")
            for epoch in range(self.num_epoch):
                total_loss = 0
                try:
                    total_loss = self.train_one_step()
                    self.epoch_losses.append(total_loss.cpu().detach().numpy())
                    pbar.update(1)
                    if total_loss < best_loss:
                        best_loss = total_loss
                        epochs_no_improve = 0
                        torch.save(
                            self.irl_model.state_dict(),
                            f"weight_vec/weights_{self.group}_{self.expert_id}.pt",
                        )
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patiance:
                        tqdm.write(f"Early stopping at epoch {epoch + 1}")
                        break
                except Exception as e:
                    tqdm.write(f"Batch {epoch + 1}: Exception occurred - {str(e)}")

                finally:
                    pbar.close()
        self.plot_losses()
        # トレーニングが終了したら、最終エポックのモデルの重みを保存
        score_data = self.test_plot()
        self.save_expert_scores()
        return score_data

    def train_one_step(self):
        with tqdm(total=len(self.train_dataloader), desc="Train_one_step", leave=False) as pbar:
            self.irl_model.train()
            for batch_idx, (states, next_states, sources) in enumerate(self.train_dataloader):
                try:
                    # データのデバイスへの移動
                    sources = sources.unsqueeze(1).to(self.device)
                    states = states.unsqueeze(1).to(self.device)
                    next_states = next_states.unsqueeze(1).to(
                        self.device
                    )  # unsqueeze(1)はチャンネル数を追加している

                    # モデルの計算
                    dens, q_hat, v_x, v_y = self.irl_model(states, next_states)

                    # optimの初期化
                    self.dens_optim.zero_grad()
                    self.r_optim.zero_grad()
                    self.v_optim.zero_grad()

                    # lossの計算(ここの実装がうまくいっていない)
                    dens_loss = negative_log_likelihood(
                        outputs=dens,
                        labels=sources,
                        lambda_reg=self.lambda_reg,
                        model=self.irl_model.dens_net,
                    )
                    log_ratio = (
                        torch.sigmoid(dens * sources).to(self.device)
                        + q_hat
                        + self.gamma * v_y
                        - v_x
                        + torch.log(self.be_ratio).to(self.device)
                    ).to(self.device)
                    q_loss = negative_log_likelihood(
                        outputs=log_ratio,
                        labels=sources,
                        lambda_reg=self.lambda_reg,
                        model=self.irl_model.reward_net,
                    )
                    v_loss = negative_log_likelihood(
                        outputs=log_ratio,
                        labels=sources,
                        lambda_reg=self.lambda_reg,
                        model=self.irl_model.state_value_net,
                    )
                    if torch.isnan(dens_loss) or torch.isnan(q_loss) or torch.isnan(v_loss):
                        continue

                    total_loss = q_loss + v_loss + dens_loss

                    # lossによる逆順伝播
                    total_loss.backward()

                    # lossの更新
                    self.dens_optim.step()
                    self.r_optim.step()
                    self.v_optim.step()

                    pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

                except Exception as e:
                    print(f"Batch {batch_idx + 1}: Exception occurred - {str(e)}")
                    traceback.print_exc()
                pbar.update(1)
        return total_loss

    def test_plot(self):
        self.irl_model.eval()
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=12,
            collate_fn=custom_collate_fun,
        )
        # expertとbaselineを分けてプロット
        score = []
        category = []
        for batch_idx, (states, next_states, source) in enumerate(test_dataloader):
            states = states.unsqueeze(1).to(self.device)
            expert_state = states[source == 1]
            baseline_state = states[source == -1]
            batch_expert_score = float(
                torch.sum(self.irl_model.reward_net(expert_state)).detach().cpu()
            )
            score.append(batch_expert_score)
            category.append("expert")
            batch_baseline_score = float(
                torch.sum(self.irl_model.reward_net(baseline_state)).detach().cpu()
            )
            score.append(batch_baseline_score)
            category.append("baseline")
        data = pd.DataFrame({"Score": score, "Category": category})
        plt.figure(figsize=(8, 6))
        sns.violinplot(x="Category", y="Score", data=data, color="gray", inner="box")
        plt.title(f"Violin Plot of Scores in {self.group}")
        plt.xlabel("Category")
        plt.ylabel("Score")
        if not os.path.exists(f"plot/{self.expert_id}"):
            os.makedirs(f"plot/{self.expert_id}")
        plt.savefig(f"plot/{self.expert_id}/{self.group}_violinplot.png")
        plt.close()
        return sum(data[data["Category"] == "expert"]["Score"])

    def plot_losses(self):
        # デバイスの確認（デバッグ用）
        # 各エポックの損失をプロット
        plt.plot(self.epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.close()

    # 時系列的に扱うために追加
    def save_expert_scores(self):
        self.irl_model.eval()  # モデルを評価モードに設定
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,  # 1つずつ処理
            collate_fn=custom_collate_fun,
        )

        scores = []  # スコアを格納するリスト
        indices = []  # 時系列順のインデックスを保持

        # 時系列順に処理
        for idx, (state, next_state, source) in enumerate(test_dataloader):
            state = state.unsqueeze(1).to(self.device)
            source = source.to(self.device)

            # expert のスコアのみ計算
            if source.item() == 1:  # source が expert の場合
                expert_score = float(self.irl_model.reward_net(state).detach().cpu())
                scores.append(expert_score)
                indices.append(idx)

        # 保存先のディレクトリを確認または作成
        save_dir = f"plot/{self.expert_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 時系列スコアをCSVに保存
        scores_df = pd.DataFrame({"Index": indices, "Expert_Score": scores})
        scores_df.to_csv(f"{save_dir}/{self.group}_expert_scores.csv", index=False)

        # スコアの推移をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(indices, scores, marker="o", label="Expert Scores")
        plt.title(f"Time Series of Expert Scores in {self.group}")
        plt.xlabel("Time Index")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/{self.group}_expert_scores_plot.png")
        plt.close()

    def save_expert_scores_with_cumulative_data(self):
        self.irl_model.eval()  # モデルを評価モードに設定

        # 初期化
        cumulative_states = []  # 前の期のデータを保持
        scores = []  # 各期のスコアを格納
        indices = []  # 時系列順のインデックスを保持

        # テストデータのデータローダー
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,  # 1つずつ処理
            collate_fn=custom_collate_fun,
        )

        # 時系列処理
        for idx, (state, _, source) in enumerate(test_dataloader):
            state = state.unsqueeze(1).to(self.device)
            source = source.to(self.device)

            # データを累積
            if source.item() == 1:  # source が expert の場合
                cumulative_states.append(state)

            # 累積データをまとめてスコア計算
            if len(cumulative_states) > 0:
                cumulative_tensor = torch.cat(cumulative_states, dim=0)  # 累積データをテンソル化
                batch_expert_score = float(
                    torch.sum(self.irl_model.reward_net(cumulative_tensor)).detach().cpu()
                )
                scores.append(batch_expert_score)
                indices.append(idx)

        # 保存先のディレクトリを確認または作成
        save_dir = f"plot/{self.expert_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 時系列スコアをCSVに保存
        scores_df = pd.DataFrame({"Index": indices, "Cumulative_Expert_Score": scores})
        scores_df.to_csv(f"{save_dir}/{self.group}_cumulative_expert_scores.csv", index=False)

        # スコアの推移をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(indices, scores, marker="o", label="Cumulative Expert Scores")
        plt.title(f"Cumulative Time Series of Expert Scores in {self.group}")
        plt.xlabel("Time Index")
        plt.ylabel("Cumulative Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/{self.group}_cumulative_expert_scores_plot.png")
        plt.close()
