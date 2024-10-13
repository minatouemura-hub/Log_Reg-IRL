import os
import random  # noqa:F401
import sys
import traceback  # noqa

import matplotlib.pyplot as plt
import torch
import torch.utils
from torch.nn import Module
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from develop.data import make_irl_dataset  # noqa: E402
from develop.dataloader import BookDataset, custom_collate_fun  # noqa: F401 E402
from logreg.models import Irl_Net  # noqa: E402
from logreg.modules import negative_log_likelihood  # noqa: E402


class Train_Irl_model(Module):
    def __init__(self, num_epoch, expert_id, gamma=0.95, lambda_reg=0.01, lr=0.001):
        super().__init__()
        if torch.backends.mps.is_available:
            self.device = "mps"
        else:
            self.device = "cpu"
        self.num_epoch = num_epoch
        self.expert_id = expert_id
        self.action_num = 4  # arbitary
        self.gamma = gamma
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.epoch_losses = []

    def train(self):
        print(f"----Training on {self.device}----")

        # データのload
        expert_path = "/Users/uemuraminato/Desktop/book_script/analysis/preprocessed/state_tras_of_{}.npy".format(  # noqa
            self.expert_id
        )
        BathPath = "/Users/uemuraminato/Desktop/book_script/analysis/preprocessed/"
        combined_dataset = make_irl_dataset(expert_path=expert_path, BasePath=BathPath)
        book_dataset = BookDataset(combined_dataset)
        self.state_space = book_dataset.calculate_state_space
        self.be_ratio = torch.tensor(book_dataset.get_source_ratio, dtype=torch.float32).to(
            self.device
        )
        self.dataloader = DataLoader(
            book_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=custom_collate_fun,
        )

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
        with tqdm(
            total=self.num_epoch, desc="Epoch"
        ) as pbar:  # epochのたびにIRLを呼び出しているから初期化されてる．
            for epoch in range(self.num_epoch):
                try:
                    total_loss = self.train_one_step()
                    self.epoch_losses.append(total_loss.cpu().detach().numpy())
                    pbar.update(1)
                except Exception as e:
                    print(f"Batch {epoch + 1}: Exception occurred - {str(e)}")
        self.plot_losses()
        # トレーニングが終了したら、最終エポックのモデルの重みを保存
        torch.save(self.irl_model.state_dict(), "weight_vec/final_model_weights.pt")

    def train_one_step(self):
        with tqdm(total=len(self.dataloader), desc="Train_one_step", leave=False) as pbar:
            self.irl_model.train()
            for batch_idx, (states, next_states, sources) in enumerate(self.dataloader):
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
                    if (q_loss is None) or (v_loss is None):
                        continue
                    total_loss = q_loss + v_loss + dens_loss

                    # lossによる逆順伝播
                    total_loss.backward()

                    # lossの更新
                    self.dens_optim.step()
                    self.r_optim.step()
                    self.v_optim.step()

                    pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})
                    pbar.update(1)
                except Exception as e:
                    print(f"Batch {batch_idx + 1}: Exception occurred - {str(e)}")
                    traceback.print_exc()
        return total_loss

    def test(self):
        # ディレクトリ内の全ての.npyファイルを取得し、特定のファイルを除外
        expert_path = "/Users/uemuraminato/Desktop/book_script/analysis/preprocessed/state_tras_of_{}.npy".format(  # noqa
            self.expert_id
        )
        BathPath = "/Users/uemuraminato/Desktop/book_script/analysis/preprocessed/"
        combined_dataset = make_irl_dataset(expert_path=expert_path, BasePath=BathPath)
        book_dataset = BookDataset(combined_dataset)
        self.state_space = book_dataset.calculate_state_space

        self.be_ratio = torch.tensor(book_dataset.get_source_ratio, dtype=torch.float32).to(
            self.device
        )
        self.dataloader = DataLoader(
            book_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fun
        )
        for batch_idx, (states, next_states, sources) in enumerate(self.dataloader):
            # データのデバイスへの移動
            sources = sources.unsqueeze(1).to(self.device)

            states = states.unsqueeze(1).to(self.device)  # [1,505,20]
            next_states = next_states.unsqueeze(1).to(
                self.device
            )  # unsqueeze(1)はチャンネル数を追加している

    def plot_losses(self):
        # デバイスの確認（デバッグ用）
        # 各エポックの損失をプロット
        plt.plot(self.epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()
