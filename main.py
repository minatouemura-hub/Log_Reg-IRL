from develop.dataloader import BookDataset, custom_collate_fun  # noqa: F401 E402
from trainer import Train_Irl_model


def main():
    for flag in ["M", "F"]:
        train_model = Train_Irl_model(num_epoch=50, group=flag, expert_id="282967")
        train_model.train()


if __name__ == "__main__":
    main()
