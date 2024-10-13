from develop.dataloader import BookDataset, custom_collate_fun  # noqa: F401 E402
from trainer import Train_Irl_model


def main():
    train_model = Train_Irl_model(
        num_epoch=70,
        expert_id="109636",
    )
    train_model.train()


if __name__ == "__main__":
    main()
