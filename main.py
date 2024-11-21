from develop.dataloader import BookDataset, custom_collate_fun  # noqa: F401 E402
from trainer import Train_Irl_model
from utils import certificate_diff


def main():
    two_score_dict = {"M_exepert_score": 0, "F_expert_score": 0}
    for flag in ["M", "F"]:
        train_model = Train_Irl_model(num_epoch=50, group=flag, expert_id="282967")
        score_list = train_model.train()
        two_score_dict[f"{flag}_expert_score"] = score_list
    certificate_diff(two_score_dict)


if __name__ == "__main__":
    main()
