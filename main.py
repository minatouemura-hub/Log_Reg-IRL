import json

from tqdm import tqdm

from develop.dataloader import BookDataset, custom_collate_fun  # noqa: F401 E402
from trainer import Train_Irl_model
from utils import extract_numbers_from_strings  # noqa
from utils import Gender_Rate, certificate_diff, choose_expert_data, plot_GenderRate


def main():
    for target in ["M", "F"]:
        two_score_dict = {"D_expert_score": [], "S_expert_score": []}
        num = 30
        expert_numbers = choose_expert_data(
            directory_path=f"/Users/uemuraminato/Desktop/book_script/vec/preproceed/{target}",
            num=num,
        )
        user_list = {"user_id": [], "gr": []}
        for expert_id in tqdm(expert_numbers, desc="experts_train", leave=True):
            S_score = 0
            D_score = 0
            for flag in ["M", "F"]:
                train_model = Train_Irl_model(num_epoch=10, group=flag, expert_id=str(expert_id))
                sum_score = train_model.train()
                if flag == target:
                    two_score_dict["S_expert_score"].append(sum_score)
                    S_score = sum_score
                else:
                    two_score_dict["D_expert_score"].append(sum_score)
                    D_score = sum_score
            gr = Gender_Rate(
                s_score=S_score, d_score=D_score, target_gender=target
            )  # [male,female] => [-1,1]
            user_list["gr"].append(float(gr))
            user_list["user_id"].append(str(expert_id))
        with open(f"plot/{target}_gr_list.json", mode="w", encoding="utf-8") as file:
            json.dump(user_list, file, ensure_ascii=False, indent=4)
        certificate_diff(two_score_dict, how="ttest", target=target)
        accuracy = plot_GenderRate(num, two_score_dict=two_score_dict, target_gender=target)
        print("判別精度：", accuracy)


if __name__ == "__main__":
    main()
