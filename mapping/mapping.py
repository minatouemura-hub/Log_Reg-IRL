import os
import sys

from utils import Map_StateValue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logreg.models import Irl_Net  # noqa


def main(usr_id):
    mapping = Map_StateValue(
        weight_path="/Users/uemuraminato/Desktop/IRL/weight_vec/final_model_weights.pt",
        pickel_path="/Users/uemuraminato/Desktop/book_script/filtered_groups.pkl",
        usr_id=usr_id,
        TimeThreshold=10,
        DiffThreshold=0.02,  # どこのタイミングまでを切るか
    )
    mapping.excute(
        state_path=f"/Users/uemuraminato/Desktop/book_script/analysis/preproceed/state_tras_of_{usr_id}.npy"  # noqa
    )
    mapping.map_correlation()


if __name__ == "__main__":
    usr_id = "1287400"
    main(usr_id=usr_id)
