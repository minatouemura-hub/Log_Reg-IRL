import os
import sys

from utils import Map_StateValue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logreg.models import Irl_Net  # noqa


def main(usr_id):
    mapping = Map_StateValue(
        weight_path="/Users/uemuraminato/Desktop/IRL/weight_vec/final_model_weights.pt",
        usr_id=usr_id,
        TimeThreshold=50,
        DiffThreshold=0.02,  # どこのタイミングまでを切るか
    )
    mapping.excute(
        state_path=f"/Users/uemuraminato/Desktop/book_script/analysis/preprocessed/state_tras_of_{usr_id}.npy"  # noqa
    )


if __name__ == "__main__":
    usr_id = "109636"
    main(usr_id=usr_id)
