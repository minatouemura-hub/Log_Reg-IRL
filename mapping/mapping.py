import os
import sys

from sklearn.metrics import r2_score  # noqa

from utils import Map_StateValue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logreg.models import Irl_Net  # noqa


def main(usr_id):
    mapping = Map_StateValue(
        weight_path="/Users/uemuraminato/Desktop/IRL/weight_vec/final_model_weights.pt",
        manneri_path=f"/Users/uemuraminato/Desktop/book_script/manneri_vec/manneri_{usr_id}.csv",
        usr_id=usr_id,
        TimeThreshold=5,
        DiffThreshold=0.02,  # どこのタイミングまでを切るか
    )
    mapping.excute(
        state_path=f"/Users/uemuraminato/Desktop/book_script/analysis/preproceed/state_tras_of_{usr_id}.npy"  # noqa
    )


if __name__ == "__main__":
    usr_id = "1087619"
    main(usr_id=usr_id)
