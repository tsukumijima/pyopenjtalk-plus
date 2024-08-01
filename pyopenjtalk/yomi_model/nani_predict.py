import os

# import pickle
from typing import List, Union

# import pandas as pd
from ..types import NJDFeature

X_COLS = ["pos", "pos_group1", "pos_group2", "pron", "ctype", "cform"]
model_dir = os.path.dirname(__file__)

# with open(os.path.join(model_dir, "nani_enc.pickle"), "rb") as f:
#     enc = pickle.load(f)

# with open(os.path.join(model_dir, "nani_model.pickle"), "rb") as f:
#     model = pickle.load(f)


def predict(input_njd: List[Union[NJDFeature, None]]) -> int:
    if input_njd == [None]:
        return 0
    else:
        # input_df = pd.DataFrame(input_njd)[X_COLS]
        # input = enc.transform(input_df)
        # return int(model.predict(input)[0])

        # 現状学習済みモデルが scikit-learn 0.24.2 でしか動かないため当面常に 0 を返す
        return 0
