import os
from typing import List, Union

import numpy as np
from onnxruntime import InferenceSession

from ..types import NJDFeature

X_COLS = ["pos", "pos_group1", "pos_group2", "pron", "ctype", "cform"]
model_dir = os.path.dirname(__file__)

# ONNX モデルをロード
# 非常に軽量なモデルのため、import 時に ONNX モデルをロードするオーバーヘッドはほとんどない
enc_session = InferenceSession(
    os.path.join(model_dir, "nani_enc.onnx"),
    providers=["CPUExecutionProvider"],
)
model_session = InferenceSession(
    os.path.join(model_dir, "nani_model.onnx"),
    providers=["CPUExecutionProvider"],
)


def predict(input_njd: List[Union[NJDFeature, None]]) -> int:
    if input_njd == [None]:
        return 0
    else:
        # 入力データを準備
        input_data = np.array(
            [[njd[col] for col in X_COLS] for njd in input_njd if njd is not None]
        )

        # OneHotEncoder で変換
        enc_input = {"input": input_data}
        enc_output = enc_session.run(None, enc_input)

        # RandomForestClassifier で予測
        model_input = {"input": enc_output[0].astype(np.float32)}
        model_output = model_session.run(None, model_input)

        return int(model_output[0][0])
