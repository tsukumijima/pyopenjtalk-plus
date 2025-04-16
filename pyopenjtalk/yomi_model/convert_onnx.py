# オリジナルの学習済みモデルが scikit-learn 0.24.2 でしか動かないことを踏まえ、ONNX 形式に変換するためのコード
# 以下のコードは Python 3.9 / scikit-learn==0.24.2 skl2onnx==1.16.0 scipy<1.9 でのみ動作を確認している

import pickle
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from .nani_predict import X_COLS


# モデルとエンコーダーをロード
with open(Path(__file__).parent / "nani_model.pickle", "rb") as f:
    model = pickle.load(f)
with open(Path(__file__).parent / "nani_enc.pickle", "rb") as f:
    enc = pickle.load(f)

# OneHotEncoder を ONNX に変換
initial_type_enc = [("input", StringTensorType([None, len(X_COLS)]))]
onnx_enc = convert_sklearn(enc, initial_types=initial_type_enc)

# OneHotEncoder の ONNX モデルを保存
with open(Path(__file__).parent / "nani_enc.onnx", "wb") as f:
    f.write(onnx_enc.SerializeToString())

# OneHotEncoder の特徴数を取得
if hasattr(enc, "get_feature_names"):
    n_features = len(enc.get_feature_names())
else:
    n_features = sum(len(categories) for categories in enc.categories_)

# RandomForestClassifier を ONNX に変換
initial_type_model = [("input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(model, initial_types=initial_type_model)

# RandomForestClassifier の ONNX モデルを保存
with open(Path(__file__).parent / "nani_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
