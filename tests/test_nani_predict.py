from typing import Any, cast

import numpy as np
import pytest

from pyopenjtalk.types import NJDFeature
from pyopenjtalk.yomi_model import nani_predict


@pytest.mark.parametrize(
    ("pronunciation", "pos_group1", "pos_group2", "expected"),
    [
        ("ヲ", "格助詞", "一般", 0),
        ("ノ", "連体化", "*", 1),
        ("デス’", "*", "*", 1),
        ("ダロ", "*", "*", 1),
    ],
)
def test_nani_predict_uses_explicit_binary_probabilities(
    pronunciation: str,
    pos_group1: str,
    pos_group2: str,
    expected: int,
) -> None:
    """旧 ORT と同じ nani_predict の判定を明示的な二クラス確率から復元できる。"""

    pytest.importorskip("onnxruntime")

    # nani_predict が参照する6フィールドだけを用意し、文法規則を経由せずモデル単体を検証
    feature = cast(
        NJDFeature,
        {
            "pos": "助詞",
            "pos_group1": pos_group1,
            "pos_group2": pos_group2,
            "pron": pronunciation,
            "ctype": "*",
            "cform": "*",
        },
    )

    assert nani_predict.predict([feature]) == expected


def test_nani_model_exposes_probability_tensor() -> None:
    """nani_model がクラスラベルに依存しない二クラス確率テンソルを返す。"""

    pytest.importorskip("onnxruntime")
    if nani_predict.enc_session is None or nani_predict.model_session is None:
        pytest.fail("ONNX Runtime sessions were not initialized")

    # ORT 1.26 で挙動が変化した「何を」の特徴量をモデルへ直接入力
    input_data = np.array([["助詞", "格助詞", "一般", "ヲ", "*", "*"]])
    encoded_features = np.asarray(
        nani_predict.enc_session.run(None, {"input": input_data})[0],
        dtype=np.float32,
    )
    model_outputs = nani_predict.model_session.run(None, {"input": encoded_features})
    probability_array = np.asarray(cast(Any, model_outputs[0]), dtype=np.float32)

    assert len(model_outputs) == 1
    np.testing.assert_allclose(
        probability_array,
        np.array([[0.6357111, 0.3642888]], dtype=np.float32),
        atol=1e-6,
    )
