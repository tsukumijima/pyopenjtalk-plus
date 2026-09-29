"""Sudachi 読み補正の接尾辞用法と tsqyomi 有効時の出力を検証する。"""

import pytest

import pyopenjtalk


@pytest.mark.parametrize(
    ("text", "expected_reading", "expected_reading_without_sudachi"),
    (
        ("支払時", "シハライジ", "シハライジ"),
        ("開催時", "カイサイジ", "カイサイジ"),
        ("その時", "ソノトキ", "ソノトキ"),
        ("困った時", "コマッタトキ", "コマッタトキ"),
        ("時をかける少女", "トキヲカケルショウジョ", "トキヲカケルショウジョ"),
    ),
)
def test_sudachi_kanji_yomi_preserves_suffix_reading(
    text: str,
    expected_reading: str,
    expected_reading_without_sudachi: str,
) -> None:
    """接尾辞としての本来の読みを保ちつつ、一般名詞の「時」に対してのみ Sudachi による読み補正が適用されることを確認する。"""

    # 発音列を連結し、形態素境界に依存せず製品へ渡る読みを確認する
    features = pyopenjtalk.run_frontend(text, use_sudachi_kanji_yomi=True)
    actual_reading = "".join(feature["read"] for feature in features)

    assert actual_reading == expected_reading

    # 補正なしの出力も固定し、Sudachi が変更する範囲を症例ごとに見える状態にする
    features_without_sudachi = pyopenjtalk.run_frontend(text, use_sudachi_kanji_yomi=False)
    actual_reading_without_sudachi = "".join(
        feature["read"] for feature in features_without_sudachi
    )

    assert actual_reading_without_sudachi == expected_reading_without_sudachi
