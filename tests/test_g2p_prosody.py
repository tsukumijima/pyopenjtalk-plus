"""Haqumei 互換の韻律記号付き音素 API (g2p_prosody / g2p_mapping_prosody) のテスト。"""

import pyopenjtalk


def test_g2p_mapping_prosody_marks_pause_and_preserves_detailed_contract() -> None:
    """g2p_mapping_prosody において、文末の句点「。」など HTS ラベル上では pau として現れない記号であっても、入力テキスト上の全句読点と文字位置 (char_span) が正しく保持されることを確認する。"""

    mapping = pyopenjtalk.g2p_mapping_prosody("あ、い。")

    assert [entry["surface"] for entry in mapping] == ["あ", "、", "い", "。"]
    assert [entry["char_span"] for entry in mapping] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert mapping[0]["phonemes"] == [{"kind": "Phoneme", "phoneme": "a", "pitch": "High"}]
    assert mapping[1]["phonemes"] == [{"kind": "Pause"}]
    assert mapping[2]["phonemes"] == [{"kind": "Phoneme", "phoneme": "i", "pitch": "High"}]
    assert mapping[3]["phonemes"] == [{"kind": "Pause"}]


def test_g2p_mapping_prosody_respects_non_pause_symbol_callback() -> None:
    """is_non_pause_symbol コールバックで括弧をポーズとして扱うように設定した場合でも、ラベル照合やマッピングが正しく動作することを確認する。"""

    mapping = pyopenjtalk.g2p_mapping_prosody("「あ」", is_non_pause_symbol=lambda _: False)

    assert [entry["phonemes"] for entry in mapping] == [
        [{"kind": "Pause"}],
        [{"kind": "Phoneme", "phoneme": "a", "pitch": "High"}],
        [{"kind": "Pause"}],
    ]
    assert pyopenjtalk.g2p_prosody("「あ」", is_non_pause_symbol=lambda _: False) == [
        "^",
        "_",
        "a",
        "_",
        "$",
    ]


def test_g2p_mapping_prosody_preserves_unknown_and_space() -> None:
    """HTS ラベルには直接現れない未知語（unk）や全角空白（sp）が、g2p_mapping_prosody においても詳細マッピング API と同様に正しい表層形と文字位置で出力されることを確認する。"""

    mapping = pyopenjtalk.g2p_mapping_prosody("𰻞𰻞　本")

    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("𰻞𰻞", (0, 2)),
        ("　", (2, 3)),
        ("本", (3, 4)),
    ]
    assert mapping[0]["is_unknown"] is True
    assert mapping[0]["phonemes"] == [{"kind": "Phoneme", "phoneme": "unk", "pitch": None}]
    assert mapping[1]["is_ignored"] is True
    assert mapping[1]["phonemes"] == [{"kind": "Phoneme", "phoneme": "sp", "pitch": None}]


def test_g2p_prosody_formats_pitch_and_accent_boundaries() -> None:
    """g2p_prosody において、BOS (^)、EOS ($)、アクセント句境界 (#) や、3種類のピッチ表記フォーマット (Default, Prefix, Numeric) がそれぞれ正しく出力されることを確認する。"""

    assert pyopenjtalk.g2p_prosody("青い空") == [
        "^",
        "a",
        "[",
        "o",
        "]",
        "i",
        "#",
        "s",
        "o",
        "]",
        "r",
        "a",
        "$",
    ]
    assert pyopenjtalk.g2p_prosody("青い空", format="Prefix") == [
        "^",
        "L_a",
        "H_o",
        "L_i",
        "#",
        "H_s",
        "H_o",
        "L_r",
        "L_a",
        "$",
    ]
    assert pyopenjtalk.g2p_prosody("青い空", format="Numeric") == [
        "^",
        "a:0",
        "o:1",
        "i:0",
        "#",
        "s:1",
        "o:1",
        "r:0",
        "a:0",
        "$",
    ]


def test_g2p_prosody_keeps_accent_boundary_across_space() -> None:
    """アクセント句の間に空白があっても、空白のない場合と同じ位置にアクセント句の境界 (#) が出ることを確認する。"""

    assert pyopenjtalk.g2p_prosody("青い 空") == [
        "^",
        "a",
        "[",
        "o",
        "]",
        "i",
        "#",
        "sp",
        "s",
        "o",
        "]",
        "r",
        "a",
        "$",
    ]


def test_g2p_prosody_marks_questions_exclamations_and_unknown_words() -> None:
    """g2p_prosody において、疑問符 (?)、感嘆符 (!)、未知語 ({unk}) の記号がそれぞれ正しく出力されることを確認する。"""

    question = pyopenjtalk.g2p_prosody("これはペンですか？")
    exclamation = pyopenjtalk.g2p_prosody("すごい！")
    unknown = pyopenjtalk.g2p_prosody("𰻞𰻞麺")

    assert question[-2:] == ["?", "$"]
    assert exclamation[-2:] == ["!", "$"]
    assert unknown[:4] == ["^", "{", "unk", "}"]


def test_g2p_prosody_returns_empty_list_for_empty_text() -> None:
    """空文字列を入力した場合、空リストが返されることを確認する。"""

    assert pyopenjtalk.g2p_prosody("") == []
