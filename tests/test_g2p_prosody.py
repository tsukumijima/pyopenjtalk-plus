"""Haqumei 互換の韻律記号付き音素 API (g2p_prosody / g2p_mapping_prosody) のテスト。"""

import pytest

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


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    ("text", "expected_prosody", "expected_accents"),
    [
        ("12", "^ju[uni$", [3]),
        ("13", "^ju]usaN$", [1]),
        ("15", "^ju]ugo$", [1]),
        ("50", "^go[ju]u$", [2]),
        ("60", "^ro[kuju]u$", [3]),
        ("70", "^na[na]juu$", [2]),
        ("80", "^ha[chiju]u$", [3]),
        ("3時", "^sa]Nji$", [1]),
        ("10時", "^ju]uji$", [1]),
        ("11時", "^ju[uichi]ji$", [4]),
        ("12時", "^ju[uni]ji$", [3]),
        ("13時", "^ju[usa]Nji$", [3]),
        ("14時", "^ju]u#yo]ji$", [1, 1]),
        ("15時", "^ju]u#go]ji$", [1, 1]),
        ("16時", "^ju[uroku]ji$", [4]),
        ("17時", "^ju[ushIchi]ji$", [4]),
        ("18時", "^ju[uhachi]ji$", [4]),
        ("19時", "^ju]u#ku]ji$", [1, 1]),
        ("20時", "^ni[ju]uji$", [2]),
        ("21時", "^ni]juu#i[chi]ji$", [1, 2]),
        ("22時", "^ni]juu#ni]ji$", [1, 1]),
        ("23時", "^ni]juu#sa]Nji$", [1, 1]),
        ("24時", "^ni]juu#yo]ji$", [1, 1]),
        ("2人", "^fU[tari$", [3]),
        ("12人", "^ju[uni]niN$", [3]),
        ("13人", "^ju[usa]NniN$", [3]),
        ("14人", "^ju]u#yo[ni]N$", [1, 2]),
        ("19人", "^ju[ukyu]uniN$", [3]),
        ("12分", "^ju[uni]fuN$", [3]),
        ("13分", "^ju[usa]NpuN$", [3]),
        ("15分", "^ju]u#go]fuN$", [1, 1]),
        ("20歳", "^ni[ju]clsai$", [2]),
        ("11回", "^ju[uiclka]i$", [5]),
        ("12回", "^ju[unika]i$", [4]),
        ("15回", "^ju]u#go[ka]i$", [1, 2]),
        ("20回", "^ni[ju]clkai$", [2]),
        ("34回", "^sa]Njuu#yo[Nka]i$", [1, 3]),
        ("100回", "^hya[clka]i$", [3]),
        ("1000回", "^se]Nkai$", [1]),
        ("10000回", "^i[chimaNkai$", [0]),
        ("12円", "^ju[uni]eN$", [3]),
        ("100円", "^hya[kueN$", [0]),
        ("1000円", "^se[NeN$", [0]),
        ("10000円", "^i[chimaNeN$", [0]),
        ("12月", "^ju[unigatsu$", [5]),
        ("12日", "^ju[uninichi$", [5]),
        ("14日", "^ju]u#yo[clka$", [1, 0]),
        ("15日", "^ju]u#go]nichi$", [1, 1]),
        ("12個", "^ju[uni]ko$", [3]),
        ("13個", "^ju[usa]Nko$", [3]),
        ("17個", "^ju[unana]ko$", [4]),
        ("12本", "^ju[uni]hoN$", [3]),
        ("15本", "^ju]u#go[hoN$", [1, 0]),
        ("12枚", "^ju[uni]mai$", [3]),
        ("15枚", "^ju]u#go[mai$", [1, 0]),
        ("15台", "^ju]u#go[dai$", [1, 0]),
        ("9年", "^kyu]uneN$", [1]),
        ("19年", "^ju[ukyu]uneN$", [3]),
        ("15球", "^ju[ugokyuu$", [0]),
        ("15週", "^ju[ugoshuu$", [0]),
        ("15戦", "^ju[ugoseN$", [0]),
        ("15層", "^ju[ugosoo$", [0]),
        ("15倍", "^ju[ugobai$", [0]),
        ("15場所", "^ju[ugobasho$", [0]),
        ("15機種", "^ju[ugoki]shu$", [4]),
        ("15地区", "^ju[ugochi]ku$", [4]),
        ("12メートル", "^ju[unime]etoru$", [4]),
        ("15メートル", "^ju[ugome]etoru$", [4]),
        ("12階", "^ju[unikai$", [0]),
        ("15階", "^ju[ugokai$", [0]),
        ("12時間", "^ju[uniji]kaN$", [4]),
        ("15時間", "^ju[ugoji]kaN$", [4]),
        ("14日目", "^ju[uyoclkame$", [6]),
        ("15日目", "^ju[ugonichime$", [6]),
    ],
)
def test_g2p_prosody_numerals_follow_counter_accent_phrases(
    text: str, expected_prosody: str, expected_accents: list[int], use_vanilla: bool
) -> None:
    """
    NHK アクセント辞典の付録の数詞と助数詞の表に合わせ、数詞のアクセント句の区切りと核を確認する。
    「12時」は1句、「22時」は2句とし、「14時」「15分」は最初に掲載されている2句の形を使う。
    「時間」のような長い助数詞は「15時間」も1句になり、1桁の「3時」「2人」の核も保つ。
    「九年」「十九人」などは許容形の「キュー」という読みを保ち、その読みに対応する区切りと核を使う。
    独自の後処理を無効にした場合も、C の数詞処理だけで同じ区切りと核になることを確認する。
    """

    prosody = pyopenjtalk.g2p_prosody(text, use_vanilla=use_vanilla)
    assert prosody.count("#") + 1 == len(expected_accents)
    assert "".join(prosody) == expected_prosody

    # 文末の尾高型は韻律記号に下降が現れないため、NJD のアクセント核も直接確認する
    features = pyopenjtalk.run_frontend(text, use_vanilla=use_vanilla)
    assert [
        feature["acc"]
        for index, feature in enumerate(features)
        if index == 0 or feature["chain_flag"] != 1
    ] == expected_accents


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
