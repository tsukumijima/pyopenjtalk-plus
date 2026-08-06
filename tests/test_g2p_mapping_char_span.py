"""`g2p_mapping()` が呼び出し元入力上の char_span を返す契約を検査する。"""

from __future__ import annotations

import pyopenjtalk


def test_g2p_mapping_char_span_covers_halfwidth_digits() -> None:
    """算用数字入力の char_span が呼び出し元座標で数字ブロックを覆う。"""

    mapping = pyopenjtalk.g2p_mapping("10分")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("十", (0, 2)),
        ("分", (2, 3)),
    ]


def test_g2p_mapping_char_span_shares_digit_block_for_kanji_expansion() -> None:
    """漢数字展開では同一数字ブロックの全 NJD ノードが同じ char_span を共有する。"""

    mapping = pyopenjtalk.g2p_mapping("123円")
    digit_spans = [entry["char_span"] for entry in mapping if entry["surface"] != "円"]
    assert digit_spans == [(0, 3), (0, 3), (0, 3), (0, 3)]
    assert mapping[-1]["char_span"] == (3, 4)


def test_g2p_mapping_char_span_projects_ascii_digit_in_chapter_title() -> None:
    """MeCab 側の全角数字を呼び出し元入力の算用数字位置へ射影する。"""

    mapping = pyopenjtalk.g2p_mapping("第1章")
    assert [entry["char_span"] for entry in mapping] == [(0, 1), (1, 2), (2, 3)]


def test_g2p_mapping_char_span_projects_halfwidth_latin_letter() -> None:
    """text2mecab 互換正規化で表記が変わっても呼び出し元座標へ戻す。"""

    mapping = pyopenjtalk.g2p_mapping("英字g")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("英字", (0, 2)),
        ("ｇ", (2, 3)),
    ]


def test_g2p_mapping_char_span_preserves_repeated_normalized_characters() -> None:
    """同じ正規化結果が反復しても、前方の入力文字を欠落させない。"""

    cases = [
        ("AＡA", [(0, 1), (1, 2), (2, 3)]),
        ("ｳﾞヴｳﾞ", [(0, 5)]),
    ]
    for text, expected_spans in cases:
        mapping = pyopenjtalk.g2p_mapping(text)
        assert [entry["char_span"] for entry in mapping] == expected_spans


def test_g2p_mapping_char_span_projects_nfkc_expansion() -> None:
    """NFKC で1文字から展開された表層全体を元の1文字へ対応させる。"""

    mapping = pyopenjtalk.g2p_mapping("㍑", normalize_mode="NFKC")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("リットル", (0, 1)),
    ]


def test_g2p_mapping_char_span_covers_odori_combination_target() -> None:
    """踊り字と後続形態素の結合結果が、消費した入力範囲全体を指す。"""

    mapping = pyopenjtalk.g2p_mapping("学生々活")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("学生", (0, 2)),
        ("生活", (2, 4)),
    ]


def test_g2p_mapping_char_span_covers_multiple_morphs_without_gap() -> None:
    """複数形態素の char_span が入力を隙間なく覆う。"""

    mapping = pyopenjtalk.g2p_mapping("猫と犬")
    assert [entry["char_span"] for entry in mapping] == [(0, 1), (1, 2), (2, 3)]


def test_g2p_mapping_char_span_covers_digit_person_compound() -> None:
    """算用数字+人が NJD で二人へ縮約しても char_span が入力を覆う。"""

    mapping = pyopenjtalk.g2p_mapping("の2人が")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("の", (0, 1)),
        ("二人", (1, 3)),
        ("が", (3, 4)),
    ]


def test_g2p_mapping_char_span_separates_digit_and_oku() -> None:
    """3億 のように digit 変換後に別語が続く場合、億を digit ブロックへ吸収しない。"""

    mapping = pyopenjtalk.g2p_mapping("3億円")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("三", (0, 1)),
        ("億", (1, 2)),
        ("円", (2, 3)),
    ]


def test_g2p_mapping_char_span_covers_digit_day_compound() -> None:
    """1日 が NJD で一日へ縮約しても char_span が算用数字位置を覆う。"""

    mapping = pyopenjtalk.g2p_mapping("1日")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("一日", (0, 2)),
    ]
