"""`g2p_mapping()` が呼び出し元入力上の char_span を返す契約を検査する。"""

from __future__ import annotations

from typing import Any

import pytest

import pyopenjtalk


def _assert_char_spans_cover_text_once(
    text: str,
    mapping: list[pyopenjtalk.SurfacePhonemeMapping],
) -> None:
    """
    (0, 0) の無視対象を除き、char_span が本文を一度ずつ覆うことを検査する。

    Args:
        text (str): `g2p_mapping()` へ渡した入力文
        mapping (list[SurfacePhonemeMapping]): `g2p_mapping()` の戻り値
    """

    expected_start = 0
    for entry in mapping:
        if entry["char_span"] == (0, 0):
            continue
        assert entry["char_span"][0] == expected_start
        assert entry["char_span"][1] > expected_start
        expected_start = entry["char_span"][1]
    assert expected_start == len(text)


def test_g2p_mapping_rejects_broken_char_span_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入力を一度ずつ覆わない char_span を公開 API の出口で拒否する。"""

    original_make_phoneme_mapping = pyopenjtalk.make_phoneme_mapping

    def return_mapping_with_gap(
        *args: Any,
        **kwargs: Any,
    ) -> list[pyopenjtalk.SurfacePhonemeMapping]:
        """実際のマッピングから先頭座標だけを壊した値を返す。"""

        mapping = original_make_phoneme_mapping(*args, **kwargs)
        mapping[0]["char_span"] = (1, 1)
        return mapping

    monkeypatch.setattr(pyopenjtalk, "make_phoneme_mapping", return_mapping_with_gap)

    with pytest.raises(ValueError, match="must cover caller text exactly once"):
        pyopenjtalk.g2p_mapping("猫")


def test_g2p_mapping_char_span_covers_halfwidth_digits() -> None:
    """算用数字入力の char_span が呼び出し元座標で数字ブロックを覆う。"""

    mapping = pyopenjtalk.g2p_mapping("10分")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("十", (0, 2)),
        ("分", (2, 3)),
    ]


def test_g2p_mapping_char_span_separates_digit_block_for_kanji_expansion() -> None:
    """漢数字展開では入力数字と挿入された位取りノードを区別する。"""

    mapping = pyopenjtalk.g2p_mapping("123円")
    digit_spans = [entry["char_span"] for entry in mapping if entry["surface"] != "円"]
    assert digit_spans == [(0, 1), (1, 2), (0, 0), (2, 3)]
    assert mapping[-1]["char_span"] == (3, 4)
    _assert_char_spans_cover_text_once("123円", mapping)


def test_g2p_mapping_char_span_projects_ascii_digit_in_chapter_title() -> None:
    """MeCab 側の全角数字を呼び出し元入力の算用数字位置へ射影する。"""

    mapping = pyopenjtalk.g2p_mapping("第1章")
    assert [entry["char_span"] for entry in mapping] == [(0, 1), (1, 2), (2, 3)]


def test_g2p_mapping_char_span_projects_halfwidth_latin_letter() -> None:
    """text2mecab() 互換の正規化で表記が変わっても、呼び出し元座標へ戻す。"""

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


def test_g2p_mapping_char_span_separates_expanded_digits_and_cho() -> None:
    """複数桁の数字展開後に続く兆を数字ブロックへ吸収しない。"""

    mapping = pyopenjtalk.g2p_mapping("１００兆円")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("百", (0, 3)),
        ("兆", (3, 4)),
        ("円", (4, 5)),
    ]


def test_g2p_mapping_char_span_keeps_zero_node_inside_digit_range() -> None:
    """NJD がゼロだけ全角数字で残しても、後続助詞の位置をずらさない。"""

    mapping = pyopenjtalk.g2p_mapping("１－２０の")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("一", (0, 1)),
        ("－", (1, 2)),
        ("二", (2, 3)),
        ("０", (3, 4)),
        ("の", (4, 5)),
    ]
    _assert_char_spans_cover_text_once("１－２０の", mapping)


def test_g2p_mapping_char_span_ignores_inserted_kanji_place_node() -> None:
    """連続する漢数字へ NJD が挿入した位取りノードをゼロ幅で返す。"""

    text = "二八パーセント（二千二年第2回"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("二", (0, 1)),
        ("十", (0, 0)),
        ("八", (1, 2)),
        ("パーセント", (2, 7)),
        ("（", (7, 8)),
        ("二", (8, 9)),
        ("千", (9, 10)),
        ("二", (10, 11)),
        ("年", (11, 12)),
        ("第", (12, 13)),
        ("二", (13, 14)),
        ("回", (14, 15)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_aligns_repeated_kanji_digits() -> None:
    """同じ漢数字を含む位取り展開でも後続語まで入力位置を維持する。"""

    text = "一一七一円"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("千", (0, 1)),
        ("百", (1, 2)),
        ("七", (2, 3)),
        ("十", (0, 0)),
        ("一", (3, 4)),
        ("円", (4, 5)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_aligns_digits_separated_by_spaces() -> None:
    """空白を除いて位取り展開された数字でも空白と各数字の位置を保つ。"""

    text = "１ ２ ３円"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("百", (0, 1)),
        ("　", (1, 2)),
        ("二", (2, 3)),
        ("十", (0, 0)),
        ("　", (3, 4)),
        ("三", (4, 5)),
        ("円", (5, 6)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_separates_compound_word_fragments() -> None:
    """連語辞書の1形態素を複数ノードへ分割しても部分表層の範囲を重複させない。"""

    text = "ありがとうございました"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("ありがとう", (0, 5)),
        ("ございました", (5, 11)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_covers_digit_day_compound() -> None:
    """1日 が NJD で一日へ縮約しても char_span が算用数字位置を覆う。"""

    mapping = pyopenjtalk.g2p_mapping("1日")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("一日", (0, 2)),
    ]


def test_g2p_mapping_char_span_covers_multiple_digit_day_compound() -> None:
    """10日 が十日へ縮約しても、後続形態素を含めて呼び出し元位置を維持する。"""

    mapping = pyopenjtalk.g2p_mapping("１０日経ちました。")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("十日", (0, 3)),
        ("経ち", (3, 5)),
        ("まし", (5, 7)),
        ("た", (7, 8)),
        ("。", (8, 9)),
    ]


def test_g2p_mapping_char_span_reserves_last_digit_for_split_day_compound() -> None:
    """24日 の分割特殊読みでも最後の数字を前段と重複させない。"""

    text = "8月24日間後"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("８月", (0, 2)),
        ("二十", (2, 3)),
        ("四日間", (3, 6)),
        ("後", (6, 7)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_aligns_katakana_number_morph() -> None:
    """NJD が数として変換するカタカナのニも数詞ブロック内で消費する。"""

    text = "1ニキロ後"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("十", (0, 1)),
        ("二", (1, 2)),
        ("キロ", (2, 4)),
        ("後", (4, 5)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_ignores_space_inside_absorbed_long_vowel() -> None:
    """JPCommon が空白越しの長音を吸収しても内部空白を重複範囲にしない。"""

    text = "次 ー"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("次ー", (0, 3)),
        ("　", (0, 0)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_ignores_space_inside_chained_kana_filler() -> None:
    """NJD が空白越しの仮名フィラーを連結しても内部空白を重複範囲にしない。"""

    text = "ゔ ぁ"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("ゔぁ", (0, 3)),
        ("　", (0, 0)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)
