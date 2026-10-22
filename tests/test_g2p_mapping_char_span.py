"""`g2p_mapping()` が呼び出し元入力上の char_span を返す契約を検査する。"""

from __future__ import annotations

from typing import Any

import pytest

import pyopenjtalk
from pyopenjtalk.types import SurfaceProsodyMapping


def _assert_char_spans_cover_text_once(
    text: str,
    mapping: list[pyopenjtalk.SurfacePhonemeMapping] | list[SurfaceProsodyMapping],
) -> None:
    """
    (0, 0) の無視対象を除き、char_span が本文を一度ずつ覆うことを検査する。

    Args:
        text (str): `g2p_mapping()` へ渡した入力文
        mapping (list[SurfacePhonemeMapping] | list[SurfaceProsodyMapping]): 音素または韻律のマッピング
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


def test_g2p_mapping_char_span_skips_standalone_halfwidth_voiced_marks() -> None:
    """
    「ﾀ━(ﾟ∀∀ﾟ)」のように前の仮名と結合しない半角の「ﾟ」「ﾞ」は、text2mecab() が除去するため、対応する要素が存在しない。
    以前は検査がこの隙間を本文の欠落とみなし、ValueError が発生していた。
    例外が発生せず、除去された文字より後ろの要素が元の入力上の位置を返すことを確認する。
    """

    text = "ﾀ━(ﾟ∀∀ﾟ)ﾞ"
    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
            ("タ", (0, 1)),
            ("━", (1, 2)),
            ("（", (2, 3)),
            ("∀", (4, 5)),
            ("∀", (5, 6)),
            ("）", (7, 8)),
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


@pytest.mark.parametrize("selector", ["\ufe00", "\ufe0f", "\U000e0100", "\U000e01ef"])
@pytest.mark.parametrize("mode", ["None", "NFC", "NFKC"])
@pytest.mark.parametrize("text", ["葛{}飾区へ行く。", "神{}戸市に住む。", "﨑{}山さん。"])
def test_variation_selector_keeps_word_and_input_range(
    text: str, selector: str, mode: pyopenjtalk.NormalizeMode
) -> None:
    """
    「葛飾区」「神戸市」の漢字に異体字セレクターが付いても、漢字の間で語を分けず、セレクターのない文と同じ読みと韻律になることを確認する。
    異体字の「﨑」と Unicode 正規化の組み合わせでも、セレクターは直前の漢字の文字範囲へ含める。
    """

    source = text.format(selector)
    reference = text.format("")
    assert pyopenjtalk.g2p(source, normalize_mode=mode) == pyopenjtalk.g2p(
        reference, normalize_mode=mode
    )
    assert pyopenjtalk.g2p_prosody(source, normalize_mode=mode) == pyopenjtalk.g2p_prosody(
        reference, normalize_mode=mode
    )
    for mapping in (
        pyopenjtalk.g2p_mapping(source, normalize_mode=mode),
        pyopenjtalk.g2p_mapping_prosody(source, normalize_mode=mode),
    ):
        _assert_char_spans_cover_text_once(source, mapping)
        assert any(
            a <= source.index(selector) - 1 and b > source.index(selector)
            for a, b in (entry["char_span"] for entry in mapping)
        )


@pytest.mark.parametrize(
    "text", ["\ufe0f", "\ufe0f神戸", "\U000e0100神戸", "か\ufe0f\ufe0fな", "😀\ufe0fです"]
)
def test_unattached_or_repeated_selectors_keep_input_coverage(text: str) -> None:
    """
    文頭の異体字セレクターには結合先がないので独立した記号として残し、漢字や絵文字に付いたセレクターは前の文字へ含めることを確認する。
    セレクターだけの入力や連続したセレクターでも文字範囲を欠かさず、マッピングが例外にならないことを確かめる。
    """

    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        _assert_char_spans_cover_text_once(text, mapping)


def test_g2p_mapping_char_span_projects_nfkc_expansion() -> None:
    """NFKC で1文字から展開された表層全体を元の1文字へ対応させる。"""

    mapping = pyopenjtalk.g2p_mapping("㍑", normalize_mode="NFKC")
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("リットル", (0, 1)),
    ]


@pytest.mark.parametrize("text", ["㎞", "㎞㎞", "㎞学校", "㍿㎞", "…学校"])
def test_nfkc_expansion_across_morphs_owns_original_character_once(text: str) -> None:
    """
    「㎞」を NFKC で「km」へ展開し、別の形態素に分かれても、元の1文字の範囲を先頭の要素だけへ割り当てることを確認する。
    同じ入力文字の範囲が重なるとマッピングと韻律 API が例外になるため、残る要素は音素を保ったゼロ幅の要素にする。
    """

    mapping = pyopenjtalk.g2p_mapping(text, normalize_mode="NFKC")
    prosody_mapping = pyopenjtalk.g2p_mapping_prosody(text, normalize_mode="NFKC")
    _assert_char_spans_cover_text_once(text, mapping)
    _assert_char_spans_cover_text_once(text, prosody_mapping)
    if text == "㎞":
        assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
            ("ｋ", (0, 1)),
            ("ｍ", (0, 0)),
        ]
    assert [p for word in mapping for p in word["phonemes"] if p not in ("sp", "pau")] == [
        p for p in pyopenjtalk.g2p(text, join=False, normalize_mode="NFKC") if p != "pau"
    ]
    assert pyopenjtalk.g2p_prosody(text, normalize_mode="NFKC")


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
    """空白で独立した数字でも、空白と各数字の位置を保つ。"""

    text = "１ ２ ３円"
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("一", (0, 1)),
        ("　", (1, 2)),
        ("二", (2, 3)),
        ("　", (3, 4)),
        ("三", (4, 5)),
        ("円", (5, 6)),
    ]
    _assert_char_spans_cover_text_once(text, mapping)


@pytest.mark.parametrize("text", ["予定1 10日まで", "1 20日", "1 24日", "1　10日", "1  10日"])
def test_number_before_spaced_day_keeps_separate_ranges(text: str) -> None:
    """
    「予定1 10日まで」の「1」と「10日」は別の数として読み、空白と日付の文字範囲が重ならないことを確認する。
    日付が数字を吸収しても、前の数へ日付の桁を割り当てず、音素と韻律にも同じ本文を残す。
    """

    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        _assert_char_spans_cover_text_once(text, mapping)
    assert [
        p
        for word in pyopenjtalk.g2p_mapping(text)
        for p in word["phonemes"]
        if p not in ("sp", "pau")
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert [
        p["phoneme"]
        for word in pyopenjtalk.g2p_mapping_prosody(text)
        for p in word["phonemes"]
        if p["kind"] == "Phoneme" and p["phoneme"] not in ("sp", "pau")
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert pyopenjtalk.g2p_prosody(text)


@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n", "\t"])
@pytest.mark.parametrize("text", ["10{}20円", "1{}2人", "今日は晴れ。{}明日は雨。"])
def test_control_whitespace_keeps_number_boundaries(text: str, separator: str) -> None:
    """
    改行やタブで区切った「10」と「20円」、「1」と「2人」を別々の数として読み、空白と同じ音素・ポーズになることを確認する。
    元の改行・タブを含む文字範囲をマッピングへ残し、数字のない2行の文でも読みとポーズを保つ。
    """

    source = text.format(separator)
    reference = text.format(" " * len(separator))
    assert pyopenjtalk.g2p(source) == pyopenjtalk.g2p(reference)
    assert pyopenjtalk.g2p(source, kana=True) == pyopenjtalk.g2p(reference, kana=True)
    assert pyopenjtalk.g2p_prosody(source) == pyopenjtalk.g2p_prosody(reference)
    for mapping in (pyopenjtalk.g2p_mapping(source), pyopenjtalk.g2p_mapping_prosody(source)):
        _assert_char_spans_cover_text_once(source, mapping)


@pytest.mark.parametrize(
    "text", ["1,000ー", "1,000ーー", "20ー", "110ー", "（200ーー）", "〇一〇ー二十"]
)
def test_number_with_absorbed_long_vowel_keeps_all_digits(text: str) -> None:
    """
    「1,000ー」や「20ー」で長音が数詞へ吸収されても、桁区切り・ゼロ・長音を含む入力の全範囲を保持することを確認する。
    位取り後の「千ー」「十ー」を通常語として1桁だけへ対応させると、残りの入力が欠けて各公開 API が例外になる。
    """

    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        _assert_char_spans_cover_text_once(text, mapping)
    assert [
        p
        for word in pyopenjtalk.g2p_mapping(text)
        for p in word["phonemes"]
        if p not in ("sp", "pau")
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert [
        p["phoneme"]
        for word in pyopenjtalk.g2p_mapping_prosody(text)
        for p in word["phonemes"]
        if p["kind"] == "Phoneme" and p["phoneme"] not in ("sp", "pau")
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert pyopenjtalk.g2p_prosody(text)


@pytest.mark.parametrize(
    "text", ["9" * 18 + "ヽ", "2〇ゞ", "9" * 18 + "ヽー", "9" * 18 + "ヽ\U000e01efー"]
)
def test_number_iteration_keeps_its_own_input_span(text: str) -> None:
    """
    数詞の末尾母音を繰り返す踊り字でも、数字の全桁と踊り字の位置を保持する。
    """

    mapping = pyopenjtalk.g2p_mapping(text)
    _assert_char_spans_cover_text_once(text, mapping)
    _assert_char_spans_cover_text_once(text, pyopenjtalk.g2p_mapping_prosody(text))
    assert [
        phone for entry in mapping for phone in entry["phonemes"] if phone not in ("pau", "unk")
    ] == [phone for phone in pyopenjtalk.g2p(text, join=False) if phone != "pau"]


@pytest.mark.parametrize("text", ["住所は13ー3ー5です。", "13ー3ー5", "23ー3ー5", "13ーー3ーー5"])
def test_number_followed_by_long_vowel_does_not_reserve_previous_digit(text: str) -> None:
    """
    「13ー3ー5」の2つ目の「三ー」を助数詞付きの縮約語と取り違えず、前の「13」へ両方の桁を割り当てることを確認する。
    長音を含む住所全体の文字範囲と音素を保ち、3種類のマッピング・韻律 API が例外なく返ることも確かめる。
    """

    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        _assert_char_spans_cover_text_once(text, mapping)
    assert [
        p for entry in pyopenjtalk.g2p_mapping(text) for p in entry["phonemes"] if p != "pau"
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert pyopenjtalk.g2p_prosody(text)


@pytest.mark.parametrize("text", ["十 十三", "十 十三駅", "1 〇二日", "1 〇・五日"])
def test_number_block_stops_at_lexical_word_and_keeps_symbolic_zero(text: str) -> None:
    """
    「十 十三」の地名「十三」を前の数詞へ含めず、「1 〇二日」の記号「〇」が数へ変わる場合も入力の位置を保つことを確認する。
    数詞の表層だけで後続の固有名詞をまとめたり、記号として解析されたゼロを落としたりすると、文字範囲が欠ける。
    """

    mapping = pyopenjtalk.g2p_mapping(text)
    _assert_char_spans_cover_text_once(text, mapping)
    _assert_char_spans_cover_text_once(text, pyopenjtalk.g2p_mapping_prosody(text))
    if text.startswith("十 十三"):
        lexical_entry = next(entry for entry in mapping if entry["surface"] == "十三")
        assert lexical_entry["char_span"] == (2, 4)
        assert lexical_entry["pron"] == "ジューソー"
    assert [p for word in mapping for p in word["phonemes"] if p not in ("sp", "pau")] == [
        p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"
    ]
    assert pyopenjtalk.g2p_prosody(text)


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


@pytest.mark.parametrize(
    "text", ["二〇日", "二〇日までに発送します。", "二〇日間", "二〇日目", "〇二日"]
)
def test_kanji_day_contraction_keeps_complete_input(text: str) -> None:
    """
    「二〇日」を「ハツカ」と読むときは、位取り後の「二十日」に元の数字と「日」の全範囲を割り当てることを確認する。
    先頭の「二」が一致しても1文字だけを消費せず、日数・序数や後続の本文でも、音素と韻律の対応を保つ。
    """

    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        _assert_char_spans_cover_text_once(text, mapping)
    assert [
        p
        for word in pyopenjtalk.g2p_mapping(text)
        for p in word["phonemes"]
        if p not in ("sp", "pau")
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert [
        p["phoneme"]
        for word in pyopenjtalk.g2p_mapping_prosody(text)
        for p in word["phonemes"]
        if p["kind"] == "Phoneme" and p["phoneme"] not in ("sp", "pau")
    ] == [p for p in pyopenjtalk.g2p(text, join=False) if p != "pau"]
    assert pyopenjtalk.g2p("二〇日", kana=True) == "ハツカ"
    assert pyopenjtalk.g2p_prosody(text)


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
