"""
辞書にないため NJD が読みを付けられなかった漢字への読みの補完 (read_unknown_kanji()) の実装。

送り仮名を伴う語 (「悪魔憑き」など) は Sudachi の語としての読みを、それ以外は Unihan の漢字ごとの音読みを使う。
"""

from sudachipy import tokenizer

from ._unihan_readings_map import UNIHAN_READINGS
from .types import NJDFeature
from .utils import get_sudachi_tokenizer, split_kana_mora


# 仮名の最後の文字から母音を引く表 (漢字の音読みを OpenJTalk の長音の表記に直すときに使う)
_VOWEL_BY_LAST_KANA: dict[str, str] = {
    **dict.fromkeys("アカサタナハマヤラワガザダバパャァ", "a"),
    **dict.fromkeys("イキシチニヒミリヰギジヂビピィ", "i"),
    **dict.fromkeys("ウクスツヌフムユルグズヅブプヴュゥ", "u"),
    **dict.fromkeys("エケセテネヘメレヱゲゼデベペェ", "e"),
    **dict.fromkeys("オコソトノホモヨロヲゴゾドボポョォ", "o"),
}


def read_unknown_kanji(
    njd_features: list[NJDFeature],
    text: str | None = None,
) -> list[NJDFeature]:
    """
    辞書にないため NJD が読みを付けられなかった漢字に、語としての読みか Unihan の音読みを付ける。
    送り仮名を伴う語 (「悪魔憑き」など) は Sudachi の語としての読みを、それ以外は Unihan の漢字ごとの音読みを使う。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        text (str | None): Sudachi で語としての読みを引くための入力テキスト。None の場合は Unihan の読みだけを使う

    Returns:
        list[NJDFeature]: 未知の漢字に読みを付けた NJDNode 用 features
    """

    def _is_kanji(character: str) -> bool:
        """
        Unihan の読みを付ける対象になる CJK 統合漢字かを判定する。

        Args:
            character (str): 判定する1文字

        Returns:
            bool: CJK 統合漢字 (拡張 A 以降と互換漢字を含む) の場合は True
        """

        codepoint = ord(character)
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x3FFFF
        )

    def _reading_to_pronunciation(reading: str) -> str:
        """
        読みの連母音 (「オウ」「エイ」など) を、OpenJTalk の発音の長音「ー」に書き換える。

        Args:
            reading (str): カタカナの読み

        Returns:
            str: 長音を「ー」で書いた発音
        """

        pronunciation: list[str] = []
        previous_vowel: str | None = None
        for mora in split_kana_mora(reading):
            if (previous_vowel, mora) in {("o", "ウ"), ("o", "オ"), ("u", "ウ"), ("e", "イ")}:
                pronunciation.append("ー")
                continue
            pronunciation.append(mora)
            if mora != "ー":
                previous_vowel = _VOWEL_BY_LAST_KANA.get(mora[-1])
        return "".join(pronunciation)

    def _find_sudachi_reading(feature_index: int) -> str | None:
        """
        Sudachi の1語が前後の既知の形態素をまたぐ場合も含めて、未知の漢字の部分の読みを取り出す。

        Args:
            feature_index (int): 未知の漢字の形態素の位置

        Returns:
            str | None: 前後の既知の形態素の読みを差し引いた読み (Sudachi の読みが一意に決まらなければ None)
        """

        # Sudachi の語のうち、この漢字を含むものより長い範囲は試さない (長い文で前後の全範囲を試すと、形態素数の3乗に比例して遅くなるため)
        unknown_surface = njd_features[feature_index]["string"]
        maximum_surface_length = max(
            (len(surface) for surface in sudachi_readings_by_surface if unknown_surface in surface),
            default=0,
        )
        if maximum_surface_length == 0:
            return None
        for candidate_start in range(feature_index, -1, -1):
            preceding_surface = "".join(
                candidate["string"] for candidate in njd_features[candidate_start:feature_index]
            )
            if len(preceding_surface) + len(unknown_surface) > maximum_surface_length:
                break
            preceding_read = "".join(
                candidate["read"] for candidate in njd_features[candidate_start:feature_index]
            )
            candidate_surface = preceding_surface
            for candidate_end in range(feature_index + 1, len(njd_features) + 1):
                candidate_surface += njd_features[candidate_end - 1]["string"]
                if len(candidate_surface) > maximum_surface_length:
                    break
                candidate_readings = sudachi_readings_by_surface.get(candidate_surface, set())
                if len(candidate_readings) != 1:
                    continue
                following_read = "".join(
                    candidate["read"]
                    for candidate in njd_features[feature_index + 1 : candidate_end]
                )
                candidate_reading = next(iter(candidate_readings))
                if candidate_reading.startswith(preceding_read) is False:
                    continue
                if following_read != "" and candidate_reading.endswith(following_read) is False:
                    continue
                extracted_read = candidate_reading[
                    len(preceding_read) : len(candidate_reading) - len(following_read)
                ]
                if extracted_read != "":
                    return extracted_read
        return None

    # NJD が記号・読点に変えた漢字だけを対象にし、本物の句読点と既知の語は変えない
    unknown_kanji_indices = [
        feature_index
        for feature_index, feature in enumerate(njd_features)
        if feature["pos"] == "記号"
        and feature["pos_group1"] == "読点"
        and any(_is_kanji(character) for character in feature["string"]) is True
    ]
    # 未知の漢字がない文では Sudachi を読み込まない (tsqyomi を使うときや Sudachi の読み補正を切ったときも、読めない漢字がなければ Sudachi は動かない)
    if len(unknown_kanji_indices) == 0:
        return njd_features

    sudachi_readings_by_surface: dict[str, set[str]] = {}
    if text is not None:
        # 送り仮名を伴う語は漢字ごとの音読みでは決まらないため、Sudachi の語としての読みを候補にする
        for morpheme in get_sudachi_tokenizer().tokenize(text, tokenizer.Tokenizer.SplitMode.C):
            reading = morpheme.reading_form()
            if reading != morpheme.surface():
                sudachi_readings_by_surface.setdefault(morpheme.surface(), set()).add(reading)

    for feature_index in unknown_kanji_indices:
        feature = njd_features[feature_index]
        reading = _find_sudachi_reading(feature_index)
        if reading is None:
            # 語として読めない未知の漢字は、すべての文字に一意な Unihan の読みがある場合だけ読みを付ける
            reading_parts: list[str] = []
            for character in feature["string"]:
                if character in UNIHAN_READINGS:
                    reading_parts.append(UNIHAN_READINGS[character])
                elif _is_kanji(character) is False:
                    reading_parts.append(character)
                else:
                    reading_parts = []
                    break
            if len(reading_parts) == 0:
                continue
            reading = "".join(reading_parts)

        feature["read"] = reading
        feature["pron"] = _reading_to_pronunciation(reading)
        feature["pos"] = "名詞"
        feature["pos_group1"] = "一般"
        feature["pos_group2"] = "*"
        feature["pos_group3"] = "*"
        feature["mora_size"] = len(split_kana_mora(feature["pron"]))
        feature["acc"] = 0
        feature["chain_flag"] = -1

    return njd_features
