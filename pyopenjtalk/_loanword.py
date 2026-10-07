"""
辞書が一般的な仮名へ置き換えた外来語の表記の復元 (restore_loanword_kana()) の実装。

表層形と読み・発音の全体が1文字ずつ対応する場合だけ戻すので、日本語で定着した別の表記や、中黒などを省いた語は変えない。
"""

from ._kana_utils import is_katakana_word
from .types import NJDFeature
from .utils import split_kana_mora


# 辞書が外来語表記を一般的な仮名へ置き換える組み合わせ
# 長い表記を先に照合し、「ヴャ」を「ヴ」だけで消費する状態を避ける
_LOANWORD_KANA_RESTORATIONS: tuple[tuple[str, str], ...] = (
    ("ヴャ", "ビャ"),
    ("ヴュ", "ビュ"),
    ("ヴョ", "ビョ"),
    ("ヴァ", "バ"),
    ("ヴィ", "ビ"),
    ("ヴェ", "ベ"),
    ("ヴォ", "ボ"),
    ("ヴ", "ブ"),
    ("スィ", "シ"),
    ("ズィ", "ジ"),
    ("テュ", "チュ"),
    ("デュ", "ヂュ"),
    ("イェ", "イエ"),
    ("シィ", "シー"),
    ("リェ", "リエ"),
    ("ニェ", "ニエ"),
    ("ヒェ", "ヒエ"),
    ("ミェ", "ミエ"),
    ("ビェ", "ビエ"),
    ("ピェ", "ピエ"),
    ("キェ", "ケ"),
    ("ギェ", "ゲ"),
    ("グゥ", "グウ"),
    ("クゥ", "クウ"),
)


def restore_loanword_kana(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    辞書が別の仮名へ置き換えた外来語の表記 (「ヴィ」を「ビ」にするなど) を、表層形から元に戻す。
    表層形と読み・発音の全体が1文字ずつ対応する場合だけ戻すので、日本語で定着した別の表記や、中黒などを省いた語は変えない。

    Args:
        njd_features (list[NJDFeature]): 復元対象の NJDNode 用 features

    Returns:
        list[NJDFeature]: 外来語の表記を戻した NJDNode 用 features
    """

    def _restore_spelling(
        surface: str,
        pronunciation: str,
    ) -> tuple[str, list[tuple[int, int, int]]] | None:
        """
        表層形と読み (または発音) を先頭から照合し、置き換えられた外来語の仮名を表層形の表記へ戻す。

        Args:
            surface (str): 外来語の表層形
            pronunciation (str): 辞書が与えた読みまたは発音

        Returns:
            tuple[str, list[tuple[int, int, int]]] | None: 表記を戻した読みまたは発音と、戻した箇所ごとの (戻す前の先頭のモーラ位置、戻す前のモーラ数、戻した後のモーラ数) の一覧 (全体が対応しないか、戻す箇所がなければ None)
        """

        restored: list[str] = []
        restored_segments: list[tuple[int, int, int]] = []
        surface_offset = 0
        pronunciation_offset = 0
        is_changed = False

        # 無声化記号は表層形に現れないため、照合位置を進めず発音側から持ち越す
        while surface_offset < len(surface):
            if pronunciation.startswith("’", pronunciation_offset):
                restored.append("’")
                pronunciation_offset += 1
                continue

            # 辞書が置き換えた組み合わせを優先し、該当しなければ同じ文字を1文字ずつ照合する
            restoration = next(
                (
                    (original, collapsed)
                    for original, collapsed in _LOANWORD_KANA_RESTORATIONS
                    if surface.startswith(original, surface_offset)
                    and pronunciation.startswith(collapsed, pronunciation_offset)
                ),
                None,
            )
            if restoration is not None:
                original, collapsed = restoration
                restored_segments.append(
                    (
                        len(split_kana_mora(pronunciation[:pronunciation_offset].replace("’", ""))),
                        len(split_kana_mora(collapsed)),
                        len(split_kana_mora(original)),
                    )
                )
                restored.append(original)
                surface_offset += len(original)
                pronunciation_offset += len(collapsed)
                is_changed = True
                continue

            surface_char = surface[surface_offset]
            if pronunciation.startswith(surface_char, pronunciation_offset) is False:
                return None
            restored.append(surface_char)
            surface_offset += 1
            pronunciation_offset += 1

        # 語末の無声化記号も、表層形との対応を崩さず保持する
        if pronunciation.startswith("’", pronunciation_offset):
            restored.append("’")
            pronunciation_offset += 1
        if pronunciation_offset != len(pronunciation) or is_changed is False:
            return None
        return "".join(restored), restored_segments

    for feature in njd_features:
        # 外来語だけへ限定し、漢字や区切り記号を含む語の読みを表層形から作らない
        surface = feature["string"]
        if is_katakana_word(surface) is False:
            continue
        restored_read = _restore_spelling(surface, feature["read"])
        if restored_read is not None:
            feature["read"] = restored_read[0]
        restored_pronunciation = _restore_spelling(surface, feature["pron"])
        if restored_pronunciation is not None:
            feature["pron"], restored_segments = restored_pronunciation
            # 無声化記号はモーラに数えない
            feature["mora_size"] = len(split_kana_mora(feature["pron"].replace("’", "")))
            # 「イエ」を「イェ」に戻すとモーラが減るので、アクセント核も戻す前と同じモーラの位置へずらす
            ## 核が戻した箇所の中にあるときは、戻した箇所の最後のモーラに置く
            accent_shift = 0
            for segment_start, collapsed_mora_count, original_mora_count in restored_segments:
                if feature["acc"] > segment_start + collapsed_mora_count:
                    accent_shift += collapsed_mora_count - original_mora_count
                elif feature["acc"] > segment_start:
                    offset_in_segment = feature["acc"] - segment_start
                    accent_shift += offset_in_segment - min(offset_in_segment, original_mora_count)
            feature["acc"] -= accent_shift

    return njd_features
