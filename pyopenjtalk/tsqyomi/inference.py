from __future__ import annotations

from dataclasses import dataclass, replace

from ..openjtalk import OpenJTalk
from ..types import MeCabMorph
from . import diagnostics
from .model import (
    ReadingPrediction,
    ReadingTarget,
    TsqyomiModel,
    get_loaded_model,
)
from .types import CandidateNode, CandidatePath, ReadingAnalysis


# 開始記号と終了記号の対応関係
_CLOSING_DELIMITER_BY_OPENING = {
    "「": "」",
    "『": "』",
    "（": "）",
    "(": ")",
    "［": "］",
    "[": "]",
    "【": "】",
    "〈": "〉",
    "《": "》",
    "“": "”",
    "‘": "’",
}

# モデルメタデータ上の読みクラス台帳は、四つ仮名と助詞の「ヲ」を発音形に正規化した上で索引する
# 2026/08/10 時点での pyopenjtalk-plus デフォルト辞書には、（恐らく従来から）この正規化が正しく行われていないエントリがあるため、
# 辞書の生の発音欄と突き合わせる前に同じ変換を通さないと、同じ意味にも関わらず一致が取れない可能性がある
# フェイルセーフとして、常にこの正規化を適用してから処理されるようにする
_PRONUNCIATION_CANONICAL_TRANSLATION = str.maketrans({"ヲ": "オ", "ヅ": "ズ", "ヂ": "ジ"})

# 辞書が「一月」を期間読みへ解析したときに、その読みを維持する後続語
## 「号」「分」「中」のように暦月と期間の両方が成り立つ接尾語は、モデルの文脈選択へ残す
_DURATION_MONTH_FOLLOWING_SURFACES = frozenset(
    {
        "間",
        "前",
        "後",
        "以内",
        "以上",
        "以下",
        "未満",
        "程度",
        "ほど",
        "くらい",
        "ぐらい",
    }
)

# 助数詞として解析されても、語彙読みとの文脈選択が必要な「何」の後続表層
_CONTEXTUAL_NAN_COUNTER_SURFACES = frozenset({"人", "色", "時", "手"})

# 「一月」の直前で暦上の月を明示する連体修飾語
_CALENDAR_MONTH_MODIFIER_SURFACES = frozenset(
    {"前年", "去年", "昨年", "今年", "来年", "翌年", "再来年", "次"}
)


@dataclass(frozen=True)
class _ResolvedTarget:
    """
    候補グラフ上で確定した1対象の位置情報と、モデルが選んだ読み。

    Attributes:
        char_span (tuple[int, int]): 正規化本文上の対象表層の半開区間
        morph_range (tuple[int, int]): 差し替え対象の形態素列における半開添字区間
        surface (str): 対象表層
        span_paths (tuple[CandidatePath, ...]): 差し替え可能な候補経路
        pronunciations (tuple[str, ...]): 候補グラフ上で到達可能な発音 (重複なし)
        selected_pronunciation (str | None): モデルが選んだ発音。推論前は None
        score_margin (float | None): モデルが選んだ1位と2位の候補スコアの差
    """

    char_span: tuple[int, int]
    morph_range: tuple[int, int]
    surface: str
    span_paths: tuple[CandidatePath, ...]
    pronunciations: tuple[str, ...]
    selected_pronunciation: str | None = None
    score_margin: float | None = None
    # 診断専用: 保護条件により辞書既定読みを維持したか
    was_preserved: bool = False

    def to_reading_target(self) -> ReadingTarget:
        """
        ONNX 推論へ渡す公開型へ変換する。

        Returns:
            ReadingTarget: `predict()` へ渡す1対象
        """

        return ReadingTarget(
            char_span=self.char_span,
            surface=self.surface,
            pronunciations=self.pronunciations,
        )

    @property
    def selected_paths(self) -> tuple[CandidatePath, ...]:
        """
        選択済み発音を実現する候補経路を返す。

        Returns:
            tuple[CandidatePath, ...]: `selected_pronunciation` に一致する候補経路
        """

        if self.selected_pronunciation is None:
            return ()
        return tuple(
            path for path in self.span_paths if path["pronunciation"] == self.selected_pronunciation
        )


def canonicalize_pronunciation(pronunciation: str) -> str:
    """
    辞書の発音形を、メタデータの読みクラス台帳と同じ正規化された表記へ揃える。

    Args:
        pronunciation (str): 辞書の発音 (pron) 欄に記載されている値

    Returns:
        str: 四つ仮名と助詞の「ヲ」を発音形に正規化した表記
    """

    return pronunciation.translate(_PRONUNCIATION_CANONICAL_TRANSLATION)


def _default_path_pronunciation(
    morphs: tuple[MeCabMorph, ...],
    morph_range: tuple[int, int],
) -> str | None:
    """
    最良経路の形態素列から対象範囲の既定発音を返す。

    Args:
        morphs (tuple[MeCabMorph, ...]): 最良経路の形態素列
        morph_range (tuple[int, int]): 対象表層の半開形態素添字区間

    Returns:
        str | None: 既定発音。読み欠落形態素が混ざる場合は None
    """

    start, end = morph_range
    segments: list[str] = []
    for morph in morphs[start:end]:
        if morph["is_ignored"] is True:
            continue
        features = morph["features"]
        if len(features) < 10:
            return None
        segments.append(features[9])
    if len(segments) == 0:
        return None
    return "".join(segments)


def _is_minute_counter_suffix(morph: MeCabMorph) -> bool:
    """助数詞の「分」かどうかを判定する。"""

    features = morph["features"]
    return (
        len(features) >= 4
        and features[1:4] == ["名詞", "接尾", "助数詞"]
        and morph["surface"] == "分"
    )


def _is_hour_counter_suffix(morph: MeCabMorph) -> bool:
    """助数詞の「時」かどうかを判定する。"""

    features = morph["features"]
    return (
        len(features) >= 4
        and features[1:4] == ["名詞", "接尾", "助数詞"]
        and morph["surface"] == "時"
    )


def _is_duration_kan_suffix(morph: MeCabMorph) -> bool:
    """時間量の接尾「間」かどうかを判定する。"""

    features = morph["features"]
    return (
        len(features) >= 4
        and features[1:4] == ["名詞", "接尾", "一般"]
        and morph["surface"] == "間"
    )


def _is_hour_duration_head_morph(morph: MeCabMorph) -> bool:
    """
    直後の「間」と結合して時間量になる見出しかどうかを判定する。

    「二時 + 間」「十時 + 間」、単独の「十時間」「何時間」などを対象とする。
    """

    surface = morph["surface"]
    features = morph["features"]
    if surface.endswith("時間") and len(features) >= 3 and features[1:3] == ["名詞", "一般"]:
        return True
    if surface == "何時" or not surface.endswith("時"):
        return False
    if len(features) >= 3 and features[1:3] == ["名詞", "一般"]:
        return True
    # 十時が固有名詞へ化けても、直後「間」なら時間量として扱う
    if len(features) >= 4 and features[1:4] == ["名詞", "固有名詞", "地域"]:
        return True
    return False


def _is_general_counter_suffix(morph: MeCabMorph) -> bool:
    """一般的な助数詞の接尾辞かどうかを判定する。"""

    features = morph["features"]
    return len(features) >= 4 and features[1:4] == ["名詞", "接尾", "助数詞"]


def _is_number_nan_morph(morph: MeCabMorph) -> bool:
    """数量疑問の「何」(名詞,数) かどうかを判定する。"""

    features = morph["features"]
    return morph["surface"] == "何" and len(features) >= 3 and features[1:3] == ["名詞", "数"]


def _is_duration_month_morph(
    morphs: tuple[MeCabMorph, ...],
    morph_index: int,
) -> bool:
    """
    後続語と暦月修飾を使い、「一月」の期間読み (ヒトツキ) を維持するか判定する。

    Args:
        morphs (tuple[MeCabMorph, ...]): MeCab の既定最良経路の形態素列
        morph_index (int): 判定対象の形態素添字

    Returns:
        bool: 辞書既定の期間読みを維持する場合は True
    """

    morph = morphs[morph_index]
    features = morph["features"]
    if (
        morph["surface"] != "一月"
        or len(features) < 3
        or features[1:3] != ["名詞", "一般"]
        or morph_index + 1 >= len(morphs)
    ):
        return False

    # 「来年の一月前」のように暦年から修飾される場合は、暦月読みをモデルへ選ばせる
    if morph_index >= 2:
        previous_morph = morphs[morph_index - 1]
        modifier_morph = morphs[morph_index - 2]
        if (
            previous_morph["surface"] == "の"
            and modifier_morph["surface"] in _CALENDAR_MONTH_MODIFIER_SURFACES
            and modifier_morph["char_span"][1] == previous_morph["char_span"][0]
            and previous_morph["char_span"][1] == morph["char_span"][0]
        ):
            return False

    # 暦年修飾がない場合に、辞書が期間読みを選んだ連続語だけを保護する
    following_morph = morphs[morph_index + 1]
    return (
        morph["char_span"][1] == following_morph["char_span"][0]
        and following_morph["surface"] in _DURATION_MONTH_FOLLOWING_SURFACES
    )


def _is_dictionary_go_suffix_morph(morph: MeCabMorph) -> bool:
    """
    MeCab 既定で接尾辞「後」(ゴ) と確定した形態素かどうかを判定する。
    """

    if morph["surface"] != "後":
        return False
    features = morph["features"]
    return (
        len(features) >= 10
        and features[1:4] == ["名詞", "接尾", "副詞可能"]
        and features[9] == "ゴ"
    )


def _is_contiguous_go_suffix_morph(
    morph: MeCabMorph,
    previous_morph: MeCabMorph,
) -> bool:
    """
    直前形態素に続く接尾辞「後」(ゴ) かどうかを判定する。

    Args:
        morph (MeCabMorph): 判定対象の形態素
        previous_morph (MeCabMorph): 直前の形態素

    Returns:
        bool: 直前形態素に続く接尾辞「後」(ゴ) なら True
    """

    return (
        previous_morph["char_span"][1] == morph["char_span"][0]
        and _is_dictionary_go_suffix_morph(morph) is True
    )


def _find_dictionary_owned_quantity_ranges(
    morphs: tuple[MeCabMorph, ...],
) -> tuple[tuple[int, int], ...]:
    """
    最良経路で読みが確定した数量・順序表現の文字範囲を返す。

    「数量 + 時 + 間」、複数数詞 + 分、数量 + 助数詞 + 後、何時何分 など、辞書既定読みが一意な区間だけを対象とする。

    Args:
        morphs (tuple[MeCabMorph, ...]): MeCab の既定最良経路の形態素列

    Returns:
        tuple[tuple[int, int], ...]: 辞書の既定発音を維持する半開文字範囲
    """

    protected_ranges: list[tuple[int, int]] = []
    morph_index = 0
    while morph_index < len(morphs):
        morph = morphs[morph_index]
        features = morph["features"]
        # 経過文脈の「一月」は MeCab 既定の ヒトツキ 読みを維持する
        if _is_duration_month_morph(morphs, morph_index) is True:
            protected_ranges.append(morph["char_span"])
            morph_index += 1
            continue

        # 辞書が「何分」を1形態素にまとめる場合は、単独の時間量問いとして既定読みを維持する
        if morph["surface"] == "何分":
            protected_ranges.append(morph["char_span"])
            morph_index += 1
            continue

        # 十時間 / 何時間 など、辞書が1形態素にまとめた時間量は既定読みを維持する
        if (
            _is_hour_duration_head_morph(morph) is True
            and morph["surface"].endswith("時間") is True
        ):
            protected_ranges.append(morph["char_span"])
            morph_index += 1
            continue

        # 二時 + 間 / 十時 + 間 など、助数詞経路に乗らない時間量分割も既定読みを維持する
        if morph_index + 1 < len(morphs):
            kan_morph = morphs[morph_index + 1]
            is_contiguous_kan = morph["char_span"][1] == kan_morph["char_span"][0]
            if (
                is_contiguous_kan is True
                and _is_hour_duration_head_morph(morph) is True
                and _is_duration_kan_suffix(kan_morph) is True
            ):
                protected_ranges.append(
                    (
                        morph["char_span"][0],
                        kan_morph["char_span"][1],
                    )
                )
                morph_index += 2
                continue

        # 何時間 + 何分 / 何 + 分 は「何時何分」と同系の数量問いなので、分側だけを保護する
        if morph["surface"] == "何時間" and morph_index + 1 < len(morphs):
            following_morph = morphs[morph_index + 1]
            is_contiguous = morph["char_span"][1] == following_morph["char_span"][0]
            if is_contiguous is True and following_morph["surface"] == "何分":
                protected_ranges.append(following_morph["char_span"])
                morph_index += 2
                continue
            if (
                is_contiguous is True
                and following_morph["surface"] == "何"
                and morph_index + 2 < len(morphs)
            ):
                minute_morph = morphs[morph_index + 2]
                is_minute_contiguous = (
                    following_morph["char_span"][1] == minute_morph["char_span"][0]
                )
                if is_minute_contiguous is True and _is_minute_counter_suffix(minute_morph) is True:
                    protected_ranges.append(
                        (
                            following_morph["char_span"][0],
                            minute_morph["char_span"][1],
                        )
                    )
                    morph_index += 3
                    continue

        # 何 + 分 (+ 後) は単独「何分」と同じく、時間量の問いとして既定読みを維持する
        if morph["surface"] == "何" and morph_index + 1 < len(morphs):
            minute_morph = morphs[morph_index + 1]
            is_contiguous = morph["char_span"][1] == minute_morph["char_span"][0]
            if is_contiguous is True and _is_minute_counter_suffix(minute_morph) is True:
                protected_ranges.append(
                    (
                        morph["char_span"][0],
                        minute_morph["char_span"][1],
                    )
                )
                morph_index += 2
                continue

            # 何時何分 (何 + 時 + 何 + 分) は時計読みの「ナンジ」経路を維持する
            if morph_index + 3 < len(morphs):
                hour_morph = morphs[morph_index + 1]
                second_nan_morph = morphs[morph_index + 2]
                minute_morph = morphs[morph_index + 3]
                is_clock_contiguous = (
                    morph["char_span"][1] == hour_morph["char_span"][0]
                    and hour_morph["char_span"][1] == second_nan_morph["char_span"][0]
                    and second_nan_morph["char_span"][1] == minute_morph["char_span"][0]
                )
                if (
                    is_clock_contiguous is True
                    and _is_hour_counter_suffix(hour_morph) is True
                    and second_nan_morph["surface"] == "何"
                    and _is_minute_counter_suffix(minute_morph) is True
                ):
                    protected_ranges.append(
                        (
                            morph["char_span"][0],
                            minute_morph["char_span"][1],
                        )
                    )
                    morph_index += 4
                    continue

            # 何 + 時 + まで + 後 は「何時まで (ナンジマデ) + 後 (ゴ)」の経路を維持する
            if morph_index + 3 < len(morphs):
                hour_morph = morphs[morph_index + 1]
                made_morph = morphs[morph_index + 2]
                following_morph = morphs[morph_index + 3]
                is_made_contiguous = (
                    morph["char_span"][1] == hour_morph["char_span"][0]
                    and hour_morph["char_span"][1] == made_morph["char_span"][0]
                    and made_morph["char_span"][1] == following_morph["char_span"][0]
                )
                if (
                    is_made_contiguous is True
                    and _is_hour_counter_suffix(hour_morph) is True
                    and made_morph["surface"] == "まで"
                    and _is_contiguous_go_suffix_morph(following_morph, made_morph) is True
                ):
                    protected_ranges.append(
                        (
                            morph["char_span"][0],
                            made_morph["char_span"][1],
                        )
                    )
                    morph_index += 3
                    continue

            # 何 + 時 + 後 は「何時」単位の経過 (ナンジ + ゴ) を維持する
            if morph_index + 2 < len(morphs):
                hour_morph = morphs[morph_index + 1]
                following_morph = morphs[morph_index + 2]
                is_hour_contiguous = morph["char_span"][1] == hour_morph["char_span"][0]
                is_following_contiguous = (
                    hour_morph["char_span"][1] == following_morph["char_span"][0]
                )
                if (
                    is_hour_contiguous is True
                    and is_following_contiguous is True
                    and _is_hour_counter_suffix(hour_morph) is True
                    and _is_contiguous_go_suffix_morph(following_morph, hour_morph) is True
                ):
                    protected_ranges.append(
                        (
                            morph["char_span"][0],
                            following_morph["char_span"][1],
                        )
                    )
                    morph_index += 3
                    continue

            # 読みが一意な「何 + 助数詞」は、モデルの誤介入を避けて辞書既定読みを維持する
            ## 「何人」「何色」「何時」「何手」は語彙読みもあるため、文脈選択の対象として残す
            counter_morph = morphs[morph_index + 1]
            if (
                is_contiguous is True
                and _is_number_nan_morph(morph) is True
                and _is_general_counter_suffix(counter_morph) is True
                and counter_morph["surface"] not in _CONTEXTUAL_NAN_COUNTER_SURFACES
            ):
                protected_ranges.append(
                    (
                        morph["char_span"][0],
                        counter_morph["char_span"][1],
                    )
                )
                morph_index += 2
                continue

        # 辞書が「何時」を1形態素にまとめる場合も、直後の「間」と合わせて時間量として扱う
        if morph["surface"] == "何時" and morph_index + 1 < len(morphs):
            duration_morph = morphs[morph_index + 1]
            duration_features = duration_morph["features"]
            is_contiguous = morph["char_span"][1] == duration_morph["char_span"][0]
            is_duration_suffix = (
                len(duration_features) >= 3
                and duration_features[1:3] == ["名詞", "接尾"]
                and duration_morph["surface"] == "間"
            )
            if is_contiguous is True and is_duration_suffix is True:
                protected_ranges.append(
                    (
                        morph["char_span"][0],
                        duration_morph["char_span"][1],
                    )
                )
                morph_index += 2
                continue

        # 数詞形態素から始まる連続列だけを数量部として扱う
        if len(features) < 3 or features[1:3] != ["名詞", "数"]:
            morph_index += 1
            continue

        number_start = morph_index
        number_end = morph_index + 1
        while number_end < len(morphs):
            next_morph = morphs[number_end]
            next_features = next_morph["features"]
            # 空白や記号を跨ぐ表記は通常の構文へ戻し、離れた語を数量表現へまとめない
            if (
                len(next_features) < 3
                or next_features[1:3] != ["名詞", "数"]
                or morphs[number_end - 1]["char_span"][1] != next_morph["char_span"][0]
            ):
                break
            number_end += 1

        # 「時」と「間」が助数詞・接尾辞として連続する場合だけ、時間量の読みが一意に定まる
        if number_end + 1 < len(morphs):
            hour_morph = morphs[number_end]
            duration_morph = morphs[number_end + 1]
            duration_features = duration_morph["features"]
            is_contiguous = (
                morphs[number_end - 1]["char_span"][1] == hour_morph["char_span"][0]
                and hour_morph["char_span"][1] == duration_morph["char_span"][0]
            )
            is_hour_counter = _is_hour_counter_suffix(hour_morph)
            is_duration_suffix = (
                len(duration_features) >= 3
                and duration_features[1:3] == ["名詞", "接尾"]
                and duration_morph["surface"] == "間"
            )
            if is_contiguous is True and is_hour_counter is True and is_duration_suffix is True:
                protected_ranges.append(
                    (
                        morphs[number_start]["char_span"][0],
                        duration_morph["char_span"][1],
                    )
                )

        # 複数数詞 + 分は同形異音語への部分介入を避け、MeCab 既定の長音読みを維持する
        if number_end > number_start + 1 and number_end < len(morphs):
            minute_morph = morphs[number_end]
            is_contiguous_minute = (
                morphs[number_end - 1]["char_span"][1] == minute_morph["char_span"][0]
            )
            is_minute_counter = _is_minute_counter_suffix(minute_morph)
            if is_contiguous_minute is True and is_minute_counter is True:
                protected_ranges.append(
                    (
                        morphs[number_start]["char_span"][0],
                        minute_morph["char_span"][1],
                    )
                )

        # 数量部を再走査せず、その直後から次の候補を探す
        morph_index = number_end

    # 接尾辞「後」(ゴ) は直前形態素に続く場合だけ保護する
    for go_morph_index, morph in enumerate(morphs):
        if go_morph_index == 0:
            continue
        if _is_contiguous_go_suffix_morph(morph, morphs[go_morph_index - 1]) is False:
            continue
        # 何時後のように既存の保護範囲に含まれる「後」は重複して登録しない
        if any(
            char_range[0] <= morph["char_span"][0] and morph["char_span"][1] <= char_range[1]
            for char_range in protected_ranges
        ):
            continue
        protected_ranges.append(morph["char_span"])

    return tuple(dict.fromkeys(protected_ranges))


def _resolve_selected_pronunciations(
    model: TsqyomiModel,
    resolved_targets: list[_ResolvedTarget],
    predictions: tuple[ReadingPrediction, ...],
    analysis: ReadingAnalysis,
) -> list[_ResolvedTarget]:
    """
    モデル予測を採用し、メタデータ指定または前接名詞に続く接尾用法では辞書既定読みを維持する。

    Args:
        model (TsqyomiModel): ロード済み tsqyomi モデル
        resolved_targets (list[_ResolvedTarget]): 推論対象列
        predictions (tuple[ReadingPrediction, ...]): モデル予測列
        analysis (ReadingAnalysis): 候補解析結果

    Returns:
        list[_ResolvedTarget]: 選択済み発音を埋めた対象列
    """

    preserve_pairs = frozenset(model.metadata.preserve_dictionary_default_pronunciations)
    selected_targets: list[_ResolvedTarget] = []
    for target, prediction in zip(resolved_targets, predictions, strict=True):
        selected_pronunciation = prediction.pronunciation
        was_preserved = False
        default_pronunciation = _default_path_pronunciation(analysis["morphs"], target.morph_range)
        default_morphs = analysis["morphs"][target.morph_range[0] : target.morph_range[1]]
        previous_morph = (
            analysis["morphs"][target.morph_range[0] - 1] if target.morph_range[0] > 0 else None
        )
        morph_before_previous = (
            analysis["morphs"][target.morph_range[0] - 2] if target.morph_range[0] > 1 else None
        )
        # 古い活用語の語幹と語尾が、動詞と平仮名一般名詞へ分裂された場合を前接名詞から除く
        ## 「虐ぐる日」の「虐+ぐる+日」で辞書既定のビを保護すると、モデルが選んだ独立読みのヒを失う
        is_previous_morph_inflection_fragment = (
            previous_morph is not None
            and morph_before_previous is not None
            and len(previous_morph["features"]) >= 3
            and previous_morph["features"][1:3] == ["名詞", "一般"]
            and len(previous_morph["surface"]) > 0
            and all("ぁ" <= character <= "ゖ" for character in previous_morph["surface"])
            and len(morph_before_previous["features"]) >= 2
            and morph_before_previous["features"][1] == "動詞"
            and morph_before_previous["char_span"][1] == previous_morph["char_span"][0]
            and previous_morph["char_span"][1] == target.char_span[0]
        )
        # 前接名詞と結合した接尾形態素は、モデル候補にない生産的な辞書既定読みを維持する
        ## 「代々家が」のように副詞的名詞の後ろの独立語を接尾辞へ誤解析した場合は、モデル介入を適用する
        is_compound_suffix_default = (
            len(default_morphs) == 1
            and len(default_morphs[0]["features"]) >= 3
            and default_morphs[0]["features"][1:3] == ["名詞", "接尾"]
            and previous_morph is not None
            and len(previous_morph["features"]) >= 3
            and previous_morph["features"][1] == "名詞"
            and previous_morph["features"][2] != "副詞可能"
            and is_previous_morph_inflection_fragment is False
            and previous_morph["char_span"][1] == target.char_span[0]
            and default_pronunciation is not None
            and default_pronunciation not in target.pronunciations
        )
        # 既定読みがモデル候補にない保護対象 (例: 接尾用法で読む位置) も辞書のまま維持する
        # 候補外の発音は後段で実現経路が見つからず交換がスキップされるため、既定読みが保たれる
        if (
            default_pronunciation is not None
            and selected_pronunciation != default_pronunciation
            and (
                (target.surface, default_pronunciation) in preserve_pairs
                or is_compound_suffix_default is True
            )
        ):
            selected_pronunciation = default_pronunciation
            was_preserved = True
        # 候補スコアの上位2件だけを使い、辞書既定読みに戻す前のモデル確信度を保存する
        sorted_scores = sorted(prediction.scores, reverse=True)
        selected_targets.append(
            replace(
                target,
                selected_pronunciation=selected_pronunciation,
                score_margin=(
                    sorted_scores[0] - sorted_scores[1] if len(prediction.scores) >= 2 else None
                ),
                was_preserved=was_preserved,
            )
        )
    return selected_targets


def select_mecab_features_with_tsqyomi(
    text: str,
    jtalk: OpenJTalk,
    *,
    include_morphs: bool = True,
) -> tuple[list[str], list[MeCabMorph]]:
    """
    ロード済みの tsqyomi モデルで選んだ読みの MeCab feature 列を返す。
    MeCab の解析 lock は候補グラフのコピー時点で解放され、モデル推論中は保持されない。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        jtalk (OpenJTalk): 候補解析に使う OpenJTalk インスタンス
        include_morphs (bool): 呼び出し元が詳細形態素列を必要とする場合は True

    Returns:
        tuple[list[str], list[MeCabMorph]]: NJD 入力用 MeCab feature 列と差し替え後の形態素列
    """

    model = get_loaded_model()
    normalized_text = jtalk.normalize_for_mecab(text)
    target_spans = _find_target_spans(
        normalized_text,
        model.metadata.surfaces_by_first_character,
    )

    # 対象表層が本文にない呼び出しが多い場合は候補グラフ生成とモデル推論を省略する
    if len(target_spans) == 0:
        if include_morphs is False:
            return jtalk.run_mecab(normalized_text), []
        return jtalk.run_mecab_detailed(normalized_text)

    processing_segments = _split_target_processing_segments(normalized_text, target_spans)
    if len(processing_segments) > 1:
        combined_features: list[str] = []
        combined_morphs: list[MeCabMorph] = []
        # 分割片ごとの診断件数を記録し、片内の相対位置を元の本文位置に戻す
        for segment_start, segment_end in processing_segments:
            diagnostic_start_index = diagnostics.record_count()
            segment_features, segment_morphs = select_mecab_features_with_tsqyomi(
                normalized_text[segment_start:segment_end],
                jtalk,
                include_morphs=include_morphs,
            )
            diagnostics.rebase_recording_char_spans(diagnostic_start_index, segment_start)
            combined_features.extend(segment_features)
            for morph in segment_morphs:
                # 分割入力の char_span は区間先頭からの相対位置なので、元の本文位置へオフセットを加算する
                adjusted_morph = morph.copy()
                adjusted_morph["char_span"] = (
                    morph["char_span"][0] + segment_start,
                    morph["char_span"][1] + segment_start,
                )
                combined_morphs.append(adjusted_morph)
        return combined_features, combined_morphs

    analysis = jtalk.analyze_mecab_candidates(normalized_text, target_spans)
    nodes_by_id = {node["node_id"]: node for node in analysis["nodes"]}
    selected_features = list(analysis["features"])
    resolved_targets: list[_ResolvedTarget] = []
    dictionary_owned_quantity_ranges = _find_dictionary_owned_quantity_ranges(analysis["morphs"])

    # メタデータ上の最長一致と既定形態素境界の両方を満たす出現だけをモデルに渡す
    for char_span in target_spans:
        surface = analysis["normalized_text"][char_span[0] : char_span[1]]
        allowed_readings = frozenset(
            model.metadata.reading_class_ids_by_surface_and_pronunciation.get(surface, {})
        )
        if len(allowed_readings) < 2:
            continue
        morph_range = _find_exact_morph_range(analysis["morphs"], char_span)
        if morph_range is None:
            # 誤読の原因切り分け (辞書・統合・モデルのどこで対象外になったか) に使うため、対象ごとに記録する
            diagnostics.record(
                diagnostics.TargetDiagnostic(
                    segment_text=analysis["normalized_text"],
                    char_span=char_span,
                    surface=surface,
                    outcome="no_exact_morph_range",
                )
            )
            continue
        # 数量表現の内部は最良経路の辞書読みで完結しており、部分表層ごとのモデル介入を行わない
        if any(
            protected_start <= char_span[0] and char_span[1] <= protected_end
            for protected_start, protected_end in dictionary_owned_quantity_ranges
        ):
            default_pronunciation = _default_path_pronunciation(
                analysis["morphs"],
                morph_range,
            )
            diagnostics.record(
                diagnostics.TargetDiagnostic(
                    segment_text=analysis["normalized_text"],
                    char_span=char_span,
                    surface=surface,
                    outcome="dictionary_default_protected",
                    reachable_pronunciations=(default_pronunciation,)
                    if default_pronunciation is not None
                    else (),
                    selected_pronunciation=default_pronunciation,
                    was_preserved=True,
                )
            )
            continue
        span_paths, is_reading_protected = _eligible_span_paths(
            analysis,
            char_span,
            surface,
            allowed_readings,
            nodes_by_id,
        )
        pronunciations = tuple(dict.fromkeys(path["pronunciation"] for path in span_paths))
        # 候補グラフ上で読み候補が2件未満なら、辞書の最良経路をそのまま維持する
        if len(pronunciations) < 2:
            diagnostics.record(
                diagnostics.TargetDiagnostic(
                    segment_text=analysis["normalized_text"],
                    char_span=char_span,
                    surface=surface,
                    outcome="reading_protected"
                    if is_reading_protected
                    else "lattice_reachable_lt2",
                    reachable_pronunciations=pronunciations,
                )
            )
            continue
        resolved_targets.append(
            _ResolvedTarget(
                char_span=char_span,
                morph_range=morph_range,
                surface=surface,
                span_paths=tuple(span_paths),
                pronunciations=pronunciations,
            )
        )

    if len(resolved_targets) > 0:
        predictions = model.predict(
            analysis["normalized_text"],
            tuple(item.to_reading_target() for item in resolved_targets),
        )
        resolved_targets = _resolve_selected_pronunciations(
            model,
            resolved_targets,
            predictions,
            analysis,
        )

        applicable_targets: list[_ResolvedTarget] = []
        for target in resolved_targets:
            # モデル候補外の辞書既定読みへ戻した対象は、既定経路を差し替えずここで処理を終える
            if (
                target.selected_pronunciation is not None
                and target.selected_pronunciation not in target.pronunciations
            ):
                diagnostics.record(
                    diagnostics.TargetDiagnostic(
                        segment_text=analysis["normalized_text"],
                        char_span=target.char_span,
                        surface=target.surface,
                        outcome="dictionary_default_protected",
                        reachable_pronunciations=(
                            target.selected_pronunciation,
                            *target.pronunciations,
                        ),
                        selected_pronunciation=target.selected_pronunciation,
                        score_margin=target.score_margin,
                        was_preserved=target.was_preserved,
                    )
                )
                continue
            applicable_targets.append(target)
        resolved_targets = applicable_targets

        # 候補グループ選択と形態素コスト再計算で同じ接続辺索引を共有する
        connection_costs = {
            (connection["left_node_id"], connection["right_node_id"]): connection["cost"]
            for connection in analysis["connections"]
        }
        # 隣接対象は接続費用をまとめて比較し、形態素を挟む対象だけを独立に選ぶ
        selected_paths: list[tuple[_ResolvedTarget, CandidatePath]] = []
        for target_group in _group_adjacent_targets(resolved_targets):
            group_paths = _select_joint_paths(connection_costs, target_group)
            if len(group_paths) == 0:
                # 接続辺が見つからずグループ全体の選択が破棄された対象を記録する
                for dropped_target in target_group:
                    diagnostics.record(
                        diagnostics.TargetDiagnostic(
                            segment_text=analysis["normalized_text"],
                            char_span=dropped_target.char_span,
                            surface=dropped_target.surface,
                            outcome="joint_path_dropped",
                            reachable_pronunciations=dropped_target.pronunciations,
                            selected_pronunciation=dropped_target.selected_pronunciation,
                            score_margin=dropped_target.score_margin,
                            was_preserved=dropped_target.was_preserved,
                        )
                    )
                continue
            selected_paths.extend(group_paths)

        applied_paths: list[tuple[_ResolvedTarget, CandidatePath]] = []
        # MeCab feature 列は元の形態素範囲を基準にするため、添字がずれないよう後方から差し替える
        for target, path in reversed(selected_paths):
            start, end = target.morph_range
            # 連続記号の展開で複数形態素が同じ feature を指す場合も、元ノードの範囲を正しく置換する
            target_feature_indices = tuple(
                feature_index
                for feature_index in analysis["feature_index_by_morph"][start:end]
                if feature_index is not None
            )
            # 対象範囲が無視形態素だけなら置換する MeCab feature が存在しない
            if len(target_feature_indices) == 0:
                diagnostics.record(
                    diagnostics.TargetDiagnostic(
                        segment_text=analysis["normalized_text"],
                        char_span=target.char_span,
                        surface=target.surface,
                        outcome="no_feature_replaced",
                        reachable_pronunciations=target.pronunciations,
                        selected_pronunciation=target.selected_pronunciation,
                        score_margin=target.score_margin,
                        was_preserved=target.was_preserved,
                    )
                )
                continue
            feature_start = target_feature_indices[0]
            feature_end = target_feature_indices[-1] + 1
            selected_features[feature_start:feature_end] = list(path["features"])
            applied_paths.append((target, path))
            diagnostics.record(
                diagnostics.TargetDiagnostic(
                    segment_text=analysis["normalized_text"],
                    char_span=target.char_span,
                    surface=target.surface,
                    outcome="applied",
                    reachable_pronunciations=target.pronunciations,
                    selected_pronunciation=target.selected_pronunciation,
                    score_margin=target.score_margin,
                    was_preserved=target.was_preserved,
                )
            )

        if include_morphs is False:
            return selected_features, []

        # 形態素列は選択経路の実コストを使い、差し替えと累積コスト計算を前方1回で済ませる
        selected_path_by_start = {
            target.morph_range[0]: (target, path) for target, path in applied_paths
        }
        selected_morphs: list[MeCabMorph] = []
        morph_index = 0
        cumulative_cost = 0
        previous_selected_node_id: int | None = None
        pending_right_link_cost: int | None = None
        while morph_index < len(analysis["morphs"]):
            selected_path = selected_path_by_start.get(morph_index)
            if selected_path is not None:
                target, path = selected_path
                node = nodes_by_id[path["node_ids"][0]]
                # 隣接候補は候補間の接続辺を使い、グループ先頭は固定された左境界を使う
                if previous_selected_node_id is None:
                    link_cost = path["left_boundary_cost"]
                else:
                    link_cost = connection_costs[(previous_selected_node_id, path["node_ids"][0])]
                cumulative_cost += link_cost
                selected_morphs.append(_replace_morph(node, link_cost, cumulative_cost))
                morph_index = target.morph_range[1]
                previous_selected_node_id = path["node_ids"][-1]
                pending_right_link_cost = path["right_link_cost"]
                continue

            base_morph = analysis["morphs"][morph_index]
            # 選択グループ直後だけは右境界の実コストへ置き換え、それ以外は最良経路の値を保つ
            link_cost = (
                pending_right_link_cost
                if pending_right_link_cost is not None
                else base_morph["link_cost"]
            )
            cumulative_cost += link_cost
            selected_morphs.append(
                _copy_morph(
                    base_morph,
                    link_cost=link_cost,
                    node_cost=cumulative_cost,
                )
            )
            morph_index += 1
            previous_selected_node_id = None
            pending_right_link_cost = None
    else:
        if include_morphs is False:
            return selected_features, []
        selected_morphs = [_copy_morph(morph) for morph in analysis["morphs"]]

    return selected_features, selected_morphs


def _find_target_spans(
    text: str,
    surfaces_by_first_character: dict[str, tuple[str, ...]],
) -> tuple[tuple[int, int], ...]:
    """
    推論対象表層を最長一致させ、内側の短い対象を除いた範囲を返す。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        surfaces_by_first_character (dict[str, tuple[str, ...]]): 先頭文字別・長さ降順の表層索引

    Returns:
        tuple[tuple[int, int], ...]: 重なりのない対象表層の半開区間 (文字位置昇順)
    """

    occurrences: list[tuple[int, int]] = []
    for start, character in enumerate(text):
        # 同じ開始位置では長い表層から確認し、最長の1件だけを対象候補にする
        for surface in surfaces_by_first_character.get(character, ()):
            if text.startswith(surface, start):
                occurrences.append((start, start + len(surface)))
                break
    selected: list[tuple[int, int]] = []
    last_selected_end = 0
    for occurrence in occurrences:
        # 出現は開始位置順なので、直前に採用した範囲の終端だけで重なりを判定できる
        if occurrence[0] < last_selected_end:
            continue
        selected.append(occurrence)
        last_selected_end = occurrence[1]
    return tuple(selected)


def _split_target_processing_segments(
    text: str,
    target_spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """
    対象を含む文だけを独立させ、連続する対象なし文をまとめた範囲を返す。
    引用符・括弧の入れ子を考慮して句点分割し、長文の前置きで MeCab 呼び出しが増えないようにする。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        target_spans (tuple[tuple[int, int], ...]): 推論対象表層の半開区間

    Returns:
        tuple[tuple[int, int], ...]: `text` 上の処理区間 (半開区間) の列。
            要素が1件のときは全文を1区間として返す
    """

    matched_openings, matched_closings = _matched_delimiter_indices(text)
    sentence_ranges: list[tuple[int, int]] = []
    sentence_start = 0
    delimiter_depth = 0
    for index, character in enumerate(text):
        # 対応が取れた括弧だけネスト深さを増やし、閉じていない開き括弧で残り全文を巻き込まない
        if index in matched_openings:
            delimiter_depth += 1
        if character in "。！？!?\n" and delimiter_depth == 0:
            sentence_ranges.append((sentence_start, index + 1))
            sentence_start = index + 1
        if index in matched_closings:
            delimiter_depth -= 1
    if sentence_start < len(text):
        sentence_ranges.append((sentence_start, len(text)))
    if len(sentence_ranges) <= 1:
        return ((0, len(text)),)

    segments: list[tuple[int, int, bool]] = []
    for range_start, range_end in sentence_ranges:
        has_target = any(
            range_start <= target_start and target_end <= range_end
            for target_start, target_end in target_spans
        )
        # 対象のない文は連結し、長い前置きで MeCab 呼び出しが文数分に増えないようにする
        if segments and has_target is False and segments[-1][2] is False:
            segments[-1] = (segments[-1][0], range_end, False)
        else:
            segments.append((range_start, range_end, has_target))
    return tuple((start, end) for start, end, _ in segments)


def _matched_delimiter_indices(text: str) -> tuple[frozenset[int], frozenset[int]]:
    """
    入れ子順まで対応した括弧と引用符の文字位置を返す。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト

    Returns:
        tuple[frozenset[int], frozenset[int]]: 対応が取れた開き括弧位置と閉じ括弧位置
    """

    expected_closings: list[tuple[str, int]] = []
    matched_openings: set[int] = set()
    matched_closings: set[int] = set()
    for index, character in enumerate(text):
        expected_closing = _CLOSING_DELIMITER_BY_OPENING.get(character)
        if expected_closing is not None:
            expected_closings.append((expected_closing, index))
            continue
        # 入れ子の途中に異なる閉じ括弧があっても、外側との誤対応は作らない
        if len(expected_closings) == 0 or expected_closings[-1][0] != character:
            continue
        _, opening_index = expected_closings.pop()
        matched_openings.add(opening_index)
        matched_closings.add(index)
    return frozenset(matched_openings), frozenset(matched_closings)


def _find_exact_morph_range(
    morphs: tuple[MeCabMorph, ...], char_span: tuple[int, int]
) -> tuple[int, int] | None:
    """
    対象範囲を隙間なく覆う既定形態素列の半開区間を返す。

    Args:
        morphs (tuple[MeCabMorph, ...]): MeCab の既定最良経路の形態素列
        char_span (tuple[int, int]): 対象表層の半開区間

    Returns:
        tuple[int, int] | None: 形態素添字の半開区間。境界が一意に定まらない場合は None
    """

    start_indices = [
        index for index, morph in enumerate(morphs) if morph["char_span"][0] == char_span[0]
    ]
    if len(start_indices) != 1:
        return None
    start = start_indices[0]
    cursor = char_span[0]
    for end in range(start, len(morphs)):
        morph_span = morphs[end]["char_span"]
        if morph_span[0] != cursor or morph_span[1] > char_span[1]:
            return None
        cursor = morph_span[1]
        if cursor == char_span[1]:
            return start, end + 1
    return None


def _eligible_span_paths(
    analysis: ReadingAnalysis,
    char_span: tuple[int, int],
    surface: str,
    allowed_readings: frozenset[str],
    nodes_by_id: dict[int, CandidateNode],
) -> tuple[list[CandidatePath], bool]:
    """
    最良経路の形態素範囲で差し替え可能な辞書候補を返す。

    Args:
        analysis (ReadingAnalysis): `analyze_mecab_candidates()` の結果
        char_span (tuple[int, int]): 対象表層の半開区間
        surface (str): 対象表層
        allowed_readings (frozenset[str]): メタデータが許可する発音
        nodes_by_id (dict[int, CandidateNode]): 候補ノード ID からノードへの索引

    Returns:
        tuple[list[CandidatePath], bool]: 許可読みを実現する候補経路と、保護候補で停止したかの診断値
    """

    paths = [
        path
        for path in analysis["paths"]
        if path["char_span"] == char_span and path["surface"] == surface
    ]
    # ユーザー辞書の保護候補やメタデータ外の読みが混在する範囲では tsqyomi による差し替えを止める
    if any(nodes_by_id[path["node_ids"][0]]["is_reading_protected"] is True for path in paths):
        return [], True
    return [
        path
        for path in paths
        if canonicalize_pronunciation(path["pronunciation"]) in allowed_readings
        and nodes_by_id[path["node_ids"][0]]["is_ignored"] is False
    ], False


def _group_adjacent_targets(
    targets: list[_ResolvedTarget],
) -> list[tuple[_ResolvedTarget, ...]]:
    """
    形態素を挟まない対象を同じ候補経路選択へまとめる。

    Args:
        targets (list[_ResolvedTarget]): モデルが選んだ読みと形態素範囲

    Returns:
        list[tuple[_ResolvedTarget, ...]]: 隣接グループごとの対象列
    """

    groups: list[list[_ResolvedTarget]] = []
    for target in sorted(targets, key=lambda item: item.morph_range):
        if groups and groups[-1][-1].morph_range[1] == target.morph_range[0]:
            groups[-1].append(target)
        else:
            groups.append([target])
    return [tuple(group) for group in groups]


def _select_joint_paths(
    connection_costs: dict[tuple[int, int], int],
    targets: tuple[_ResolvedTarget, ...],
) -> list[tuple[_ResolvedTarget, CandidatePath]]:
    """
    選択済み読みを実現する候補経路の組を、固定した外側経路の費用で決める。

    Args:
        connection_costs (dict[tuple[int, int], int]): 候補ノード間の接続費用索引
        targets (tuple[_ResolvedTarget, ...]): 同一グループ内の隣接対象

    Returns:
        list[tuple[_ResolvedTarget, CandidatePath]]: 各対象と選ばれた候補経路の対
    """

    if len(targets) == 1:
        # 保護対象の既定読みが候補外の場合は実現経路が存在しないため、交換せず辞書のまま維持する
        if len(targets[0].selected_paths) == 0:
            return []
        best_path = min(
            targets[0].selected_paths,
            key=lambda path: (path["boundary_cost"], path["path_id"]),
        )
        return [(targets[0], best_path)]

    # 各候補までの最小費用だけを次の対象に引き継ぎ、候補数の直積を作らずに最小経路を求める
    states = [
        (
            path["left_boundary_cost"],
            (path["path_id"],),
            (path,),
        )
        for path in targets[0].selected_paths
    ]
    for target in targets[1:]:
        next_states: list[tuple[int, tuple[int, ...], tuple[CandidatePath, ...]]] = []
        for right_path in target.selected_paths:
            candidates: list[tuple[int, tuple[int, ...], tuple[CandidatePath, ...]]] = []
            for cost, path_ids, paths in states:
                connection_cost = connection_costs.get(
                    (paths[-1]["node_ids"][-1], right_path["node_ids"][0])
                )
                if connection_cost is None:
                    continue
                candidates.append(
                    (
                        cost + connection_cost,
                        (*path_ids, right_path["path_id"]),
                        (*paths, right_path),
                    )
                )
            if len(candidates) > 0:
                next_states.append(min(candidates, key=lambda state: (state[0], state[1])))
        states = next_states
        if len(states) == 0:
            break

    if len(states) == 0:
        return []
    _, _, best_paths = min(
        states,
        key=lambda state: (
            state[0] + state[2][-1]["right_boundary_cost"],
            state[1],
        ),
    )
    return list(zip(targets, best_paths))


def _replace_morph(node: CandidateNode, link_cost: int, node_cost: int) -> MeCabMorph:
    """
    選択した辞書ノードの feature を NJD 入力と詳細形態素へ反映する。

    Args:
        node (CandidateNode): 採用する辞書候補ノード
        link_cost (int): 直前ノードから候補ノードへの単語コスト込み局所コスト
        node_cost (int): 候補ノードまでの累積コスト

    Returns:
        MeCabMorph: 候補ノードの surface・feature・コスト情報を反映した形態素
    """

    return MeCabMorph(
        surface=node["surface"],
        features=node["feature"].split(","),
        char_span=node["char_span"],
        pos_id=node["pos_id"],
        left_id=node["left_id"],
        right_id=node["right_id"],
        word_cost=node["word_cost"],
        link_cost=link_cost,
        node_cost=node_cost,
        is_unknown=node["is_unknown"],
        is_ignored=node["is_ignored"],
        dictionary_index=node["dictionary_index"],
    )


def _copy_morph(
    morph: MeCabMorph,
    *,
    link_cost: int | None = None,
    node_cost: int | None = None,
) -> MeCabMorph:
    """
    詳細形態素を独立した feature 列とともに複製する。

    Args:
        morph (MeCabMorph): 複製元の詳細形態素
        link_cost (int | None): 上書きする局所コスト。None の場合は元の値
        node_cost (int | None): 上書きする累積コスト。None の場合は元の値

    Returns:
        MeCabMorph: 必要なコストだけを上書きした複製
    """

    return MeCabMorph(
        surface=morph["surface"],
        features=list(morph["features"]),
        char_span=morph["char_span"],
        pos_id=morph["pos_id"],
        left_id=morph["left_id"],
        right_id=morph["right_id"],
        word_cost=morph["word_cost"],
        link_cost=morph["link_cost"] if link_cost is None else link_cost,
        node_cost=morph["node_cost"] if node_cost is None else node_cost,
        is_unknown=morph["is_unknown"],
        is_ignored=morph["is_ignored"],
        dictionary_index=morph["dictionary_index"],
    )
