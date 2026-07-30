"""tsqyomi の推論対象を1文と最大系列長の範囲へ収める。"""

from __future__ import annotations

from collections.abc import Callable, Sequence


_SENTENCE_TERMINATORS = frozenset("。！？!?\n\r")
_TRAILING_CLOSERS = frozenset("」』）】〉》〕］}】〗〙〛")
_TARGET_START_MARKER = "【"
_TARGET_END_MARKER = "】"
# サブワード境界の局所的な系列長変動だけを追加確認する文字半径
_TOKEN_BOUNDARY_PROBE_RADIUS = 32


def _find_sentence_span(text: str, char_span: tuple[int, int]) -> tuple[int, int]:
    """
    対象範囲を含む1文の半開区間を返す。

    Args:
        text (str): 入力本文
        char_span (tuple[int, int]): 対象語の半開区間

    Returns:
        tuple[int, int]: 対象語を含む文の半開区間
    """

    target_start, target_end = char_span
    sentence_start = 0
    # 対象の直前から逆走し、直前の文末とそれに続く閉じ括弧の次を文頭にする
    for char_index in range(target_start - 1, -1, -1):
        if text[char_index] in _SENTENCE_TERMINATORS:
            sentence_start = char_index + 1
            # 直前文の終端に属する閉じ括弧を、新しい文の先頭から除外する
            while sentence_start < target_start and text[sentence_start] in _TRAILING_CLOSERS:
                sentence_start += 1
            break

    sentence_end = len(text)
    # 対象の直後から最初の文末を探し、連続記号と閉じ括弧までを同じ文に含める
    for char_index in range(target_end, len(text)):
        if text[char_index] in _SENTENCE_TERMINATORS:
            sentence_end = char_index + 1
            # 連続する終端記号と直後の閉じ括弧も同じ発話単位へ含める
            while sentence_end < len(text) and (
                text[sentence_end] in _SENTENCE_TERMINATORS
                or text[sentence_end] in _TRAILING_CLOSERS
            ):
                sentence_end += 1
            break
    return sentence_start, sentence_end


def _mark_target(text: str, char_span: tuple[int, int]) -> str:
    """対象範囲をモデルが学習したマーカーで囲む。

    Args:
        text (str): 対象語を含む入力本文
        char_span (tuple[int, int]): 対象語の半開区間

    Returns:
        str: 対象マーカーを挿入した本文
    """

    target_start, target_end = char_span
    return (
        text[:target_start]
        + _TARGET_START_MARKER
        + text[target_start:target_end]
        + _TARGET_END_MARKER
        + text[target_end:]
    )


def build_model_context(
    text: str,
    char_span: tuple[int, int],
    candidate_pronunciations: Sequence[str],
    encoded_length: Callable[[str, str], int],
    model_max_length: int,
) -> str:
    """
    対象語を含む1文を切り出し、モデルの最大系列長へ収める。

    Args:
        text (str): 入力本文
        char_span (tuple[int, int]): 対象語の半開区間
        candidate_pronunciations (Sequence[str]): 比較する候補発音
        encoded_length (Callable[[str, str], int]): 本文と候補発音を符号化した系列長を返す関数
        model_max_length (int): モデルへ渡せる最大系列長

    Returns:
        str: 対象マーカーを付けたモデル入力本文

    Raises:
        ValueError: 候補が空、対象範囲が不正、または対象語と候補だけで最大系列長を超える場合
    """

    target_start, target_end = char_span
    # 候補が空の場合は系列長の最大値を定義できないため、入力契約の違反として明示的に拒否
    if len(candidate_pronunciations) == 0:
        raise ValueError("candidate_pronunciations must not be empty")
    # 範囲外アクセスや空の対象を後段の文境界探索へ持ち込まない
    if target_start < 0 or target_end > len(text) or target_start >= target_end:
        raise ValueError("char_span must identify a non-empty range within text")

    sentence_start, sentence_end = _find_sentence_span(text, char_span)
    # 元文の対象位置を切り出した1文内の位置へ変換してからマーカーを付ける
    sentence = text[sentence_start:sentence_end]
    sentence_target_span = (target_start - sentence_start, target_end - sentence_start)
    marked_sentence = _mark_target(sentence, sentence_target_span)
    maximum_encoded_length = max(
        encoded_length(marked_sentence, pronunciation) for pronunciation in candidate_pronunciations
    )
    if maximum_encoded_length <= model_max_length:
        return marked_sentence

    # 長い1文では対象語の左右を同程度残し、系列長が増える通常の範囲を二分探索する
    target_text = sentence[sentence_target_span[0] : sentence_target_span[1]]
    minimum_context = _TARGET_START_MARKER + target_text + _TARGET_END_MARKER
    if (
        max(
            encoded_length(minimum_context, pronunciation)
            for pronunciation in candidate_pronunciations
        )
        > model_max_length
    ):
        raise ValueError("target and candidate pronunciation exceed the model maximum length")

    left_available = sentence_target_span[0]
    right_available = len(sentence) - sentence_target_span[1]
    maximum_radius = max(left_available, right_available)
    lower_radius = 0
    upper_radius = maximum_radius
    best_radius = 0
    best_context = minimum_context
    while lower_radius <= upper_radius:
        radius = (lower_radius + upper_radius) // 2
        window_start = max(0, sentence_target_span[0] - radius)
        window_end = min(len(sentence), sentence_target_span[1] + radius)
        window = sentence[window_start:window_end]
        window_target_span = (
            sentence_target_span[0] - window_start,
            sentence_target_span[1] - window_start,
        )
        marked_window = _mark_target(window, window_target_span)
        if (
            max(
                encoded_length(marked_window, pronunciation)
                for pronunciation in candidate_pronunciations
            )
            <= model_max_length
        ):
            best_radius = radius
            best_context = marked_window
            lower_radius = radius + 1
        else:
            upper_radius = radius - 1

    # サブワード分割は境界文字の追加で系列長が一時的に減るため、二分探索直後だけ実測で補正
    ## 全半径の線形探索は数千文字の入力で秒単位になるため、局所的な境界変動へ範囲を限定する
    probe_upper_radius = min(maximum_radius, best_radius + _TOKEN_BOUNDARY_PROBE_RADIUS)
    for radius in range(best_radius + 1, probe_upper_radius + 1):
        window_start = max(0, sentence_target_span[0] - radius)
        window_end = min(len(sentence), sentence_target_span[1] + radius)
        window = sentence[window_start:window_end]
        window_target_span = (
            sentence_target_span[0] - window_start,
            sentence_target_span[1] - window_start,
        )
        marked_window = _mark_target(window, window_target_span)
        if (
            max(
                encoded_length(marked_window, pronunciation)
                for pronunciation in candidate_pronunciations
            )
            <= model_max_length
        ):
            best_context = marked_window
    return best_context
