"""記号のポーズ判定カスタマイズを検証するテスト。"""

from __future__ import annotations

import pyopenjtalk


def test_default_is_non_pause_symbol_accepts_quotes_and_brackets() -> None:
    """デフォルトの非ポーズ記号判定関数において、各種括弧や引用符が True（ポーズを挿入しない記号）と判定され、句読点などは False と判定されることを確認する。"""

    non_pause_symbols = "「」『』（）()【】［］[]〈〉《》〔〕｛｝{}\"'”“’‘"
    assert all(pyopenjtalk.default_is_non_pause_symbol(symbol) for symbol in non_pause_symbols)
    assert pyopenjtalk.default_is_non_pause_symbol("、") is False
    assert pyopenjtalk.default_is_non_pause_symbol("。") is False
    assert pyopenjtalk.default_is_non_pause_symbol("…") is False


def test_custom_non_pause_symbol_rule_adds_and_removes_pause() -> None:
    """カスタムの非ポーズ記号判定関数を渡すことで、通常はポーズにならない括弧に pau を挿入したり、句読点から pau を除去できることを確認する。"""

    def opening_quote_is_pause(surface: str) -> bool:
        """「「」（開き鉤括弧）をポーズ対象（False）として扱い、それ以外の括弧は非ポーズとするテスト用判定関数。"""

        if surface == "「":
            return False
        return pyopenjtalk.default_is_non_pause_symbol(surface)

    custom_mapping = pyopenjtalk.g2p_mapping(
        "「あ」",
        is_non_pause_symbol=opening_quote_is_pause,
    )
    assert [entry["phonemes"] for entry in custom_mapping] == [["pau"], ["a"], []]
    assert [entry["is_ignored"] for entry in custom_mapping] == [False, False, True]
    assert [entry["char_span"] for entry in custom_mapping] == [(0, 1), (1, 2), (2, 3)]

    no_pause_mapping = pyopenjtalk.g2p_mapping(
        "あ、い。",
        is_non_pause_symbol=lambda _surface: True,
    )
    assert [entry["phonemes"] for entry in no_pause_mapping] == [["a"], [], ["i"], []]
    assert [entry["is_ignored"] for entry in no_pause_mapping] == [False, True, False, True]
    assert [entry["char_span"] for entry in no_pause_mapping] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
