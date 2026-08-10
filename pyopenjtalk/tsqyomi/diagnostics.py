from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal


# 対象がモデル推論や feature 差し替えに至らなかった理由、または適用結果の分類
TargetDiagnosticOutcome = Literal[
    "applied",
    "no_exact_morph_range",
    "reading_protected",
    "dictionary_default_protected",
    "lattice_reachable_lt2",
    "joint_path_dropped",
    "no_feature_replaced",
]

TARGET_OUTCOMES: tuple[TargetDiagnosticOutcome, ...] = (
    "applied",  # モデル選択が特徴列へ適用された
    "no_exact_morph_range",  # 最良経路の形態素境界が対象範囲と一致しない
    "reading_protected",  # ユーザー辞書の保護候補が範囲に混在し差し替えを止めた
    "dictionary_default_protected",  # 前接名詞との複合用法で、接尾辞の辞書既定読みを維持
    "lattice_reachable_lt2",  # 候補グラフ上で到達可能な発音が2件未満
    "joint_path_dropped",  # 隣接対象グループの接続辺が見つからず選択が破棄された
    "no_feature_replaced",  # 対象範囲が無視形態素だけで置換する MeCab feature が無い
)


@dataclass(frozen=True)
class TargetDiagnostic:
    """
    1対象の統合経路診断。

    Attributes:
        segment_text (str): 対象を処理した分割区間の正規化本文
        char_span (tuple[int, int]): 正規化本文上の対象表層の半開区間
        surface (str): 対象表層
        outcome (TargetDiagnosticOutcome): `TARGET_OUTCOMES` のいずれか
        reachable_pronunciations (tuple[str, ...]): 候補グラフ上で到達可能だった発音
        selected_pronunciation (str | None): モデル (保護規則適用後) が選んだ発音
        score_margin (float | None): モデルが選んだ1位と2位の候補スコアの差
        was_preserved (bool): 保護条件により辞書既定読みを維持したか
    """

    segment_text: str
    char_span: tuple[int, int]
    surface: str
    outcome: TargetDiagnosticOutcome
    reachable_pronunciations: tuple[str, ...] = ()
    selected_pronunciation: str | None = None
    score_margin: float | None = None
    was_preserved: bool = False


_records: ContextVar[tuple[TargetDiagnostic, ...] | None] = ContextVar(
    "tsqyomi_diagnostic_records",
    default=None,
)


def start_recording() -> None:
    """診断の収集を開始する (既存の記録は破棄する)。"""

    _records.set(())


def stop_recording() -> list[TargetDiagnostic]:
    """
    診断の収集を終了し、記録を返す。

    Returns:
        list[TargetDiagnostic]: 収集した対象別診断 (処理順)
    """

    collected = _records.get()
    _records.set(None)
    return list(collected) if collected is not None else []


def is_recording() -> bool:
    """
    収集中かを返す。

    Returns:
        bool: `start_recording()` 済みで未終了なら True
    """

    return _records.get() is not None


def record_count() -> int:
    """
    収集中の診断件数を返す。

    Returns:
        int: 収集中の診断件数。収集中でなければ 0
    """

    records = _records.get()
    return len(records) if records is not None else 0


def record(diagnostic: TargetDiagnostic) -> None:
    """
    収集中のときだけ診断を1件追加する。

    Args:
        diagnostic (TargetDiagnostic): 追加する対象別診断
    """

    records = _records.get()
    if records is not None:
        _records.set((*records, diagnostic))


def rebase_recording_char_spans(start_index: int, char_offset: int) -> None:
    """
    分割入力へ相対化された診断位置を元の本文位置へ戻す。

    Args:
        start_index (int): 位置を変換する診断の開始添字
        char_offset (int): 分割片の本文先頭が元の本文で始まる位置
    """

    records = _records.get()
    if records is None:
        return
    # 再帰した分割片の診断だけに親本文のオフセットを加え、先に処理した片の位置はそのまま保持する
    _records.set(
        (
            *records[:start_index],
            *(
                replace(
                    diagnostic,
                    char_span=(
                        diagnostic.char_span[0] + char_offset,
                        diagnostic.char_span[1] + char_offset,
                    ),
                )
                for diagnostic in records[start_index:]
            ),
        )
    )
