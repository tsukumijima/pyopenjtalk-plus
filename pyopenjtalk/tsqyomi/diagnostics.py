"""tsqyomi 統合経路の対象別診断を opt-in で記録する。

モデルの読み選択が製品出力まで生き残ったか、どの経路で捨てられたかを対象単位で観測する。
既定では何も記録せず、評価・監査ツールが `start_recording()` を呼んだときだけ収集する。
"""

from __future__ import annotations

from dataclasses import dataclass


# 対象がモデル推論・特徴列差し替えへ到達できなかった理由、または適用結果の分類
TARGET_OUTCOMES = (
    "applied",  # モデル選択が特徴列へ適用された
    "no_exact_morph_range",  # 最良経路の形態素境界が対象範囲と一致しない
    "reading_protected",  # ユーザー辞書の保護候補が範囲に混在し差し替えを止めた
    "lattice_reachable_lt2",  # 候補グラフ上で到達可能な発音が2件未満
    "joint_path_dropped",  # 隣接対象グループの接続辺が見つからず選択が破棄された
    "no_feature_replaced",  # 対象範囲が無視形態素だけで置換する feature が無い
)


@dataclass(frozen=True)
class TargetDiagnostic:
    """
    1対象の統合経路診断。

    Attributes:
        segment_text (str): 対象を処理した分割区間の正規化本文
        char_span (tuple[int, int]): 分割区間内の対象表層の半開区間
        surface (str): 対象表層
        outcome (str): `TARGET_OUTCOMES` のいずれか
        reachable_pronunciations (tuple[str, ...]): 候補グラフ上で到達可能だった発音
        selected_pronunciation (str | None): モデル (保護規則適用後) が選んだ発音
        was_preserved (bool): 構造保全ペアで辞書既定読みへ差し戻されたか
    """

    segment_text: str
    char_span: tuple[int, int]
    surface: str
    outcome: str
    reachable_pronunciations: tuple[str, ...] = ()
    selected_pronunciation: str | None = None
    was_preserved: bool = False


_records: list[TargetDiagnostic] | None = None


def start_recording() -> None:
    """診断の収集を開始する (既存の記録は破棄する)。"""

    global _records
    _records = []


def stop_recording() -> list[TargetDiagnostic]:
    """
    診断の収集を終了し、記録を返す。

    Returns:
        list[TargetDiagnostic]: 収集した対象別診断 (処理順)
    """

    global _records
    collected = _records if _records is not None else []
    _records = None
    return collected


def is_recording() -> bool:
    """
    収集中かを返す。

    Returns:
        bool: `start_recording()` 済みで未終了なら True
    """

    return _records is not None


def record(diagnostic: TargetDiagnostic) -> None:
    """
    収集中のときだけ診断を1件追加する。

    Args:
        diagnostic (TargetDiagnostic): 追加する対象別診断
    """

    if _records is not None:
        _records.append(diagnostic)
