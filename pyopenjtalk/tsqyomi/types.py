from __future__ import annotations

from typing_extensions import TypedDict

from ..types import MeCabMorph


class CandidateNode(TypedDict):
    """
    MeCab 候補グラフから Python 側へコピーした辞書ノード。
    `OpenJTalk.analyze_mecab_candidates()` が Lattice 走査結果から構築する。
    """

    node_id: int  # MeCab Lattice 上のノード ID
    surface: str  # 表層形
    feature: str  # MeCab feature 文字列 (カンマ区切り)
    pronunciation: str  # feature から抽出した発音 (カタカナ)
    char_span: tuple[int, int]  # 正規化本文上の半開区間
    pos_id: int  # 品詞 ID
    left_id: int  # 左文脈 ID
    right_id: int  # 右文脈 ID
    word_cost: int  # 単語コスト
    dictionary_index: int  # 辞書インデックス (0=システム, 1..N=ユーザー辞書)
    is_unknown: bool  # 未知語か
    is_ignored: bool  # "記号,空白" か
    is_reading_protected: bool  # tsqyomi 差し替えから保護するユーザー辞書候補か


class CandidatePath(TypedDict):
    """
    1つの読みを実現する解析内で一意な候補経路。
    """

    path_id: int  # 同一 `(char_span, pronunciation)` 内での安定ソート用 ID
    node_ids: tuple[int, ...]  # 経路を構成する候補ノード ID 列
    char_span: tuple[int, int]  # 対象表層の半開区間
    surface: str  # 対象表層
    pronunciation: str  # 候補経路の発音
    features: tuple[str, ...]  # 候補ノード surface 列
    left_boundary_cost: int  # 左外側 MeCab ノードとの境界コスト
    right_boundary_cost: int  # 右外側 MeCab ノードとの境界コスト
    right_link_cost: int  # 右外側 MeCab ノードの単語コストを含む局所コスト
    boundary_cost: int  # 左境界 + 候補内部連接 + 右境界の合計


class CandidateConnection(TypedDict):
    """
    2つの候補ノードを接続する MeCab の局所費用。
    """

    left_node_id: int  # 左候補ノード ID
    right_node_id: int  # 右候補ノード ID
    cost: int  # MeCab connector 由来の連接コスト


class ReadingAnalysis(TypedDict):
    """
    tsqyomi 読み選択へ渡す固定済みの最良経路と候補グラフ。
    `OpenJTalk.analyze_mecab_candidates()` の戻り値。
    """

    normalized_text: str  # MeCab 入力と同じ規則で正規化した本文
    features: tuple[str, ...]  # 最良経路の MeCab feature 列 ("記号,空白" 除外)
    morphs: tuple[MeCabMorph, ...]  # 最良経路の詳細形態素列
    nodes: tuple[CandidateNode, ...]  # 公開対象 span に一致する候補ノード
    paths: tuple[CandidatePath, ...]  # 公開対象 span の候補経路
    connections: tuple[CandidateConnection, ...]  # 公開候補ノード間の接続辺
