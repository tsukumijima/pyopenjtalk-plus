from typing import TypedDict


class NJDFeature(TypedDict):
    """OpenJTalk の形態素解析結果・アクセント推定結果を表す型"""

    string: str  # 表層形
    pos: str  # 品詞
    pos_group1: str  # 品詞細分類1
    pos_group2: str  # 品詞細分類2
    pos_group3: str  # 品詞細分類3
    ctype: str  # 活用型
    cform: str  # 活用形
    orig: str  # 原形
    read: str  # 読み
    pron: str  # 発音形式
    acc: int  # アクセント型 (0: 平板型, 1-n: n番目のモーラにアクセント核)
    mora_size: int  # モーラ数
    chain_rule: str  # アクセント結合規則
    chain_flag: int  # アクセント句の連結フラグ
