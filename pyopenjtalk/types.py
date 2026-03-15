from typing import TypedDict


class NJDFeature(TypedDict):
    """
    OpenJTalk の形態素解析結果・アクセント推定結果を表す型。
    """

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


class MeCabMorph(TypedDict):
    """
    MeCab の形態素解析結果。
    通常の run_mecab() が返す feature 文字列に加え、
    MeCab の Lattice ノードから取得した未知語フラグやコスト情報を含む。
    """

    surface: str  # 表層形
    feature: str  # MeCab から返される feature 文字列
    is_unknown: bool  # MeCab が未知語と判定したか (stat == MECAB_UNK_NODE)
    is_ignored: bool  # OpenJTalk パイプラインで無視されるトークンか (記号,空白)
    pos_id: int  # 品詞 ID (現在の naist-jdic ベースの辞書では左文脈 ID / 右文脈 ID と同一値)
    word_cost: int  # 単語コスト (辞書に登録されたコスト、低いほど出現しやすい)
    cost: int  # BOS からこのノードまでの最小累積コスト


class WordPhonemeDetail(TypedDict):
    """
    形態素と対応する音素列のマッピング（未知語・無視トークン情報付き）。make_phoneme_mapping() の戻り値。
    make_phoneme_mapping() に morphs 引数が渡された場合は MeCab の未知語判定情報・無視トークン情報が付与され、
    make_phoneme_mapping() に morphs 引数が省略された場合は is_unknown=False, is_ignored は音素列から推定される。

    Note: MeCabMorph.is_ignored（記号や空白かどうかの判定）とは判定基準が異なる。
    MeCabMorph.is_ignored は、MeCab の feature に「記号,空白」が含まれているかで判定する。
    WordPhonemeDetail.is_ignored は、Cython で音素のマッピングを行った結果、対応する音素列が空なら True になる
    (Haqumei の map.phonemes.is_empty() と同じ判定ロジック)。
    つまり、記号や空白だけでなく、文頭にある 'ー' など音素が割り当てられないトークンも is_ignored=True になる。
    """

    word: str  # 表層形
    phonemes: list[str]  # 対応する音素列
    is_unknown: bool  # MeCab が未知語と判定したか
    is_ignored: bool  # OpenJTalk が音素を生成しなかったか (元の音素列が空)
