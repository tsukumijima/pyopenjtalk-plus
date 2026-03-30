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
    acc: int  # アクセント核位置 (0: 平板型, 1-n: n番目のモーラにアクセント核)
    # acc: chain_flag=1 で前の語と連結された場合、アクセント句の先頭語の acc が
    ## chain_rule に基づいて句全体のアクセント核位置に更新される
    ## (先頭語以外の acc は更新されないので、句全体の核位置は先頭語の acc を参照する)
    mora_size: int  # モーラ数
    chain_rule: str  # アクセント結合規則 (C1-C5/F1-F5/P1-P2 等)
    # chain_rule: njd_set_accent_type が chain_flag=1 のノードを連結する際に、
    ## 連結後のアクセント核位置をどう計算するかを決めるルール文字列
    ## 主なルール:
    ##   C1: 先頭モーラ数 + 後続語の acc（名詞結合用）
    ##   C2: 先頭モーラ数 + 1
    ##   C3: 先頭モーラ数
    ##   C4: 0（平板化）
    ##   F2: 先頭語が平板型の場合のみ先頭モーラ数 + add_type
    ##   P1/P2: 接頭語用の特殊ルール
    chain_flag: int  # アクセント句連結フラグ
    # chain_flag: njd_set_accent_phrase が品詞・活用に基づいて設定する
    ## -1: njd_set_accent_phrase がループの先頭ノードを処理しないため残る初期値。0 と同義
    ##  0: 新しいアクセント句の開始（前の語とは別のアクセント句）
    ##  1: 前の語と同じアクセント句に連結（助詞・助動詞・接尾語など）


class MeCabMorph(TypedDict):
    """
    MeCab の形態素解析結果。
    通常の run_mecab() が返す feature 文字列に加え、
    MeCab の Lattice ノードから取得した未知語フラグやコスト情報を含む。
    """

    surface: str  # 表層形
    features: list[str]  # MeCab feature 文字列の分割リスト（13 列目以降はカスタムフィールド）
    # features: 既知語は 12 列、未知語は 8 列（読み/発音/acc/chain_rule がない）
    pos_id: int  # 品詞 ID (pos-id.def で定義。品詞4分類による粗い分類で、文脈 ID とは別物)
    left_id: int  # 左文脈 ID (left-id.def で定義。連接コスト行列のインデックスとして使われる)
    right_id: int  # 右文脈 ID (right-id.def で定義。連接コスト行列のインデックスとして使われる)
    word_cost: int  # 単語コスト (辞書に登録されたコスト。低いほど出現しやすい)
    is_unknown: bool  # MeCab が未知語と判定したか (stat == MECAB_UNK_NODE)
    is_ignored: bool  # OpenJTalk パイプラインで無視されるトークンか ("記号,空白")


class SurfacePhonemeMapping(TypedDict):
    """
    形態素と対応する音素列のマッピング（未知語・無視トークン情報付き）。
    事実上 NJDFeature のスーパーセットとなっているが、意味明確化のため一部フィールドの名称を変更している。

    NOTE: MeCabMorph.is_ignored（記号や空白かどうかの判定）とは判定基準が異なる。
    MeCabMorph.is_ignored は、MeCab の feature に「記号,空白」が含まれているかで判定する。
    SurfacePhonemeMapping.is_ignored は、Cython で音素のマッピングを行った結果、対応する音素列が空なら True になる
    (Haqumei の map.phonemes.is_empty() と同じ判定ロジック)。
    つまり、記号や空白だけでなく、文頭にある 'ー' など音素が割り当てられないトークンも is_ignored=True になる。
    """

    surface: str  # 表層形
    phonemes: list[str]  # 対応する音素列
    features: list[str]  # MeCab feature 文字列の分割リスト（13 列目以降はカスタムフィールド）
    # features: 既知語は 12 列、未知語は 8 列（読み/発音/acc/chain_rule がない）
    ## make_phoneme_mapping() が morphs 付きで呼ばれた場合、アライメントで対応する MeCab morph の features を転写する
    ## morphs なしの場合や、数字正規化・踊り字展開で morph と NJD の surface が一致しない場合は空リスト
    # --- NJDFeature から取れるものと同一値 ---
    pos: str  # 品詞
    pos_group1: str  # 品詞細分類1
    pos_group2: str  # 品詞細分類2
    pos_group3: str  # 品詞細分類3
    ctype: str  # 活用型
    cform: str  # 活用形
    orig: str  # 原形
    read: str  # 読み
    pron: str  # 発音形式
    accent_nucleus: int  # アクセント核位置 (0: 平板型, 1-n: n番目のモーラにアクセント核)
    mora_count: int  # モーラ数
    chain_rule: str  # アクセント結合規則 (C1-C5/F1-F5/P1-P2 等)
    chain_flag: int  # アクセント句連結フラグ
    # --- 未知語・無視トークン情報 ---
    is_unknown: bool  # MeCab が未知語と判定したか
    is_ignored: bool  # OpenJTalk が音素を生成しなかったか（元の音素列が空）
