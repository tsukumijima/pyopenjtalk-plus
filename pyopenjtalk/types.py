from typing_extensions import TypedDict


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
    `char_span` は Lattice ノードの surface ポインタが sentence バッファ内のどこを指すかを Unicode 半開区間へ写した値。
    その座標系は `text2mecab()` による正規化後の本文であり、`g2p_mapping()` 呼び出し時の入力文とは表記が異なる場合がある。
    """

    surface: str  # 表層形
    features: list[str]  # MeCab feature 文字列の分割リスト（13 列目以降はカスタムフィールド）
    # features: 既知語は 12 列、未知語は 8 列（読み/発音/acc/chain_rule がない）
    char_span: tuple[int, int]  # `text2mecab()` による正規化後の本文上の半開区間
    pos_id: int  # 品詞 ID (pos-id.def で定義。品詞4分類による粗い分類で、文脈 ID とは別物)
    left_id: int  # 左文脈 ID (left-id.def で定義。連接コスト行列のインデックスとして使われる)
    right_id: int  # 右文脈 ID (right-id.def で定義。連接コスト行列のインデックスとして使われる)
    word_cost: int  # 単語コスト (辞書に登録されたコスト。低いほど出現しやすい)
    link_cost: int  # 直前ノードからこの形態素へ遷移する局所コスト (連接コストと単語コストを含む)
    node_cost: int  # BOS からこの形態素までの累積コスト (MeCab の最短経路計算後の値)
    is_unknown: bool  # MeCab が未知語と判定したか (stat == MECAB_UNK_NODE)
    is_ignored: bool  # OpenJTalk パイプラインで無視されるトークンか ("記号,空白")
    # 0 はシステム辞書、1..N は userdic の読込順、255 は未知語・制御ノード
    dictionary_index: int


class MeCabNBestPath(TypedDict):
    """
    MeCab の n-best 候補1パス分の形態素解析結果。
    features は run_njd_from_mecab() に渡せるフィルタ済み feature 文字列で、
    morphs は候補パス内の全トークンを MeCabMorph と同じ詳細形式で保持する。
    """

    features: list[str]  # run_njd_from_mecab() に渡せる feature 文字列 ("記号,空白" は除外)
    morphs: list[MeCabMorph]  # 候補パス内の全トークン (記号,空白も含む)
    path_cost: int  # BOS を除く候補パス上の全ノード (EOS/EON 含む) の局所コスト合計。morphs の link_cost 合計とは一致しない


class MeCabLatticeCandidate(TypedDict):
    """
    `_mecab_node_to_cost_candidate()` / `analyze_mecab_candidates()` が Lattice 走査中に構築する内部候補ノード。
    `tsqyomi.types.CandidateNode` へ昇格する前段であり、公開 API には出ない。
    """

    surface: str  # 表層形
    features: list[str]  # MeCab feature 文字列の分割リスト（13 列目以降はカスタムフィールド）
    # features: 既知語は 12 列、未知語は 8 列（読み/発音/acc/chain_rule がない）
    char_span: tuple[int, int]  # MeCabMorph と同じ座標系 (MeCab 正規化本文上の半開区間)
    pos_id: int  # 品詞 ID (pos-id.def で定義。品詞4分類による粗い分類で、文脈 ID とは別物)
    left_id: int  # 左文脈 ID (left-id.def で定義。連接コスト行列のインデックスとして使われる)
    right_id: int  # 右文脈 ID (right-id.def で定義。連接コスト行列のインデックスとして使われる)
    word_cost: int  # 単語コスト (辞書に登録されたコスト。低いほど出現しやすい)
    is_unknown: bool  # MeCab が未知語と判定したか (stat == MECAB_UNK_NODE)
    is_ignored: bool  # OpenJTalk パイプラインで無視されるトークンか ("記号,空白")
    is_reading_protected: bool  # tsqyomi 差し替えから保護するユーザー辞書候補か
    # 0 はシステム辞書、1..N は userdic の読込順、255 は未知語・制御ノード
    dictionary_index: int
    local_replacement_cost: int | None  # 最良経路の外側を固定した差し替え経路費用
    left_boundary_cost: int | None  # 左外側最良経路ノードからこの候補への MeCab 連接コスト
    right_boundary_cost: int | None  # この候補から右外側最良経路ノードへの MeCab 連接コスト
    right_link_cost: int | None  # この候補から右外側最良経路ノードへの単語コストを含む MeCab 連接コスト # fmt: skip


class JPCommonMappingEntry(TypedDict):
    """
    Cython 側 `make_phoneme_mapping()` が JPCommon 走査前に構築する内部マッピング1件。
    `SurfacePhonemeMapping` より `features` / `is_unknown` / `is_ignored` が欠けており、Python 側 API には出ない。
    """

    surface: str  # NJD 後処理後の表層形
    phonemes: list[str]  # 対応する音素列
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


class SurfacePhonemeMapping(TypedDict):
    """
    形態素と対応する音素列のマッピング（未知語・無視トークン情報付き）。
    事実上 NJDFeature のスーパーセットとなっているが、意味明確化のため一部フィールドの名称を変更している。

    NOTE: MeCabMorph.is_ignored（記号や空白かどうかの判定）とは判定基準が異なる。
    MeCabMorph.is_ignored は、MeCab の feature に「記号,空白」が含まれているかで判定する。
    SurfacePhonemeMapping.is_ignored は、Cython で音素のマッピングを行った結果、対応する音素列が空なら True になる
    (Haqumei の map.phonemes.is_empty() と同じ判定ロジック)。
    つまり、記号や空白だけでなく、文頭にある 'ー' など音素が割り当てられないトークンも is_ignored=True になる。

    NOTE: char_span の座標系は make_phoneme_mapping() の呼び出し方で決まる。
    フィールド名は MeCabMorph.char_span と同じだが、MeCabMorph は MeCab 正規化本文上の半開区間を指す。
    g2p_mapping(text=...) は内部で morphs 付き make_phoneme_mapping() を呼ぶため、
    返る char_span は呼び出し元に渡した text 上の半開区間になる (表記差がある場合は射影する)。
    morphs 付きで make_phoneme_mapping() を直接呼んだ場合、caller_text を渡せばその文字列上の半開区間、
    省略すれば MeCab 正規化本文上の半開区間になる。
    morphs を省略した make_phoneme_mapping() では、NJD 後処理後の surface を先頭から連結した
    文字列上の半開区間になる。これは入力文の座標系ではない。
    """

    surface: str  # NJD 後処理後の表層形
    phonemes: list[str]  # 対応する音素列
    features: list[str]  # MeCab feature 文字列の分割リスト（13 列目以降はカスタムフィールド）
    # features: 既知語は 12 列、未知語は 8 列（読み/発音/acc/chain_rule がない）
    ## make_phoneme_mapping() が morphs 付きで呼ばれた場合、アライメントで対応する MeCab morph の features を転写する
    ## morphs なしの場合や、数字正規化・踊り字展開で morph と NJD の surface が一致しない場合は空リスト
    char_span: tuple[int, int]  # 半開区間 [start, end)
    # char_span: morphs 省略時は NJD surface 連結文字列上の添字
    ## morphs 指定時は MeCab morph の char_span をアライメント結果に合わせて合成する
    ## caller_text を渡した場合は MeCab 座標から caller_text 上へ射影する
    ## 対応 morph を特定できない entry は (0, 0)
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


class UserDictionaryEntry(TypedDict):
    """
    OpenJTalk 用のユーザー辞書と読み保護の指定を表す型。
    """

    dic_path: str  # ユーザー辞書ファイル (.dic) のパス
    is_reading_protected: bool  # tsqyomi による MeCab feature 差し替えから保護するか
