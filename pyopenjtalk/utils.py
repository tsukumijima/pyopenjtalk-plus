import unicodedata
from threading import Lock, local
from typing import Any, Literal, Union

from sudachipy import dictionary, tokenizer

from .openjtalk import OpenJTalk
from .types import NJDFeature
from .yomi_model.nani_predict import predict


# 小書き仮名の集合 (モーラ分割で前の文字と結合される文字)
_SMALL_KANA = frozenset("ャュョァィゥェォ")

# 濁音→清音の 1 文字マップ (detect_odori_unit での清音化に使用)
_SEION_CHAR_MAP: dict[str, str] = {
    "が": "か",
    "ぎ": "き",
    "ぐ": "く",
    "げ": "け",
    "ご": "こ",
    "ざ": "さ",
    "じ": "し",
    "ず": "す",
    "ぜ": "せ",
    "ぞ": "そ",
    "だ": "た",
    "ぢ": "ち",
    "づ": "つ",
    "で": "て",
    "ど": "と",
    "ば": "は",
    "び": "ひ",
    "ぶ": "ふ",
    "べ": "へ",
    "ぼ": "ほ",
    "ガ": "カ",
    "ギ": "キ",
    "グ": "ク",
    "ゲ": "ケ",
    "ゴ": "コ",
    "ザ": "サ",
    "ジ": "シ",
    "ズ": "ス",
    "ゼ": "セ",
    "ゾ": "ソ",
    "ダ": "タ",
    "ヂ": "チ",
    "ヅ": "ツ",
    "デ": "テ",
    "ド": "ト",
    "バ": "ハ",
    "ビ": "ヒ",
    "ブ": "フ",
    "ベ": "ヘ",
    "ボ": "ホ",
    "ヴ": "ウ",
}

# 助動詞「う」の不自然な長音化を抑制するための段判定マップ
_DAN_MAP: dict[str, Literal["a", "i", "u", "e", "o"]] = {
    "ア": "a",
    "カ": "a",
    "サ": "a",
    "タ": "a",
    "ナ": "a",
    "ハ": "a",
    "マ": "a",
    "ヤ": "a",
    "ラ": "a",
    "ワ": "a",
    "ガ": "a",
    "ザ": "a",
    "ダ": "a",
    "バ": "a",
    "パ": "a",
    "ァ": "a",
    "イ": "i",
    "キ": "i",
    "シ": "i",
    "チ": "i",
    "ニ": "i",
    "ヒ": "i",
    "ミ": "i",
    "リ": "i",
    "ギ": "i",
    "ジ": "i",
    "ヂ": "i",
    "ビ": "i",
    "ピ": "i",
    "ィ": "i",
    "ウ": "u",
    "ク": "u",
    "ス": "u",
    "ツ": "u",
    "ヌ": "u",
    "フ": "u",
    "ム": "u",
    "ユ": "u",
    "ル": "u",
    "グ": "u",
    "ズ": "u",
    "ヅ": "u",
    "ブ": "u",
    "プ": "u",
    "ヴ": "u",
    "ゥ": "u",
    "エ": "e",
    "ケ": "e",
    "セ": "e",
    "テ": "e",
    "ネ": "e",
    "ヘ": "e",
    "メ": "e",
    "レ": "e",
    "ゲ": "e",
    "ゼ": "e",
    "デ": "e",
    "ベ": "e",
    "ペ": "e",
    "ェ": "e",
    "オ": "o",
    "コ": "o",
    "ソ": "o",
    "ト": "o",
    "ノ": "o",
    "ホ": "o",
    "モ": "o",
    "ヨ": "o",
    "ロ": "o",
    "ヲ": "o",
    "ゴ": "o",
    "ゾ": "o",
    "ド": "o",
    "ボ": "o",
    "ポ": "o",
    "ォ": "o",
}

# Sudachi の Dictionary はスレッド間で共有可能だが、Tokenizer はスレッドセーフでないため
# Dictionary をモジュールレベルで一度だけ生成し、Tokenizer のみスレッドごとに遅延初期化する
_SUDACHI_DICTIONARY: Union[dictionary.Dictionary, None] = None
_SUDACHI_DICTIONARY_LOCK = Lock()
_SUDACHI_TOKENIZER_LOCAL = local()


def _get_sudachi_tokenizer() -> tokenizer.Tokenizer:
    """
    現在のスレッドに紐づく Sudachi Tokenizer を取得する。

    Sudachi の Dictionary はスレッド間で共有可能だが、Tokenizer はスレッドセーフでないため、
    Dictionary はモジュールレベルで一度だけ生成し、Tokenizer のみスレッドごとに遅延初期化する。

    Returns:
        tokenizer.Tokenizer: 遅延初期化済みの Sudachi tokenizer。
    """

    global _SUDACHI_DICTIONARY
    sudachi_tokenizer = getattr(_SUDACHI_TOKENIZER_LOCAL, "tokenizer", None)
    if sudachi_tokenizer is None:
        with _SUDACHI_DICTIONARY_LOCK:
            if _SUDACHI_DICTIONARY is None:
                _SUDACHI_DICTIONARY = dictionary.Dictionary()
        sudachi_tokenizer = _SUDACHI_DICTIONARY.create()
        _SUDACHI_TOKENIZER_LOCAL.tokenizer = sudachi_tokenizer
    return sudachi_tokenizer


def normalize_text(
    text: str,
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
) -> str:
    """
    指定された方式で Unicode 正規化を行う。

    Args:
        text (str): 正規化対象のテキスト。
        normalize_mode (Literal["None", "NFC", "NFKC"]): 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する。
            デフォルトは `"None"` 。

    Returns:
        str: 正規化後のテキスト。正規化不要な場合は元の文字列をそのまま返す。

    Raises:
        ValueError: `normalize_mode` に未対応の値が指定された場合。
    """

    if normalize_mode not in ("None", "NFC", "NFKC"):
        raise ValueError("normalize_mode must be one of 'None', 'NFC', or 'NFKC'")
    if normalize_mode == "None":
        return text

    normalized_form: Literal["NFC", "NFKC"] = "NFC" if normalize_mode == "NFC" else "NFKC"
    if unicodedata.is_normalized(normalized_form, text) is True:
        return text
    return unicodedata.normalize(normalized_form, text)


def merge_njd_marine_features(
    njd_features: list[NJDFeature], marine_results: dict[str, Any]
) -> list[NJDFeature]:
    features = []

    marine_accs = marine_results["accent_status"]
    marine_chain_flags = marine_results["accent_phrase_boundary"]

    assert len(njd_features) == len(marine_accs) == len(marine_chain_flags), (
        "Invalid sequence sizes in njd_results, marine_results"
    )

    for node_index, njd_feature in enumerate(njd_features):
        _feature = {}
        for feature_key in njd_feature.keys():
            if feature_key == "acc":
                _feature["acc"] = int(marine_accs[node_index])
            elif feature_key == "chain_flag":
                _feature[feature_key] = int(marine_chain_flags[node_index])
            else:
                _feature[feature_key] = njd_feature[feature_key]
        features.append(_feature)
    return features


def modify_kanji_yomi(
    text: str,
    pyopen_njd: list[NJDFeature],
    target_kanji_set: frozenset[str],
) -> list[NJDFeature]:
    """
    Sudachi を用いて、複数の読みを持つ漢字の読みを補正する。
    Sudachi の形態素解析結果と NJD の形態素を逆順で突合し、
    対象漢字の pron / read を Sudachi の読みで上書きする。

    Args:
        text (str): 読み対象となるテキスト。
        pyopen_njd (list[NJDFeature]): OpenJTalk の形態素解析結果。
        target_kanji_set (frozenset[str]): 複数の読みを持つ対象漢字の集合。

    Returns:
        list[NJDFeature]: 漢字の読み補正を適用した形態素解析結果。
            突合に失敗した場合は元の形態素をそのまま返す。
    """

    if len(target_kanji_set) == 0:
        return pyopen_njd
    if any(feature["orig"] in target_kanji_set for feature in pyopen_njd) is False:
        return pyopen_njd

    sudachi_yomi = sudachi_analyze(text, target_kanji_set)
    return_njd = []

    for feature in reversed(pyopen_njd):
        if feature["orig"] in target_kanji_set:
            try:
                correct_yomi = sudachi_yomi.pop()
            except IndexError:
                return pyopen_njd
            if correct_yomi[0] != feature["orig"]:
                return pyopen_njd
            if correct_yomi[0] == "方" and correct_yomi[1] == "ホウ":
                correct_yomi[1] = "ホオ"
            feature["pron"] = correct_yomi[1]
            feature["read"] = correct_yomi[1]
            return_njd.append(feature)
        else:
            return_njd.append(feature)

    return_njd.reverse()
    return return_njd


def sudachi_analyze(text: str, target_kanji_set: frozenset[str]) -> list[list[str]]:
    """
    複数の読み方をする漢字の読みを Sudachi で形態素解析した結果をリストで返す。

    Args:
        text (str): 読み対象となるテキスト。
        target_kanji_set (frozenset[str]): 複数の読みを持つ対象漢字の集合。

    Returns:
        list[list[str]]: 漢字とその読み方のリスト。
            例: 「風がこんな風に吹く」→ [["風", "カゼ"], ["風", "フウ"]]
    """

    if len(target_kanji_set) == 0:
        return []

    text = text.replace("ー", "")
    tokenizer_obj = _get_sudachi_tokenizer()
    mode = tokenizer.Tokenizer.SplitMode.C
    m_list = tokenizer_obj.tokenize(text, mode)
    yomi_list = [[m.surface(), m.reading_form()] for m in m_list if m.surface() in target_kanji_set]
    return yomi_list


def predict_nani_reading(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    ONNX モデルを用いて、単独形態素として出現した「何」の読みを補正する。

    Args:
        njd_features (list[NJDFeature]): OpenJTalk の形態素解析結果。

    Returns:
        list[NJDFeature]: 「何」の読み補正を適用した形態素解析結果。
    """

    if any(feature["orig"] == "何" for feature in njd_features) is False:
        return njd_features

    for feature_index, current_feature in enumerate(njd_features):
        if current_feature["orig"] != "何":
            continue

        next_feature = (
            njd_features[feature_index + 1] if feature_index + 1 < len(njd_features) else None
        )
        is_read_nan = predict([next_feature])
        yomi = "ナン" if is_read_nan == 1 else "ナニ"
        current_feature["pron"] = yomi
        current_feature["read"] = yomi

    return njd_features


def suppress_unnatural_auxiliary_u_long_vowel(
    njd_features: list[NJDFeature],
) -> list[NJDFeature]:
    """
    助動詞「う」が不自然に長音化されたケースを打ち消す。
    OpenJTalk (NJD) は、動詞または助動詞の直後に来る助動詞「う」を無条件で長音化することがある。
    このうち、直前語末がア段・イ段・エ段のケースでは不自然な読みになることが多いため、`pron` を `"ー"` から `"ウ"` に戻す。
    ref: https://github.com/tsukumijima/pyopenjtalk-plus/issues/6#issuecomment-4067840409

    Args:
        njd_features (list[NJDFeature]): OpenJTalk の形態素解析結果。

    Returns:
        list[NJDFeature]: 不自然な長音化を補正した形態素解析結果。
    """

    if len(njd_features) < 2:
        return njd_features

    for feature_index in range(len(njd_features) - 1):
        current_feature = njd_features[feature_index]
        next_feature = njd_features[feature_index + 1]

        if next_feature["pron"] != "ー" or next_feature["read"] != "ウ":
            continue

        current_pron = current_feature["pron"].rstrip("’")
        if current_pron == "":
            continue

        previous_dan = _DAN_MAP.get(current_pron[-1])
        if previous_dan in ("a", "i", "e"):
            next_feature["pron"] = "ウ"

    return njd_features


def retreat_acc_nuc(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    長母音、重母音、撥音がアクセント核に来た場合にひとつ前のモーラにアクセント核がズレるルールの実装

    Args:
        njd_features (list[NJDFeature]): run_frontend() の結果

    Returns:
        list[NJDFeature]: 修正後の njd_features
    """

    if not njd_features:
        return njd_features

    inappropriate_for_nuclear_chars = ["ー", "ッ", "ン"]
    delete_youon = str.maketrans("", "", "ャュョァィゥェォ")
    phase_len = 0
    acc = 0
    head = njd_features[0]

    for _, njd in enumerate(njd_features):
        # アクセント境界直後の node (chain_flag 0 or -1) にアクセント核の位置の情報が入っている
        if njd["chain_flag"] in [0, -1]:
            head = njd
            acc = njd["acc"]
            phase_len = 0

        phase_len += njd["mora_size"]
        pron = njd["pron"].translate(delete_youon)
        if len(pron) == 0:
            pron = njd["pron"]

        if acc > 0:
            if acc <= njd["mora_size"]:
                try:
                    nuc_pron = pron[acc - 1]
                except IndexError:
                    nuc_pron = pron[0]
                if nuc_pron in inappropriate_for_nuclear_chars:
                    head["acc"] += -1
                acc = -1
            else:
                acc = acc - njd["mora_size"]

    return njd_features


def modify_acc_after_chaining(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    品詞「特殊・マス」は直前に接続する動詞にアクセント核がある場合、アクセント核を「ま」に移動させる法則がある
    書きます → か[きま]す, 参ります → ま[いりま]す
    書いております → [か]いております

    Args:
        njd_features (list[NJDFeature]): run_frontend() の結果

    Returns:
        list[NJDFeature]: 修正後の njd_features
    """

    if not njd_features:
        return njd_features

    acc = 0
    is_after_nuc = False
    phase_len = 0
    head = njd_features[0]

    for njd in njd_features:
        # アクセント境界直後の node (chain_flag 0 or -1) にアクセント核の位置の情報が入っている
        if njd["chain_flag"] in [0, -1]:
            is_after_nuc = False
            head = njd
            acc = njd["acc"]
            phase_len = 0
        # acc = 0 の場合は「特殊・マス」は存在しないと考えてよい
        if acc == 0:
            continue
        elif is_after_nuc:
            if njd["ctype"] == "特殊・マス":
                head["acc"] = phase_len + 1 if njd["cform"] != "未然形" else phase_len + 2
            elif njd["ctype"] == "特殊・ナイ":
                head["acc"] = phase_len
            elif njd["orig"] in ["れる", "られる", "すぎる", "せる", "させる"]:
                head["acc"] = phase_len + njd["acc"]
            else:
                is_after_nuc = False
                acc = 0
            phase_len += njd["mora_size"]

        else:
            phase_len += njd["mora_size"]
            if acc <= njd["mora_size"]:
                is_after_nuc = True
            else:
                acc = acc - njd["mora_size"]

    return njd_features


def revert_pron_to_read(
    njd_features: list[NJDFeature],
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
) -> list[NJDFeature]:
    """
    辞書によって自動的に正規化・変換された発音 (pron) を、元のテキスト通りの読み (read) に復元する。

    Args:
        njd_features (list[NJDFeature]): OpenJTalk の形態素解析結果
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルトは False 。
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルトは False 。
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルトは False 。

    Returns:
        list[NJDFeature]: 発音復元後の形態素解析結果
    """

    for feature in njd_features:
        is_should_revert = use_read_as_pron
        # 辞書が自動的に長音化した発音を復元
        # pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元
        if revert_long_vowels is True and "ー" in feature["pron"] and "ー" not in feature["orig"]:
            is_should_revert = True
        # 四つ仮名の発音統合を復元
        if revert_yotsugana is True and ("ヅ" in feature["read"] or "ヂ" in feature["read"]):
            is_should_revert = True
        if is_should_revert is True:
            feature["pron"] = feature["read"]

    return njd_features


def split_kana_mora(text: str) -> list[str]:
    """
    カタカナ/ひらがな文字列をモーラ単位に分割する。
    小書き仮名 (ャュョァィゥェォ) は前の文字と結合して1モーラとして扱う。

    Args:
        text (str): 分割対象のカタカナ/ひらがな文字列

    Returns:
        list[str]: モーラ単位に分割されたリスト
    """

    chars = list(text)
    result: list[str] = []
    idx = 0
    while idx < len(chars):
        char = chars[idx]
        if idx + 1 < len(chars) and chars[idx + 1] in _SMALL_KANA:
            result.append(char + chars[idx + 1])
            idx += 2
        else:
            result.append(char)
            idx += 1
    return result


def detect_odori_unit(read: str) -> Union[int, None]:
    """
    読み文字列を清音化し、末尾の繰り返し単位 (周期) を検出する。
    「々」の展開で直前トークンが既に踊り字展開済みの場合に、繰り返しの基底単位を特定するために使う。

    例: 「サマザマ」→ 清音化→ 「サマサマ」→ モーラ ["サ","マ","サ","マ"] → 周期 2 (「サマ」が繰り返し)

    Args:
        read (str): 直前トークンの読み (カタカナ)

    Returns:
        Union[int, None]: 繰り返し周期 (モーラ数)。検出できなかった場合は None
    """

    # 濁音を全て清音に変換
    seion_read = "".join(_SEION_CHAR_MAP.get(ch, ch) for ch in read)
    moras = split_kana_mora(seion_read)
    mora_count = len(moras)
    if mora_count < 2:
        return None

    # 後ろ半分が前半分と一致する最小の単位を探す
    for period in range(1, mora_count // 2 + 1):
        first_half = moras[mora_count - period * 2 : mora_count - period]
        second_half = moras[mora_count - period :]
        if first_half == second_half:
            return period
    return None


def process_odori_features(
    njd_features: list[NJDFeature],
    jtalk: Union[OpenJTalk, None] = None,
) -> list[NJDFeature]:
    """
    踊り字（々）と一の字点（ゝ、ゞ、ヽ、ヾ）の読みを適切に処理する後処理関数

    OpenJTalk の挙動に合わせて、連続する踊り字を処理する
    踊り字の数に応じて読みを繰り返す：
    - 「叙々苑」→「ジョジョエン」
    - 「叙々々苑」→「ジョジョジョエン」
    - 「叙々々々苑」→「ジョジョジョジョエン」

    また、複数漢字や複数トークンの場合は、前の読みをそのまま使用する：
    - 「部分々々」→「ブブンブブン」
    - 「其他々々」→「ソノホカソノホカ」
    - 「前進々々」→「ゼンシンゼンシン」

    さらに、単独の踊り字で直前のトークンが複数漢字の場合は、適宜直前と直後の漢字を使って再解析：
    - 「結婚式々場」→「ケッコンシキシキジョウ」
    - 「民主々義」→「ミンシュシュギ」
    - 「学生々活」→「ガクセイセイカツ」

    一の字点（ゝ、ゞ、ヽ、ヾ）は直前の文字を繰り返す：
    - 「こゝろ」→「こころ」
    - 「みすゞ」→「みすず」
    - 「づゝ」→「づつ」
    - 「ぶゞ漬け」→「ぶぶ漬け」

    Args:
        njd_features (list[NJDFeature]): OpenJTalk の形態素解析結果
        jtalk (Union[OpenJTalk, None], optional): OpenJTalk インスタンス。
            単独の踊り字の直前の漢字を再解析する場合に使用。デフォルトは None。

    Returns:
        list[NJDFeature]: 踊り字の読みを修正した形態素解析結果
    """

    def is_dancing(orig: str) -> bool:
        """文字列が踊り字のみで構成されているかを判定する

        Args:
            orig (str): 判定対象の文字列

        Returns:
            bool: 踊り字のみで構成されている場合は True
        """
        return set(orig) == {"々"}

    def is_odoriji(orig: str) -> bool:
        """文字列が一の字点のみで構成されているかを判定する

        Args:
            orig (str): 判定対象の文字列

        Returns:
            bool: 一の字点のみで構成されている場合は True
        """
        return set(orig) <= {"ゝ", "ゞ", "ヽ", "ヾ"}

    def count_odori(orig: str) -> int:
        """文字列に含まれる踊り字の数をカウントする

        Args:
            orig (str): カウント対象の文字列

        Returns:
            int: 踊り字の数
        """
        return orig.count("々")

    def is_kanji_token(token: NJDFeature) -> bool:
        """トークンが漢字を含むかを判定する

        Args:
            token (NJDFeature): 判定対象のトークン

        Returns:
            bool: 漢字を含む場合は True
        """
        # 品詞が記号の場合は False
        if token["pos"] == "記号":
            return False
        # 原形に漢字が含まれているかを判定
        return any(0x4E00 <= ord(c) <= 0x9FFF for c in token["orig"])

    def is_single_kanji_token(token: NJDFeature) -> bool:
        """トークンが1文字の漢字で構成されているかを判定する

        Args:
            token (NJDFeature): 判定対象のトークン

        Returns:
            bool: 1文字の漢字で構成されている場合は True
        """
        return (
            is_kanji_token(token)
            and len(token["orig"]) == 1
            and 0x4E00 <= ord(token["orig"][0]) <= 0x9FFF
        )

    def needs_reanalysis(
        odori_feature: NJDFeature,
        prev_feature: NJDFeature,
        next_feature: Union[NJDFeature, None] = None,
    ) -> tuple[bool, str, Union[str, None]]:
        """踊り字の直前の漢字を再解析する必要があるかを判定

        Args:
            odori_feature (NJDFeature): 踊り字のトークン
            prev_feature (NJDFeature): 直前のトークン
            next_feature (Union[NJDFeature, None], optional): 後続のトークン

        Returns:
            tuple[bool, str, Union[str, None]]: (再解析が必要か, 再解析する漢字, 後続の漢字)
        """
        # 踊り字が単独（1文字）でない場合は再解析不要
        if count_odori(odori_feature["orig"]) != 1:
            return False, "", None

        # 直前のトークンが漢字を含まない場合は再解析不要
        if not is_kanji_token(prev_feature):
            return False, "", None

        # 直前のトークンが複数文字で構成されている場合
        if len(prev_feature["orig"]) > 1:
            # 直前のトークンの最後の漢字を抽出
            last_char = prev_feature["orig"][-1]
            if 0x4E00 <= ord(last_char) <= 0x9FFF:
                # 後続のトークンが1文字の漢字の場合は、その漢字も含めて再解析
                if next_feature is not None and is_single_kanji_token(next_feature):
                    return True, last_char, next_feature["orig"]
                # それ以外の場合は最後の漢字のみを再解析
                return True, last_char, None

        return False, "", None

    def reanalyze_kanji(kanji: str, jtalk: OpenJTalk) -> list[NJDFeature]:
        """漢字を再解析して読みを取得

        Args:
            kanji (str): 解析対象の漢字
            jtalk (OpenJTalk): OpenJTalk インスタンス

        Returns:
            list[NJDFeature]: 解析結果
        """
        features = jtalk.run_frontend(kanji)
        return features

    def process_odoriji(
        odori_feature: NJDFeature,
        prev_feature: NJDFeature,
    ) -> NJDFeature:
        """一の字点の読みを処理する

        Args:
            odori_feature (NJDFeature): 一の字点のトークン
            prev_feature (NJDFeature): 直前のトークン

        Returns:
            NJDFeature: 読みを修正したトークン
        """
        # 直前のトークンの読みを取得
        # 読みとモーラサイズを1文字ずつに分解
        prev_read_chars = []
        prev_pron_chars = []
        prev_mora_sizes = []

        # カタカナを1文字ずつに分解
        i = 0
        while i < len(prev_feature["read"]):
            char = prev_feature["read"][i]
            # 小書き文字の処理
            if i + 1 < len(prev_feature["read"]) and prev_feature["read"][i + 1] in {"ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ"}:  # fmt: skip
                prev_read_chars.append(char + prev_feature["read"][i + 1])
                i += 2
            else:
                prev_read_chars.append(char)
                i += 1

        # 無声化サインなどアクセント用の記号はモーラとして扱わないように除去してから分割する。
        # これを行わないと、「マス’」のような読みが「マ」「ス」「’」の3文字に分割され、
        # モーラ数や一の字点展開時の読みが不整合になる。
        prev_pron_source = prev_feature["pron"].replace("’", "")
        # 万が一、除去の結果として空文字列になってしまった場合は、読みの情報を失わないように
        # read 側を pron のソースとして利用する。
        if prev_pron_source == "":
            prev_pron_source = prev_feature["read"]

        i = 0
        while i < len(prev_pron_source):
            char = prev_pron_source[i]
            # 小書き文字の処理
            if i + 1 < len(prev_pron_source) and prev_pron_source[i + 1] in {"ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ"}:  # fmt: skip
                prev_pron_chars.append(char + prev_pron_source[i + 1])
                i += 2
            else:
                prev_pron_chars.append(char)
                i += 1

        # モーラサイズを文字数に応じて分配
        mora_per_char = prev_feature["mora_size"] / len(prev_read_chars)
        prev_mora_sizes = [mora_per_char] * len(prev_read_chars)

        # 最後の文字の読みを取得
        prev_read = prev_read_chars[-1]
        prev_pron = prev_pron_chars[-1]
        prev_mora_size = prev_mora_sizes[-1]

        # 濁点化のマッピング (単一文字 + 拗音)
        dakuten_map = {
            "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
            "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
            "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
            "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ",
            "か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご",
            "さ": "ざ", "し": "じ", "す": "ず", "せ": "ぜ", "そ": "ぞ",
            "た": "だ", "ち": "ぢ", "つ": "づ", "て": "で", "と": "ど",
            "は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ",
            # 拗音のマッピング
            "キャ": "ギャ", "キュ": "ギュ", "キョ": "ギョ",
            "シャ": "ジャ", "シュ": "ジュ", "ショ": "ジョ",
            "チャ": "ヂャ", "チュ": "ヂュ", "チョ": "ヂョ",
            "ヒャ": "ビャ", "ヒュ": "ビュ", "ヒョ": "ビョ",
            "きゃ": "ぎゃ", "きゅ": "ぎゅ", "きょ": "ぎょ",
            "しゃ": "じゃ", "しゅ": "じゅ", "しょ": "じょ",
            "ちゃ": "ぢゃ", "ちゅ": "ぢゅ", "ちょ": "ぢょ",
            "ひゃ": "びゃ", "ひゅ": "びゅ", "ひょ": "びょ",
        }  # fmt: skip

        # 濁点の逆引きマッピング
        dakuten_reverse_map = {v: k for k, v in dakuten_map.items()}

        # 一の字点の種類を判定
        # ゞ/ヾ が含まれているかで強制濁音化を判定
        is_forced_voiced = False
        for char in odori_feature["orig"]:
            if char in ("ゞ", "ヾ"):
                is_forced_voiced = True
                break
            if char in ("ゝ", "ヽ"):
                break

        # 対象モーラが単一の仮名 grapheme か判定する
        # 一の字点 (ゝ, ゞ, ヽ, ヾ) は歴史的に「直前の仮名1文字」を
        # 繰り返す記号であり、拗音 (きゃ, しゃ 等) のような
        # 複数仮名からなるモーラに対して使われる例はほぼ存在しない
        is_single_grapheme_mora = not any(char in _SMALL_KANA for char in prev_read)

        if is_forced_voiced is True:
            # 濁音の踊り字 (ゞ, ヾ): 強制的に濁音化
            voiced_read: str = dakuten_map.get(prev_read) or prev_read
            voiced_pron: str = dakuten_map.get(prev_pron) or prev_pron
            odori_feature["read"] = voiced_read
            odori_feature["pron"] = voiced_pron
            odori_feature["mora_size"] = int(prev_mora_size)
        else:
            # 清音の踊り字 (ゝ, ヽ)
            if is_single_grapheme_mora is True:
                # 対象が単一文字の場合: 清音化
                seion_read: str = dakuten_reverse_map.get(prev_read) or prev_read
                seion_pron: str = dakuten_reverse_map.get(prev_pron) or prev_pron
                odori_feature["read"] = seion_read
                odori_feature["pron"] = seion_pron
            else:
                # 対象が拗音などの複数文字の場合: 濁点を維持する
                odori_feature["read"] = prev_read
                odori_feature["pron"] = prev_pron
            odori_feature["mora_size"] = int(prev_mora_size)

        # 記号扱いにすると後の処理で誤作動するケースがありそうな気がするので、適当に一般名詞としておく
        if odori_feature["pos"] == "記号":
            odori_feature["pos"] = "名詞"
            odori_feature["pos_group1"] = "一般"
            odori_feature["pos_group2"] = "*"
            odori_feature["pos_group3"] = "*"
            odori_feature["ctype"] = "*"
            odori_feature["cform"] = "*"

        return odori_feature

    i = 0
    while i < len(njd_features):
        if is_dancing(njd_features[i]["orig"]):
            # 単独の踊り字で再解析が必要な場合
            if i > 0 and jtalk is not None:
                next_feature = njd_features[i + 1] if i + 1 < len(njd_features) else None
                needs_reanalysis_flag, target_kanji, next_kanji = needs_reanalysis(
                    njd_features[i], njd_features[i - 1], next_feature
                )
                if needs_reanalysis_flag:
                    # 後続の漢字も含めて再解析する場合
                    if next_kanji is not None:
                        analyzed = reanalyze_kanji(target_kanji + next_kanji, jtalk)
                        # 再解析結果は直前の語の一部を繰り返して合成した語なので、
                        # 直前の語に連結させる (chain_flag=1)
                        # (reanalyze_kanji は独立テキストとして解析するため先頭が -1 になる)
                        if len(analyzed) > 0:
                            analyzed[0]["chain_flag"] = 1
                        # 再解析結果を踊り字トークンに反映し、後続の漢字トークンを削除
                        njd_features[i : i + 2] = analyzed
                        i += len(analyzed)
                        continue
                    else:
                        # 最後の漢字のみを再解析
                        analyzed = reanalyze_kanji(target_kanji, jtalk)
                        # 再解析結果を踊り字トークンに反映
                        njd_features[i] = analyzed[0]
                        # 踊り字は直前の語の繰り返しなので連結させる
                        njd_features[i]["chain_flag"] = 1
                        # 記号扱いにすると後の処理で誤作動するケースがありそうな気がするので、適当に一般名詞としておく
                        njd_features[i]["pos"] = "名詞"
                        njd_features[i]["pos_group1"] = "一般"
                        njd_features[i]["pos_group2"] = "*"
                        njd_features[i]["pos_group3"] = "*"
                        njd_features[i]["ctype"] = "*"
                        njd_features[i]["cform"] = "*"
                        i += 1
                        continue

            # 連続する踊り字トークンを特定
            start = i
            end = i
            total_odori = 0
            while end < len(njd_features) and is_dancing(njd_features[end]["orig"]):
                total_odori += count_odori(njd_features[end]["orig"])
                end += 1

            # 直前トークンが「々」で終わる場合 (既に踊り字展開済み)、
            # 清音ベースで繰り返し周期を検出して展開する
            if i > 0 and njd_features[i - 1]["orig"].endswith("々"):
                prev = njd_features[i - 1]
                period = detect_odori_unit(prev["read"])
                if period is not None:
                    raw_read_moras = split_kana_mora(prev["read"])
                    raw_pron_moras = split_kana_mora(prev["pron"])
                    # 読みが空の場合はゼロ除算を避けるためスキップ
                    if len(raw_read_moras) >= period and len(raw_read_moras) > 0:
                        unit_read = "".join(raw_read_moras[len(raw_read_moras) - period :])
                        unit_pron = "".join(raw_pron_moras[len(raw_pron_moras) - period :])
                        unit_mora = (prev["mora_size"] // len(raw_read_moras)) * period
                        base_acc = prev["acc"]

                        current_feat = njd_features[i]
                        current_odori = count_odori(current_feat["orig"])
                        current_feat["read"] = unit_read * current_odori
                        current_feat["pron"] = unit_pron * current_odori
                        current_feat["mora_size"] = unit_mora * current_odori
                        current_feat["acc"] = base_acc
                        current_feat["chain_flag"] = 1
                        if current_feat["pos"] == "記号":
                            current_feat["pos"] = "名詞"
                            current_feat["pos_group1"] = "一般"
                            current_feat["pos_group2"] = "*"
                            current_feat["pos_group3"] = "*"
                            current_feat["ctype"] = "*"
                            current_feat["cform"] = "*"
                        i += 1
                        continue

            # 直前の漢字トークンを遡行して収集
            # 記号・フィラー・感動詞をハード境界として設定し、
            # 遠方の無関係な単語を誤参照するアライメント問題を防ぐ
            normal_list: list[NJDFeature] = []
            j = start - 1
            collected_chars = 0
            needed_chars = min(total_odori, 8)
            while j >= 0:
                target = njd_features[j]
                # 記号・フィラー・感動詞はハード境界として停止
                if target["pos"] in ("記号", "フィラー", "感動詞"):
                    break
                if is_kanji_token(target):
                    normal_list.append(target)
                    collected_chars += len(target["orig"])
                    if collected_chars >= needed_chars:
                        break
                else:
                    # 漢字でないトークンに到達した場合も停止
                    break
                j -= 1
            normal_list.reverse()  # 元の順序に戻す

            # 前に適切な漢字がない場合はスキップ
            if not normal_list:
                i = end
                continue

            # 置換用の読みを決定
            # 単一漢字の場合は踊り字の数に応じて繰り返し、
            # 複数漢字の場合はそのまま使用
            is_single_kanji = len(normal_list) == 1 and len(normal_list[0]["orig"]) == 1
            if is_single_kanji:
                # 単一漢字の場合
                base_read = normal_list[0]["read"]
                base_pron = normal_list[0]["pron"]
                base_mora_size = normal_list[0]["mora_size"]
            else:
                # 複数漢字の場合
                base_read = "".join(item["read"] for item in normal_list)
                base_pron = "".join(item["pron"] for item in normal_list)
                base_mora_size = sum(item["mora_size"] for item in normal_list)

            # 直前トークンの acc を踊り字の読み繰り返しに引き継ぐ
            base_acc = normal_list[0]["acc"]

            # 連続する踊り字トークンを処理
            processed_odori = 0
            for j in range(start, end):
                current_odori = count_odori(njd_features[j]["orig"])
                if is_single_kanji:
                    # 単一漢字の場合は踊り字の数に応じて繰り返す
                    njd_features[j]["read"] = base_read * current_odori
                    njd_features[j]["pron"] = base_pron * current_odori
                    njd_features[j]["mora_size"] = base_mora_size * current_odori
                else:
                    # 複数漢字の場合はそのまま使用
                    njd_features[j]["read"] = base_read
                    njd_features[j]["pron"] = base_pron
                    njd_features[j]["mora_size"] = base_mora_size
                # 踊り字は直前の語の繰り返しなので acc を引き継ぎ、連結させる
                njd_features[j]["acc"] = base_acc
                njd_features[j]["chain_flag"] = 1

                processed_odori += current_odori

                # 記号扱いにすると後の処理で誤作動するケースがありそうな気がするので、適当に一般名詞としておく
                if njd_features[j]["pos"] == "記号":
                    njd_features[j]["pos"] = "名詞"
                    njd_features[j]["pos_group1"] = "一般"
                    njd_features[j]["pos_group2"] = "*"
                    njd_features[j]["pos_group3"] = "*"
                    njd_features[j]["ctype"] = "*"
                    njd_features[j]["cform"] = "*"

            i = end
        elif is_odoriji(njd_features[i]["orig"]):
            # 一の字点の処理
            if i > 0:
                # 直前が記号の場合は、絵文字や装飾的なケースとみなして踊り字展開を行わず、
                # OpenJTalk の生の解析結果を尊重してそのまま残す
                direct_prev = njd_features[i - 1]
                if direct_prev["pos"] != "記号":
                    # 前方のトークンを探索する
                    # これにより「こゝろ」「みすゞ」などの通常の一の字点利用では直前の仮名を基準に処理できる
                    prev_index = i - 1
                    while prev_index >= 0:
                        prev_token = njd_features[prev_index]
                        if prev_token["pos"] != "記号" and prev_token["mora_size"] > 0:
                            break
                        prev_index -= 1

                    # 有効な直前トークンが存在する場合のみ一の字点の処理を行い、
                    # 見つからない場合は raw の解析結果を尊重して変更しない
                    if prev_index >= 0:
                        njd_features[i] = process_odoriji(
                            njd_features[i],
                            njd_features[prev_index],
                        )
            i += 1
        else:
            i += 1

    return njd_features
