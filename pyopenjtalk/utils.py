import unicodedata
from threading import Lock, local
from typing import Any, Literal

from sudachipy import dictionary, tokenizer

from ._itaiji_map import ITAIJI_MAP
from ._kana_utils import is_katakana_word
from ._unihan_readings_map import UNIHAN_READINGS
from .openjtalk import OpenJTalk
from .types import IuPronunciation, NJDFeature, NormalizeMode
from .yomi_model.nani_predict import predict


# 小書き仮名の集合 (モーラ分割で前の文字と結合される文字)
_SMALL_KANA = frozenset("ャュョァィゥェォ")

# 辞書が外来語表記を一般的な仮名へ置き換える組み合わせ
# 長い表記を先に照合し、「ヴャ」を「ヴ」だけで消費する状態を避ける
_LOANWORD_KANA_RESTORATIONS: tuple[tuple[str, str], ...] = (
    ("ヴャ", "ビャ"),
    ("ヴュ", "ビュ"),
    ("ヴョ", "ビョ"),
    ("ヴァ", "バ"),
    ("ヴィ", "ビ"),
    ("ヴェ", "ベ"),
    ("ヴォ", "ボ"),
    ("ヴ", "ブ"),
    ("スィ", "シ"),
    ("ズィ", "ジ"),
    ("テュ", "チュ"),
    ("デュ", "ヂュ"),
    ("イェ", "イエ"),
    ("シィ", "シー"),
    ("リェ", "リエ"),
    ("ニェ", "ニエ"),
    ("ヒェ", "ヒエ"),
    ("ミェ", "ミエ"),
    ("ビェ", "ビエ"),
    ("ピェ", "ピエ"),
    ("キェ", "ケ"),
    ("ギェ", "ゲ"),
    ("グゥ", "グウ"),
    ("クゥ", "クウ"),
)

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

# 直後に接尾辞「国」が続くと、「国」を「コク」でなく「ノクニ」と読む令制国の名前 (例: 石見国 = イワミノクニ)
## 「中国」「外国」のように MeCab が1語として解析する語は、接尾辞の条件で対象から外れる
_OLD_PROVINCE_NAMES = frozenset(
    {
        # 畿内
        "山城",
        "大和",
        "河内",
        "和泉",
        "摂津",
        # 東海道
        "伊賀",
        "伊勢",
        "志摩",
        "尾張",
        "三河",
        "遠江",
        "駿河",
        "伊豆",
        "甲斐",
        "相模",
        "武蔵",
        "安房",
        "上総",
        "下総",
        "常陸",
        # 東山道
        "近江",
        "美濃",
        "飛騨",
        "信濃",
        "上野",
        "下野",
        "陸奥",
        "出羽",
        # 北陸道
        "若狭",
        "越前",
        "加賀",
        "能登",
        "越中",
        "越後",
        "佐渡",
        # 山陰道
        "丹波",
        "丹後",
        "但馬",
        "因幡",
        "伯耆",
        "出雲",
        "石見",
        "隠岐",
        # 山陽道
        "播磨",
        "美作",
        "備前",
        "備中",
        "備後",
        "安芸",
        "周防",
        "長門",
        # 南海道
        "紀伊",
        "淡路",
        "阿波",
        "讃岐",
        "伊予",
        "土佐",
        # 西海道
        "筑前",
        "筑後",
        "豊前",
        "豊後",
        "肥前",
        "肥後",
        "日向",
        "大隅",
        "薩摩",
        "壱岐",
        "対馬",
        # 明治期に分立した国
        "岩代",
        "磐城",
        "陸前",
        "陸中",
        "羽前",
        "羽後",
        # 北海道の11国
        "渡島",
        "後志",
        "胆振",
        "石狩",
        "天塩",
        "北見",
        "日高",
        "十勝",
        "釧路",
        "根室",
        "千島",
        # 記紀や風土記などに現れる古い表記
        "大倭",
        "御野",
        "諏方",
        "石城",
        "石背",
        "多禰",
        "筑紫",
        "末廬",
        "高志",
        "上毛野",
        "下毛野",
        "三野",
        "針間",
        "吉備",
        "科野",
    }
)

# 仮名の最後の文字から母音を引く表 (漢字の音読みを OpenJTalk の長音の表記に直すときに使う)
_VOWEL_BY_LAST_KANA: dict[str, str] = {
    **dict.fromkeys("アカサタナハマヤラワガザダバパャァ", "a"),
    **dict.fromkeys("イキシチニヒミリヰギジヂビピィ", "i"),
    **dict.fromkeys("ウクスツヌフムユルグズヅブプヴュゥ", "u"),
    **dict.fromkeys("エケセテネヘメレヱゲゼデベペェ", "e"),
    **dict.fromkeys("オコソトノホモヨロヲゴゾドボポョォ", "o"),
}

# 文脈による読み補正 (modify_context_reading()) で、前後の語から読みを決めるときに使う語の集合
## 「〜の下」を「モト」と読ませる抽象名詞
_ABSTRACT_NO_PREDECESSORS = frozenset(
    {
        "支配",
        "統治",
        "指導",
        "指揮",
        "監督",
        "管理",
        "監視",
        "命令",
        "号令",
        "庇護",
        "保護",
        "援助",
        "協力",
        "後援",
        "統制",
        "占領",
        "名",
        "法",
        "条件",
        "前提",
        "仮定",
        "原則",
        "方針",
        "契約",
        "規定",
        "制度",
        "計画",
        "設定",
        "愛情",
        "信頼",
        "理解",
        "合意",
        "影響",
        "配慮",
        "恩師",
        "陛下",
        "殿下",
        "親方",
    }
)
## 「方」を「ガタ」と読ませる敬称・複数の前接語
_HONORIFIC_PLURAL_PREDECESSORS = frozenset(
    {"皆様", "皆", "みんな", "あなた", "先生", "奥様", "お客様", "親御", "殿"}
)
## 「前」を「ゼン」と読ませる前接語
_ZEN_PREDECESSORS = frozenset(
    {
        "紀元",
        "門",
        "生",
        "就学",
        "出生",
        "公判",
        "産",
        "術",
        "食",
        "陸",
        "膝蓋",
        "祝典",
        "患難",
        "停滞",
        "閉塞",
        "寒帯",
        "温暖",
        "暴露",
    }
)
## 「橋」を「キョー」と読ませる構造種別の前接語
_BRIDGE_TYPE_PREDECESSORS = frozenset(
    {
        "高架",
        "可動",
        "水管",
        "人道",
        "跨道",
        "跨線",
        "連絡",
        "斜張",
        "張",
        "河口",
        "併用",
        "吊",
        "桁",
        "鉄道",
        "歩道",
        "陸",
        "仮設",
        "アーチ",
        "トラス",
        "ラーメン",
    }
)
## 「寺」を「デラ」と読ませる前接語
_DERA_PREDECESSORS = frozenset(
    {"縁切", "駆け込み", "田舎", "猫", "だるま", "隠れ", "峯", "山", "花"}
)
## 「寺」を「ジ」と読ませる実証済みの前接語
_JI_PREDECESSORS = frozenset({"霊山"})
## 名詞直後の後部要素へ与える複合語の読み (read, pron, 対象の品詞細分類。None は品詞を問わない)
_COMPOUND_SUFFIX_READINGS = {
    "不足": ("ブソク", "ブソク", "サ変接続"),
    "焼": ("ヤキ", "ヤキ", "接尾"),
    "峡": ("キョウ", "キョー", "一般"),
    "屯": ("トン", "トン", "サ変接続"),
    "角形": ("カクケイ", "カクケー", "一般"),
    "通": ("ドオリ", "ドーリ", "固有名詞"),
    "旗": ("キ", "キ", "一般"),
    "環": ("カン", "カン", None),
    "洞": ("ドウ", "ドー", None),
    "湖": ("コ", "コ", None),
    "唇": ("シン", "シン", None),
    "印": ("イン", "イン", None),
    "塚": ("ズカ", "ズカ", None),
    "小屋": ("ゴヤ", "ゴヤ", None),
    "部屋": ("ベヤ", "ベヤ", None),
    "付": ("ツキ", "ツキ", "接尾"),
    "金": ("キン", "キン", "一般"),
    "公": ("コウ", "コー", "一般"),
    "硬": ("コウ", "コー", None),
}

# 後ろの語を独立したアクセント句として読む、指示的な漢語の接頭辞 (「本論文」の「本」など)
## 収集した例で、後ろの語が接頭辞と切れて発音されていたものだけを持つ
_INDEPENDENT_PREFIXES = frozenset({"本", "当", "同", "全"})

# 旧字体・異体字を OpenJTalk の辞書で使われる通用字へ一括で置き換えるための変換表
_ITAIJI_TRANSLATION = str.maketrans(ITAIJI_MAP)

# Sudachi の Dictionary はスレッド間で共有可能だが、Tokenizer はスレッドセーフでないため
# Dictionary をモジュールレベルで一度だけ生成し、Tokenizer のみスレッドごとに遅延初期化する
_SUDACHI_DICTIONARY: dictionary.Dictionary | None = None
_SUDACHI_DICTIONARY_LOCK = Lock()
_SUDACHI_TOKENIZER_LOCAL = local()


def _get_sudachi_tokenizer() -> tokenizer.Tokenizer:
    """
    現在のスレッドに紐づく Sudachi Tokenizer を取得する。

    Sudachi の Dictionary はスレッド間で共有可能だが、Tokenizer はスレッドセーフでないため、
    Dictionary はモジュールレベルで一度だけ生成し、Tokenizer のみスレッドごとに遅延初期化する。

    Returns:
        tokenizer.Tokenizer: 遅延初期化済みの Sudachi tokenizer
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
    normalize_mode: NormalizeMode = "None",
) -> str:
    """
    指定された方式で Unicode 正規化を行う。

    Args:
        text (str): 正規化対象のテキスト
        normalize_mode (NormalizeMode): 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`

    Returns:
        str: 正規化後のテキスト。正規化不要な場合は元の文字列をそのまま返す

    Raises:
        ValueError: `normalize_mode` に未対応の値が指定された場合
    """

    if normalize_mode not in ("None", "NFC", "NFKC"):
        raise ValueError("normalize_mode must be one of 'None', 'NFC', or 'NFKC'")
    if normalize_mode == "None":
        return text

    normalized_form: Literal["NFC", "NFKC"] = "NFC" if normalize_mode == "NFC" else "NFKC"
    if unicodedata.is_normalized(normalized_form, text) is True:
        return text
    return unicodedata.normalize(normalized_form, text)


def normalize_itaiji(
    text: str,
    target_characters: frozenset[str] | None = None,
) -> str:
    """
    旧字体と異体字を、OpenJTalk の辞書で使われる通用字へ置き換える。

    Args:
        text (str): 正規化対象のテキスト
        target_characters (frozenset[str] | None): 置き換えてよい文字。None の場合はすべての異体字を置き換える

    Returns:
        str: 通用字へ置き換えたテキスト
    """

    if target_characters is not None:
        return "".join(
            ITAIJI_MAP.get(character, character) if character in target_characters else character
            for character in text
        )
    return text.translate(_ITAIJI_TRANSLATION)


def normalize_unknown_itaiji(text: str, inference_jtalk: OpenJTalk) -> str:
    """
    辞書で読めない異体字だけを通用字へ置き換える。

    Args:
        text (str): 正規化済みの入力テキスト
        inference_jtalk (OpenJTalk): 異体字を読めるかどうかを確かめる OpenJTalk インスタンス

    Returns:
        str: 辞書で読めない異体字だけを通用字へ置き換えたテキスト
    """

    # 異体字を含まない通常の文では、追加の形態素解析をしない
    fully_normalized_text = normalize_itaiji(text)
    if fully_normalized_text == text:
        return text

    # ユーザー辞書を含む今の辞書で読める字形は、その字形に固有の読み・品詞・アクセントを残すため置き換えない
    # 同じ異体字が既知語と未知語の両方に現れることがあるので、置き換えるのは未知語の位置にある字だけにする
    # 未知語の位置は MeCab 向けに正規化した本文上の位置なので、入力の1文字ずつを正規化して入力上の位置へ対応付ける
    _, morphs = inference_jtalk.run_mecab_detailed(text)
    normalized_characters = [inference_jtalk.normalize_for_mecab(character) for character in text]
    if "".join(normalized_characters) != inference_jtalk.normalize_for_mecab(text):
        ## 1文字ずつの正規化と本文全体の正規化が食い違う入力では位置を対応付けられないので、未知語の字の集合で置き換える
        unknown_characters = frozenset(
            "".join(morph["surface"] for morph in morphs if morph["is_unknown"] is True)
        )
        return normalize_itaiji(text, unknown_characters)
    original_index_by_mecab_index = [
        index
        for index, normalized_character in enumerate(normalized_characters)
        for _ in normalized_character
    ]
    unknown_positions = {
        original_index_by_mecab_index[mecab_index]
        for morph in morphs
        if morph["is_unknown"] is True
        for mecab_index in range(*morph["char_span"])
    }
    return "".join(
        normalize_itaiji(character) if index in unknown_positions else character
        for index, character in enumerate(text)
    )


def merge_njd_marine_features(
    njd_features: list[NJDFeature], marine_results: dict[str, Any]
) -> list[NJDFeature]:
    """
    marine のアクセント推定結果を NJDFeature 列へ反映する。

    Args:
        njd_features (list[NJDFeature]): OpenJTalk の NJD 処理結果
        marine_results (dict[str, Any]): marine の `predict()` 戻り値。
            `accent_status` と `accent_phrase_boundary` を参照する

    Returns:
        list[NJDFeature]: `acc` と `chain_flag` を marine 推定値で上書きした NJDFeature 列

    Raises:
        ValueError: `njd_features` と marine 結果の長さが一致しない場合
    """

    features = []

    marine_accs = marine_results["accent_status"]
    marine_chain_flags = marine_results["accent_phrase_boundary"]

    if len(njd_features) != len(marine_accs) or len(njd_features) != len(marine_chain_flags):
        raise ValueError("Invalid sequence sizes in njd_results, marine_results")

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
    Sudachi の形態素解析結果と NJD の形態素を逆順で突合し、対象漢字の pron / read を Sudachi の読みで上書きする。
    接尾辞の読みは上書き対象から除外するが、形態素境界を保つため対応する Sudachi 候補は消費する。

    Args:
        text (str): 読み対象となるテキスト
        pyopen_njd (list[NJDFeature]): OpenJTalk の形態素解析結果
        target_kanji_set (frozenset[str]): 複数の読みを持つ対象漢字の集合

    Returns:
        list[NJDFeature]: 漢字の読み補正を適用した形態素解析結果。
            突合に失敗した場合は元の形態素をそのまま返す
    """

    if len(target_kanji_set) == 0:
        return pyopen_njd
    if any(feature["orig"] in target_kanji_set for feature in pyopen_njd) is False:
        return pyopen_njd

    sudachi_yomi = sudachi_analyze(text, target_kanji_set)
    corrected_njd = [feature.copy() for feature in pyopen_njd]

    # 全対象の対応が確認できるまではコピーだけを変更する
    ## 逆順照合の途中で失敗しても、呼び出し元が渡した NJD features へ半端な補正を残さない
    for feature in reversed(corrected_njd):
        if feature["orig"] in target_kanji_set:
            try:
                correct_yomi = sudachi_yomi.pop()
            except IndexError:
                return pyopen_njd
            if correct_yomi[0] != feature["orig"]:
                return pyopen_njd

            # OpenJTalk が接尾辞として確定した読みは、前接語との結合を反映した結果なので保持する
            ## Sudachi の単漢字読みは「支払時」の「時」を一般名詞の「トキ」として返すため、ここで上書きすると
            ## 文脈解析済みの「ジ」を失う一方、「その時」のような非自立名詞には既存の補正を適用できる
            if feature["pos_group1"] == "接尾":
                continue

            # Sudachi の「ホウ」を正書法の読みとして保持し、発音のみ長音表記へ変換する
            ## 他の語と同様に両方を Sudachi の「ホウ」で上書きすると発音の長音化「ホー」が失われ、
            ## かといって read まで「ホー」にすると正書法のかな読みが崩れるため、pron だけを差し替える
            if correct_yomi == ["方", "ホウ"]:
                feature["read"] = "ホウ"
                feature["pron"] = "ホー"
            else:
                feature["read"] = correct_yomi[1]
                feature["pron"] = correct_yomi[1]
            # 読みの差し替えでモーラ数が変わるため、後続のアクセント計算が参照する値を同じ発音へ揃える
            feature["mora_size"] = len(split_kana_mora(feature["pron"]))

    # Sudachi 側に未対応の対象語が残る場合も、形態素境界が一致していないと判断する
    if len(sudachi_yomi) > 0:
        return pyopen_njd
    return corrected_njd


def sudachi_analyze(text: str, target_kanji_set: frozenset[str]) -> list[list[str]]:
    """
    複数の読み方をする漢字の読みを Sudachi で形態素解析した結果をリストで返す。

    Args:
        text (str): 読み対象となるテキスト
        target_kanji_set (frozenset[str]): 複数の読みを持つ対象漢字の集合

    Returns:
        list[list[str]]: 漢字とその読み方のリスト
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


def is_high_confidence_nani_context(next_feature: NJDFeature | None) -> bool:
    """
    後続形態素だけで「何」を「ナニ」と確定できる文脈か判定する。

    Args:
        next_feature (NJDFeature | None): 「何」の次にある NJD feature

    Returns:
        bool: ナニと確定できる場合は True
    """

    if next_feature is None:
        return False
    return (
        next_feature["orig"] in {"を", "が", "に", "も", "より"}
        or next_feature["orig"] == "する"
        or (
            next_feature["string"] == "で"
            and next_feature["pos"] == "助動詞"
            and next_feature["ctype"] == "特殊・ダ"
        )
    )


def predict_nani_reading(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    ONNX モデルを用いて、単独形態素として出現した「何」の読みを補正する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)

    Returns:
        list[NJDFeature]: 「何」の読み補正を適用した NJDNode 用 features
    """

    if any(feature["orig"] == "何" for feature in njd_features) is False:
        return njd_features

    for feature_index, current_feature in enumerate(njd_features):
        if current_feature["orig"] != "何":
            continue

        next_feature = (
            njd_features[feature_index + 1] if feature_index + 1 < len(njd_features) else None
        )
        # 後続形態素で読みが確定する文脈と、既定読みを保つ格助詞「で」はモデル判定を省く
        ## 格助詞「で」は、単一形態素「何で」が担っていた製品既定の「ナン」を分割後も維持する
        is_high_confidence_nani = is_high_confidence_nani_context(next_feature)
        should_keep_default_nan = (
            next_feature is not None
            and next_feature["orig"] == "で"
            and next_feature["pos"] == "助詞"
            and next_feature["pos_group1"] == "格助詞"
        )
        if is_high_confidence_nani is True:
            is_read_nan = 0
        elif should_keep_default_nan is True:
            is_read_nan = 1
        else:
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
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)

    Returns:
        list[NJDFeature]: 不自然な長音化を補正した NJDNode 用 features
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
    長音、促音、撥音がアクセント核に来た場合に、核位置を1モーラ前へずらす。

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
    品詞「特殊・マス」は直前に接続する動詞にアクセント核がある場合、アクセント核を「ま」に移動させる法則がある。
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
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False

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


def normalize_iu(
    njd_features: list[NJDFeature],
    pronunciation: IuPronunciation,
) -> list[NJDFeature]:
    """
    動詞「言う」と、「という」などの定型表現に含まれる「イウ」の発音を、指定した方式に揃える。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        pronunciation (IuPronunciation): 「イウ」をどう発音するかの方式

    Returns:
        list[NJDFeature]: 「イウ」の発音を揃えた NJDNode 用 features
    """

    is_kanji_only = pronunciation in {"KanjiIu", "KanjiYuu", "KanjiYuuBase"}
    is_base_only = pronunciation in {"YuuBase", "KanjiYuuBase"}
    replacement = "イ" if pronunciation in {"Iu", "KanjiIu"} else "ユ"

    def _replace_at(feature: NJDFeature, index: int) -> None:
        """
        発音の指定した位置が「言う」の「イ」(または「ユ」) なら、選んだ方式の発音に置き換える。

        Args:
            feature (NJDFeature): 書き換える NJDNode 用 feature
            index (int): 発音の中の「イ」または「ユ」の位置
        """

        if index < 0 or index >= len(feature["pron"]):
            return
        if feature["pron"][index] not in {"イ", "ユ"}:
            return
        if is_base_only is True and feature["pron"][index + 1 : index + 2] != "ウ":
            return
        if is_base_only is True:
            feature["pron"] = feature["pron"][:index] + "ユー" + feature["pron"][index + 2 :]
            return
        feature["pron"] = feature["pron"][:index] + replacement + feature["pron"][index + 1 :]

    for feature in njd_features:
        original = feature["orig"]
        if is_kanji_only is True and "言" not in original and "云" not in original:
            continue

        # 定型表現では「言う」にあたる音の位置が語形ごとに決まっているので、表層形ごとに置き換える位置を固定する
        if feature["pos"] == "連体詞" and original in {
            "こういう",
            "そういう",
            "どういう",
            "ああいう",
        }:
            _replace_at(feature, 2)
            continue
        if original.startswith(("ていう", "という")):
            _replace_at(feature, 1)
            continue
        if original.startswith(("っていう", "とかいう")):
            _replace_at(feature, 2)
            continue
        if original.startswith(("あっという", "アッという", "あっと言う", "アッと言う")):
            _replace_at(feature, 3)
            continue

        is_target_pos = (
            (feature["pos"] == "動詞" and feature["pos_group1"] == "自立")
            or (
                feature["pos"] == "形容詞"
                and feature["pos_group1"].endswith("自立")
                and feature["ctype"] == "形容詞・アウオ段"
            )
            or (feature["pos"] == "副詞" and feature["pos_group1"] == "一般")
        )
        if is_target_pos is False:
            continue

        if feature["pron"] == "イウ" or original.startswith(("いう", "言う", "云う")):
            _replace_at(feature, 0)
            continue
        if "言う" in original:
            # 複合語では後ろ側の「言う」にあたる音を対象にし、語幹側にある同じ音の並びは変えない
            for index in range(len(feature["pron"]) - 2, -1, -1):
                if feature["pron"][index] in {"イ", "ユ"} and feature["pron"][index + 1] in {
                    "ウ",
                    "ッ",
                    "エ",
                    "オ",
                    "ー",
                }:
                    _replace_at(feature, index)
                    break

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


def modify_context_reading(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    すぐ前後の形態素だけで読みが1つに決まる語 (「駆け込み寺」の「デラ」、「先生方」の「ガタ」など) の読みを書き換える。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features

    Returns:
        list[NJDFeature]: 前後の語から読みを補正した NJDNode 用 features
    """

    def _set_reading(
        feature: NJDFeature,
        reading: str,
        pronunciation: str | None = None,
    ) -> None:
        """
        読みと発音を書き換え、書き換えた発音からモーラ数を数え直す。

        Args:
            feature (NJDFeature): 書き換える NJDNode 用 feature
            reading (str): 新しい読み
            pronunciation (str | None): 新しい発音。None の場合は読みと同じにする
        """

        feature["read"] = reading
        feature["pron"] = reading if pronunciation is None else pronunciation
        feature["mora_size"] = len(split_kana_mora(feature["pron"]))

    for index, feature in enumerate(njd_features):
        previous = njd_features[index - 1] if index > 0 else None
        previous_previous = njd_features[index - 2] if index > 1 else None
        following = njd_features[index + 1] if index + 1 < len(njd_features) else None
        surface = feature["string"]

        # 直後の語で意味が確定する少数の表現を、閉じた表層形の集合で判定する
        if surface == "一見" and following is not None and following["string"] == "さん":
            _set_reading(feature, "イチゲン")
        elif (
            surface == "一声"
            and following is not None
            and following["string"]
            in {
                "かけ",
                "掛け",
                "かける",
                "掛ける",
            }
        ):
            _set_reading(feature, "ヒトコエ")
        elif surface == "一行" and following is not None and following["string"] in {"ごと", "毎"}:
            _set_reading(feature, "イチギョウ", "イチギョー")
        elif surface == "兵" and following is not None and following["string"] in {"ども", "共"}:
            _set_reading(feature, "ツワモノ")
        elif (
            surface == "如何"
            and following is not None
            and following["string"]
            in {
                "で",
                "です",
                "でし",
                "でしょ",
            }
        ):
            _set_reading(feature, "イカガ")

        # 直前の語が読みを確定する敬称・仏号・定型表現を表層形で判定する
        elif (
            surface == "仏"
            and previous is not None
            and previous["string"]
            in {
                "阿弥陀",
                "釈迦",
                "大日",
                "薬師",
                "毘盧遮那",
            }
        ):
            _set_reading(feature, "ブツ")
        elif (
            surface == "方"
            and feature["pos_group1"] == "接尾"
            and previous is not None
            and previous["string"] in _HONORIFIC_PLURAL_PREDECESSORS
        ):
            _set_reading(feature, "ガタ")
        elif surface == "前" and previous is not None and previous["string"] in _ZEN_PREDECESSORS:
            _set_reading(feature, "ゼン")
        elif surface == "様" and previous is not None and previous["string"] == "同じ":
            _set_reading(feature, "ヨウ", "ヨー")
        elif (
            surface == "下"
            and feature["pos_group1"] == "一般"
            and previous is not None
            and previous["string"] == "の"
            and previous_previous is not None
            and previous_previous["string"] in _ABSTRACT_NO_PREDECESSORS
        ):
            _set_reading(feature, "モト")

        # 「橋」は構造種別なら「キョウ」（「キョー」）と読ませ、名詞に続く接尾辞用法は「バシ」と読ませる
        elif (
            surface == "橋"
            and previous is not None
            and previous["string"] in _BRIDGE_TYPE_PREDECESSORS
        ):
            _set_reading(feature, "キョウ", "キョー")
        elif (
            surface == "橋"
            and feature["pos_group1"] == "接尾"
            and previous is not None
            and previous["pos"] == "名詞"
        ):
            _set_reading(feature, "バシ")
        elif surface == "寺" and previous is not None and previous["string"] in _DERA_PREDECESSORS:
            _set_reading(feature, "デラ")
        # 「寺」の読みは前接語によって「ジ」と「デラ」に分かれるため、実証済みの複合語だけを閉じた集合で「ジ」へ補正する
        elif surface == "寺" and previous is not None and previous["string"] in _JI_PREDECESSORS:
            _set_reading(feature, "ジ")

        # 名詞へ直接続く後部要素は、助詞を挟んだ独立用法と区別して複合語の読みへ変える
        elif (
            surface in _COMPOUND_SUFFIX_READINGS
            and previous is not None
            and previous["pos"] == "名詞"
        ):
            reading, pronunciation, required_pos_group1 = _COMPOUND_SUFFIX_READINGS[surface]
            if required_pos_group1 is None or feature["pos_group1"] == required_pos_group1:
                _set_reading(feature, reading, pronunciation)
        # 辞書が人名と解析する「記念章」だけを「ショウ」へ補正し、人名の「章」は「アキラ」と読むように残す
        elif (
            surface == "章"
            and feature["pos_group1"] == "固有名詞"
            and previous is not None
            and previous["string"] == "記念"
        ):
            _set_reading(feature, "ショウ", "ショー")

        # 「等」は代名詞に続けば「ラ」、自立した名詞に続けば「トウ」（「トー」）、活用語や形式名詞では既定の「ナド」を残す
        elif surface == "等" and previous is not None and previous["pos_group1"] == "代名詞":
            _set_reading(feature, "ラ")
        elif (
            surface == "等"
            and previous is not None
            and previous["pos_group1"]
            in {
                "一般",
                "サ変接続",
                "固有名詞",
                "接尾",
            }
        ):
            _set_reading(feature, "トウ", "トー")

    return njd_features


def restore_loanword_kana(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    辞書が別の仮名へ置き換えた外来語の表記 (「ヴィ」を「ビ」にするなど) を、表層形から元に戻す。
    表層形と読み・発音の全体が1文字ずつ対応する場合だけ戻すので、日本語で定着した別の表記や、中黒などを省いた語は変えない。

    Args:
        njd_features (list[NJDFeature]): 復元対象の NJDNode 用 features

    Returns:
        list[NJDFeature]: 外来語の表記を戻した NJDNode 用 features
    """

    def _restore_spelling(
        surface: str,
        pronunciation: str,
    ) -> tuple[str, list[tuple[int, int, int]]] | None:
        """
        表層形と読み (または発音) を先頭から照合し、置き換えられた外来語の仮名を表層形の表記へ戻す。

        Args:
            surface (str): 外来語の表層形
            pronunciation (str): 辞書が与えた読みまたは発音

        Returns:
            tuple[str, list[tuple[int, int, int]]] | None: 表記を戻した読みまたは発音と、戻した箇所ごとの (戻す前の先頭のモーラ位置、戻す前のモーラ数、戻した後のモーラ数) の一覧 (全体が対応しないか、戻す箇所がなければ None)
        """

        restored: list[str] = []
        restored_segments: list[tuple[int, int, int]] = []
        surface_offset = 0
        pronunciation_offset = 0
        is_changed = False

        # 無声化記号は表層形に現れないため、照合位置を進めず発音側から持ち越す
        while surface_offset < len(surface):
            if pronunciation.startswith("’", pronunciation_offset):
                restored.append("’")
                pronunciation_offset += 1
                continue

            # 辞書が置き換えた組み合わせを優先し、該当しなければ同じ文字を1文字ずつ照合する
            restoration = next(
                (
                    (original, collapsed)
                    for original, collapsed in _LOANWORD_KANA_RESTORATIONS
                    if surface.startswith(original, surface_offset)
                    and pronunciation.startswith(collapsed, pronunciation_offset)
                ),
                None,
            )
            if restoration is not None:
                original, collapsed = restoration
                restored_segments.append(
                    (
                        len(split_kana_mora(pronunciation[:pronunciation_offset].replace("’", ""))),
                        len(split_kana_mora(collapsed)),
                        len(split_kana_mora(original)),
                    )
                )
                restored.append(original)
                surface_offset += len(original)
                pronunciation_offset += len(collapsed)
                is_changed = True
                continue

            surface_char = surface[surface_offset]
            if pronunciation.startswith(surface_char, pronunciation_offset) is False:
                return None
            restored.append(surface_char)
            surface_offset += 1
            pronunciation_offset += 1

        # 語末の無声化記号も、表層形との対応を崩さず保持する
        if pronunciation.startswith("’", pronunciation_offset):
            restored.append("’")
            pronunciation_offset += 1
        if pronunciation_offset != len(pronunciation) or is_changed is False:
            return None
        return "".join(restored), restored_segments

    for feature in njd_features:
        # 外来語だけへ限定し、漢字や区切り記号を含む語の読みを表層形から作らない
        surface = feature["string"]
        if is_katakana_word(surface) is False:
            continue
        restored_read = _restore_spelling(surface, feature["read"])
        if restored_read is not None:
            feature["read"] = restored_read[0]
        restored_pronunciation = _restore_spelling(surface, feature["pron"])
        if restored_pronunciation is not None:
            feature["pron"], restored_segments = restored_pronunciation
            # 無声化記号はモーラに数えない
            feature["mora_size"] = len(split_kana_mora(feature["pron"].replace("’", "")))
            # 「イエ」を「イェ」に戻すとモーラが減るので、アクセント核も戻す前と同じモーラの位置へずらす
            ## 核が戻した箇所の中にあるときは、戻した箇所の最後のモーラに置く
            accent_shift = 0
            for segment_start, collapsed_mora_count, original_mora_count in restored_segments:
                if feature["acc"] > segment_start + collapsed_mora_count:
                    accent_shift += collapsed_mora_count - original_mora_count
                elif feature["acc"] > segment_start:
                    offset_in_segment = feature["acc"] - segment_start
                    accent_shift += offset_in_segment - min(offset_in_segment, original_mora_count)
            feature["acc"] -= accent_shift

    return njd_features


def modify_old_province_yomi(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    令制国の名前に続く接尾辞「国」の読みを「コク」から「ノクニ」に変える。
    名前の側の読みは変えないので、名前そのものを誤読する語は辞書で直す必要がある。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features

    Returns:
        list[NJDFeature]: 旧国名に続く「国」の読みを補正した NJDNode 用 features
    """

    for index in range(1, len(njd_features)):
        feature = njd_features[index]
        if (
            feature["string"] != "国"
            or feature["pos_group1"] != "接尾"
            or njd_features[index - 1]["string"] not in _OLD_PROVINCE_NAMES
        ):
            continue

        feature["read"] = "ノクニ"
        feature["pron"] = "ノクニ"
        feature["mora_size"] = 3

    return njd_features


def read_unknown_kanji(
    njd_features: list[NJDFeature],
    text: str | None = None,
) -> list[NJDFeature]:
    """
    辞書にないため NJD が読みを付けられなかった漢字に、語としての読みか Unihan の音読みを付ける。
    送り仮名を伴う語 (「悪魔憑き」など) は Sudachi の語としての読みを、それ以外は Unihan の漢字ごとの音読みを使う。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        text (str | None): Sudachi で語としての読みを引くための入力テキスト。None の場合は Unihan の読みだけを使う

    Returns:
        list[NJDFeature]: 未知の漢字に読みを付けた NJDNode 用 features
    """

    def _is_kanji(character: str) -> bool:
        """
        Unihan の読みを付ける対象になる CJK 統合漢字かを判定する。

        Args:
            character (str): 判定する1文字

        Returns:
            bool: CJK 統合漢字 (拡張 A 以降と互換漢字を含む) の場合は True
        """

        codepoint = ord(character)
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x3FFFF
        )

    def _reading_to_pronunciation(reading: str) -> str:
        """
        読みの連母音 (「オウ」「エイ」など) を、OpenJTalk の発音の長音「ー」に書き換える。

        Args:
            reading (str): カタカナの読み

        Returns:
            str: 長音を「ー」で書いた発音
        """

        pronunciation: list[str] = []
        previous_vowel: str | None = None
        for mora in split_kana_mora(reading):
            if (previous_vowel, mora) in {("o", "ウ"), ("o", "オ"), ("u", "ウ"), ("e", "イ")}:
                pronunciation.append("ー")
                continue
            pronunciation.append(mora)
            if mora != "ー":
                previous_vowel = _VOWEL_BY_LAST_KANA.get(mora[-1])
        return "".join(pronunciation)

    def _find_sudachi_reading(feature_index: int) -> str | None:
        """
        Sudachi の1語が前後の既知の形態素をまたぐ場合も含めて、未知の漢字の部分の読みを取り出す。

        Args:
            feature_index (int): 未知の漢字の形態素の位置

        Returns:
            str | None: 前後の既知の形態素の読みを差し引いた読み (Sudachi の読みが一意に決まらなければ None)
        """

        for candidate_start in range(feature_index, -1, -1):
            preceding_read = "".join(
                candidate["read"] for candidate in njd_features[candidate_start:feature_index]
            )
            for candidate_end in range(feature_index + 1, len(njd_features) + 1):
                candidate_surface = "".join(
                    candidate["string"] for candidate in njd_features[candidate_start:candidate_end]
                )
                candidate_readings = sudachi_readings_by_surface.get(candidate_surface, set())
                if len(candidate_readings) != 1:
                    continue
                following_read = "".join(
                    candidate["read"]
                    for candidate in njd_features[feature_index + 1 : candidate_end]
                )
                candidate_reading = next(iter(candidate_readings))
                if candidate_reading.startswith(preceding_read) is False:
                    continue
                if following_read != "" and candidate_reading.endswith(following_read) is False:
                    continue
                extracted_read = candidate_reading[
                    len(preceding_read) : len(candidate_reading) - len(following_read)
                ]
                if extracted_read != "":
                    return extracted_read
        return None

    # NJD が記号・読点に変えた漢字だけを対象にし、本物の句読点と既知の語は変えない
    unknown_kanji_indices = [
        feature_index
        for feature_index, feature in enumerate(njd_features)
        if feature["pos"] == "記号"
        and feature["pos_group1"] == "読点"
        and any(_is_kanji(character) for character in feature["string"]) is True
    ]
    # 未知の漢字がない文では Sudachi を読み込まない (tsqyomi を使うときや Sudachi の読み補正を切ったときも、読めない漢字がなければ Sudachi は動かない)
    if len(unknown_kanji_indices) == 0:
        return njd_features

    sudachi_readings_by_surface: dict[str, set[str]] = {}
    if text is not None:
        # 送り仮名を伴う語は漢字ごとの音読みでは決まらないため、Sudachi の語としての読みを候補にする
        for morpheme in _get_sudachi_tokenizer().tokenize(text, tokenizer.Tokenizer.SplitMode.C):
            reading = morpheme.reading_form()
            if reading != morpheme.surface():
                sudachi_readings_by_surface.setdefault(morpheme.surface(), set()).add(reading)

    for feature_index in unknown_kanji_indices:
        feature = njd_features[feature_index]
        reading = _find_sudachi_reading(feature_index)
        if reading is None:
            # 語として読めない未知の漢字は、すべての文字に一意な Unihan の読みがある場合だけ読みを付ける
            reading_parts: list[str] = []
            for character in feature["string"]:
                if character in UNIHAN_READINGS:
                    reading_parts.append(UNIHAN_READINGS[character])
                elif _is_kanji(character) is False:
                    reading_parts.append(character)
                else:
                    reading_parts = []
                    break
            if len(reading_parts) == 0:
                continue
            reading = "".join(reading_parts)

        feature["read"] = reading
        feature["pron"] = _reading_to_pronunciation(reading)
        feature["pos"] = "名詞"
        feature["pos_group1"] = "一般"
        feature["pos_group2"] = "*"
        feature["pos_group3"] = "*"
        feature["mora_size"] = len(split_kana_mora(feature["pron"]))
        feature["acc"] = 0
        feature["chain_flag"] = -1

    return njd_features


def split_prefix_accent_phrase(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    「本論文」「当ホテル」のような指示的な漢語の接頭辞の後ろの語を、独立したアクセント句にする。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features

    Returns:
        list[NJDFeature]: 接頭辞の後ろでアクセント句を分けた NJDNode 用 features
    """

    # 接頭辞と同じアクセント句につながっている後ろの語だけを、新しいアクセント句の始まりにする
    for index in range(1, len(njd_features)):
        previous = njd_features[index - 1]
        if (
            njd_features[index]["chain_flag"] == 1
            and previous["pos"] == "接頭詞"
            and previous["string"] in _INDEPENDENT_PREFIXES
        ):
            njd_features[index]["chain_flag"] = 0

    return njd_features


def detect_odori_unit(read: str) -> int | None:
    """
    読み文字列を清音化し、末尾の繰り返し単位 (周期) を検出する。
    「々」の展開で直前トークンが既に踊り字展開済みの場合に、繰り返しの基底単位を特定するために使う。

    例: 「サマザマ」→ 清音化→ 「サマサマ」→ モーラ ["サ","マ","サ","マ"] → 周期 2 (「サマ」が繰り返し)

    Args:
        read (str): 直前トークンの読み (カタカナ)

    Returns:
        int | None: 繰り返し周期 (モーラ数)。検出できなかった場合は None
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
    jtalk: OpenJTalk | None = None,
) -> list[NJDFeature]:
    """
    踊り字（々）と一の字点（ゝ、ゞ、ヽ、ヾ）の読みを適切に処理する後処理関数。

    OpenJTalk の挙動に合わせて、連続する踊り字を処理する。
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
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。
            None の場合は MeCab 再解析が必要な踊り字処理を省略する

    Returns:
        list[NJDFeature]: 踊り字の読みを修正した NJDNode 用 features
    """

    def is_dancing(orig: str) -> bool:
        """
        文字列が踊り字のみで構成されているかを判定する。

        Args:
            orig (str): 判定対象の文字列

        Returns:
            bool: 踊り字のみで構成されている場合は True
        """
        return set(orig) == {"々"}

    def is_odoriji(orig: str) -> bool:
        """
        文字列が一の字点のみで構成されているかを判定する。

        Args:
            orig (str): 判定対象の文字列

        Returns:
            bool: 一の字点のみで構成されている場合は True
        """
        return set(orig) <= {"ゝ", "ゞ", "ヽ", "ヾ"}

    def count_odori(orig: str) -> int:
        """
        文字列に含まれる踊り字の数をカウントする。

        Args:
            orig (str): カウント対象の文字列

        Returns:
            int: 踊り字の数
        """
        return orig.count("々")

    def is_kanji_token(token: NJDFeature) -> bool:
        """
        トークンが漢字を含むかを判定する。

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
        """
        トークンが1文字の漢字で構成されているかを判定する。

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
        next_feature: NJDFeature | None = None,
    ) -> tuple[bool, str, str | None]:
        """
        踊り字の直前の漢字を再解析する必要があるかを判定する。

        Args:
            odori_feature (NJDFeature): 踊り字のトークン
            prev_feature (NJDFeature): 直前のトークン
            next_feature (NJDFeature | None, optional): 後続のトークン

        Returns:
            tuple[bool, str, str | None]: (再解析が必要か, 再解析する漢字, 後続の漢字)
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
        """
        漢字を再解析して読みを取得する。

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
        """
        一の字点の読みを処理する。

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

        # 無声化記号 (’) などはモーラとして扱わないよう除去してから分割する
        ## これを行わないと「マス’」のような発音が「マ」「ス」「’」の3文字に分割され、
        ## モーラ数や一の字点展開時の読みが不整合になる
        prev_pron_source = prev_feature["pron"].replace("’", "")
        # 万が一除去の結果として空文字列になった場合は、読みの情報を失わないよう read 側を代替とする
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

        # 踊り字は元の MeCab 解析で「記号」に分類されるが、展開後は実体を持つ単語となるため「名詞,一般」へ変更して後続処理での誤作動を防ぐ
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
                        # 踊り字の展開結果を後続の NJD 処理で通常の形態素として扱わせるため、「名詞,一般」へ変更する
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
                        # 展開後の踊り字を後続の NJD 処理で通常の形態素として扱わせるため、「名詞,一般」へ変更する
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

                # 展開後の踊り字を後続の NJD 処理で通常の形態素として扱わせるため、「名詞,一般」へ変更する
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
