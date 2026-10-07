import difflib
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
## 「前」を「ゼン」と読ませる後続の役職名と段階の語
_ZEN_SUCCESSORS = frozenset({"会長", "大統領", "段階", "理事長", "社長", "総裁", "首相"})
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
        "道路",
        "段",
        "ＰＣ",
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
# 「縁」を「フチ」と読む器物や形状を表す前接語
_FUCHI_PREDECESSORS = frozenset(
    {"カップ", "器", "堀", "崖", "径", "屏風", "内側", "口", "皿", "模様", "火鉢", "金"}
)
# 「ひがみ入ってます」の名詞用法も動詞として解析されるため、複合動詞の「入る」は確認できた前接語に限る
_IRU_COMPOUND_PREDECESSORS = frozenset({"走り", "攻め", "折り", "分け"})
# 「大」は後続する語によって「オー」と「ダイ」が分かれるため、表層形で判定する
_OO_SUCCESSORS = frozenset(
    {"にぎわい", "丸髷", "地主", "旦那", "泥坊", "津波", "掃除", "番狂わせ", "番頭", "違い"}
)
# 「御」は「御言葉」の「オ」と「御住所」の「ゴ」を後続する語で分ける
_O_SUCCESSORS = frozenset(
    {
        "仕置",
        "屋敷",
        "帰り",
        "急ぎ",
        "手際",
        "支払い",
        "楽しみ",
        "気の毒",
        "田植祭",
        "神籤",
        "粗末",
        "言葉",
        "近く",
        "隣",
        "嬢",
    }
)
# 「の」を挟んで道具が前に来る「柄」は「エ」、刀剣が前に来る「柄」は「ツカ」と読む
_TOOL_HANDLE_PREDECESSORS = frozenset(
    {
        "うちわ",
        "やり",
        "傘",
        "剃刀",
        "団扇",
        "提灯",
        "斧",
        "柄杓",
        "槍",
        "洋傘",
        "箒",
        "薙刀",
        "鋏",
        "鋤",
        "鋸",
        "鍬",
        "錫杖",
    }
)
_SWORD_HILT_PREDECESSORS = frozenset(
    {
        "刀",
        "剣",
        "大刀",
        "太刀",
        "小剣",
        "懐剣",
        "木剣",
        "短刀",
        "短剣",
        "脇差",
        "軍刀",
        "長脇差",
        "鎧通し",
    }
)
# 「盲」「聾」の音読みを、学校種別や障害種別を表す語との列挙に限って適用する
_DISABILITY_ENUMERATION_TERMS = frozenset(
    {
        "盲",
        "聾",
        "ろう",
        "盲学校",
        "聾学校",
        "ろう学校",
        "養護学校",
        "視覚障害",
        "聴覚障害",
        "知的障害",
        "肢体不自由",
        "病弱",
    }
)
# 空間を比較する「より外」は「ソト」なので、「ホカ」への補正は動詞・代名詞と打ち消しの組に限る
_NEGATIVE_ORIGINALS = frozenset({"ない", "無い", "ぬ", "ん", "まい", "ず"})
# 「何にも知らない」「何にもならない」のように、打ち消しと組んで「ナンニモ」と読む述語に限る
## 「何にも似ていない」「何にも代えがたい」は格助詞の「に」を保ち、「ナニニモ」と読む
_NANNIMO_PREDICATES = frozenset(
    {
        "知る",
        "わかる",
        "分かる",
        "分る",
        "する",
        "出来る",
        "なる",
        "言う",
        "やる",
        "食べる",
        "聞く",
        "答える",
        "ある",
        "ない",
        "無い",
        "面白い",
    }
)
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

# 長音を戻すとき、発音の「ー」に対応する読みの文字がこれらの仮名なら、その仮名へ戻す
_LONG_VOWEL_KANA = frozenset("アイウエオァィゥェォ")

# 四つ仮名を戻すときの、読みの文字と発音の文字の組み合わせ
_YOTSUGANA_PAIRS = frozenset({("ヅ", "ズ"), ("ヂ", "ジ")})

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
    特殊拍上のアクセント核と、前部末型の結合で準特殊拍に来た核を1モーラ前へずらす。
    準特殊拍はア段に続く「イ」と、発音に無声化記号が付いた拍を対象とする。
    NHK アクセント辞典の「警戒心」「仙台市」と同じ補正を、C3 で結合する語に適用する。

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

    for index, njd in enumerate(njd_features):
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
                # C3 が置いた前部末の核だけ、二重母音と無声化拍の補正も適用する
                ## 「クイ」などの母音連続は仮名だけでは二重母音と確定できないため、ア段＋イに限定する
                next_njd = njd_features[index + 1] if index + 1 < len(njd_features) else None
                is_c3_boundary = (
                    acc == njd["mora_size"]
                    and next_njd is not None
                    and next_njd["chain_flag"] == 1
                    and next_njd["chain_rule"] == "C3"
                )
                # 無声化記号を拍数から除き、前部末の拍とその直前の母音を調べる
                if is_c3_boundary is True:
                    pron = pron.replace("’", "")
                try:
                    nuc_pron = pron[acc - 1]
                except IndexError:
                    nuc_pron = pron[0]
                previous_kana = njd["pron"].replace("’", "")[-2:-1]
                is_quasi_special_mora = (
                    len(pron) >= 2
                    and nuc_pron == "イ"
                    and (_DAN_MAP.get(previous_kana) == "a" or previous_kana == "ャ")
                ) or njd["pron"].endswith("’")
                if nuc_pron in inappropriate_for_nuclear_chars or (
                    is_c3_boundary is True and is_quasi_special_mora is True
                ):
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
    辞書が発音 (pron) で書き換えた長音と四つ仮名を、元のテキスト通りの読み (read) の表記に戻す。
    revert_long_vowels と revert_yotsugana は、名前が指す違いだけを文字単位で戻し、助詞の「ワ」「エ」、連濁、無声化の記号などは発音のまま残す。

    Args:
        njd_features (list[NJDFeature]): OpenJTalk の形態素解析結果
        use_read_as_pron (bool): True の場合、全ての発音を読みで丸ごと上書きする。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
        revert_long_vowels (bool): True の場合、辞書が長音にした母音を、読みの仮名へ戻す。
            元の表記に「ー」がある語は、書かれた長音なので戻さない。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、辞書が「ズ」「ジ」にした四つ仮名を、読みの「ヅ」「ヂ」へ戻す。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False

    Returns:
        list[NJDFeature]: 発音復元後の形態素解析結果
    """

    for feature in njd_features:
        if use_read_as_pron is True:
            feature["pron"] = feature["read"]
            continue
        read = feature["read"]
        # 無声化の記号「’」は読みにないので、位置を合わせる間だけ外しておく
        pronunciation_chars = [char for char in feature["pron"] if char != "’"]
        if read == "".join(pronunciation_chars):
            continue
        is_long_vowel_target = revert_long_vowels is True and "ー" not in feature["orig"]
        for read_char, pronunciation_index in _align_read_with_pronunciation(
            read, pronunciation_chars
        ):
            pronunciation_char = pronunciation_chars[pronunciation_index]
            is_long_vowel = pronunciation_char == "ー" and read_char in _LONG_VOWEL_KANA
            is_yotsugana = (read_char, pronunciation_char) in _YOTSUGANA_PAIRS
            if (is_long_vowel_target is True and is_long_vowel is True) or (
                revert_yotsugana is True and is_yotsugana is True
            ):
                pronunciation_chars[pronunciation_index] = read_char
        reverted_chars = iter(pronunciation_chars)
        feature["pron"] = "".join(
            char if char == "’" else next(reverted_chars) for char in feature["pron"]
        )

    return njd_features


def _align_read_with_pronunciation(
    read: str, pronunciation_chars: list[str]
) -> list[tuple[str, int]]:
    """
    読みと発音で、同じ位置にあるのに文字が違う箇所を、読みの文字と発音の位置の組で返す。
    長さが同じなら先頭から1文字ずつ対応させ、違えば一致しない区間のうち長さが同じものだけを対応させる。

    Args:
        read (str): 形態素の読み
        pronunciation_chars (list[str]): 無声化の記号を除いた形態素の発音の文字

    Returns:
        list[tuple[str, int]]: 文字が違う箇所の、読みの文字と発音の位置の組
    """

    if len(read) == len(pronunciation_chars):
        blocks = [(0, len(read), 0)]
    else:
        # 「アトリウム」と「アトリューム」のように長さが違う語は、一致する部分を除いた区間でだけ対応を取る
        matcher = difflib.SequenceMatcher(None, read, "".join(pronunciation_chars), autojunk=False)
        blocks = [
            (read_start, read_end, pronunciation_start)
            for tag, read_start, read_end, pronunciation_start, pronunciation_end in matcher.get_opcodes()
            if tag == "replace" and read_end - read_start == pronunciation_end - pronunciation_start
        ]
    return [
        (read[read_index], pronunciation_start + read_index - read_start)
        for read_start, read_end, pronunciation_start in blocks
        for read_index in range(read_start, read_end)
        if read[read_index] != pronunciation_chars[pronunciation_start + read_index - read_start]
    ]


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
    前後の形態素で読みが決まる語 (「駆け込み寺」の「デラ」、「先生方」の「ガタ」など) の読みを書き換える。
    「より外にない」の「ホカ」は、後続する打ち消しの語も確かめてから適用する。

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
        読みと発音を書き換え、モーラ数の変更に合わせてアクセント句の核の位置を補正する。

        Args:
            feature (NJDFeature): 書き換える NJDNode 用 feature
            reading (str): 新しい読み
            pronunciation (str | None): 新しい発音。None の場合は読みと同じにする
        """

        # 保護された読みは後段で復元されるため、そのまま残してアクセント核も維持する
        if feature.get("is_reading_protected", False) is True:
            return

        # 読みが合っている場合は、NJD が付けた無声化記号もそのまま使う
        new_pronunciation = reading if pronunciation is None else pronunciation
        if feature["read"] == reading and feature["pron"].replace("’", "") == new_pronunciation:
            return

        old_mora_size = feature["mora_size"]
        old_pronunciation = feature["pron"].replace("’", "")
        feature["read"] = reading
        feature["pron"] = new_pronunciation
        feature["mora_size"] = len(split_kana_mora(feature["pron"]))

        # NJD が数えた句の核を、変更後も同じ後続モーラに置く
        if feature["mora_size"] != old_mora_size:
            head = index
            while head > 0 and njd_features[head]["chain_flag"] == 1:
                head -= 1
            preceding_mora_size = sum(node["mora_size"] for node in njd_features[head:index])
            accent = njd_features[head]["acc"]
            if accent > preceding_mora_size + old_mora_size:
                njd_features[head]["acc"] += feature["mora_size"] - old_mora_size
            elif accent > preceding_mora_size:
                # 短い湖名では、「ミズウミ」の短縮で消える拍の核を前の要素の末尾に置く
                ## C1 の加算で得た核を「コ」に丸めると尾高型へ変わるので、3モーラ以下の下がり目を保つ
                if (
                    feature["string"] == "湖"
                    and old_pronunciation == "ミズウミ"
                    and new_pronunciation == "コ"
                    and preceding_mora_size + feature["mora_size"] <= 3
                    and accent > preceding_mora_size + feature["mora_size"]
                ):
                    njd_features[head]["acc"] = preceding_mora_size
                else:
                    njd_features[head]["acc"] = preceding_mora_size + min(
                        accent - preceding_mora_size, feature["mora_size"]
                    )

    for index, feature in enumerate(njd_features):
        previous = njd_features[index - 1] if index > 0 else None
        previous_previous = njd_features[index - 2] if index > 1 else None
        following = njd_features[index + 1] if index + 1 < len(njd_features) else None
        surface = feature["string"]

        # 単漢字の音読みをコストで優先すると文学作品の訓読みも変わるため、福祉・教育の語との列挙だけを補正する
        if (
            surface in {"盲", "聾"}
            and feature["pos"] == "名詞"
            and not feature.get("is_reading_protected", False)
            and _is_disability_enumeration(njd_features, index)
        ):
            _set_reading(
                feature,
                "モウ" if surface == "盲" else "ロウ",
                "モー" if surface == "盲" else "ロー",
            )
            # 「モー」「ロー」は2モーラの頭高型なので、元の訓読みの核を引き継がずに設定する
            feature["acc"] = 1

        # 「好き」の行のコストを下げると「旅好き」も変わるため、最上級を表す「一番」に続く形容動詞だけを補正する
        if (
            surface == "好き"
            and feature["pos_group1"] == "接尾"
            and feature["pos_group2"] == "形容動詞語幹"
            and previous is not None
            and previous["string"] == "一番"
        ):
            _set_reading(feature, "スキ")

        # 「何にも」の副詞の行を優先すると「何にも依存しない」も変わるため、打ち消しの述語で読みを選ぶ
        if (
            surface == "何"
            and feature["pos"] == "名詞"
            and index + 2 < len(njd_features)
            and njd_features[index + 1]["string"] == "に"
            and njd_features[index + 1]["pos_group1"] == "格助詞"
            and njd_features[index + 2]["string"] == "も"
            and njd_features[index + 2]["pos_group1"] == "係助詞"
            and _is_negative_nannimo_context(njd_features, index + 3)
        ):
            _set_reading(feature, "ナン")
        # 直後の語で意味が確定する少数の表現を、閉じた表層形の集合で判定する
        elif surface == "一見" and following is not None and following["string"] == "さん":
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
        elif surface == "町" and following is not None and following["string"] == "史":
            _set_reading(feature, "チョウ", "チョー")
        elif surface == "一" and following is not None and following["string"] == "しょ":
            _set_reading(feature, "イッ")
        # 「面を伏せる」はラケットにも使うため、恥じる動作や号令、古風な命令形「上げよ」で顔を表す場合だけを補正する
        ## 接尾辞の「面」と連体修飾を受ける「面」は表面も表すため、独立した名詞か、程度を表す「〜のあまり」の直後に限る
        elif (
            surface == "面"
            and feature["pos"] == "名詞"
            and (
                (feature["pos_group1"] == "一般" and feature["chain_flag"] != 1)
                or (
                    feature["pos_group1"] == "接尾"
                    and previous is not None
                    and previous["string"] == "あまり"
                    and previous_previous is not None
                    and previous_previous["pos_group1"] == "連体化"
                )
            )
            and not feature.get("is_reading_protected", False)
            and (previous is None or previous["pos_group1"] != "連体化")
            and following is not None
            and following["string"] == "を"
            and following["pos_group1"] == "格助詞"
            and index + 2 < len(njd_features)
            and njd_features[index + 2]["pos"] == "動詞"
            and njd_features[index + 2]["orig"] in {"伏せる", "上げる"}
        ):
            predicate = njd_features[index + 2]
            is_face_movement = predicate["orig"] == "上げる" and predicate["cform"] == "命令ｙｏ"
            # 恥じる気持ちや号令は読点を挟んで動作につながるので、句点やかぎ括弧までの前方を調べる
            ## 別の文の感情や命令が、表面を動かす説明に及ばないよう、文の境界で区切る
            for preceding in reversed(njd_features[:index]):
                if preceding["string"] in {"。", "！", "？", "!", "?", "「", "」"}:
                    break
                if preceding["orig"] in {"恥ずかしい", "恥じる", "恥じらう", "照れる", "号令"}:
                    is_face_movement = True
                    break
            if is_face_movement:
                # 「〜のあまり」に「面」が接尾辞として誤結合した場合は、顔を表す名詞を別のアクセント句にする
                if feature["pos_group1"] == "接尾":
                    feature["chain_flag"] = 0
                _set_reading(feature, "オモテ")
                # 「オモテ」は3モーラの尾高型なので、「メン」「ツラ」のアクセント核を引き継がずに設定する
                feature["acc"] = 3
        # 「声明」は政治的な発表にも使われるため、「声明を唱える」と受身の「声明が唱えられる」だけを補正する
        ## 宗教団体の政治的な声明も「セイメイ」と読むので、文中の宗教語の有無では判定せず、後続する動詞の原形が「唱える」かどうかで判定する
        ## 「声明が唱える理念」は声明文自体が主語になるので、「が」の後は「られる」が続く受身形に限定する
        elif (
            surface == "声明"
            and feature["pos"] == "名詞"
            and following is not None
            and following["string"] in {"を", "が"}
            and following["pos_group1"] == "格助詞"
            and index + 2 < len(njd_features)
            and njd_features[index + 2]["pos"] == "動詞"
            and njd_features[index + 2]["orig"] == "唱える"
            and (
                following["string"] == "を"
                or (
                    index + 3 < len(njd_features)
                    and njd_features[index + 3]["orig"] == "られる"
                    and njd_features[index + 3]["pos_group1"] == "接尾"
                )
            )
            and not feature.get("is_reading_protected", False)
        ):
            _set_reading(feature, "ショウミョウ", "ショーミョー")
            # 平板の「セイメイ」から頭高型の「ショーミョー」へ変わるため、アクセント核を1に設定する
            feature["acc"] = 1
        # 「麻布」は地名にも使われるため、生成り・織りの修飾や、染色・製織・漆加工の対象となる局所的な関係だけを補正する
        ## 文中に織物や染色の語があるだけでは、教室の所在地なども変わるので、助詞を挟んだ直前・直後の関係に限定する
        elif (
            surface == "麻布"
            and feature["pos"] == "名詞"
            and not feature.get("is_reading_protected", False)
        ):
            is_cloth_modifier = (
                previous is not None
                and previous["string"] == "の"
                and previous["pos_group1"] == "連体化"
                and previous_previous is not None
                and previous_previous["string"] in {"生成り", "織り"}
            )
            particle_index = index + 1
            if following is not None and following["string"] == "など":
                particle_index += 1
            is_cloth_processing = False
            if (
                particle_index + 1 < len(njd_features)
                and njd_features[particle_index]["pos"] == "助詞"
                and njd_features[particle_index]["string"] in {"を", "は"}
            ):
                predicate = njd_features[particle_index + 1]
                is_cloth_processing = (
                    (predicate["pos"] == "動詞" and predicate["orig"] in {"染める", "織る"})
                    or (
                        predicate["string"] == "染色"
                        and predicate["pos_group1"] == "サ変接続"
                        and particle_index + 2 < len(njd_features)
                        and njd_features[particle_index + 2]["orig"] == "する"
                        and njd_features[particle_index + 2]["pos"] == "動詞"
                    )
                    or (
                        predicate["string"] == "漆"
                        and predicate["pos"] == "名詞"
                        and particle_index + 2 < len(njd_features)
                        and njd_features[particle_index + 2]["string"] == "で"
                        and njd_features[particle_index + 2]["pos_group1"] == "格助詞"
                    )
                )
            if is_cloth_modifier or is_cloth_processing:
                # 「アサヌノ」は平板なので、元の語の中にある核は外し、「など」に由来する後続の核はモーラ数の変更に合わせて保つ
                old_mora_size = feature["mora_size"]
                _set_reading(feature, "アサヌノ")
                if feature["acc"] <= old_mora_size:
                    feature["acc"] = 0
        # 「二十前」は年齢を表す用法が多いため、数詞の「二」「十」に接尾辞「前」が続く形を「ハタチ」にする
        ## 「二十前半」「二十前後」「二十時前」と「百二十」などの末尾の「二十」は、形態素の区切りで区別して数詞の読みを保つ
        elif (
            surface == "二"
            and feature["pos_group1"] == "数"
            and following is not None
            and following["string"] == "十"
            and following["pos_group1"] == "数"
            and following["chain_flag"] == 1
            and index + 2 < len(njd_features)
            and njd_features[index + 2]["string"] == "前"
            and njd_features[index + 2]["pos_group1"] == "接尾"
            and njd_features[index + 2]["read"] == "マエ"
            and njd_features[index + 2]["chain_flag"] == 1
            and (previous is None or previous["pos_group1"] != "数")
            and not feature.get("is_reading_protected", False)
            and not following.get("is_reading_protected", False)
            and (
                index + 3 == len(njd_features)
                or not njd_features[index + 3]["string"].startswith(("半", "後"))
            )
        ):
            # 片方だけを変えると数詞の読みが混ざるため、両方の読みが保護されていない場合にまとめて変更する
            ## 合計は3モーラのままなので、「前」のアクセント核とアクセント句の区切りを保つ
            feature["read"] = feature["pron"] = "ハタ"
            feature["mora_size"] = 2
            following["read"] = following["pron"] = "チ"
            following["mora_size"] = 1
        # 「八つ、九つ、十」のように和語の数詞で数え上げた後の「十」は、「ジュウ」でなく「トオ」と読む
        ## 「十」に助数詞や数が続く場合は、漢語の数詞として NJD の数詞処理の読みを保つ
        elif (
            surface == "十"
            and feature["pos_group1"] == "数"
            and previous is not None
            and (
                previous["string"] == "九つ"
                or (
                    previous["string"] == "、"
                    and previous_previous is not None
                    and previous_previous["string"] == "九つ"
                )
            )
            and (
                following is None
                or (following["pos_group1"] != "数" and following["pos_group2"] != "助数詞")
            )
        ):
            _set_reading(feature, "トオ", "トー")
        # 「返戻金型」は「返戻金」に「型」が付く表現なので、鋳型の「金型」と分けて「キン」を保つ
        elif (
            surface == "金"
            and following is not None
            and following["string"] == "型"
            and (previous is None or previous["string"] != "返戻")
        ):
            _set_reading(feature, "カナ")
        # 証書の「金五万円」「金拾万円」のように金額の前に付く「金」は、金銭の「カネ」でなく「キン」と読む
        ## 「金」の「キン」と「カネ」は同じ名詞でコストが近く、直後の数詞だけでは辞書の連接で決まらない
        ## 大字の「壱」「参」「伍」は辞書で数でない名詞として解析されるため、表層で数詞と見なす
        elif (
            surface == "金"
            and following is not None
            and following["pos"] == "名詞"
            and (following["pos_group1"] == "数" or following["string"] in {"壱", "参", "伍"})
        ):
            _set_reading(feature, "キン")
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
        # 「就学前の児童」は句として「マエ」、「就学前教育」は複合語として「ゼン」と読む
        elif (
            surface == "前"
            and previous is not None
            and previous["pos"] == "名詞"
            and previous["pos_group1"] == "サ変接続"
            and following is not None
            and following["pos"] == "助詞"
        ):
            _set_reading(feature, "マエ")
        elif surface == "前" and previous is not None and previous["string"] in _ZEN_PREDECESSORS:
            _set_reading(feature, "ゼン")
        # 「前会長」「前首相」のように直後に役職が続く「前」は、前任を表す「ゼン」と読む
        elif surface == "前" and following is not None and following["string"] in _ZEN_SUCCESSORS:
            _set_reading(feature, "ゼン")
        elif surface == "様" and previous is not None and previous["string"] == "同じ":
            _set_reading(feature, "ヨウ", "ヨー")
        # 「届きます様に」「言う様な」のように述語に続く助動詞「ようだ」の用法では、「様」を「ヨウ」と読む
        ## 「ヨウ」のコストだけを下げると「帰る様を描く」の「サマ」まで変わるため、「に」「な」が続く形に絞る
        ## 動作の様子を指す「描く様には驚く」も同じ形になるが、助動詞の用法を優先する
        elif (
            surface == "様"
            and previous is not None
            and previous["pos"] in {"動詞", "形容詞", "助動詞"}
            and previous["cform"] in {"基本形", "連体形"}
            and following is not None
            and (
                (following["string"] == "に" and following["pos"] == "助詞")
                or (following["string"] == "な" and following["pos"] == "助動詞")
            )
        ):
            _set_reading(feature, "ヨウ", "ヨー")
        # 「代」のコストだけを下げると代金などの「ダイ」も変わるため、神々の時代を表す「神の代」に絞る
        ## 短い連語の登録は「代理」「代行」「代弁」にも届くため、独立した名詞の「代」と前後の品詞を確かめる
        elif (
            surface == "代"
            and feature["pos"] == "名詞"
            and feature["pos_group1"] == "一般"
            and previous is not None
            and previous["string"] == "の"
            and previous["pos"] == "助詞"
            and previous["pos_group1"] == "連体化"
            and previous_previous is not None
            and previous_previous["string"] == "神"
            and previous_previous["pos"] == "名詞"
            and previous_previous["pos_group1"] == "一般"
            and (following is None or following["pos"] in {"助詞", "助動詞", "記号"})
        ):
            _set_reading(feature, "ヨ")
        elif (
            surface == "下"
            and feature["pos_group1"] == "一般"
            and previous is not None
            and previous["string"] == "の"
            and previous_previous is not None
            and previous_previous["string"] in _ABSTRACT_NO_PREDECESSORS
        ):
            _set_reading(feature, "モト")
        # 「下四桁」「上二桁」のように数字の桁の位置を指す「下」「上」は、「シタ」「ウエ」でなく「シモ」「カミ」と読む
        elif (
            surface in {"下", "上"}
            and following is not None
            and following["pos_group1"] == "数"
            and index + 2 < len(njd_features)
            and njd_features[index + 2]["string"] == "桁"
        ):
            _set_reading(feature, "シモ" if surface == "下" else "カミ")

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
        # 「受者」の「受」も動詞として解析されるため、1文字の語幹に続く「者」は辞書の読みを保つ
        elif (
            surface == "者"
            and previous is not None
            and previous["pos_group1"] == "自立"
            and len(previous["string"]) > 1
        ):
            _set_reading(feature, "モノ")
        elif (
            surface == "入っ"
            and previous is not None
            and previous["pos"] == "動詞"
            and previous["cform"] == "連用形"
            and previous["string"] in _IRU_COMPOUND_PREDECESSORS
        ):
            _set_reading(feature, "イッ")

        # 「寺小屋」は「テラコヤ」、「いつか公になる」は「オーヤケ」と読み、名詞の後の一律の補正から分ける
        elif surface == "小屋" and previous is not None and previous["string"] == "寺":
            _set_reading(feature, "コヤ")
        elif (
            surface == "公"
            and feature["pos_group1"] == "一般"
            and previous is not None
            and previous["pos_group1"] == "副詞可能"
        ):
            _set_reading(feature, "オオヤケ", "オーヤケ")

        # 「大津波」と「御言葉」は同じ文脈 ID の候補をコストだけでは使い分けられないため、後続語で読みを選ぶ
        elif (
            surface == "大"
            and feature["pos_group1"] == "名詞接続"
            and following is not None
            and following["string"] in _OO_SUCCESSORS
        ):
            _set_reading(feature, "オオ", "オー")
        elif (
            surface == "御"
            and feature["pos_group1"] == "名詞接続"
            and following is not None
            and following["string"] in _O_SUCCESSORS
        ):
            _set_reading(feature, "オ")

        # 「柄」は同じ名詞の候補に「ガラ」「エ」「ツカ」があるので、「の」の前の道具名で分ける
        elif (
            surface == "柄"
            and previous is not None
            and previous["string"] == "の"
            and previous_previous is not None
        ):
            if previous_previous["string"] in _TOOL_HANDLE_PREDECESSORS:
                _set_reading(feature, "エ")
            elif previous_previous["string"] in _SWORD_HILT_PREDECESSORS:
                _set_reading(feature, "ツカ")
        elif (
            surface == "縁"
            and previous is not None
            and previous["string"] == "の"
            and previous_previous is not None
            and previous_previous["string"] in _FUCHI_PREDECESSORS
        ):
            _set_reading(feature, "フチ")

        # 「尼」の前が名詞という条件だけでは「毎日尼を見る」まで「ニ」になるため、実証済みの仏教語に限る
        elif surface == "尼" and previous is not None and previous["string"] in {"修道", "比丘"}:
            _set_reading(feature, "ニ")
        elif (
            surface == "茶屋"
            and previous is not None
            and previous["pos"] == "名詞"
            and previous["pos_group1"] in {"一般", "固有名詞", "サ変接続"}
        ):
            _set_reading(feature, "ジャヤ")
        # 「芭蕉翁」は「オー」と読むが、「明日翁が来る」のような独立用法は「オキナ」を保つ
        elif (
            surface == "翁"
            and previous is not None
            and previous["pos"] == "名詞"
            and previous["pos_group1"] in {"一般", "固有名詞", "サ変接続"}
        ):
            _set_reading(feature, "オウ", "オー")

        # コストを下げても「処」が「ショ」に戻る文があるため、活用語に続く場所の「トコロ」は文脈で確定する
        elif (
            surface == "処"
            and previous is not None
            and (previous["pos_group1"] == "自立" or previous["pos"] == "助動詞")
        ):
            _set_reading(feature, "トコロ")

        # 「識って」「仰しゃる」「て了った」は動詞の異表記として読む
        elif (
            surface == "於"
            and following is not None
            and following["string"] in {"て", "ては", "ても"}
        ):
            _set_reading(feature, "オイ")
        elif (
            surface == "識"
            and following is not None
            and following["string"] in {"って", "った", "り"}
        ):
            _set_reading(feature, "シ")
        # MeCab は「仰しゃる」を「仰」「しゃ」「る」に分けるため、直後の「しゃ」も対象に含める
        elif (
            surface == "仰"
            and following is not None
            and following["string"] in {"しゃ", "しゃっ", "しゃる", "しゃい", "しゃら", "しゃり"}
        ):
            _set_reading(feature, "オッ")
        elif surface == "了" and previous is not None and previous["string"] in {"て", "で"}:
            _set_reading(feature, "シマ")
        # 形容詞の終止形に「た」は続かないので、MeCab が形容詞の「強い」に「た」を付けた「強いた」は、動詞「強いる」の過去形として「シイ」と読む
        ## 動詞「強いる」の「シイ」の行のコストを下げると、「強い風」「強い国」の形容詞まで「シイ」になるので、後続の「た」で決める
        elif (
            surface == "強い"
            and feature["pos"] == "形容詞"
            and following is not None
            and following["pos"] == "助動詞"
            and following["string"] == "た"
        ):
            _set_reading(feature, "シイ")
        # 候文の「申し上げ候」のように動詞の連用形に続く「候」は、名詞の「コウ」でなく補助動詞の「ソウロウ」と読む
        elif (
            surface == "候"
            and previous is not None
            and previous["pos"] == "動詞"
            and previous["cform"] == "連用形"
        ):
            _set_reading(feature, "ソウロウ", "ソーロー")

        # 「より外にない」は選択肢の「ホカ」、物体の位置を比べる場合は「ソト」と読む
        elif (
            surface == "外"
            and previous is not None
            and previous["string"] == "より"
            and previous_previous is not None
            and (previous_previous["pos"] == "動詞" or previous_previous["pos_group1"] == "代名詞")
        ):
            for later_feature in njd_features[index + 1 :]:
                if later_feature["string"] in {"。", "．", "！", "？"}:
                    break
                if later_feature["orig"] in _NEGATIVE_ORIGINALS:
                    _set_reading(feature, "ホカ")
                    break

        # 学位の「博士」は「ハクシ」と読み、人を指す「広瀬博士」の「ハカセ」を保つ
        elif surface == "博士" and (
            (following is not None and following["string"] in {"学位", "論文", "号", "課程"})
            or (
                previous is not None
                and previous["string"]
                in {"医学", "工学", "理学", "農学", "薬学", "文学", "法学", "経済学", "大学院"}
            )
        ):
            _set_reading(feature, "ハクシ")

        # 名詞へ直接続く後部要素は、助詞を挟んだ独立用法と区別して複合語の読みへ変える
        ## 数詞に続く「部屋」は助数詞なので、NJD の数詞処理が選んだ「ヘヤ」を保つ
        ## 日付に続く「付」は辞書の「ヅケ」を保つ
        ## 「十日」は地名、「一日」「四日」は副詞可能名詞になるため、数字と「日」からなる表層形も日付として扱う
        elif (
            surface in _COMPOUND_SUFFIX_READINGS
            and previous is not None
            and previous["pos"] == "名詞"
            and (surface != "部屋" or previous["pos_group1"] != "数")
            and not (
                surface == "付"
                and previous["string"].endswith("日")
                and (
                    previous["pos_group2"] == "助数詞"
                    or (
                        len(previous["string"]) > 1
                        and all(
                            char in "一二三四五六七八九十〇零0123456789０１２３４５６７８９"
                            for char in previous["string"][:-1]
                        )
                    )
                )
            )
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


def _is_disability_enumeration(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「盲」「聾」が、中黒または読点を挟んで学校種別や障害種別と並んでいるかを判定する。
    「知的」「障害」のように複数の形態素に分かれた語も照合し、句点や無関係な語で隔てられた文脈は対象外とする。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        index (int): 「盲」または「聾」の位置

    Returns:
        bool: 福祉・教育の語との列挙なら True
    """

    for direction in (-1, 1):
        separator = index + direction
        if (
            not 0 <= separator < len(njd_features)
            or njd_features[separator]["pos"] != "記号"
            or njd_features[separator]["string"] not in {"・", "、"}
        ):
            continue
        cursor = separator + direction
        term = ""
        while 0 <= cursor < len(njd_features) and njd_features[cursor]["pos"] == "名詞":
            surface = njd_features[cursor]["string"]
            term = surface + term if direction < 0 else term + surface
            if term in _DISABILITY_ENUMERATION_TERMS:
                return True
            if not any(
                candidate.endswith(term) if direction < 0 else candidate.startswith(term)
                for candidate in _DISABILITY_ENUMERATION_TERMS
            ):
                break
            cursor += direction
    return False


def _is_negative_nannimo_context(njd_features: list[NJDFeature], start: int) -> bool:
    """
    「何に」「も」の後の最初の述語が、確認できた打ち消しの用法かを判定する。
    「何にも答えてもらっていない」の補助動詞と「何にもすることができず」の可能表現をたどる。
    それ以外の名詞や別の自立動詞へ進んだ場合は対象外とする。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        start (int): 判定を始める形態素の位置

    Returns:
        bool: 「ナンニモ」と読む打ち消しの用法なら True
    """

    predicate_found = False
    sahen_found = False
    for index in range(start, len(njd_features)):
        feature = njd_features[index]
        # 読点・引用符や節をつなぐ助詞で区切り、「何にも似るが知らない」の後半の否定を切り離す
        if feature["pos"] == "記号":
            return False
        if not predicate_found:
            # 「何にもしない」の「しない」が名詞と解析された場合も、この打ち消しの表現として扱う
            if index == start and feature["string"] == "しない" and feature["pos"] == "名詞":
                return True
            if feature["pos_group1"] == "接続助詞":
                return False
            if feature["pos"] not in {"動詞", "形容詞", "助動詞"}:
                sahen_found |= feature["pos_group1"] == "サ変接続"
                continue
            # 「何にも依存しない」の「し」は「依存する」の一部なので、格助詞の「に」を保つ
            if feature["orig"] not in _NANNIMO_PREDICATES or (
                feature["orig"] == "する" and sahen_found
            ):
                return False
            predicate_found = True
        # 「何にもすることができず」の「ことが」に続く可能の述語と、その否定まで確かめる
        elif (
            feature["string"] in {"こと", "事"}
            and feature["pos_group1"] == "非自立"
            and index + 2 < len(njd_features)
            and njd_features[index + 1]["string"] == "が"
            and njd_features[index + 2]["string"] in {"でき", "出来"}
        ):
            return _is_negative_nannimo_context(njd_features, index + 2)
        elif not (
            feature["pos"] == "助動詞"
            or (feature["pos"] == "動詞" and feature["pos_group1"] == "非自立")
            or (feature["pos"] == "助詞" and feature["string"] in {"て", "で", "は", "も"})
        ):
            return False
        # 「ない」「無い」「知らん」「知りません」「知らず」は、原形と品詞で確認する
        if feature["pos"] in {"助動詞", "形容詞"} and feature["orig"] in _NEGATIVE_ORIGINALS:
            return True
    return False


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

        # Sudachi の語のうち、この漢字を含むものより長い範囲は試さない (長い文で前後の全範囲を試すと、形態素数の3乗に比例して遅くなるため)
        unknown_surface = njd_features[feature_index]["string"]
        maximum_surface_length = max(
            (len(surface) for surface in sudachi_readings_by_surface if unknown_surface in surface),
            default=0,
        )
        if maximum_surface_length == 0:
            return None
        for candidate_start in range(feature_index, -1, -1):
            preceding_surface = "".join(
                candidate["string"] for candidate in njd_features[candidate_start:feature_index]
            )
            if len(preceding_surface) + len(unknown_surface) > maximum_surface_length:
                break
            preceding_read = "".join(
                candidate["read"] for candidate in njd_features[candidate_start:feature_index]
            )
            candidate_surface = preceding_surface
            for candidate_end in range(feature_index + 1, len(njd_features) + 1):
                candidate_surface += njd_features[candidate_end - 1]["string"]
                if len(candidate_surface) > maximum_surface_length:
                    break
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
