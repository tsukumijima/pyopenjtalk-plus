import re

from .types import NJDFeature


"""
    もともとagpl3でだしちゃったので、コメントのみ持ってきて再実装
    NHK 日本語アクセント辞典を参考に、日本語方言や話者特有の訛り・アクセントの差分を適用する。
    区分は付録 NHK 日本語アクセント辞典 125p を参照した。
    持っていない人のためにも、細かくコメントを残しておく。

    Args:
        kata_list (list[str]): 単語単位の単語のカタカナ読みのリスト
        accent_list (list[str]): 単語単位の単語のアクセントのリスト
        pos_list (list[str]): 単語単位の単語の品詞 (Part-Of-Speech) のリスト
        dialect_rule (DialectRule): 適用対象の方言ルール。
            例えば DialectRule.Kansai 指定時はアクセントが京阪式になる。
        speaking_style_rules (list[SpeakingStyleRule]): 適用対象の喋り方ルールのリスト。
            例えば SpeakingStyleRule.ConvertBToV はバ行をヴァ行に変換し、外国語風の訛りを作る。

    Returns:
        tuple[list[str], list[str]]: 修正された kata_list と accent_list
"""

"""
    日本語方言の区分は以下の通り。
    - 本土方言
        - 八丈方言
        - 東部方言
        - 西部方言
            - 近畿方言 => DialectRule.Kansai
        - 九州方言 => DialectRule.Kyushu

    以下、厳密でない方言もしくは喋り方 (SpeakingStyleRule) の実装。
    - ConvertBToV: モーラ "b" を "v" に変換する
    - ConvertTToTs: モーラ "t" を "ts" に変換する
    - ConvertDToR: モーラ "d" を "r" に変換し、アクセントを平型にする
    - ConvertRToD: モーラ "r" を "d" に変換し、アクセントを頭高型にする
    - ConvertSToZ: モーラ "s" を "z" に、モーラ "sh" を "j" に変換し、アクセントを頭高型にする
    - ConvertToHatsuonbin: 単語の先頭以外の "na", "no", "ra", "ru" を "N" に変換する (撥音便化)

    - ExtendFirstMora: 文章の1モーラ目を長音化し、アクセントを頭高型にする / "やはり、" => "やーはり" (HLLL)
    - GeminationFirstMora: 文章の1モーラ目を促音化し、アクセントを頭高型にする / "やはり、" => "やっはり" (HLLL)
    - RemoveFirstMora: 文章の1モーラ目を "っ" に変換し、アクセントを平型にする / "やはり、" => "っはり" (LHH)
    - DiphthongFirstMora: 各単語の最初を連母音にし、アクセントを頭高型にする ("e" は "ei", "o" は "ou" になる) /
      "俺のターン。" => "おぅれのターン" / "先生。" => "せぃんせい"

    - LastMoraAccentH: 最後の単語の終端のモーラをアクセント核にする
    - LastWordAccent1: 最後の単語のアクセントを頭高型にする

    - AddYouonA: 各単語に最初にア段が出てきた時、"ァ" をつけ "ァ" をアクセント核にする /
      "そうさ、ボクの仕業さ。悪く思うなよ" => "そうさぁ。ボクの仕業さぁ。わぁるく思うなぁよ"
      ("ァ" は "ア" に置き換えられるので "ー" でも "ア" でもよいが、わかりやすくするため "ァ" とした)
    - AddYouonI: 各単語に最初にイ段が出てきた時、"ィ" をつけアクセントを頭高型にする /
      "しまった。にげられた。" => "しぃまった。にぃげられた"
    - AddYouonE: 各単語に最初にエ段が出てきた時、"ェ" をつけ "ェ" をアクセント核にする /
      "へえ、それで" => "へェえ、それェでェ"
    - AddYouonO: 各単語に最初にオ段が出てきた時、"ぉ" をつけアクセントを頭高型にする /
      "ようこそ。" => "よぉうこぉそ。"

    - BabyTalkStyle: "s" を "ch" に変換する (幼児語風)
      (幼児語のネイティブ話者つまり幼児の喋る幼児語でなく、我々大人の喋る (イメージする) 幼児語である)
"""
# 事前に正規表現パターンをコンパイル
__KYUSHU_HATSUON_PATTERN = re.compile("[ヌニムモミ]+")
__YOUON_PATTERN = re.compile("[ァィゥェォャュョヮ]+")
__A_DAN_PATTERN = re.compile("[アカサタナハマヤラワガダバパ]|[ャヮ]+")
__I_DAN_PATTERN = re.compile("[イキシチニヒミリギジビピ]|ィ+")
__E_DAN_PATTERN = re.compile("[エケセテネヘメレゲデベペ]|ェ+")
__O_DAN_PATTERN = re.compile("[オコソトノホモヨロゴゾドボポ]|[ョォ]+")


def modify_kyusyu_hougen(njd: list[NJDFeature]) -> list[NJDFeature]:
    # 九州方言
    modified_njd = []

    for features in njd:
        # 九州のほぼ全域で "e" を "ye" と発音する: 付録 131p
        features["pron"] = features["pron"].replace("エ", "イェ")

        # 九州のほぼ全域で "s e" を "sh e", "z e" を "j e" と発音する: 付録 132p
        features["pron"] = features["pron"].replace("セ", "シェ")
        features["pron"] = features["pron"].replace("ゼ", "ジェ")

        # 発音化: 語末の "ヌ", "ニ", "ム", "モ, "ミ" などが発音 "ンN" で表される: 付録 132p
        num = len(features["pron"])
        if __KYUSHU_HATSUON_PATTERN.fullmatch(str(features["pron"])[num - 1]):
            features["pron"] = str(features["pron"])[: num - 1] + "ン"

        modified_njd.append(features)

    return modified_njd


def modify_kansai_hougen(njd: list[NJDFeature]) -> list[NJDFeature]:
    # 近畿方言 (関西弁)

    modified_njd = []
    for features in njd:
        # 1泊の名詞を長音化し2泊で発音する
        if features["pos"] == "名詞" and len(features["pron"]) == 1:
            if features["pron"] not in ["!", "?", "'"]:
                features["pron"] = features["pron"] + "ー"

        modified_njd.append(features)

    return modified_njd


def modify_kansai_accent(njd: list[NJDFeature]) -> list[NJDFeature]:
    """
    NHK 日本語アクセント辞典を参考に、京阪式アクセントの差分を適用する。
    東京式と京阪式の対応表は付録 146p を参照した。
    持っていない人のためにも、細かくコメントを残しておく。
    """

    modified_njd = []
    for features in njd:
        # 分類が名詞の場合
        if features["pos"] == "名詞":
            # 一音の場合(長音可で2泊化)されている
            if len(features["pron"]) == 2 and str(features["pron"])[1] == "ー":
                # 平型の場合頭高型に
                if features["acc"] == 0:
                    features["acc"] = 1

                # 頭高型の場合全て低く
                if features["acc"] == 1:
                    features["acc"] = 2

            # ニ音の場合
            elif len(features["pron"]) == 2:
                # 平型の場合全て高く
                if features["acc"] == 0:
                    features["acc"] = 1

                # 尾高型の場合頭高型に
                if features["acc"] == 2:
                    features["acc"] = 1

        # 分類が動詞の場合
        elif features["pos"] == "動詞":
            # ニ音の場合
            if len(features["pron"]) == 2:
                # 平型の場合全て高く
                if features["acc"] == 0:
                    # features["acc"] = 0
                    features["acc"] = 1

                # 頭高型の場合尾高型に？(忘れた)
                if features["acc"] == 1:
                    features["acc"] = 2

            # 三音の場合
            if len(features["pron"]) == 3:
                # 平型の場合全て高く
                if features["acc"] == 0:
                    # features["acc"] = 0
                    features["acc"] = 1

                # 中高型の場合尾高型に
                if features["acc"] == 2:
                    features["acc"] = 3

        # 分類が形容詞の場合
        elif features["pos"] == "形容詞":
            # ニ音の場合
            if len(features["pron"]) == 2:
                # 頭高型の場合尾高型に
                if features["acc"] == 1:
                    features["acc"] = 2
            # 三音の場合
            if len(features["pron"]) == 3:
                # 平型の場合頭高に
                if features["acc"] == 0:
                    features["acc"] = 1
                # 中高型の場合頭高に
                if features["acc"] == 2:
                    features["acc"] = 1

        else:
            features["acc"] = features["acc"] + 1
            # 一泊ずれの法則
            # https://www.akenotsuki.com/kyookotoba/accent/taihi.html

        modified_njd.append(features)

    return modified_njd


# ここから特に参考資料はないが表現の幅が広がったり、話者の特性を再現できそうなもの


def convert_tt2t_style(njd: list[NJDFeature]) -> list[NJDFeature]:
    # タ行をツァ行に変換する
    modified_njd = []

    for features in njd:
        if "タ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("タ", "ツァ")
        if "チ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("チ", "ツィ")
        if "テ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("テ", "ツェ")
        if "ト" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ト", "ツォ")

        modified_njd.append(features)

    return modified_njd


def convert_d2r_style(njd: list[NJDFeature]) -> list[NJDFeature]:
    # ダ行をラ行に変換し (ヂを除く) 、アクセントを平型にする
    modified_njd = []

    for features in njd:
        if "ダ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ダ", "ラ")
        if "デ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("デ", "レ")
        if "ド" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ド", "ロ")
        # アクセントを平型に変更
        features["acc"] = 0

        modified_njd.append(features)

    return modified_njd


def convert_s2z_style(njd: list[NJDFeature]) -> list[NJDFeature]:
    # サ行をザ行に、シャ行をジャ行に変換し、アクセントを頭高型にする
    modified_njd = []

    for features in njd:
        if "サ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("サ", "ザ")
        if "スィ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("スィ", "ズィ")
        if "ス" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ス", "ズ")
        if "セ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("セ", "ゼ")
        if "ソ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ソ", "ゾ")
        if "シャ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("シャ", "ジャ")
        if "シ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("シ", "ジ")
        if "シュ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("シュ", "ジュ")
        if "シェ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("シェ", "ジェ")
        if "ショ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ショ", "ジョ")
        # アクセントを頭高型に変更
        features["acc"] = 1

        modified_njd.append(features)

    return modified_njd


def convert_hatsuonbin_style(njd: list[NJDFeature]) -> list[NJDFeature]:
    # 単語の先頭以外の "na", "no", "ra", "ru" を "N" に変換する (撥音便化)
    modified_njd = []

    for features in njd:
        # 1文字以外の時
        if len(str(features["pron"])) != 1:
            # 各単語先頭は置き換えない
            if "ナ" in str(features["pron"][1:]):
                features["pron"] = features["pron"].replace("ナ", "ン")
            elif "ノ" in str(features["pron"][1:]):
                features["pron"] = features["pron"].replace("ノ", "ン")
            # 一種ずつしか撥音化しない
            elif "ル" in str(features["pron"][1:]):
                features["pron"] = features["pron"].replace("ル", "ン")
            elif "ラ" in str(features["pron"][1:]):
                features["pron"] = features["pron"].replace("ラ", "ン")

        modified_njd.append(features)

    return modified_njd


def convert_babytalk_style(njd: list[NJDFeature]) -> list[NJDFeature]:
    # "s" を "ch" に変換する (幼児語風)

    modified_njd = []

    for features in njd:
        if "サ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("サ", "チャ")
        if "シ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("シ", "チ")
        if "ス" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ス", "チュ")
        if "セ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("セ", "チェ")
        if "ソ" in str(features["pron"]):
            features["pron"] = features["pron"].replace("ソ", "チョ")

        modified_njd.append(features)

    return modified_njd
