"""
前後の形態素で読みが決まる語の読み補正 (modify_context_reading()) の実装。

辞書の生起コストだけでは使い分けられない語 (「駆け込み寺」の「デラ」、「先生方」の「ガタ」など) を、前後の語・品詞・同じ文の語から判定して読みを書き換える。
規則は1つの形態素ごとに順に試し、同じ表層形に読みの候補が複数ある規則は、最初に当たった規則だけを適用する。
"""

from .types import NJDFeature
from .utils import split_kana_mora


# 文脈による読み補正で、前後の語から読みを決めるときに使う語の集合
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
# 「方」を「ガタ」と読ませる敬称・複数の前接語
_HONORIFIC_PLURAL_PREDECESSORS = frozenset(
    {"皆様", "皆", "みんな", "あなた", "先生", "奥様", "お客様", "親御", "殿"}
)
# 「前」を「ゼン」と読ませる前接語
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
# 「前」を「ゼン」と読ませる後続の役職名と段階の語
_ZEN_SUCCESSORS = frozenset({"会長", "大統領", "段階", "理事長", "社長", "総裁", "首相"})
# 「橋」を「キョー」と読ませる構造種別の前接語
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
# 「寺」を「デラ」と読ませる前接語
_DERA_PREDECESSORS = frozenset(
    {"縁切", "駆け込み", "田舎", "猫", "だるま", "隠れ", "峯", "山", "花"}
)
# 「寺」を「ジ」と読ませる実証済みの前接語
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
# 名詞直後の後部要素へ与える複合語の読み (read, pron, 対象の品詞細分類)
## 品詞細分類の None は品詞を問わないことを表す
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

# 直前か直後の語の表層形だけで読みが決まる語の読み (表層形ごとに、品詞細分類の条件・直前の語・直後の語・読み・発音の組を並べる)
## 品詞細分類・直前の語・直後の語の None はその条件を問わず、発音の None は読みと同じ発音にすることを表す
## 同じ表層形の組は上から順に試し、最初に当たった組の読みを使う
_ADJACENT_WORD_READINGS: dict[
    str,
    tuple[tuple[str | None, frozenset[str] | None, frozenset[str] | None, str, str | None], ...],
] = {
    # 直後の語で意味が確定する少数の表現を、閉じた表層形の集合で判定する
    "一見": ((None, None, frozenset({"さん"}), "イチゲン", None),),
    "一声": ((None, None, frozenset({"かけ", "掛け", "かける", "掛ける"}), "ヒトコエ", None),),
    "一行": ((None, None, frozenset({"ごと", "毎"}), "イチギョウ", "イチギョー"),),
    "町": ((None, None, frozenset({"史"}), "チョウ", "チョー"),),
    "一": ((None, None, frozenset({"しょ"}), "イッ", None),),
    "兵": ((None, None, frozenset({"ども", "共"}), "ツワモノ", None),),
    "如何": ((None, None, frozenset({"で", "です", "でし", "でしょ"}), "イカガ", None),),
    # 直前の語が読みを確定する敬称・仏号・定型表現を表層形で判定する
    "仏": ((None, frozenset({"阿弥陀", "釈迦", "大日", "薬師", "毘盧遮那"}), None, "ブツ", None),),
    "方": (("接尾", _HONORIFIC_PLURAL_PREDECESSORS, None, "ガタ", None),),
    "前": (
        (None, _ZEN_PREDECESSORS, None, "ゼン", None),
        # 「前会長」「前首相」のように直後に役職が続く「前」は、前任を表す「ゼン」と読む
        (None, None, _ZEN_SUCCESSORS, "ゼン", None),
    ),
    "様": ((None, frozenset({"同じ"}), None, "ヨウ", "ヨー"),),
    # 「橋」は構造種別なら「キョウ」（「キョー」）と読ませる (名詞に続く接尾辞用法の「バシ」は _read_bashi_suffix() で読ませる)
    "橋": ((None, _BRIDGE_TYPE_PREDECESSORS, None, "キョウ", "キョー"),),
    "寺": (
        (None, _DERA_PREDECESSORS, None, "デラ", None),
        # 「寺」の読みは前接語によって「ジ」と「デラ」に分かれるため、実証済みの複合語だけを閉じた集合で「ジ」へ補正する
        (None, _JI_PREDECESSORS, None, "ジ", None),
    ),
    # 「寺小屋」は「テラコヤ」と読み、名詞の後の一律の補正から分ける
    "小屋": ((None, frozenset({"寺"}), None, "コヤ", None),),
    # 「大津波」と「御言葉」は同じ文脈 ID の候補をコストだけでは使い分けられないため、後続語で読みを選ぶ
    "大": (("名詞接続", None, _OO_SUCCESSORS, "オオ", "オー"),),
    "御": (("名詞接続", None, _O_SUCCESSORS, "オ", None),),
    # 「尼」の前が名詞という条件だけでは「毎日尼を見る」まで「ニ」になるため、実証済みの仏教語に限る
    "尼": ((None, frozenset({"修道", "比丘"}), None, "ニ", None),),
    # 「識って」「仰しゃる」「て了った」は動詞の異表記として読む
    "於": ((None, None, frozenset({"て", "ては", "ても"}), "オイ", None),),
    "識": ((None, None, frozenset({"って", "った", "り"}), "シ", None),),
    # MeCab は「仰しゃる」を「仰」「しゃ」「る」に分けるため、直後の「しゃ」も対象に含める
    "仰": (
        (
            None,
            None,
            frozenset({"しゃ", "しゃっ", "しゃる", "しゃい", "しゃら", "しゃり"}),
            "オッ",
            None,
        ),
    ),
    "了": ((None, frozenset({"て", "で"}), None, "シマ", None),),
    # 学位の「博士」は「ハクシ」と読み、人を指す「広瀬博士」の「ハカセ」を保つ
    "博士": (
        (None, None, frozenset({"学位", "論文", "号", "課程"}), "ハクシ", None),
        (
            None,
            frozenset({"医学", "工学", "理学", "農学", "薬学", "文学", "法学", "経済学", "大学院"}),
            None,
            "ハクシ",
            None,
        ),
    ),
    # 辞書が人名と解析する「記念章」だけを「ショウ」へ補正し、人名の「章」は「アキラ」と読むように残す
    "章": (("固有名詞", frozenset({"記念"}), None, "ショウ", "ショー"),),
}


def modify_context_reading(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    前後の形態素で読みが決まる語 (「駆け込み寺」の「デラ」、「先生方」の「ガタ」など) の読みを書き換える。
    「より外にない」の「ホカ」は、後続する打ち消しの語も確かめてから適用する。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features

    Returns:
        list[NJDFeature]: 前後の語から読みを補正した NJDNode 用 features
    """

    for index in range(len(njd_features)):
        # 前後の語との組み合わせで読みが決まる規則は、当たったものをすべて同じ形態素に重ねて適用する
        for independent_rule in (
            _read_hair_tying_motoyui,
            _read_verdict_kokubyaku,
            _read_awakening_kaigen,
            _read_sonohoka,
            _read_comparative_yoi,
            _read_figurative_asu,
            _read_petal_kaben,
            _read_dull_niburu,
            _read_water_surface_omote,
            _read_ametsuchi,
            _read_municipal_enumeration,
            _read_origin_moto,
            _read_water_master_nushi,
            _read_musashi_name,
            _read_old_loom_hata,
            _read_central_kaname,
            _read_melodic_fushi,
            _read_enduring_taeru,
            _read_shrine_yashiro,
            _read_impurity_kegare,
            _read_giving_up_ne,
            _read_cloth_beniiro,
            _read_disability_enumeration,
            _read_ichiban_suki,
        ):
            independent_rule(njd_features, index)

        # 同じ表層形に読みの候補が複数ある規則は、上から順に試して最初に当たった規則だけを適用する
        for exclusive_rule in (
            _read_nannimo,
            _read_face_omote,
            _read_capsized_kutsugae,
            _read_edge_kiwa,
            _read_chanted_shomyo,
            _read_hemp_cloth_asanuno,
            _read_age_before_twenty,
            _read_age_past_twenty,
            _read_counted_too,
            _read_kin,
            _read_mae_after_sahen_noun,
            _read_by_adjacent_word,
            _read_you_auxiliary,
            _read_kami_era_yo,
            _read_abstract_moto,
            _read_digit_position,
            _read_bashi_suffix,
            _read_mono,
            _read_compound_iru,
            _read_ooyake,
            _read_handle_e_or_tsuka,
            _read_rim_fuchi,
            _read_teahouse_jaya,
            _read_honorific_ou,
            _read_tokoro,
            _read_shiita,
            _read_sourou,
            _read_hoka,
            _read_compound_suffix,
            _read_ra_or_tou,
        ):
            if exclusive_rule(njd_features, index):
                break

    return njd_features


def _feature_at(njd_features: list[NJDFeature], position: int) -> NJDFeature | None:
    """
    指定した位置の形態素を返し、形態素列の範囲外なら None を返す。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        position (int): 取り出す形態素の位置 (負の値や末尾より後ろも受け付ける)

    Returns:
        NJDFeature | None: 指定した位置の形態素 (範囲外なら None)
    """

    return njd_features[position] if 0 <= position < len(njd_features) else None


def _set_reading(
    njd_features: list[NJDFeature],
    index: int,
    reading: str,
    pronunciation: str | None = None,
) -> None:
    """
    読みと発音を書き換え、モーラ数の変更に合わせてアクセント句の核の位置を補正する。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 書き換える形態素の位置
        reading (str): 新しい読み
        pronunciation (str | None): 新しい発音 (None なら読みと同じにする)
    """

    feature = njd_features[index]

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


def _set_accent(njd_features: list[NJDFeature], index: int, accent: int) -> None:
    """
    アクセント句の先頭にある語のアクセント核を設定し、同じ句に結合した後続の助詞・助動詞の結合規則を計算し直す。
    「など」「より」「です」のように平板型の語に続くと核を自分の上へ移す助詞があるため、語の核だけを書き換えると、句の核の位置が平板型かどうかの変化に追従しない。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): アクセント核を設定する語の位置
        accent (int): 語の中でのアクセント核の位置 (0 は平板型)
    """

    feature = njd_features[index]
    feature["acc"] = accent
    # NJD の結合規則と同じく、F2 は平板型のとき、F3 は起伏型のとき、F4 は常に、核を句のそれまでのモーラ数に加算位置を足した位置へ移し、F5 と C4 は平板型にする
    ## NJD は直前の語の品詞名に規則の品詞名が含まれるかで照合するので、「助動詞」の後では「動詞」の規則も当たる
    phrase_mora_size = feature["mora_size"]
    for cursor in range(index + 1, len(njd_features)):
        node = njd_features[cursor]
        if node["chain_flag"] != 1 or node["pos"] not in {"助詞", "助動詞"}:
            break
        for rule in node["chain_rule"].split("/"):
            part_of_speech, separator, suffix = rule.partition("%")
            if separator and part_of_speech not in njd_features[cursor - 1]["pos"]:
                continue
            rule_name, _, offset = (suffix if separator else rule).partition("@")
            if (
                (rule_name == "F2" and feature["acc"] == 0)
                or (rule_name == "F3" and feature["acc"] != 0)
                or rule_name == "F4"
            ):
                feature["acc"] = phrase_mora_size + (int(offset) if offset else 0)
            elif rule_name in {"F5", "C4"}:
                feature["acc"] = 0
            break
        phrase_mora_size += node["mora_size"]


def _sentence_words(njd_features: list[NJDFeature], index: int) -> set[str]:
    """
    指定した形態素を含む文 (句点・感嘆符・疑問符で区切った範囲) にある語の原形を集める。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        index (int): 文の中にある形態素の位置

    Returns:
        set[str]: 同じ文にある形態素の原形の集合
    """

    start = index
    while start > 0 and njd_features[start - 1]["string"] not in {"。", "！", "？"}:
        start -= 1
    end = index + 1
    while end < len(njd_features) and njd_features[end]["string"] not in {"。", "！", "？"}:
        end += 1
    return {node["orig"] for node in njd_features[start:end]}


def _read_hair_tying_motoyui(njd_features: list[NJDFeature], index: int) -> None:
    """
    髪を結う用法の「元結」だけを「モトユイ」にし、別の読みを明示した文章や複合語の一部では元の読みを保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    # 名詞のコストを一律に下げると「もっとい」の注記がある箇所も変わるため、独立した名詞で、同じ文に髪を結う手掛かりがある場合に絞る
    if (
        feature["string"] == "元結"
        and feature["pos"] == "名詞"
        and feature["pos_group1"] == "一般"
        and (previous is None or previous["pos"] not in {"名詞", "接頭詞"})
        and not feature.get("is_reading_protected", False)
        and _is_hair_tying_motoyui(njd_features, index)
    ):
        _set_reading(njd_features, index, "モトユイ")
        _set_accent(njd_features, index, 3)


def _is_hair_tying_motoyui(njd_features: list[NJDFeature], position: int) -> bool:
    """
    読みの明示がない段落で、対象の「元結」と同じ文に髪を結う手掛かりがあるかを判定する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        position (int): 形態素列での「元結」の位置

    Returns:
        bool: 髪を結う用法として「モトユイ」を優先する場合は True
    """

    # 読みの注記は後続する同じ語にも適用されるので、段落に「もっとい」の表記があれば元の読みを保つ
    paragraph = "".join(node["string"] for node in njd_features)
    if "もっとい" in paragraph or "モットイ" in paragraph:
        return False

    # 読点では同じ文が続くため、句点・疑問符・感嘆符で範囲を区切って髪を結う手掛かりを調べる
    boundaries = {"。", "？", "！", "?", "!", "\n", "\r", "\r\n"}
    start = position
    while start > 0 and njd_features[start - 1]["string"] not in boundaries:
        start -= 1
    end = position + 1
    while end < len(njd_features) and njd_features[end]["string"] not in boundaries:
        end += 1
    return any(
        node["orig"]
        in {
            "髷",
            "髻",
            "髪",
            "黒髪",
            "頭髪",
            "長髪",
            "結髪",
            "髪結い",
            "髪結",
            "結う",
            "侍",
            "力士",
        }
        for node in njd_features[start:end]
    )


def _read_verdict_kokubyaku(njd_features: list[NJDFeature], index: int) -> None:
    """
    色の「黒白」は「クロシロ」を保ち、是非を判定する述語が続く場合だけ「コクビャク」を選ぶ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 写真や弔事の水引の色を表す「黒白」もあるため、「コクビャク」の生起コストを一律に下げず、格助詞と直後の述語で限定する
    if (
        feature["string"] == "黒白"
        and following is not None
        and following["string"] in {"を", "が"}
        and index + 2 < len(njd_features)
        and (
            (
                following["string"] == "を"
                and njd_features[index + 2]["orig"] in {"つける", "付ける", "争う", "決める"}
            )
            or njd_features[index + 2]["string"] == "明らか"
        )
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "コクビャク")
        _set_accent(njd_features, index, 0)


def _read_awakening_kaigen(njd_features: list[NJDFeature], index: int) -> None:
    """
    目を開く医療の用法は「カイガン」を保ち、技芸の習得や悟りの対象を伴う用法を「カイゲン」にする。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 宗教語が文中にあるだけでは眼を開く動作と区別できないため、直前の習得対象か悟りを得る述語で判定する
    ## 技芸の用法にも「カイガン」の読みがあるため、直後に括弧による注記がある場合は元の読みを残す
    if (
        feature["string"] == "開眼"
        and (following is None or following["string"] not in {"（", "("})
        and (
            (
                previous is not None
                and previous["string"] == "に"
                and previous_previous is not None
                and previous_previous["string"]
                in {
                    "真髄",
                    "技芸",
                    "悟り",
                    "境地",
                    "禅",
                    "音楽",
                    "芸術",
                    "剣術",
                    "書道",
                    "茶道",
                    "ワイン",
                    "ゴルフ",
                }
            )
            or (
                previous is not None
                and previous["string"] == "て"
                and index >= 4
                and njd_features[index - 2]["orig"] == "得る"
                and njd_features[index - 3]["string"] == "を"
                and njd_features[index - 4]["string"] == "悟り"
            )
        )
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "カイゲン")
        _set_accent(njd_features, index, 0)


def _read_sonohoka(njd_features: list[NJDFeature], index: int) -> None:
    """
    追加の内容を示す「その他に」は「ソノホカ」と読み、分類の名称として扱う「その他」は「ソノタ」を保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    following = _feature_at(njd_features, index + 1)
    # 「その他に分類する」も同じ助詞になるため、直後の分類・区分の動詞を確認してから補正する
    if (
        njd_features[index]["string"] == "その他"
        and following is not None
        and following["string"] == "に"
        and (
            index + 2 == len(njd_features)
            or njd_features[index + 2]["orig"]
            not in {"分類", "区分", "分類する", "区分する", "含める", "入れる"}
        )
    ):
        _set_reading(njd_features, index, "ソノホカ")


def _read_comparative_yoi(njd_features: list[NJDFeature], index: int) -> None:
    """
    比較の副詞「より」に続く「良い」を「ヨイ」にし、独立した「良い」の辞書の読みを保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    surface = njd_features[index]["string"]
    previous = _feature_at(njd_features, index - 1)
    # 複合語が1つの形態素になった場合も、単独の「良い」のコスト調整では届かない同じ比較表現として扱う
    if surface == "より良い":
        _set_reading(njd_features, index, "ヨリヨイ")
    elif (
        surface == "良い"
        and previous is not None
        and previous["string"] == "より"
        and previous["pos"] == "副詞"
    ):
        _set_reading(njd_features, index, "ヨイ")


def _read_figurative_asu(njd_features: list[NJDFeature], index: int) -> None:
    """
    将来を指す「明日の社会・国」は「アス」を選び、日付を指す「明日の会議」「明日の天気」は「アシタ」を保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 名詞全体のコストでは日付用法も変わるため、連体助詞と社会・国家を表す直後の名詞で限定する
    if (
        feature["string"] == "明日"
        and following is not None
        and following["string"] == "の"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["string"] in {"日本", "社会", "未来", "国", "世界"}
        and (
            index + 3 == len(njd_features)
            or njd_features[index + 3]["pos"] in {"助詞", "助動詞", "記号"}
        )
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "アス")
        _set_accent(njd_features, index, 2)


def _read_petal_kaben(njd_features: list[NJDFeature], index: int) -> None:
    """
    植物の部位として独立した「花弁」を「カベン」と読み、料理名や読みの注記が続く形の「ハナビラ」を保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # コストを一律に下げると料理の「花弁大根」や「花弁（はなびら）」まで変わるため、助詞・助動詞に続く形に絞る
    if (
        feature["string"] == "花弁"
        and following is not None
        and following["pos"] in {"助詞", "助動詞"}
        and not feature.get("is_reading_protected", False)
    ):
        has_particle_accent = feature["acc"] > feature["mora_size"]
        _set_reading(njd_features, index, "カベン")
        # 語の内部の核だけを平板にし、既に後続助詞にある核はモーラ数の補正後の位置を保つ
        ## 平板化で「など」の F2 規則が新たに働くため、結合済みの助詞・助動詞を順に計算し直す
        if not has_particle_accent:
            _set_accent(njd_features, index, 0)


def _read_dull_niburu(njd_features: list[NJDFeature], index: int) -> None:
    """
    刃物の切れ味を述べる「鈍る」は「ニブル」と読み、身体がなまる用法の読みを保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    # 動詞全体のコストは習得した技能や身体の用法にも届くため、切れ味を主語か話題にした形で限定する
    if (
        feature["orig"] == "鈍る"
        and previous is not None
        and previous["string"] in {"が", "は", "も"}
        and previous_previous is not None
        and previous_previous["string"] == "切れ味"
        and feature["read"].startswith("ナマ")
    ):
        _set_reading(njd_features, index, "ニブ" + feature["read"][2:])


def _read_water_surface_omote(njd_features: list[NJDFeature], index: int) -> None:
    """
    水面を指す「水の面」を「ミズノオモテ」と読み、観点を表す「水の面で」や描写の動詞を伴わない「ミノモ」を保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    surface = feature["string"]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 全体を1語にした辞書の行も扱うが、用水などの語の途中を除き、同じ節の最初の動詞が水面の描写の場合だけ補正する
    if (
        (
            (
                surface == "水の面"
                and (previous is None or previous["pos"] not in {"名詞", "接頭詞"})
            )
            or (
                surface == "面"
                and previous is not None
                and previous["string"] == "の"
                and previous_previous is not None
                and previous_previous["string"] == "水"
                and feature["pos_group1"] == "一般"
            )
        )
        and following is not None
        and following["string"] in {"に", "を", "が", "は"}
        and not feature.get("is_reading_protected", False)
    ):
        for node in njd_features[index + 2 :]:
            if node["pos"] == "記号":
                break
            if node["pos"] == "動詞" and node["pos_group1"] == "自立":
                if node["orig"] in {
                    "映る",
                    "浮かぶ",
                    "揺れる",
                    "覆う",
                    "蔽う",
                    "見つめる",
                    "眺める",
                }:
                    _set_reading(
                        njd_features, index, "ミズノオモテ" if surface == "水の面" else "オモテ"
                    )
                    # 「水」は平板で「面」は尾高なので、連語なら6モーラ目、単独の面なら3モーラ目に核を置く
                    _set_accent(njd_features, index, 6 if surface == "水の面" else 3)
                break


def _read_ametsuchi(njd_features: list[NJDFeature], index: int) -> None:
    """
    古語の連語「天地の道」は「アメツチ」と読み、荷物の上下などを表す一般語の「天地」は「テンチ」を保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 短い連語を辞書へ足すと道具などの途中にも届くため、独立した道が続く形態素の境界で限定する
    if (
        feature["string"] == "天地"
        and following is not None
        and following["string"] == "の"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["string"] == "道"
        and njd_features[index + 2]["pos_group1"] == "一般"
        and (
            index + 3 == len(njd_features)
            or njd_features[index + 3]["pos"] in {"助詞", "助動詞", "記号"}
        )
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "アメツチ")
        _set_accent(njd_features, index, 1)


def _read_municipal_enumeration(njd_features: list[NJDFeature], index: int) -> None:
    """
    行政区分を市と並べる町・村は「チョウ」「ソン」と読み、単独の町・村や地名の接尾辞は辞書の読みを保つ。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    surface = feature["string"]
    # 読点や中黒で続く名詞だけを調べ、並立助詞でつないだ表現や文中の離れた市、都市名から単独の町・村を音読みへ変えることを防ぐ
    if (
        surface in {"町", "村"}
        and feature["pos_group1"] == "一般"
        and _is_municipal_enumeration(njd_features, index)
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(
            njd_features,
            index,
            "チョウ" if surface == "町" else "ソン",
            "チョー" if surface == "町" else "ソン",
        )
        _set_accent(njd_features, index, 1)


def _is_municipal_enumeration(njd_features: list[NJDFeature], position: int) -> bool:
    """
    独立した市・町・村が読点や中黒で続く場合に、市を含む行政区分の列挙かを判定する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        position (int): 町または村の形態素の位置

    Returns:
        bool: 市を含む行政区分の列挙であれば True
    """

    names = {njd_features[position]["string"]}
    # 「市や町」「市と村」などの普段の言い方は訓読みを保ち、読点や中黒で区分を並べた列挙だけをたどる
    for step in (-1, 1):
        cursor = position + step
        while 0 <= cursor + step < len(njd_features):
            separator = njd_features[cursor]
            node = njd_features[cursor + step]
            if (
                separator["string"] not in {"、", "・"}
                or node["string"] not in {"市", "町", "村"}
                or node["pos"] != "名詞"
                or node["pos_group1"] != "一般"
            ):
                break
            names.add(node["string"])
            cursor += 2 * step
    return "市" in names and len(names) > 1


def _read_origin_moto(njd_features: list[NJDFeature], index: int) -> None:
    """
    成り立ちを調べる「本を正す」の「本」を「モト」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 「モト」のコストを下げると書物の「本を読む」も変わるため、成り立ちを調べる「本を正す」に絞る
    if (
        feature["string"] == "本"
        and feature["pos"] == "名詞"
        and not feature.get("is_reading_protected", False)
        and following is not None
        and following["string"] == "を"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["orig"] == "正す"
    ):
        _set_reading(njd_features, index, "モト")
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 2)


def _read_water_master_nushi(njd_features: list[NJDFeature], index: int) -> None:
    """
    副詞の「主として」を一律に変えず、湖・沼・池に住む主を表す連体修飾だけで「ヌシ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    # 名詞の「主」の読みは保持し、湖名は1語になる場合もあるため末尾の「湖」を認める
    ## 「電池」なども含む末尾の「池」へは条件を広げない
    if (
        feature["string"] == "主として"
        and feature["pos"] == "副詞"
        and not feature.get("is_reading_protected", False)
        and previous is not None
        and previous["string"] == "の"
        and previous["pos_group1"] == "連体化"
        and previous_previous is not None
        and previous_previous["pos"] == "名詞"
        and (
            previous_previous["string"] in {"湖", "沼", "池"}
            or previous_previous["string"].endswith("湖")
        )
    ):
        _set_reading(njd_features, index, "ヌシトシテ")
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 1)


def _read_musashi_name(njd_features: list[NJDFeature], index: int) -> None:
    """
    名前の提示か人名の接尾辞がある場合だけ、数詞3語の「六三四」を1句の「ムサシ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    # 「六三四」を辞書で優先すると数字列も変わるため、名前の提示か人名の接尾辞がある場合だけ補正する
    ## 数詞3語を1句にまとめ、前後に数詞がある長い数字列と保護された読みはそのままにする
    if (
        njd_features[index]["string"] == "六"
        and index + 2 < len(njd_features)
        and [node["string"] for node in njd_features[index : index + 3]] == ["六", "三", "四"]
        and all(node["pos_group1"] == "数" for node in njd_features[index : index + 3])
        and not any(
            node.get("is_reading_protected", False) for node in njd_features[index : index + 3]
        )
        and (previous is None or previous["pos_group1"] != "数")
        and (index + 3 == len(njd_features) or njd_features[index + 3]["pos_group1"] != "数")
        and (
            (
                previous is not None
                and previous["string"] in {"は", "が"}
                and previous_previous is not None
                and previous_previous["string"] in {"名前", "氏名", "名"}
            )
            or (
                index + 3 < len(njd_features)
                and njd_features[index + 3]["string"] in {"くん", "君", "さん"}
                and njd_features[index + 3]["pos_group1"] == "接尾"
            )
            or (
                index + 5 < len(njd_features)
                and njd_features[index + 3]["string"] == "と"
                and njd_features[index + 4]["orig"] == "いう"
                and njd_features[index + 5]["string"] in {"名前", "氏名", "名"}
            )
        )
    ):
        for offset, reading in enumerate(("ム", "サ", "シ")):
            node = njd_features[index + offset]
            node["read"] = node["pron"] = reading
            node["mora_size"] = 1
            node["acc"] = 1 if offset == 0 else 0
            if offset > 0:
                node["chain_flag"] = 1


def _read_old_loom_hata(njd_features: list[NJDFeature], index: int) -> None:
    """
    「古い」が直接修飾する独立した「機」は、織機を表す「ハタ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    # 「ハタ」の生起コストを下げると「機を逃さず」の「キ」まで変わるため、修飾する形容詞で限定する
    ## 「機械」「機会」などの複合語と、「輸送機」のような接尾辞の「機」は対象に入らない
    if (
        feature["string"] == "機"
        and feature["pos"] == "名詞"
        and feature["pos_group1"] != "接尾"
        and previous is not None
        and previous["orig"] == "古い"
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "ハタ")
        # 「ハタ」は2モーラの尾高型なので、独立したアクセント句では核を2に設定する
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 2)


def _read_central_kaname(njd_features: list[NJDFeature], index: int) -> None:
    """
    「守備の要として」「組織の要となる」のように、名詞と「の」に続いて「と」で受ける「要」は、中心となる部分を表す「カナメ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 「説明の要はない」「刷新の要は、今や」のように「は」で受ける形は必要を表す「ヨウ」にもなるため、「と」が続く形に限る
    if (
        feature["string"] == "要"
        and feature["pos"] == "名詞"
        and previous is not None
        and previous["string"] == "の"
        and previous["pos"] == "助詞"
        and previous_previous is not None
        and previous_previous["pos"] == "名詞"
        and following is not None
        and following["string"] == "と"
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "カナメ")
        # 「カナメ」は平板型なので、独立したアクセント句では核を0に設定する
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 0)


def _read_melodic_fushi(njd_features: list[NJDFeature], index: int) -> None:
    """
    「歌の節」「唄の節」のように歌の旋律を指す「節」は、「フシ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    # 「テーマ曲の節を参照」のように文書の区切りを指す「セツ」もあるため、前の名詞を「歌」「唄」に限る
    if (
        feature["string"] == "節"
        and feature["pos"] == "名詞"
        and previous is not None
        and previous["string"] == "の"
        and previous_previous is not None
        and previous_previous["string"] in {"歌", "唄"}
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "フシ")
        # 「フシ」は2モーラの尾高型なので、独立したアクセント句では核を2に設定する
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 2)


def _read_enduring_taeru(njd_features: list[NJDFeature], index: int) -> None:
    """
    「痛みに堪える」「苦難にも堪えて」のように、苦痛を表す名詞を「に」で受ける「堪える」は、辛抱する意味の「タエル」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    # 「笑いを堪える」の「コラエル」は「を」で受けるため、この形に入らない
    ## 「寒さに体が慣れず、冷え込みが堪えた」の「コタエル」と区別するため、「に」と動詞の間には「も」と副詞だけを認める
    ## 寒さ・暑さは「身にこたえる」の意味でも「に」で受けるため、名詞には含めない
    if (
        feature["orig"] == "堪える"
        and feature["pos"] == "動詞"
        and feature["read"].startswith(("コタエ", "コラエ"))
        and not feature.get("is_reading_protected", False)
    ):
        cursor = index - 1
        while cursor >= 0 and (
            njd_features[cursor]["pos"] == "副詞"
            or (njd_features[cursor]["string"] == "も" and njd_features[cursor]["pos"] == "助詞")
        ):
            cursor -= 1
        if (
            cursor > 0
            and njd_features[cursor]["string"] == "に"
            and njd_features[cursor]["pos"] == "助詞"
            and njd_features[cursor - 1]["orig"] in {"痛み", "苦しみ", "苦痛", "苦難"}
        ):
            original_accent = feature["acc"]
            _set_reading(njd_features, index, "タエ" + feature["read"][3:])
            # 辞書の「コタエ」「コラエ」と「タエ」は、どれも語末から同じ位置に核がある (「3/3」と「2/2」)
            ## NJD は活用形と後続の助動詞 (「堪えた」の「タ＼エタ」、「堪えます」の「タエマ＼ス」) に合わせて核を決めるので、先頭の1モーラが減った分だけ核を前へずらす
            if feature["chain_flag"] != 1 and original_accent > 0:
                feature["acc"] = original_accent - 1


def _read_shrine_yashiro(njd_features: list[NJDFeature], index: int) -> None:
    """
    「この社は神を祀る」のように、神社を指して「この」「その」「あの」で受ける独立した「社」は、「ヤシロ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 会社を指す「この社は」と同じ形になるため、同じ文に祭祀の語があり、会社の活動を示す語がない場合に限る
    ## 「その社史」「その社号」のような複合語の一部と区別するため、直後が助詞の場合に限る
    ## 「歴史」は会社の沿革にも使うので、祭祀の語には含めない
    if (
        feature["string"] == "社"
        and feature["pos"] == "名詞"
        and previous is not None
        and previous["pos"] == "連体詞"
        and previous["string"] in {"この", "その", "あの"}
        and following is not None
        and following["pos"] == "助詞"
        and not feature.get("is_reading_protected", False)
    ):
        sentence_words = _sentence_words(njd_features, index)
        if sentence_words & {
            "神社",
            "社殿",
            "鳥居",
            "祭祀",
            "祀る",
            "神",
            "屋根神",
        } and not sentence_words & {
            "会社",
            "企業",
            "創業",
            "社員",
            "新聞",
            "出版",
            "記者",
            "営業",
            "売上",
        }:
            _set_reading(njd_features, index, "ヤシロ")


def _read_impurity_kegare(njd_features: list[NJDFeature], index: int) -> None:
    """
    「汚れを祓う」のように、祓う動作の対象になる名詞の「汚れ」は、「ヨゴレ」でなく「ケガレ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 「靴の汚れを清める」「物の汚れを除き清める」のように「清める」は物に付いた汚れにも使うため、「祓う」が「を」を挟んで直後に続く形に限る
    if (
        feature["string"] == "汚れ"
        and feature["pos"] == "名詞"
        and following is not None
        and following["string"] == "を"
        and following["pos"] == "助詞"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["orig"] == "祓う"
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "ケガレ")
        # 「ケガレ」は尾高型と平板型の両方で読まれるので、独立したアクセント句では平板型の核0を設定する
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 0)


def _read_giving_up_ne(njd_features: list[NJDFeature], index: int) -> None:
    """
    訓練や苦痛に耐えられず弱音を吐く「音を上げる」の「音」は、「オト」でなく「ネ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 音量を大きくする「音を上げる」と同じ形になるため、同じ文に苦痛を表す語があり、音響を表す語がない場合に限る
    if (
        feature["string"] == "音"
        and feature["pos"] == "名詞"
        and following is not None
        and following["string"] == "を"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["orig"] == "上げる"
        and not feature.get("is_reading_protected", False)
    ):
        sentence_words = _sentence_words(njd_features, index)
        if sentence_words & {
            "訓練",
            "厳しい",
            "苦しい",
            "苦痛",
            "疲れる",
            "疲れ",
            "限界",
            "寒さ",
            "暑さ",
            "辛い",
        } and not sentence_words & {
            "音量",
            "ボリューム",
            "スピーカー",
            "録画",
            "音楽",
            "雑音",
            "聞こえる",
            "鳴る",
            "イヤホン",
            "オーディオ",
            "テレビ",
            "ラジオ",
            "再生",
            "大音量",
            "サウンド",
            "BGM",
        }:
            _set_reading(njd_features, index, "ネ")
            # 「ネ」は平板型なので、独立したアクセント句では核を0に設定する
            if feature["chain_flag"] != 1:
                _set_accent(njd_features, index, 0)


def _read_cloth_beniiro(njd_features: list[NJDFeature], index: int) -> None:
    """
    「紅色の着物」「薄紅色のドレス」のように衣服や布の色を指す「紅色」は、「コウショク」でなく「ベニイロ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 「紅色細菌」のような専門語は「コウショク」と読むので、辞書のコストは変えず、「の」に続く衣服や布の名詞で限定する
    ## 「紅色の帯模様」は生き物の模様にも使うため、「帯」は対象に含めない
    if (
        feature["string"] == "紅色"
        and following is not None
        and following["string"] == "の"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["string"]
        in {"敷物", "着物", "布", "衣", "袴", "旗", "衣装", "ドレス"}
        and not feature.get("is_reading_protected", False)
    ):
        _set_reading(njd_features, index, "ベニイロ")
        # 「ベニイロ」は平板型なので、独立したアクセント句では核を0に設定する
        if feature["chain_flag"] != 1:
            _set_accent(njd_features, index, 0)
        # 接頭辞「薄」と結合した「ウスベニイロ」も平板型なので、「コウショク」の結合で句の先頭に付いた核を外す
        elif (
            previous is not None
            and previous["string"] == "薄"
            and previous["pos"] == "接頭詞"
            and previous["chain_flag"] != 1
        ):
            previous["acc"] = 0


def _read_disability_enumeration(njd_features: list[NJDFeature], index: int) -> None:
    """
    「盲・聾・養護学校」のように福祉・教育の語と列挙された「盲」「聾」を、音読みの「モウ」「ロウ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    surface = feature["string"]
    # 単漢字の音読みをコストで優先すると文学作品の訓読みも変わるため、福祉・教育の語との列挙だけを補正する
    if (
        surface in {"盲", "聾"}
        and feature["pos"] == "名詞"
        and not feature.get("is_reading_protected", False)
        and _is_disability_enumeration(njd_features, index)
    ):
        _set_reading(
            njd_features,
            index,
            "モウ" if surface == "盲" else "ロウ",
            "モー" if surface == "盲" else "ロー",
        )
        # 「モー」「ロー」は2モーラの頭高型なので、元の訓読みの核を引き継がずに設定する
        _set_accent(njd_features, index, 1)


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


def _read_ichiban_suki(njd_features: list[NJDFeature], index: int) -> None:
    """
    最上級を表す「一番」に続く形容動詞の「好き」を、接尾辞の「ズキ」でなく「スキ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    # 「好き」の行のコストを下げると「旅好き」も変わるため、最上級を表す「一番」に続く形容動詞だけを補正する
    if (
        feature["string"] == "好き"
        and feature["pos_group1"] == "接尾"
        and feature["pos_group2"] == "形容動詞語幹"
        and previous is not None
        and previous["string"] == "一番"
    ):
        _set_reading(njd_features, index, "スキ")


def _read_nannimo(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「何にも知らない」のように打ち消しの述語と組む「何にも」の「何」を、「ナン」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    # 「何にも」の副詞の行を優先すると「何にも依存しない」も変わるため、打ち消しの述語で読みを選ぶ
    if (
        feature["string"] == "何"
        and feature["pos"] == "名詞"
        and index + 2 < len(njd_features)
        and njd_features[index + 1]["string"] == "に"
        and njd_features[index + 1]["pos_group1"] == "格助詞"
        and njd_features[index + 2]["string"] == "も"
        and njd_features[index + 2]["pos_group1"] == "係助詞"
        and _is_negative_nannimo_context(njd_features, index + 3)
    ):
        _set_reading(njd_features, index, "ナン")
        return True
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


def _read_face_omote(njd_features: list[NJDFeature], index: int) -> bool:
    """
    恥じる動作や号令、古風な命令形「上げよ」で顔を表す「面を伏せる」「面を上げる」の「面」を、「オモテ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 「面を伏せる」はラケットにも使うため、恥じる動作や号令、古風な命令形「上げよ」で顔を表す場合だけを補正する
    ## 接尾辞の「面」と連体修飾を受ける「面」は表面も表すため、独立した名詞か、程度を表す「〜のあまり」の直後に限る
    if not (
        feature["string"] == "面"
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
        return False

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
        _set_reading(njd_features, index, "オモテ")
        # 「オモテ」は3モーラの尾高型なので、「メン」「ツラ」のアクセント核を引き継がずに設定する
        _set_accent(njd_features, index, 3)
    return True


def _read_capsized_kutsugae(njd_features: list[NJDFeature], index: int) -> bool:
    """
    船・舟・ボートが主語の「覆っ」を、覆い隠す「オオッ」でなく転覆を表す「クツガエッ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    # 「覆っ」のコストで「覆る」を優先すると雲や布が覆う文も変わるため、船・舟・ボートが主語の転覆だけを補正する
    ## 「覆う」「覆われる」「覆い尽くす」は覆い隠す意味でも使うので、活用形が重なる「覆っ」に限定する
    if not (
        feature["string"] == "覆っ"
        and feature["pos"] == "動詞"
        and feature["orig"] == "覆う"
        and previous is not None
        and previous["string"] == "が"
        and previous["pos_group1"] == "格助詞"
        and previous_previous is not None
        and previous_previous["string"] in {"船", "舟", "ボート"}
        and previous_previous["pos"] == "名詞"
        and not feature.get("is_reading_protected", False)
    ):
        return False

    is_covering = False
    # 「海面を船が覆った」のような語順も他動詞なので、同じ節の主語より前にある目的語を調べる
    ## 動詞や文の境界を越えた目的語は別の述語に属するため、そこで探索を区切る
    for preceding in reversed(njd_features[: index - 2]):
        if preceding["pos"] in {"動詞", "形容詞"} or preceding["string"] in {
            "。",
            "！",
            "？",
            "!",
            "?",
            "「",
            "」",
        }:
            break
        if preceding["string"] == "を" and preceding["pos_group1"] == "格助詞":
            is_covering = True
            break
    # 「船が覆っていた海面」は海面を覆う連体修飾にもなるため、助動詞や補助動詞の後に自立した名詞がある場合は既定の読みを保つ
    ## 「覆ったこと」のように出来事を名詞化する形は、転覆を表す用法にもなるので対象に残す
    ## 「覆って乗員が落ちた」の「て」は次の節をつなぐので、非自立の動詞が続く場合だけ同じ述語の一部としてたどる
    ending = index + 1
    while ending < len(njd_features):
        node = njd_features[ending]
        if not (
            node["pos"] == "助動詞"
            or (node["pos"] == "動詞" and node["pos_group1"] == "非自立")
            or (
                node["string"] == "て"
                and node["pos_group1"] == "接続助詞"
                and ending + 1 < len(njd_features)
                and njd_features[ending + 1]["pos"] == "動詞"
                and njd_features[ending + 1]["pos_group1"] == "非自立"
            )
        ):
            break
        ending += 1
    if (
        ending < len(njd_features)
        and njd_features[ending]["pos"] == "名詞"
        and njd_features[ending]["pos_group1"] != "非自立"
    ):
        is_covering = True
    if not is_covering:
        _set_reading(njd_features, index, "クツガエッ")
        # 自動詞「覆る」は「ガ」の後に下がるため、原形と活用型も「覆る」に合わせ、アクセント核を3に設定する
        feature["orig"] = "覆る"
        feature["ctype"] = "五段・ラ行"
        feature["acc"] = 3
    return True


def _read_edge_kiwa(njd_features: list[NJDFeature], index: int) -> bool:
    """
    崖・淵・水の端を表す「〜の際」の「際」を、時や場合を表す「サイ」でなく「キワ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    # 「キワ」の生起コストを下げると「その際」「出発の際」の読みも変わるため、崖・淵・水の端を表す「〜の際」だけを補正する
    ## 「洪水の際」「排水の際」は時や場合を表すので、「の」の前が独立した名詞「崖」「淵」「水」である場合に限定する
    if not (
        feature["string"] == "際"
        and feature["pos"] == "名詞"
        and previous is not None
        and previous["string"] == "の"
        and previous["pos_group1"] == "連体化"
        and previous_previous is not None
        and previous_previous["string"] in {"崖", "淵", "水"}
        and previous_previous["pos"] == "名詞"
        and not feature.get("is_reading_protected", False)
    ):
        return False

    # 非自立名詞「サイ」の結合で前のアクセント句に付いた核を外し、「淵の」など前の語にある核は保つ
    if feature["chain_flag"] == 1:
        head = index - 1
        while head > 0 and njd_features[head]["chain_flag"] == 1:
            head -= 1
        preceding_mora_size = sum(node["mora_size"] for node in njd_features[head:index])
        if njd_features[head]["acc"] > preceding_mora_size:
            njd_features[head]["acc"] = 0
    # 端を表す「キワ」は一般名詞なので、時や場合を表す「サイ」と区別して別のアクセント句で尾高型にする
    _set_reading(njd_features, index, "キワ")
    feature["pos_group1"] = "一般"
    feature["pos_group2"] = "*"
    feature["chain_rule"] = "C3"
    feature["chain_flag"] = 0
    _set_accent(njd_features, index, 2)
    return True


def _read_chanted_shomyo(njd_features: list[NJDFeature], index: int) -> bool:
    """
    法要で唱える「声明」を、発表の「セイメイ」でなく「ショウミョウ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 「声明」は政治的な発表にも使われるため、「声明を唱える」と受身の「声明が唱えられる」だけを補正する
    ## 宗教団体の政治的な声明も「セイメイ」と読むので、文中の宗教語の有無では判定せず、後続する動詞の原形が「唱える」かどうかで判定する
    ## 「声明が唱える理念」は声明文自体が主語になるので、「が」の後は「られる」が続く受身形に限定する
    if not (
        feature["string"] == "声明"
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
        return False

    _set_reading(njd_features, index, "ショウミョウ", "ショーミョー")
    # 平板の「セイメイ」から頭高型の「ショーミョー」へ変わるため、アクセント核を1に設定する
    _set_accent(njd_features, index, 1)
    return True


def _read_hemp_cloth_asanuno(njd_features: list[NJDFeature], index: int) -> bool:
    """
    材料の「麻布」を、地名の「アザブ」でなく「アサヌノ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 「麻布」は地名にも使われるため、生成り・織りの修飾や、染色・製織・漆加工の対象となる局所的な関係だけを補正する
    ## 文中に織物や染色の語があるだけでは、教室の所在地なども変わるので、助詞を挟んだ直前・直後の関係に限定する
    if not (
        feature["string"] == "麻布"
        and feature["pos"] == "名詞"
        and not feature.get("is_reading_protected", False)
    ):
        return False

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
        _set_reading(njd_features, index, "アサヌノ")
        if feature["acc"] <= old_mora_size:
            _set_accent(njd_features, index, 0)
    return True


def _read_age_before_twenty(njd_features: list[NJDFeature], index: int) -> bool:
    """
    年齢を表す「二十前」の「二十」を、数の「ニジュウ」でなく「ハタチ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 「二十前」は年齢を表す用法が多いため、数詞の「二」「十」に接尾辞「前」が続く形を「ハタチ」にする
    ## 「二十前半」「二十前後」「二十時前」と「百二十」などの末尾の「二十」は、形態素の区切りで区別して数詞の読みを保つ
    if not (
        feature["string"] == "二"
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
        return False

    # 片方だけを変えると数詞の読みが混ざるため、両方の読みが保護されていない場合にまとめて変更する
    ## 合計は3モーラのままなので、「前」のアクセント核とアクセント句の区切りを保つ
    feature["read"] = feature["pron"] = "ハタ"
    feature["mora_size"] = 2
    following["read"] = following["pron"] = "チ"
    following["mora_size"] = 1
    return True


def _read_age_past_twenty(njd_features: list[NJDFeature], index: int) -> bool:
    """
    単位を省いた年齢を表す「二十を過ぎる」の「二十」を、数の「ニジュウ」でなく「ハタチ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 「二十」の語を優先すると「二十を数える」も「ハタチ」になるため、単位を省いた「二十を過ぎる」に絞る
    ## 同じ節に回数・得点・時刻などがある場合は数量とし、前に数詞がある「百二十」も補正しない
    if not (
        feature["string"] == "二"
        and feature["pos_group1"] == "数"
        and following is not None
        and following["string"] == "十"
        and following["pos_group1"] == "数"
        and following["chain_flag"] == 1
        and index + 3 < len(njd_features)
        and njd_features[index + 2]["string"] == "を"
        and njd_features[index + 3]["orig"] == "過ぎる"
        and (previous is None or previous["pos_group1"] != "数")
        and not feature.get("is_reading_protected", False)
        and not following.get("is_reading_protected", False)
        and _is_age_past_twenty_context(njd_features, index)
    ):
        return False

    # 合計3モーラと形態素数を保ち、両方の読みをまとめて変更する
    feature["read"] = feature["pron"] = "ハタ"
    feature["mora_size"] = 2
    feature["acc"] = 1
    following["read"] = following["pron"] = "チ"
    following["mora_size"] = 1
    return True


def _is_age_past_twenty_context(njd_features: list[NJDFeature], start: int) -> bool:
    """
    単位を省いた「二十」の前方を同じ節の中で調べ、年齢として補正できるかを判定する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features
        start (int): 数詞「二」がある形態素の位置

    Returns:
        bool: 年齢を表す名詞が先に見つかるか、数量や時刻を表す名詞がない場合は True
    """

    # 「5：20」のコロンは節の境界にもなるため、前が数詞なら時刻の数の並びとして先に除外する
    if (
        start >= 2
        and njd_features[start - 1]["string"] in {":", "："}
        and njd_features[start - 2]["pos_group1"] == "数"
    ):
        return False

    for node in reversed(njd_features[:start]):
        if node["pos"] == "記号":
            break
        if node["pos"] != "名詞":
            continue
        if node["string"] in {"年齢", "齢", "歳", "才"}:
            return True
        if node["string"] in {
            "回数",
            "得点",
            "点数",
            "スコア",
            "数",
            "人数",
            "個数",
            "数量",
            "件数",
            "時刻",
            "時間",
            "分",
            "秒",
            "カウント",
            "打数",
            "失点",
            "年数",
            "番号",
            "ページ",
        }:
            return False
    return True


def _read_counted_too(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「八つ、九つ、十」のように和語の数詞で数え上げた後の「十」を、「ジュウ」でなく「トオ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 「十」に助数詞や数が続く場合は、漢語の数詞として NJD の数詞処理の読みを保つ
    if (
        feature["string"] == "十"
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
        _set_reading(njd_features, index, "トオ", "トー")
        return True
    return False


def _read_kin(njd_features: list[NJDFeature], index: int) -> bool:
    """
    鋳型の「金型」の「金」を「カナ」、証書の「金五万円」のように金額の前に付く「金」を「キン」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    if njd_features[index]["string"] != "金":
        return False
    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 「返戻金型」は「返戻金」に「型」が付く表現なので、鋳型の「金型」と分けて「キン」を保つ
    if (
        following is not None
        and following["string"] == "型"
        and (previous is None or previous["string"] != "返戻")
    ):
        _set_reading(njd_features, index, "カナ")
        return True
    # 証書の「金五万円」「金拾万円」のように金額の前に付く「金」は、金銭の「カネ」でなく「キン」と読む
    ## 「金」の「キン」と「カネ」は同じ名詞でコストが近く、直後の数詞だけでは辞書の連接で決まらない
    ## 大字の「壱」「参」「伍」は辞書で数でない名詞として解析されるため、表層で数詞と見なす
    if (
        following is not None
        and following["pos"] == "名詞"
        and (following["pos_group1"] == "数" or following["string"] in {"壱", "参", "伍"})
    ):
        _set_reading(njd_features, index, "キン")
        return True
    return False


def _read_mae_after_sahen_noun(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「就学前の児童」のようにサ変接続の名詞と助詞に挟まれた「前」を、句として「マエ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 「就学前の児童」は句として「マエ」、「就学前教育」は複合語として「ゼン」と読む
    ## 「就学」は「ゼン」と読ませる前接語にも含まれるので、この規則を前接語の表より先に試す
    if (
        njd_features[index]["string"] == "前"
        and previous is not None
        and previous["pos"] == "名詞"
        and previous["pos_group1"] == "サ変接続"
        and following is not None
        and following["pos"] == "助詞"
    ):
        _set_reading(njd_features, index, "マエ")
        return True
    return False


def _read_by_adjacent_word(njd_features: list[NJDFeature], index: int) -> bool:
    """
    直前か直後の語の表層形だけで読みが決まる語を、_ADJACENT_WORD_READINGS の表から引いて読みを書き換える。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    for (
        pos_group1,
        previous_words,
        following_words,
        reading,
        pronunciation,
    ) in _ADJACENT_WORD_READINGS.get(feature["string"], ()):
        if (
            (pos_group1 is None or feature["pos_group1"] == pos_group1)
            and (
                previous_words is None
                or (previous is not None and previous["string"] in previous_words)
            )
            and (
                following_words is None
                or (following is not None and following["string"] in following_words)
            )
        ):
            _set_reading(njd_features, index, reading, pronunciation)
            return True
    return False


def _read_you_auxiliary(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「届きます様に」「言う様な」のように述語に続く助動詞「ようだ」の用法の「様」を、敬称の「サマ」でなく「ヨウ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    following = _feature_at(njd_features, index + 1)
    # 「ヨウ」のコストだけを下げると「帰る様を描く」の「サマ」まで変わるため、「に」「な」が続く形に絞る
    ## 動作の様子を指す「描く様には驚く」も同じ形になるが、助動詞の用法を優先する
    if (
        njd_features[index]["string"] == "様"
        and previous is not None
        and previous["pos"] in {"動詞", "形容詞", "助動詞"}
        and previous["cform"] in {"基本形", "連体形"}
        and following is not None
        and (
            (following["string"] == "に" and following["pos"] == "助詞")
            or (following["string"] == "な" and following["pos"] == "助動詞")
        )
    ):
        _set_reading(njd_features, index, "ヨウ", "ヨー")
        return True
    return False


def _read_kami_era_yo(njd_features: list[NJDFeature], index: int) -> bool:
    """
    神々の時代を表す「神の代」の「代」を、代金や世代の「ダイ」でなく「ヨ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    following = _feature_at(njd_features, index + 1)
    # 「代」のコストだけを下げると代金などの「ダイ」も変わるため、神々の時代を表す「神の代」に絞る
    ## 短い連語の登録は「代理」「代行」「代弁」にも届くため、独立した名詞の「代」と前後の品詞を確かめる
    if (
        feature["string"] == "代"
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
        _set_reading(njd_features, index, "ヨ")
        return True
    return False


def _read_abstract_moto(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「支配の下」「条件の下」のように抽象名詞と「の」に続く「下」を、「シタ」でなく「モト」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    if (
        feature["string"] == "下"
        and feature["pos_group1"] == "一般"
        and previous is not None
        and previous["string"] == "の"
        and previous_previous is not None
        and previous_previous["string"] in _ABSTRACT_NO_PREDECESSORS
    ):
        _set_reading(njd_features, index, "モト")
        return True
    return False


def _read_digit_position(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「下四桁」「上二桁」のように数字の桁の位置を指す「下」「上」を、「シタ」「ウエ」でなく「シモ」「カミ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    surface = njd_features[index]["string"]
    following = _feature_at(njd_features, index + 1)
    if (
        surface in {"下", "上"}
        and following is not None
        and following["pos_group1"] == "数"
        and index + 2 < len(njd_features)
        and njd_features[index + 2]["string"] == "桁"
    ):
        _set_reading(njd_features, index, "シモ" if surface == "下" else "カミ")
        return True
    return False


def _read_bashi_suffix(njd_features: list[NJDFeature], index: int) -> bool:
    """
    名詞に続く接尾辞の「橋」を「バシ」と読む (構造種別の前接語に続く「キョウ」は _ADJACENT_WORD_READINGS の表で先に読む)。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    if (
        feature["string"] == "橋"
        and feature["pos_group1"] == "接尾"
        and previous is not None
        and previous["pos"] == "名詞"
    ):
        _set_reading(njd_features, index, "バシ")
        return True
    return False


def _read_mono(njd_features: list[NJDFeature], index: int) -> bool:
    """
    2文字以上の自立語に続く「者」を、「シャ」でなく「モノ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    # 「受者」の「受」も動詞として解析されるため、1文字の語幹に続く「者」は辞書の読みを保つ
    if (
        njd_features[index]["string"] == "者"
        and previous is not None
        and previous["pos_group1"] == "自立"
        and len(previous["string"]) > 1
    ):
        _set_reading(njd_features, index, "モノ")
        return True
    return False


def _read_compound_iru(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「走り入って」「攻め入って」のように複合動詞を作る「入っ」を、「ハイッ」でなく「イッ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    if (
        njd_features[index]["string"] == "入っ"
        and previous is not None
        and previous["pos"] == "動詞"
        and previous["cform"] == "連用形"
        and previous["string"] in _IRU_COMPOUND_PREDECESSORS
    ):
        _set_reading(njd_features, index, "イッ")
        return True
    return False


def _read_ooyake(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「いつか公になる」のように副詞可能名詞に続く「公」を、「オーヤケ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    previous = _feature_at(njd_features, index - 1)
    # 名詞の後の一律の補正 (_COMPOUND_SUFFIX_READINGS の「コウ」) から分けるため、その規則より先に試す
    if (
        feature["string"] == "公"
        and feature["pos_group1"] == "一般"
        and previous is not None
        and previous["pos_group1"] == "副詞可能"
    ):
        _set_reading(njd_features, index, "オオヤケ", "オーヤケ")
        return True
    return False


def _read_handle_e_or_tsuka(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「柄」は同じ名詞の候補に「ガラ」「エ」「ツカ」があるので、「の」の前の道具名で分ける。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    if not (
        njd_features[index]["string"] == "柄"
        and previous is not None
        and previous["string"] == "の"
        and previous_previous is not None
    ):
        return False

    if previous_previous["string"] in _TOOL_HANDLE_PREDECESSORS:
        _set_reading(njd_features, index, "エ")
    elif previous_previous["string"] in _SWORD_HILT_PREDECESSORS:
        _set_reading(njd_features, index, "ツカ")
    return True


def _read_rim_fuchi(njd_features: list[NJDFeature], index: int) -> bool:
    """
    器物や形状を表す名詞と「の」に続く「縁」を、「エン」でなく「フチ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    if (
        njd_features[index]["string"] == "縁"
        and previous is not None
        and previous["string"] == "の"
        and previous_previous is not None
        and previous_previous["string"] in _FUCHI_PREDECESSORS
    ):
        _set_reading(njd_features, index, "フチ")
        return True
    return False


def _read_teahouse_jaya(njd_features: list[NJDFeature], index: int) -> bool:
    """
    名詞に続く「茶屋」を、連濁した「ジャヤ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    if (
        njd_features[index]["string"] == "茶屋"
        and previous is not None
        and previous["pos"] == "名詞"
        and previous["pos_group1"] in {"一般", "固有名詞", "サ変接続"}
    ):
        _set_reading(njd_features, index, "ジャヤ")
        return True
    return False


def _read_honorific_ou(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「芭蕉翁」のように名詞に続く「翁」を、「オキナ」でなく「オー」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    # 「芭蕉翁」は「オー」と読むが、「明日翁が来る」のような独立用法は「オキナ」を保つ
    if (
        njd_features[index]["string"] == "翁"
        and previous is not None
        and previous["pos"] == "名詞"
        and previous["pos_group1"] in {"一般", "固有名詞", "サ変接続"}
    ):
        _set_reading(njd_features, index, "オウ", "オー")
        return True
    return False


def _read_tokoro(njd_features: list[NJDFeature], index: int) -> bool:
    """
    活用語に続く場所を表す「処」を、「ショ」でなく「トコロ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    # コストを下げても「処」が「ショ」に戻る文があるため、活用語に続く場所の「トコロ」は文脈で確定する
    if (
        njd_features[index]["string"] == "処"
        and previous is not None
        and (previous["pos_group1"] == "自立" or previous["pos"] == "助動詞")
    ):
        _set_reading(njd_features, index, "トコロ")
        return True
    return False


def _read_shiita(njd_features: list[NJDFeature], index: int) -> bool:
    """
    形容詞の「強い」に「た」が付いた「強いた」を、動詞「強いる」の過去形として「シイ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    following = _feature_at(njd_features, index + 1)
    # 形容詞の終止形に「た」は続かないので、MeCab が形容詞の「強い」に「た」を付けた「強いた」は、動詞「強いる」の過去形として「シイ」と読む
    ## 動詞「強いる」の「シイ」の行のコストを下げると、「強い風」「強い国」の形容詞まで「シイ」になるので、後続の「た」で決める
    if (
        feature["string"] == "強い"
        and feature["pos"] == "形容詞"
        and following is not None
        and following["pos"] == "助動詞"
        and following["string"] == "た"
    ):
        _set_reading(njd_features, index, "シイ")
        return True
    return False


def _read_sourou(njd_features: list[NJDFeature], index: int) -> bool:
    """
    候文の「申し上げ候」のように動詞の連用形に続く「候」を、名詞の「コウ」でなく補助動詞の「ソウロウ」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    if (
        njd_features[index]["string"] == "候"
        and previous is not None
        and previous["pos"] == "動詞"
        and previous["cform"] == "連用形"
    ):
        _set_reading(njd_features, index, "ソウロウ", "ソーロー")
        return True
    return False


def _read_hoka(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「より外にない」は選択肢の「ホカ」、物体の位置を比べる場合は「ソト」と読む。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    previous_previous = _feature_at(njd_features, index - 2)
    if not (
        njd_features[index]["string"] == "外"
        and previous is not None
        and previous["string"] == "より"
        and previous_previous is not None
        and (previous_previous["pos"] == "動詞" or previous_previous["pos_group1"] == "代名詞")
    ):
        return False

    for later_feature in njd_features[index + 1 :]:
        if later_feature["string"] in {"。", "．", "！", "？"}:
            break
        if later_feature["orig"] in _NEGATIVE_ORIGINALS:
            _set_reading(njd_features, index, "ホカ")
            break
    return True


def _read_compound_suffix(njd_features: list[NJDFeature], index: int) -> bool:
    """
    名詞へ直接続く後部要素は、助詞を挟んだ独立用法と区別して複合語の読みへ変える。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    feature = njd_features[index]
    surface = feature["string"]
    previous = _feature_at(njd_features, index - 1)
    # 数詞に続く「部屋」は助数詞なので、NJD の数詞処理が選んだ「ヘヤ」を保つ
    ## 日付に続く「付」は辞書の「ヅケ」を保つ
    ## 「十日」は地名、「一日」「四日」は副詞可能名詞になるため、数字と「日」からなる表層形も日付として扱う
    if not (
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
        return False

    reading, pronunciation, required_pos_group1 = _COMPOUND_SUFFIX_READINGS[surface]
    if required_pos_group1 is None or feature["pos_group1"] == required_pos_group1:
        _set_reading(njd_features, index, reading, pronunciation)
    return True


def _read_ra_or_tou(njd_features: list[NJDFeature], index: int) -> bool:
    """
    「等」は代名詞に続けば「ラ」、自立した名詞に続けば「トウ」（「トー」）、活用語や形式名詞では既定の「ナド」を残す。

    Args:
        njd_features (list[NJDFeature]): 補正対象の NJDNode 用 features
        index (int): 補正する形態素の位置

    Returns:
        bool: 規則の条件に当たり、後続の規則を試さない場合は True
    """

    previous = _feature_at(njd_features, index - 1)
    if njd_features[index]["string"] != "等" or previous is None:
        return False
    if previous["pos_group1"] == "代名詞":
        _set_reading(njd_features, index, "ラ")
        return True
    if previous["pos_group1"] in {"一般", "サ変接続", "固有名詞", "接尾"}:
        _set_reading(njd_features, index, "トウ", "トー")
        return True
    return False
