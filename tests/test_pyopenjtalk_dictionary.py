"""
pyopenjtalk-plus のデフォルト辞書の読みが正しいことを検証するテスト。
辞書の以下のカテゴリの問題を検証する:
  - 辞書未登録語の追加
  - 辞書誤登録・読み誤りの修正
  - コスト設定問題の調整
  - 形態素分割ミスの改善
  - 連濁・発音形の差異の修正

各テストケースは (入力テキスト, 期待される g2p() の kana 出力) のタプルで構成される。
g2p(text, kana=True) は発音形（pron フィールド）を返すため、期待値も発音形（オウ→オー 等）で記述する。
"""

import csv
import re
from functools import lru_cache
from pathlib import Path

import pytest

import pyopenjtalk


KATAKANA_SURFACE_RE = re.compile(r"^[ァ-ヴーヽヾ]+$")
# 発音をモーラに区切るパターン (VOICEVOX ENGINE がユーザー辞書のモーラ数を数えるときと同じもの)
## 同梱の辞書では、OpenJTalk の JPCommon がモーラ表で区切った結果とすべての行で一致する
MORA_RE = re.compile(
    r"(?:"
    r"[イ][ェ]|[ヴ][ャュョ]|[クグトド][ゥ]|[テデ][ィャュョ]|[デ][ェ]|[クグ][ヮ]|"
    r"[キシチニヒミリギジヂビピ][ェャュョ]|[シ][ィ]|"
    r"[クツフヴグ][ァ]|[ウクスツフヴグズヅ][ィ]|[ウクツフヴグ][ェォ]|[フ][ュ]|"
    r"[ァ-ヴー]"
    r")"
)


# ============================================================
# 辞書未登録語の追加テスト
# report 2.1, 4.1, 9.7, Appendix の ✅ 判定エントリ
# ============================================================

UNREGISTERED_WORDS = [
    # 一般語
    ("今帝", "キンテー"),
    ("擦音", "サツオン"),
    ("導水溝", "ドースイコー"),
    ("端迷惑", "ハタメーワク"),
    ("巡詣", "ジュンケー"),
    ("繭層", "ケンソー"),
    ("汎太平洋", "ハンタイヘーヨー"),
    ("魚醤", "ギョショー"),
    ("桑果", "ソーカ"),
    ("津液", "シンエキ"),
    ("鬼手", "キシュ"),
    ("業前", "ワザマエ"),
    ("増募", "ゾーボ"),
    ("拭払", "ショクフツ"),
    ("藐視", "ビョーシ"),
    ("凝望", "ギョーボー"),
    ("偵伺", "テーシ"),
    ("親跳ね", "オヤッパネ"),
    ("強面", "コワモテ"),
    ("神籬", "ヒモロギ"),
    ("掌打", "ショーダ"),
    ("基肥", "モトゴエ"),
    ("憎嫉", "ゾーシツ"),
    ("憂い事", "ウレイゴト"),
    # 漢方・医学用語
    ("牡丹皮", "ボタンピ"),
    ("虚労", "キョロー"),
    # 四字熟語
    ("唯唯諾諾", "イーダクダク"),
    ("花朝月夕", "カチョーゲッセキ"),
    ("錦衣玉食", "キンイギョクショク"),
    ("百折不撓", "ヒャクセツフトー"),
    ("兵戈無用", "ヒョーガムヨー"),
    ("慈心不殺", "ジシンフセツ"),
    ("眺望絶佳", "チョーボーゼッカ"),
    ("妄評多罪", "モーヒョータザイ"),
    ("一樹百穫", "イチジュヒャッカク"),
    ("華夷秩序", "カイチツジョ"),
    # 仏教用語（複合語としてのみ登録）
    ("二河白道", "ニガビャクドー"),
    ("清浄寂滅", "ショージョージャクメツ"),
    ("白衣観音", "ビャクエカンノン"),
    # 植物名
    ("羽衣枝垂", "ハゴロモシダレ"),
    # 文語
    ("彪蔚", "ヒューウツ"),
    # Appendix A-1.4 の追加エントリ
    ("褐輪", "カツリン"),
    ("辺偶", "ヘングー"),
    # Appendix A-2.2, A-2.3 の追加エントリ
    ("四魂", "シコン"),
    ("浅頭筋", "セントーキン"),
    ("皮筋", "ヒキン"),
    ("冥熏", "メークン"),
    ("逓騎哨", "テーキショー"),
    ("聚慎", "ジュシン"),
    ("墓闕", "ボケツ"),
    ("高頤", "コーイ"),
    ("吏部", "リブ"),
    ("登華殿", "トーカデン"),
    ("洪水吐", "コーズイバキ"),
    ("堤体", "ツツミタイ"),
    ("逓伝哨", "テーデンショー"),
    ("力皇", "リキオー"),
    ("宵々々山", "ヨイヨイヨイヤマ"),
    ("山建", "ヤマタテ"),
    # 9.7 JSUT 追加エントリ
    ("建坪率", "ケンペーリツ"),
    ("直刀", "チョクトー"),
    ("源汰", "ゲンタ"),
    # Round 3 追加エントリ
    ("清浄光寺", "ショージョーコージ"),
    ("二環路", "ニカンロ"),
    ("内閉鎖筋", "ナイヘーサキン"),
    ("逓自転車哨", "テージテンシャショー"),
    ("抽分銭", "チューモンセン"),
    ("洪", "コー"),
    # Section 4.1#11 / A-2.3 追加
    ("玄妙五種香", "ゲンミョーゴシュコー"),
    ("切痕", "セッコン"),
    # Section 2.3#2 千切る 活用形
    ("千切る", "チギル"),
    ("千切って", "チギッテ"),
    # Appendix A-2.4#2: 古文の連語
    ("奉り候", "タテマツリソーロー"),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("出穂の時期だ", "シュッスイノジキダ"),
        ("意見を擦り合わせる", "イケンヲスリアワセル"),
        ("両手を擦り合わせた", "リョーテヲスリアワセタ"),
        ("ここで憩える", "ココデイコエル"),
        ("ここで憩えない", "ココデイコエナイ"),
        ("ここで憩えればよい", "ココデイコエレバヨイ"),
        ("邸内の案内図", "テーナイノアンナイズ"),
        ("艶麗な装い", "エンレーナヨソオイ"),
        ("専門に特化した講座", "センモンニトッカシタコーザ"),
        ("内袋を取り替えた", "ウチブクロヲトリカエタ"),
        ("中袋を縫い付ける", "ナカブクロヲヌイツケル"),
        ("封筒の中袋を開く", "フートーノナカブクロヲヒラク"),
        ("横走りに渡る", "ヨコバシリニワタル"),
        ("縦走りの幕を使う", "タテバシリノマクヲツカウ"),
        ("諸尊を拝する", "ショソンヲハイスル"),
        ("四角形の面積を求める", "シカッケーノメンセキヲモトメル"),
        ("日田市を訪ねる", "ヒタシヲタズネル"),
        ("お白石を奉納する", "オシライシヲホーノースル"),
        ("お白石持を見学する", "オシライシモチヲケンガクスル"),
        ("お白石持ちに参加する", "オシライシモチニサンカスル"),
        ("讃容の里を訪れる", "サヨノサトヲオトズレル"),
        ("白石さんと話す", "シライシサントハナス"),
        ("総馬さんに会う", "ソーマサンニアウ"),
        ("十二四郎さんに会う", "ジューニシローサンニアウ"),
        ("開式を待つ", "カイシキヲマツ"),
        ("十重二十重に巻く", "トエハタエニマク"),
        ("百十重の層", "ヒャクジュージューノソー"),
        ("江山さんに頼む", "エヤマサンニタノム"),
        ("田中汎さんに頼む", "タナカヒロシサンニタノム"),
    ],
)
def test_compound_and_conjugated_dictionary_readings(text: str, expected: str) -> None:
    """複合語と可能動詞を語の読みで選び、数詞と人名の読みを保つ。"""

    assert (
        pyopenjtalk.g2p(text, kana=True, use_sudachi_kanji_yomi=False, predict_nani=False)
        == expected
    )


@pytest.mark.parametrize(
    "text, expected",
    UNREGISTERED_WORDS,
    ids=[t[0] for t in UNREGISTERED_WORDS],
)
def test_unregistered_words(text: str, expected: str) -> None:
    """辞書に未登録だった語が正しく読めることを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("茶菓を用意する。", "サカヲヨーイスル。"),
        ("茶菓子を買う。", "チャガシヲカウ。"),
        ("茶会に出る。", "チャカイニデル。"),
        ("茶道を学ぶ。", "サドーヲマナブ。"),
        ("茶碗を洗う。", "チャワンヲアラウ。"),
    ],
)
def test_saka_default_preserves_other_tea_compounds(text: str, expected: str) -> None:
    """
    茶と菓子を表す「茶菓」が「サカ」と読まれ、同じ「茶」で始まる「茶菓子」「茶会」「茶道」「茶碗」の読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十束剣を抜いた。", "トツカノツルギヲヌイタ。"),
        ("十拳剣を抜いた。", "トツカノツルギヲヌイタ。"),
        ("十束の剣を抜いた。", "トツカノツルギヲヌイタ。"),
        ("薪を十束運ぶ。", "タキギヲジュッタバハコブ。"),
        ("天地の詞を写した。", "アメツチノコトバヲウツシタ。"),
        ("天地の袋を縫った。", "アメツチノフクロヲヌッタ。"),
        ("天地がひっくり返る。", "テンチガヒックリカエル。"),
        ("天地無用と書く。", "テンチムヨートカク。"),
        ("天地の道を説く。", "テンチノミチヲトク。"),
    ],
)
def test_fixed_sword_and_ametsuchi_expressions(text: str, expected: str) -> None:
    """
    剣の名称の「十束剣」「十拳剣」「十束の剣」が、数詞「十」と助数詞「束」や名詞「拳」「剣」に分かれず、「トツカノツルギ」と読まれることを確認する。
    「天地の詞」「天地の袋」が、一般語「天地」の「テンチ」や単独の「詞」の「シ」と競合しても、「アメツチノコトバ」「アメツチノフクロ」と読まれることを確認する。
    薪の数量を表す「十束」は「ジュッタバ」と読まれ、「天地がひっくり返る」「天地無用」「天地の道」の「天地」は「テンチ」のまま読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("沼の主な生き物を調べる。", "ヌマノオモナイキモノヲシラベル。"),
        ("湖の主流は東に向かう。", "ミズウミノシュリューワヒガシニムカウ。"),
        ("池の主な用途は灌漑です。", "イケノオモナヨートワカンガイデス。"),
        ("主として野菜を食べる。", "シュトシテヤサイヲタベル。"),
        ("私の主として使う辞書だ。", "ワタシノシュトシテツカウジショダ。"),
    ],
)
def test_noun_followed_by_shu_keeps_word_boundaries(text: str, expected: str) -> None:
    """
    「沼の主な生き物」「湖の主流」のように水辺の名詞と「の」の後に「主」で始まる語が続く入力で、「主な」「主流」の語の区切りが保たれ、「オモナ」「シュリュー」と読まれることを確認する。
    「湖の主」「沼の主」を1語で登録すると、後ろの「な」「流」まで「ヌシ」と読む経路が選ばれるため、「〜の主」を辞書の行で直さない理由としてこの区切りを守る。
    副詞の「主として」は、直前に「の」があっても「シュトシテ」と読まれることも確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("沖する煙を見た。", "チュースルケムリヲミタ。"),
        ("噴煙が天に沖する。", "フンエンガテンニチュースル。"),
        ("沖した煙が消えた。", "チューシタケムリガキエタ。"),
        ("沖す煙が見える。", "チュースケムリガミエル。"),
        ("沖すれば見える。", "チュースレバミエル。"),
        ("沖に船が浮かぶ。", "オキニフネガウカブ。"),
        ("沖縄へ行く。", "オキナワエイク。"),
    ],
)
def test_chuusuru_verb_preserves_oki_noun(text: str, expected: str) -> None:
    """
    煙が高く上がる意味の「沖する」「沖した」などの入力で、名詞「沖」と後続語への分割より動詞の活用形が選ばれ、「沖」が「チュー」と読まれることを確認する。
    動詞の活用形のコストを下げても、海を表す「沖に船が浮かぶ」の「沖」は「オキ」、地名の「沖縄」は「オキナワ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize("text", ["天地の詞", "天地の袋"])
def test_genitive_phrase_entries_keep_separate_accent_phrases(text: str) -> None:
    """
    「天地の詞」「天地の袋」の入力で、語全体を1つの辞書行に登録しても、「:」区切りに従って2つのアクセント句が作られることを確認する。
    全体を1つの句にまとめると後半のアクセント核が変わるため、「天地」は「アメツチ」の1モーラ目、「詞」「袋」は3モーラ目に核が置かれることを確認する。
    """

    features = pyopenjtalk.run_frontend(text)

    assert len(features) == 2
    assert features[1]["chain_flag"] == 0
    assert (features[0]["read"], features[0]["acc"]) == ("アメツチノ", 1)
    assert (features[1]["acc"], features[1]["mora_size"]) == (3, 3)


@pytest.mark.xfail(strict=True, reason="「湖の主として」の「主として」が副詞の行で読まれる")
def test_water_nushi_before_toshite_reading() -> None:
    """
    湖に長く住む生き物を指す「湖の主として知られる」が、副詞「主として」の辞書行と競合しても、「主」を「ヌシ」と読むことを確認する。
    「湖の主」を1語で登録すると「湖の主な」「湖の主流」の区切りまで崩れるため、辞書の行では直していない。
    """

    assert (
        pyopenjtalk.g2p("湖の主として知られる大鯉を釣った。", kana=True)
        == "ミズウミノヌシトシテシラレルオーゴイヲツッタ。"
    )


@pytest.mark.xfail(strict=True, reason="慣用句の「音を上げる」が音量と同じ「オト」で読まれる")
def test_newoageru_idiom_reading() -> None:
    """
    降参する意味の「難題に音を上げる」が、名詞「音」の「オト」と競合しても、慣用句として「ネヲアゲル」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p("難題に音を上げる。", kana=True) == "ナンダイニネヲアゲル。"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("スピーカーの音を上げた。", "スピーカーノオトヲアゲタ。"),
        ("録音の音を上げる。", "ロクオンノオトヲアゲル。"),
        ("雑音に負けないように音を上げた。", "ザツオンニマケナイヨーニオトヲアゲタ。"),
    ],
)
def test_sound_volume_readings_preserved(text: str, expected: str) -> None:
    """
    スピーカーや録音の音量を上げる入力で、降参する意味の慣用句と同じ「音を上げる」という表記でも、「音」が「オト」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("黒白の写真を飾る。", "クロシロノシャシンヲカザル。"),
        ("黒白の模様を描く。", "クロシロノモヨーヲエガク。"),
        ("白黒をつける。", "シロクロヲツケル。"),
    ],
)
def test_color_readings_preserved(text: str, expected: str) -> None:
    """
    写真や模様の「黒白」が、同じ表記の「コクビャク」の辞書行と競合しても「クロシロ」と読まれ、「白黒をつける」は「シロクロヲツケル」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


# ============================================================
# 辞書誤登録・読み誤りの修正テスト
# report 4.3 の ✅ 判定エントリ
# ============================================================

READING_FIXES = [
    # 「いう」を含む指示表現は、一般的な発話に近い既定発音を維持する
    ("こういう", "コーユウ"),
    ("そういう", "ソーユウ"),
    ("出歯亀", "デバガメ"),
    ("朱泥", "シュデー"),
    ("緩手", "カンシュ"),
    ("坂路", "ハンロ"),
    ("河川敷", "カセンジキ"),
    ("南砂町", "ミナミスナマチ"),
    # 利用者辞書に依存せず、デフォルト辞書だけで一般的な読みを選ぶ
    ("一寸待って", "チョットマッテ"),
    ("素振りをする", "スブリヲスル"),
    ("不動明王を拝む", "フドーミョーオーヲオガム"),
    ("大太鼓を叩く", "オーダイコヲタタク"),
    ("研究所へ行く", "ケンキュージョエイク"),
    ("二遊間を守る", "ニユーカンヲマモル"),
    ("鶏小屋を掃除する", "トリゴヤヲソージスル"),
    ("地魚のみを使う", "ジザカナノミヲツカウ"),
    ("一生物の道具", "イッショーモノノドーグ"),
    ("東京人です", "トーキョージンデス"),
    ("日本の伝統です", "ニホンノデントーデス"),
    ("１服飲む", "イップクノム"),
    ("ラーメン橋の耐震設計を確認する", "ラーメンキョーノタイシンセッケーヲカクニンスル"),
    ("魚を乄る", "サカナヲシメル"),
    # 希少な複合語・地名の候補が一般的な分割経路や人名文脈を上書きしないことも確認する
    (
        "もちろん、小舟をつかえば倭館まではすぐである。",
        "モチロン、コブネヲツカエバワカンマデワスグデアル。",
    ),
    ("ひと揃いずつ持っている。", "ヒトソロイズツモッテイル。"),
    ("白飯とみそ汁を食べる。", "シロメシトミソシルヲタベル。"),
    # 同コスト問題の修正
    ("温く", "ヌルク"),
    ("芳しかっ", "カンバシカッ"),
    # 歴史地名・複合語・外来語の読み修正
    ("大倭", "ヤマト"),
    ("後志", "シリベシ"),
    ("御野", "ミノ"),
    ("末廬", "マツラ"),
    ("石城", "イワキ"),
    ("茶畑", "チャバタケ"),
    ("内分泌", "ナイブンピ"),
    ("ウォーミング", "ウォーミング"),
]


@pytest.mark.parametrize(
    "text, expected",
    READING_FIXES,
    ids=[t[0] for t in READING_FIXES],
)
def test_reading_fixes(text: str, expected: str) -> None:
    """辞書の読み誤りが修正されていることを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


def test_ramen_bridge_uses_default_compound_path() -> None:
    """一般土木用語を、利用者辞書の完全一致行に頼らず複合語として読む。"""

    features = pyopenjtalk.run_frontend("ラーメン橋の耐震設計")
    assert [
        (feature["string"], feature["pos_group1"], feature["read"]) for feature in features[:2]
    ] == [
        ("ラーメン", "一般", "ラーメン"),
        ("橋", "接尾", "キョウ"),
    ]


def test_ban_keeps_both_general_reading_candidates() -> None:
    """ネット用語のバンと英字略称のビーエーエヌを同じ表層から供給する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates("ＢＡＮ", ((0, 3),))
    pronunciations = {
        path["pronunciation"] for path in analysis["paths"] if path["char_span"] == (0, 3)
    }
    features = {
        path["features"][0]
        for path in analysis["paths"]
        if path["char_span"] == (0, 3) and path["surface"] == "ＢＡＮ"
    }

    # 読みの候補を追加しても品詞は変えず、後続の「する」「された」の形態素構造を維持する
    assert {"バン", "ビーエーエヌ"} <= pronunciations
    assert len(features) > 0
    assert all(",名詞,一般," in feature for feature in features)

    # 高い生起コストを設定することで通常時の読みを維持し、tsqyomi が文脈から選ぶ候補としてのみ追加する
    assert pyopenjtalk.g2p("ＢＡＮ", kana=True) == "ビーエーエヌ"
    assert pyopenjtalk.g2p("アカウントをＢＡＮする", kana=True) == "アカウントヲビーエーエヌスル"
    assert (
        pyopenjtalk.g2p("アカウントをＢＡＮされた", kana=True) == "アカウントヲビーエーエヌサレタ"
    )
    assert pyopenjtalk.g2p("ＢＡＮという略称", kana=True) == "ビーエーエヌトイウリャクショー"


@pytest.mark.parametrize("surface", ("〆る", "乄る"))
def test_transferred_general_entries_keep_morphology_and_accent(surface: str) -> None:
    """〆/乄 系の動詞を、「締める」と同型の一段動詞として解析する。"""

    features = pyopenjtalk.run_frontend(surface)
    assert [
        (feature["string"], feature["pos"], feature["ctype"], feature["read"])
        for feature in features
    ] == [
        (surface, "動詞", "一段", "シメル"),
    ]
    assert (features[0]["acc"], features[0]["mora_size"]) == (2, 3)


def test_region_name_does_not_override_person_name_context() -> None:
    """地域名の「武昌」が人名文脈を上書きしないことを検証する。"""

    # 人名の正読は別途補完の余地を残し、地域名の「ブショー」が選ばれないことだけを固定する
    assert "ブショー" not in pyopenjtalk.g2p("守屋武昌防衛局長", kana=True)


def test_man_old_character_keeps_general_and_family_name_uses() -> None:
    """旧字体の「萬」を一般名詞と姓のどちらでもヨロズと読む。"""

    isolated_features = pyopenjtalk.run_frontend("萬")
    family_name_features = pyopenjtalk.run_frontend("萬さん")

    # 単独表記では一般名詞を優先し、人名接尾辞が続く文脈では姓の形態素を維持する
    assert pyopenjtalk.g2p("萬", kana=True, use_vanilla=True) == "ヨロズ"
    assert isolated_features[0]["pos_group1"] == "一般"
    assert pyopenjtalk.g2p("萬さん", kana=True, use_vanilla=True) == "ヨロズサン"
    assert family_name_features[0]["pos_group1"] == "固有名詞"
    assert family_name_features[0]["pos_group2"] == "人名"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("米粉", "コメコ"),
        ("米粉です", "コメコデス"),
        ("米粉パン", "コメコパン"),
        ("国産米粉を使う", "コクサンコメコヲツカウ"),
    ],
)
def test_komeko_dominates_split_paths(text: str, expected: str) -> None:
    """「米粉」が「ベイフン」や「米」+「粉」の分割ではなく、「コメコ」として優先的に読まれることを確認する。"""

    # 同表層の候補間だけでなく、「米」+「粉」の分割経路にも勝つことを公開 API で確認する
    assert pyopenjtalk.g2p(text, kana=True) == expected


def _heteronym_pronunciations(surface: str) -> list[str]:
    """
    heteronyms.csv にある表層の発音を、重複を保ったまま返す。

    Args:
        surface (str): 検索する表層

    Returns:
        list[str]: 出現順の発音
    """

    dictionary_directory = Path(pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"))
    dictionary_path = dictionary_directory / "heteronyms.csv"
    pronunciations: list[str] = []
    with dictionary_path.open(encoding="utf-8", newline="") as dictionary_file:
        for row in csv.reader(dictionary_file):
            if len(row) > 12 and row[0] == surface:
                pronunciations.append(row[12])
    return pronunciations


@lru_cache(maxsize=1)
def _naist_jdic_surfaces() -> frozenset[str]:
    """naist-jdic.csv に存在する表層集合を返す。

    Returns:
        frozenset[str]: 辞書に存在する表層の集合
    """

    dictionary_directory = Path(pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"))
    dictionary_path = dictionary_directory / "naist-jdic.csv"
    with dictionary_path.open(encoding="utf-8", newline="") as dictionary_file:
        return frozenset(row[0] for row in csv.reader(dictionary_file) if len(row) > 12)


@lru_cache(maxsize=1)
def _unidic_csj_rows() -> tuple[tuple[str, ...], ...]:
    """unidic-csj.csv の全エントリを返す。"""

    dictionary_directory = Path(pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"))
    dictionary_path = dictionary_directory / "unidic-csj.csv"
    with dictionary_path.open(encoding="utf-8", newline="") as dictionary_file:
        return tuple(tuple(row) for row in csv.reader(dictionary_file) if len(row) > 12)


@pytest.mark.parametrize(
    "dictionary_name",
    [
        "naist-jdic.csv",
        "unidic-csj.csv",
        "heteronyms.csv",
        "rare_syllables.csv",
        "fillers.csv",
        "symbols.csv",
    ],
)
def test_dictionary_accent_field_matches_pronunciation(dictionary_name: str) -> None:
    """辞書のアクセント欄の句数とモーラ数が、発音と食い違っていないことを確認する (食い違うとアクセント核の位置がずれる)。"""

    dictionary_directory = Path(pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"))
    violations: list[str] = []
    with (dictionary_directory / dictionary_name).open(
        encoding="utf-8", newline=""
    ) as dictionary_file:
        for row in csv.reader(dictionary_file):
            # アクセントを持たない記号などの行 (「*/*」) は対象にしない
            if len(row) < 14 or re.fullmatch(r"\d+/\d+(:\d+/\d+)*", row[13]) is None:
                continue
            accent_phrases = row[13].split(":")
            if any(len(row[index].split(":")) != len(accent_phrases) for index in (10, 11, 12)):
                violations.append(f"句数が食い違う: {row}")
                continue
            for pronunciation, accent in zip(row[12].split(":"), accent_phrases):
                nucleus, mora_count = (int(value) for value in accent.split("/"))
                # 無声化記号の「’」はモーラに数えない
                moras = MORA_RE.findall(pronunciation.replace("’", ""))
                if "".join(moras) != pronunciation.replace("’", ""):
                    violations.append(f"発音をモーラに区切れない: {row}")
                elif len(moras) != mora_count or nucleus > mora_count:
                    violations.append(
                        f"アクセント欄が発音のモーラ数 {len(moras)} と合わない: {row}"
                    )

    assert violations == []


def test_unidic_katakana_common_nouns_beat_unknown_candidates() -> None:
    """UniDic のカタカナ一般名詞が未知語候補より優先される。"""

    katakana_common_noun_rows = [
        row
        for row in _unidic_csj_rows()
        if row[4] == "名詞"
        and row[5] == "一般"
        and KATAKANA_SURFACE_RE.fullmatch(row[0]) is not None
    ]

    # 辞書に行を追加してもテストが壊れないよう、件数ではなく「未知語のコスト 8360 より低い」ことだけを確かめる
    ## 件数の下限は、抽出条件の誤りで対象が空になり、何も確かめないままテストが通ってしまうのを防ぐためのもの
    assert len(katakana_common_noun_rows) > 30000
    assert all(int(row[3]) <= 8359 for row in katakana_common_noun_rows)


@pytest.mark.parametrize(
    ("text", "expected_read", "expected_acc", "expected_mora_size"),
    [
        ("オイチョカブ", "オイチョカブ", 3, 5),
        ("ヌマエビ", "ヌマエビ", 2, 4),
        ("アイゴ", "アイゴ", 0, 3),
        ("アウトカム", "アウトカム", 3, 5),
        ("アオアシ", "アオアシ", 0, 4),
    ],
)
def test_unidic_katakana_common_nouns_keep_known_word_accents(
    text: str,
    expected_read: str,
    expected_acc: int,
    expected_mora_size: int,
) -> None:
    """UniDic 由来のカタカナ一般名詞が未知語として扱われず、辞書に登録された正しいアクセントで解析されることを確認する。"""

    features, morphs = pyopenjtalk.run_frontend_detailed(text)

    assert len(features) == 1
    assert features[0]["read"] == expected_read
    assert (features[0]["acc"], features[0]["mora_size"]) == (expected_acc, expected_mora_size)
    assert len(morphs) == 1
    assert morphs[0]["is_unknown"] is False
    assert morphs[0]["word_cost"] == 8359


@pytest.mark.parametrize(
    ("text", "expected_morphemes"),
    [
        ("アカアシ", [("アカ", "アカ"), ("アシ", "アシ")]),
        ("クマヤナギ", [("クマ", "クマ"), ("ヤナギ", "ヤナギ")]),
        ("アカクラゲ", [("アカ", "アカ"), ("クラゲ", "クラゲ")]),
    ],
)
def test_unidic_katakana_common_noun_costs_keep_lower_cost_split_paths(
    text: str,
    expected_morphemes: list[tuple[str, str]],
) -> None:
    """カタカナ語のコスト調整後も、「アカクラゲ」のように分割した方が自然な複合語では無理に1語に結合せず、適切な形態素分割が維持されることを確認する。"""

    features = pyopenjtalk.run_frontend(text)

    assert [(feature["string"], feature["read"]) for feature in features] == expected_morphemes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("乾死にした", "ヒジニシタ"),
        ("干死にした", "ヒジニシタ"),
        ("早死にした", "ハヤジニシタ"),
        ("溺れ死にした", "オボレジニシタ"),
        ("犬死にした", "イヌジニシタ"),
        ("若死にした", "ワカジニシタ"),
        ("飢え死にした", "ウエジニシタ"),
        ("餓え死にした", "ウエジニシタ"),
        ("切死にした", "キリジニシタ"),
        ("怨み死にした", "ウラミジニシタ"),
        ("恨み死にした", "ウラミジニシタ"),
        ("斬死にした", "キリジニシタ"),
        ("焦がれ死にした", "コガレジニシタ"),
        ("討死にした", "ウチジニシタ"),
        ("野垂死にした", "ノタレジニシタ"),
        ("飢死にした", "ウエジニシタ"),
    ],
)
def test_death_ni_spelling_candidates_do_not_duplicate_particle_reading(
    text: str,
    expected: str,
) -> None:
    """「討死にした」「飢え死にした」などの複合動詞において、送り仮名の「に」と助詞の「に」が重複して誤読されないことを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("アルクィン", "アルクィン"),
        ("ウィ〜ン", "ウィーン"),
        ("ウェストモーランド", "ウェストモーランド"),
        ("ウォ〜ン", "ウォーン"),
        ("ウフィッツィ", "ウフィッツィ"),
        ("エキストラオーディナレー", "エキストラオーディナレー"),
        ("エスクィリーノ", "エスクィリーノ"),
        ("エトキシフェニル", "エトキシフェニル"),
        ("カーディスィーヤ", "カーディシーヤ"),
        ("クィリーナーリス", "クィリーナーリス"),
        ("チオホスフェイト", "チオホスフェイト"),
        ("テトラエチルピロホスフェイト", "テトラエチルピロホスフェイト"),
        ("ネブカドネツァル", "ネブカドネツァル"),
        ("ヒンドゥ", "ヒンドゥ"),
        ("フュルジャンス", "フュルジャンス"),
        ("プレグナジェン", "プレグナジェン"),
    ],
)
def test_unidic_rare_syllable_entries_keep_g2p_pronunciation(
    text: str,
    expected: str,
) -> None:
    """「クィ」「ツァ」「ドゥ」など UniDic 由来の稀少な音節を含む外来語が、意図した通りの発音で出力されることを確認する。"""

    # 無声化記号は njd_set_unvoiced_vowel が文脈で挿入するため、稀音節の照合対象から除く
    features = pyopenjtalk.run_frontend(text, use_vanilla=True)
    pronunciation = "".join(
        feature["pron"].replace("’", "") for feature in features if feature["pron"] != "、"
    )
    assert pronunciation == expected


def test_unidic_rare_syllable_entry_keeps_default_loanword_spelling() -> None:
    """「スィ」のような特殊な外来語の仮名表記が、デフォルトの後処理後も一般的な仮名（「シ」など）に潰されずに保持されることを確認する。"""

    assert pyopenjtalk.g2p("カーディスィーヤ", kana=True) == "カーディスィーヤ"


@pytest.mark.parametrize(
    ("text", "expected_read", "expected_pron"),
    [
        ("一隊", "イッタイ", "イッタイ"),
        ("全家", "ゼンカ", "ゼンカ"),
        ("小羊", "コヒツジ", "コヒツジ"),
        ("小葉", "ショウヨウ", "ショーヨー"),
        ("律法", "リッポウ", "リッポー"),
        ("心切り", "シンキリ", "シンキリ"),
        ("聖別", "セイベツ", "セーベツ"),
        ("過越", "スギコシ", "スギコシ"),
    ],
)
def test_unidic_compound_cost_adjustments_select_whole_word(
    text: str,
    expected_read: str,
    expected_pron: str,
) -> None:
    """コストを調整した複合語（「一隊」「小羊」など）が不自然に単語分割されず、1つの単語として正しい読みとアクセントで解析されることを確認する。"""

    features = pyopenjtalk.run_frontend(text)

    assert [(feature["string"], feature["read"], feature["pron"]) for feature in features] == [
        (text, expected_read, expected_pron)
    ]
    assert pyopenjtalk.g2p(text, kana=True) == expected_pron


@pytest.mark.parametrize(
    ("text", "expected_features", "expected_pron"),
    [
        ("大連", [("大連", "ダイレン", "ダイレン", "固有名詞")], "ダイレン"),
        (
            "東日本",
            [("東日本", "ヒガシニホン", "ヒガシニホン", "固有名詞")],
            "ヒガシニホン",
        ),
        ("陵", [("陵", "ミササギ", "ミササギ", "一般")], "ミササギ"),
        ("殿", [("殿", "トノ", "トノ", "一般")], "トノ"),
    ],
)
def test_naist_context_exact_adjustments_select_expected_candidates(
    text: str,
    expected_features: list[tuple[str, str, str, str]],
    expected_pron: str,
) -> None:
    """品詞や文脈 ID に応じてコスト調整された単語（「陵」「殿」など）が、意図した通りの候補として優先選択されることを確認する。"""

    features = pyopenjtalk.run_frontend(text)

    assert [
        (feature["string"], feature["read"], feature["pron"], feature["pos_group1"])
        for feature in features
    ] == expected_features
    assert pyopenjtalk.g2p(text, kana=True) == expected_pron


@pytest.mark.parametrize(
    ("text", "expected_pron"),
    [
        ("登坂", "トサカ"),
        ("東日本学園北海道医療大学", "ヒガシニッポンガクエンホッカイドーイリョーダイガク"),
        ("陵墓", "リョーボ"),
        ("殿様", "トノサマ"),
    ],
)
def test_naist_context_exact_adjustments_keep_negative_controls(
    text: str,
    expected_pron: str,
) -> None:
    """コスト調整対象の語と表層や読みが似ている周辺の語（「陵墓」「殿様」など）において、コスト調整の影響を受けずに本来の読みが維持されることを確認する（ネガティブテスト）。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected_pron


@pytest.mark.parametrize(
    ("text", "expected_pron"),
    [
        ("蜘蛛", "クモ"),
        ("蜘蛛が出た", "クモガデタ"),
        ("蜘蛛の巣", "クモノス"),
        ("黒後家蜘蛛", "クロゴケグモ"),
        ("女郎蜘蛛", "ジョローグモ"),
        ("水蜘蛛", "ミズグモ"),
    ],
)
def test_kumo_rendaku_reading_is_limited_to_compounds(text: str, expected_pron: str) -> None:
    """連濁形の「グモ」が単独の「蜘蛛」に選ばれず、「黒後家蜘蛛」などの複合語でだけ使われることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected_pron


@pytest.mark.parametrize(
    ("text", "expected_pron"),
    [
        ("毒蛇", "ドクヘビ"),
        ("毒蛇に噛まれた", "ドクヘビニカマレタ"),
        ("羽毛布団", "ウモーブトン"),
        ("夏布団", "ナツブトン"),
        ("布団", "フトン"),
        ("芋焼酎", "イモジョーチュー"),
        ("麦焼酎", "ムギショーチュー"),
        ("焼酎", "ショーチュー"),
    ],
)
def test_compound_entries_select_rendaku_and_native_readings(text: str, expected_pron: str) -> None:
    """「毒蛇」が「ドクヘビ」と読まれ、「羽毛布団」「芋焼酎」などの複合語だけが連濁し、単独の「布団」「焼酎」や「麦焼酎」は連濁しないことを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected_pron


@pytest.mark.parametrize(
    ("text", "expected_pron"),
    [
        ("的を射る", "マトヲイル"),
        ("度々", "タビタビ"),
        ("芋を蒸す", "イモヲムス"),
        ("謀反を起こす", "ムホンヲオコス"),
        ("花を摘む", "ハナヲツム"),
        ("煩いが多い", "ワズライガオーイ"),
        ("母家", "オモヤ"),
        ("鍛冶場", "カジバ"),
        ("総力戦", "ソーリョクセン"),
        ("津軽三味線", "ツガルジャミセン"),
    ],
)
def test_cost_adjusted_readings_beat_competing_candidates(text: str, expected_pron: str) -> None:
    """コストを下げた読み（「度々」の「タビタビ」など）と追加した複合語が、別の読みや分割に負けずに選ばれることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected_pron


@pytest.mark.parametrize(
    ("surface", "expected_pronunciations", "expected_row_count"),
    [
        ("米粉", {"コメコ", "ビーフン"}, 2),
        ("一分", {"イチブ", "イチブン", "イップン"}, 3),
        ("経緯", {"イキサツ", "ケーイ", "タテヌキ", "タテヨコ", "ユクタテ"}, 5),
        ("最高値", {"サイコーチ", "サイタカネ"}, 2),
        ("艶やか", {"アデヤカ", "ツヤヤカ"}, 2),
    ],
)
def test_moved_heteronyms_keep_all_pronunciations(
    surface: str,
    expected_pronunciations: set[str],
    expected_row_count: int,
) -> None:
    """naist-jdic.csv から移した同形異音語の全候補を保持する。"""

    # 移動元へ戻ると重複候補の費用競合を再発させるため、表層が残らないことまで固定する
    assert surface not in _naist_jdic_surfaces()
    pronunciations = _heteronym_pronunciations(surface)

    assert len(pronunciations) == expected_row_count
    assert set(pronunciations) == expected_pronunciations


# ============================================================
# コスト設定問題の修正テスト
# report 2.2, 4.2 の ✅ 判定エントリ
# ============================================================

COST_ADJUSTMENTS = [
    # 一般的な訓読みが選択されるべきケース
    ("火傷", "ヤケド"),
    ("柵", "サク"),
    ("殺陣", "タテ"),
    ("紫蘇", "シソ"),
    ("不治", "フジ"),
    # 一般名詞が人名より優先されるべきケース
    ("百花", "ヒャッカ"),
    ("蓮", "ハス"),
    # コスト差が大きすぎるケース
    ("頭数", "アタマカズ"),
    ("仇", "カタキ"),
    ("私事", "ワタクシゴト"),
    ("馬鈴薯", "バレーショ"),
    ("蔵", "クラ"),
    # 現代語として自然な読みが選択されるべきケース
    ("戯れ", "タワムレ"),
    ("牛骨", "ギューコツ"),
    ("目前", "モクゼン"),
    ("殿", "トノ"),
    ("斜塔", "シャトー"),
    ("赤帯", "アカオビ"),
    ("半魚人", "ハンギョジン"),
    ("一昨年", "オトトシ"),
    ("日向市", "ヒューガシ"),
    ("黄土色", "オードイロ"),
    ("赤銅色", "シャクドーイロ"),
    ("山吹色", "ヤマブキイロ"),
    # 人名のコストが異常に低い問題
    ("亮", "リョー"),
    ("長江", "チョーコー"),
    # 連語の読み
    ("そこら中", "ソコラジュー"),
    # 施術の読み
    ("施術", "セジュツ"),
    # 抱きの読み
    ("抱き", "ダキ"),
    # 茸の読み
    ("茸", "キノコ"),
    # 零の読み
    ("零", "ゼロ"),
    # 若布の読み
    ("若布", "ワカメ"),
    # 鯨のコスト調整
    ("鯨", "クジラ"),
    # Round 2: 追加のコスト調整
    ("百間", "ヒャッケン"),
    ("三百", "サンビャク"),
    ("軍兵", "グンピョー"),
    ("暇", "ヒマ"),
    ("早急", "ソーキュー"),
    ("柄", "ガラ"),
    ("擦って", "スッテ"),
]


@pytest.mark.parametrize(
    "text, expected",
    COST_ADJUSTMENTS,
    ids=[t[0] for t in COST_ADJUSTMENTS],
)
def test_cost_adjustments(text: str, expected: str) -> None:
    """コスト調整により正しい読みが選択されることを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


# ============================================================
# 形態素分割ミスの改善テスト
# report 2.3, 4.4 の ✅ 判定エントリ
# ============================================================

MORPHEME_FIXES = [
    ("天晴れ", "アッパレ"),
    ("千切れ", "チギレ"),
    ("ガス欠", "ガスケツ"),
    ("七五三", "シチゴサン"),
    ("ボロ家", "ボロヤ"),
    # Round 2: 追加の形態素分割改善
    ("弐撃決殺", "ニゲキケッサツ"),
    ("八八歩", "ハチハチフ"),
    ("伊弉諾", "イザナギ"),
    ("父っつぁん", "トッツァン"),
    ("十時間", "ジュージカン"),
    ("何時間", "ナンジカン"),
    ("何分", "ナンプン"),
    ("数分", "スーフン"),
    ("二時間", "ニジカン"),
    ("三時間", "サンジカン"),
    ("四時間", "ヨジカン"),
    ("五時間", "ゴジカン"),
    ("六時間", "ロクジカン"),
    ("七時間", "ナナジカン"),
    ("八時間", "ハチジカン"),
    ("九時間", "クジカン"),
    ("ガイド下生検", "ガイドカセーケン"),
    ("日仏", "ニチフツ"),
    ("耐風性", "タイフーセー"),
    ("上腸間膜動脈", "ジョーチョーカンマクドーミャク"),
    ("糸直刃", "イトスグハ"),
    ("細直刃", "ホソスグハ"),
    ("中直刃", "チュースグハ"),
    ("広直刃", "ヒロスグハ"),
    ("中間値", "チューカンチ"),
    ("中間種", "チューカンシュ"),
    ("主日", "シュジツ"),
    ("火叩き", "ヒハタキ"),
    ("小球性貧血", "ショーキューセーヒンケツ"),
    ("大球性貧血", "ダイキューセーヒンケツ"),
    ("正球性貧血", "セーキューセーヒンケツ"),
    ("赤球", "アカダマ"),
    ("勝負球", "ショーブダマ"),
    ("お米券", "オコメケン"),
    ("受信人払い", "ジュシンニンバライ"),
    ("秘密裏", "ヒミツリ"),
    ("栄えある", "ハエアル"),
    ("夕焼け空", "ユーヤケゾラ"),
    ("登坂車線", "トーハンシャセン"),
    ("皇陵", "コーリョー"),
    ("皇道派", "コードーハ"),
    ("居城", "キョジョー"),
    ("城内", "ジョーナイ"),
    ("誤作動", "ゴサドー"),
    ("誤検知", "ゴケンチ"),
    ("長寿命", "チョージュミョー"),
    ("五大明王", "ゴダイミョーオー"),
    ("上の方", "ウエノホー"),
    ("欠損歯", "ケッソンシ"),
    ("構造相転移", "コーゾーソーテンイ"),
    ("十二面体", "ジューニメンタイ"),
    ("彫刻刀", "チョーコクトー"),
    ("英文法", "エーブンポー"),
    ("過去問", "カコモン"),
    ("短答", "タントー"),
    ("和英辞典", "ワエージテン"),
    ("西方浄土", "サイホージョード"),
    ("憂い目", "ウイメ"),
    ("𠮟責", "シッセキ"),
    ("傍聴人", "ボーチョーニン"),
    ("受取人", "ウケトリニン"),
    ("注文書", "チューモンショ"),
    ("掌底", "ショーテー"),
    ("留置所", "リューチジョ"),
    ("留置場", "リューチジョー"),
    ("独自性", "ドクジセー"),
    ("円錐形", "エンスイケー"),
    ("楕円形", "ダエンケー"),
    ("半時間", "ハンジカン"),
    ("寂として", "セキトシテ"),
    ("就職口", "シューショクグチ"),
    ("尼さん", "アマサン"),
    ("後の世", "ノチノヨ"),
    ("微調整", "ビチョーセー"),
    ("甘味料", "カンミリョー"),
    ("人工甘味料", "ジンコーカンミリョー"),
    ("一寸先", "イッスンサキ"),
    ("作業衣", "サギョーイ"),
    ("四十七士", "シジューシチシ"),
    ("固めの杯", "カタメノサカズキ"),
    ("従妹", "イトコ"),
    ("一文無し", "イチモンナシ"),
    ("一か八か", "イチカバチカ"),
    ("一財産", "ヒトザイサン"),
    ("先見の明", "センケンノメー"),
    ("現在形", "ゲンザイケー"),
    ("竹馬の友", "チクバノトモ"),
    ("過去帳", "カコチョー"),
    ("長風呂", "ナガブロ"),
    ("お局", "オツボネ"),
    ("寝ぼけ眼", "ネボケマナコ"),
    ("亜麻色", "アマイロ"),
    ("オレンジ色", "オレンジイロ"),
    ("瑠璃色", "ルリイロ"),
    ("群青色", "グンジョーイロ"),
    ("藤色", "フジイロ"),
    ("和太鼓", "ワダイコ"),
    ("三叉神経", "サンサシンケー"),
    ("我が輩", "ワガハイ"),
    ("依頼人", "イライニン"),
    ("行商人", "ギョーショーニン"),
    ("罰が当たる", "バチガアタル"),
    ("海の幸", "ウミノサチ"),
    ("六根清浄", "ロッコンショージョー"),
    ("蛇の道は蛇", "ジャノミチワヘビ"),
    ("目覚まし時計", "メザマシドケー"),
    ("証券取引所", "ショーケントリヒキジョ"),
    ("東京証券取引所", "トーキョーショーケントリヒキジョ"),
    ("食用油", "ショクヨーアブラ"),
    ("葛根湯", "カッコントー"),
    ("あがり性", "アガリショー"),
    ("不空", "フクー"),
    ("不空訳", "フクーヤク"),
    ("数分後", "スーフンゴ"),
    ("四分後", "ヨンプンゴ"),
    ("二十分", "ニジュップン"),
    ("三十分", "サンジュップン"),
    ("四十分", "ヨンジュップン"),
    ("五十分", "ゴジュップン"),
    ("六十分", "ロクジュップン"),
    ("七十分", "ナナジュップン"),
    ("八十分", "ハチジュップン"),
    ("九十分", "キュージュップン"),
    ("寝惚けて", "ネボケテ"),
    ("細工は流々", "サイクワリューリュー"),
]


def test_nanjikan_dictionary_candidate_keeps_single_accent_phrase() -> None:
    """何時間の推定候補が1形態素・1アクセント句として辞書へ反映される。"""

    features = pyopenjtalk.run_frontend("何時間")

    assert features == [
        {
            "string": "何時間",
            "pos": "名詞",
            "pos_group1": "一般",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "何時間",
            "read": "ナンジカン",
            "pron": "ナンジカン",
            "acc": 3,
            "mora_size": 5,
            "chain_rule": "C1",
            "chain_flag": -1,
        }
    ]


def test_nampun_isolated_and_question_context_keep_expected_readings() -> None:
    """「何分」が単独では「ナンプン」、疑問文（「何分かかりますか」）では「ナンフン」と正しく読まれることを確認する。"""

    # 単独「何分」は辞書エントリ由来で「ナンプン」
    # 文中の数量疑問では NJD 処理で「ナンフン」になる
    assert pyopenjtalk.g2p("何分", kana=True) == "ナンプン"
    assert pyopenjtalk.g2p("何分かかりますか。", kana=True) == "ナンフンカカリマスカ。"


def test_yonpun_duration_candidate_wins_over_minor_place_reading() -> None:
    """時間量の「四分」が、奈良県橿原市の局地的な地名（「シブ」）に誤爆せず、「ヨンプン」と読まれることを確認する。"""

    assert pyopenjtalk.g2p("四分", kana=True) == "ヨンプン"
    assert pyopenjtalk.g2p("四分かかります。", kana=True) == "ヨンプンカカリマス。"
    assert pyopenjtalk.g2p("四分程度です。", kana=True) == "ヨンプンテードデス。"
    assert pyopenjtalk.g2p("四分以内です。", kana=True) == "ヨンプンイナイデス。"
    assert pyopenjtalk.g2p("四分前です。", kana=True) == "ヨンプンマエデス。"
    assert pyopenjtalk.g2p("四分でも待ちます。", kana=True) == "ヨンプンデモマチマス。"


@pytest.mark.parametrize(
    "surface",
    ["二時間", "三時間", "四時間", "五時間", "六時間", "七時間", "八時間", "九時間"],
)
def test_hour_duration_compounds_keep_single_dictionary_morpheme(surface: str) -> None:
    """「二時間」から「九時間」までの時間量が、分割されずに1形態素（名詞,一般）として解析されることを確認する。"""

    features = pyopenjtalk.run_frontend(surface)
    assert len(features) == 1
    assert features[0]["string"] == surface
    assert features[0]["pos_group1"] == "一般"


@pytest.mark.parametrize(
    "surface",
    ["二十分", "三十分", "四十分", "五十分", "六十分", "七十分", "八十分", "九十分"],
)
def test_tens_of_minutes_keep_single_dictionary_morpheme(surface: str) -> None:
    """「二十分」から「九十分」までの時間量が、分割されずに1形態素（名詞,一般）として解析されることを確認する。"""

    # 分割経路でも発音だけは正しくなるため、形態素数まで固定してモデル介入の再発を検出する
    features = pyopenjtalk.run_frontend(surface)
    assert len(features) == 1
    assert features[0]["string"] == surface


def test_ball_suffix_uses_productive_kyuu_reading() -> None:
    """漢語や外来語に接尾辞「球」が続く複合語（「ボール球」「樹脂球」など）では、「キュー」と発音されることを確認する。"""

    assert pyopenjtalk.g2p("ボール球", kana=True) == "ボールキュー"
    assert pyopenjtalk.g2p("樹脂球", kana=True) == "ジュシキュー"
    assert pyopenjtalk.g2p("練習用球", kana=True) == "レンシューヨーキュー"


def test_ball_rule_does_not_reach_general_noun_or_counter() -> None:
    """接尾辞以外の単独の「球」（「高い球」など）では「タマ」、助数詞の「球」（「七球目」など）では「キュー」と正しく読み分けられることを確認する。"""

    assert pyopenjtalk.g2p("高い球", kana=True) == "タカイタマ"
    assert pyopenjtalk.g2p("速い球", kana=True) == "ハヤイタマ"
    assert pyopenjtalk.g2p("七球目", kana=True) == "ナナキューメ"


def test_ball_suffix_keeps_rendaku_after_japanese_inflection() -> None:
    """動詞の連用形に続く「球」（「捨て球」「釣り球」など）では、連濁して「ダマ」と読まれることを確認する。"""

    assert pyopenjtalk.g2p("捨て球", kana=True) == "ステダマ"
    assert pyopenjtalk.g2p("釣り球", kana=True) == "ツリダマ"
    assert pyopenjtalk.g2p("送り球", kana=True) == "オクリダマ"


def test_ball_suffix_keeps_lexicalized_dama_compounds() -> None:
    """「決め球」「隠し球」「勝負球」など、個別の辞書登録や文脈によって「ダマ」と読む複合語が意図通り発音されることを確認する。"""

    assert pyopenjtalk.g2p("決め球", kana=True) == "キメダマ"
    assert pyopenjtalk.g2p("隠し球", kana=True) == "カクシダマ"
    assert pyopenjtalk.g2p("見せ球", kana=True) == "ミセダマ"
    assert pyopenjtalk.g2p("勝負球", kana=True) == "ショーブダマ"


def test_compound_final_fusoku_uses_rendaku_reading() -> None:
    """「資金不足」「睡眠不足」のように名詞に続く複合語の「不足」は、連濁して「ブソク」と読まれることを確認する。"""

    assert pyopenjtalk.g2p("資金不足です。", kana=True) == "シキンブソクデス。"
    assert pyopenjtalk.g2p("睡眠不足です。", kana=True) == "スイミンブソクデス。"
    assert pyopenjtalk.g2p("運動不足です。", kana=True) == "ウンドーブソクデス。"
    assert pyopenjtalk.g2p("労働力不足です。", kana=True) == "ロードーリョクブソクデス。"


def test_independent_fusoku_keeps_unvoiced_reading() -> None:
    """「情報が不足する」のように単独の動詞句として使われる場合や、「不足分」のような語では、清音の「フソク」という読みが維持されることを確認する。"""

    assert pyopenjtalk.g2p("情報が不足しています。", kana=True) == "ジョーホーガフソクシテイマス。"
    assert pyopenjtalk.g2p("不足分を補います。", kana=True) == "フソクブンヲオギナイマス。"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("三分の一です。", "サンブンノイチデス。"),
        ("３分の１です。", "サンブンノイチデス。"),
        ("五分の一です。", "ゴブンノイチデス。"),
    ],
)
def test_fraction_denominator_uses_bun_reading(text: str, expected: str) -> None:
    """「3分の1」のように数値と「の」に挟まれた分数の「分」は、時間量の「フン」ではなく「ブン」と読まれることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "surface", "expected_mora_size"),
    [
        ("三分の一です。", "三分", 4),
        ("五分の一です。", "五分", 3),
        ("３分の１です。", "分", 2),
    ],
)
def test_fraction_denominator_mora_size(
    text: str,
    surface: str,
    expected_mora_size: int,
) -> None:
    """分母の読みを「ブン」へ補正した後も、形態素のモーラ数が正しく保たれることを確認する。"""

    features = pyopenjtalk.run_frontend(text)
    denominator = next(feature for feature in features if feature["string"] == surface)

    assert denominator["mora_size"] == expected_mora_size


def test_numeral_reading_correction_can_be_disabled() -> None:
    """数詞の読み補正を無効にした場合（modify_numeral_reading=False）は、OpenJTalk 本来の読みとアクセントが維持されることを確認する。"""

    text = "三分の一と〇〇"
    expected_pronunciations = ["サンブ", "ノ", "イチ", "ト", "、", "、"]

    features = pyopenjtalk.run_frontend(text, use_vanilla=True)
    detailed_features, _ = pyopenjtalk.run_frontend_detailed(text, use_vanilla=True)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    low_level_features = jtalk.run_frontend(text, modify_numeral_reading=False)

    assert [feature["pron"] for feature in features] == expected_pronunciations
    assert detailed_features == features
    assert low_level_features == features
    assert pyopenjtalk.g2p("三分の一", kana=True, use_vanilla=True) == "サンブノイチ"
    assert pyopenjtalk.g2p_mapping("三分の一", use_vanilla=True)[0]["phonemes"] == [
        "s",
        "a",
        "N",
        "b",
        "u",
    ]


def test_non_fraction_contexts_do_not_use_bun_reading() -> None:
    """「五分の休憩」のように直後が数値でない文脈では、分数の「ブン」に誤補正されないことを確認する。"""

    assert pyopenjtalk.g2p("五分の休憩です。", kana=True) == "ゴブノキューケーデス。"
    assert pyopenjtalk.g2p("五分後です。", kana=True) == "ゴブゴデス。"


def test_repeated_placeholder_circle_uses_maru_reading() -> None:
    """2文字以上連続する伏字の「〇」（「〇〇町」など）は、数値の「レイ」ではなく「マル」と読まれることを確認する。"""

    assert pyopenjtalk.g2p("住所は〇〇町です。", kana=True) == "ジューショワマルマルマチデス。"
    assert pyopenjtalk.g2p("氏名は〇〇〇です。", kana=True) == "シメーワマルマルマルデス。"


@pytest.mark.parametrize(
    ("text", "expected_phrase"),
    [
        # 後ろに何も続かない「〇〇」は MeCab が記号として返すが、名詞として1つのアクセント句にまとまる
        ("〇〇", ("4", "4")),
        ("〇〇です。", ("6", "2")),
        ("住所は〇〇町です。", ("8", "4")),
        ("氏名は〇〇〇です。", ("8", "4")),
    ],
)
def test_repeated_placeholder_circle_accent_follows_njd_chaining(
    text: str, expected_phrase: tuple[str, str]
) -> None:
    """連続する「〇」のアクセント句が、NJD の結合どおり「マル＼マル」「マルマル＼マチ」の形になることを確認する。"""

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    # 「マ」の音素が属するアクセント句の、モーラ数と核の位置
    first_ma = next(label for label in labels if "-m+a=" in label and "/F:" in label)
    phrase = re.search(r"/F:(\w+)_(\w+)", first_ma)

    assert phrase is not None
    assert phrase.groups() == expected_phrase


def test_single_circle_keeps_numeric_reading() -> None:
    """「〇円」のように単独で現れる「〇」は、伏字ではなく数値として「レー」と読まれることを確認する。"""

    assert pyopenjtalk.g2p("〇円です。", kana=True) == "レーエンデス。"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("五十", "ゴジュー"),
        ("五十です", "ゴジューデス"),
        ("二十から五十", "ニジューカラゴジュー"),
        ("十九", "ジューキュー"),
        ("十九の春", "ジューキューノハル"),
        ("七百", "ナナヒャク"),
        ("七百です", "ナナヒャクデス"),
        ("四十九", "ヨンジューキュー"),
        ("四十九です", "ヨンジューキューデス"),
        ("四十九日", "シジュークニチ"),
        ("十九歳", "ジューキューサイ"),
        ("百万円", "ヒャクマンエン"),
        ("七百駅", "シチヒャクエキ"),
        ("十九町", "ジュックチョー"),
        ("四十九町", "シジュクチョー"),
        ("四十九駅", "シジュクエキ"),
    ],
)
def test_numerals_and_explicit_place_names(text: str, expected: str) -> None:
    """「十九」は数詞で読み、「十九町」は地名の「ジュックチョー」で読む。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ⅺ", "ジューイチ"),
        ("ⅺ", "ジューイチ"),
        ("Ⅻ", "ジューニ"),
        ("ⅻ", "ジューニ"),
        ("第Ⅺ章", "ダイジューイチショー"),
        ("第ⅺ章", "ダイジューイチショー"),
        ("Ⅻ型", "ジューニガタ"),
        ("ⅻ型", "ジューニガタ"),
        ("ロッキーⅣ", "ロッキーヨン"),
    ],
)
def test_roman_numerals_eleven_and_twelve(text: str, expected: str) -> None:
    """「Ⅺ」「Ⅻ」の読みを出力し、既存の「Ⅳ」の読みも保つ。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected
    assert "{unk}" not in pyopenjtalk.g2p_prosody(text)


def test_chosakuken_keeps_natural_geminated_pronunciation() -> None:
    """「著作権」の発音において、TTS でより自然な促音化された「チョサッケン」という発音が維持されることを確認する。"""

    assert (
        pyopenjtalk.g2p("今日は著作権を学びます。", kana=True)
        == "キョーワチョサッケンヲマナビマス。"
    )


_DURATION_MORPHEME_SURFACES: frozenset[str] = frozenset(
    {
        "十時間",
        "何時間",
        "何分",
        "数分",
        "数分後",
        "四分後",
        "二時間",
        "三時間",
        "四時間",
        "五時間",
        "六時間",
        "七時間",
        "八時間",
        "九時間",
        "二十分",
        "三十分",
        "四十分",
        "五十分",
        "六十分",
        "七十分",
        "八十分",
        "九十分",
    }
)
_DURATION_MORPHEME_SENTENCE_TEMPLATES: tuple[str, ...] = (
    "{surface}かかります。",
    "あと{surface}です。",
    "約{surface}待ってください。",
    "{surface}程度かかります。",
    "{surface}後に届きます。",
)


def _duration_morpheme_fix_entries() -> tuple[tuple[str, str], ...]:
    """2026/08/11 時間量修正エントリだけを MORPHEME_FIXES から抽出する。"""

    return tuple(
        (surface, expected_kana)
        for surface, expected_kana in MORPHEME_FIXES
        if surface in _DURATION_MORPHEME_SURFACES
    )


@pytest.mark.parametrize(("surface", "expected_isolated_kana"), _duration_morpheme_fix_entries())
@pytest.mark.parametrize("template", _DURATION_MORPHEME_SENTENCE_TEMPLATES)
def test_duration_morpheme_fixes_embedded_in_sentences_match_vanilla_baseline(
    surface: str,
    expected_isolated_kana: str,
    template: str,
) -> None:
    """時間量の辞書エントリを様々な文型へ埋め込んだ場合でも、意図通りの読みが正しく維持されることを確認する。"""

    if "{surface}後に" in template and (
        surface.endswith("時間") is True or surface.endswith("後") is True
    ):
        pytest.skip("時間単位または「後」終端の表層は「後」付き文型の対象外")

    text = template.format(surface=surface)
    vanilla = pyopenjtalk.g2p(text, kana=True, use_vanilla=True)
    actual = pyopenjtalk.g2p(text, kana=True)
    assert actual == vanilla
    # 「何分」は NJD で「ナンフン」となり、「後」終端の表層は単独読みの包含検査に適さない
    if surface != "何分" and surface.endswith("後") is False:
        assert expected_isolated_kana in actual


@pytest.mark.parametrize(
    "text, expected",
    MORPHEME_FIXES,
    ids=[t[0] for t in MORPHEME_FIXES],
)
def test_morpheme_fixes(text: str, expected: str) -> None:
    """形態素分割が改善され正しい読みが得られることを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


# ============================================================
# 連濁・発音形の差異テスト
# report 4.5 の ✅ 判定エントリ
# ============================================================

RENDAKU_FIXES = [
    ("茗荷谷", "ミョーガダニ"),
    ("三票", "サンピョー"),
]


@pytest.mark.parametrize(
    "text, expected",
    RENDAKU_FIXES,
    ids=[t[0] for t in RENDAKU_FIXES],
)
def test_rendaku_fixes(text: str, expected: str) -> None:
    """連濁・発音形の差異が修正されていることを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


# ============================================================
# デグレッション防止テスト
# 辞書修正により一般的な単語の読みが壊れていないことを確認する
# 日常会話・学校・医療・会社・家族・食べ物・動物・色・天気・自然・
# 身体・感情・職業・数詞・時間・地名・文・熟語・形容詞・動詞・
# 複合語・外来語・辞書修正関連の周辺語・敬称付き人名・姓名文脈から収録
# ============================================================

DEGRESSION_CHECKS = [
    # === 日常会話 ===
    ("おはようございます", "オハヨーゴザイマス"),
    ("こんにちは", "コンニチワ"),
    ("こんばんは", "コンバンワ"),
    ("おやすみなさい", "オヤスミナサイ"),
    ("ありがとうございます", "アリガトーゴザイマス"),
    ("すみません", "スミマセン"),
    ("お疲れ様です", "オツカレサマデス"),
    ("いただきます", "イタダキマス"),
    # === 学校 ===
    ("学校", "ガッコー"),
    ("先生", "センセー"),
    ("生徒", "セート"),
    ("教室", "キョーシツ"),
    ("教科書", "キョーカショ"),
    # === 医療 ===
    ("病院", "ビョーイン"),
    ("医者", "イシャ"),
    ("看護師", "カンゴシ"),
    ("薬", "クスリ"),
    ("手術", "シュジュツ"),
    # === 会社 ===
    ("会社", "カイシャ"),
    ("社長", "シャチョー"),
    ("部長", "ブチョー"),
    ("会議", "カイギ"),
    ("書類", "ショルイ"),
    # === 家族 ===
    ("父", "チチ"),
    ("母", "ハハ"),
    ("兄", "アニ"),
    ("姉", "アネ"),
    ("弟", "オトート"),
    ("妹", "イモート"),
    ("祖父", "ソフ"),
    ("祖母", "ソボ"),
    # === 食べ物 ===
    ("御飯", "ゴハン"),
    ("味噌汁", "ミソシル"),
    ("豆腐", "トーフ"),
    ("納豆", "ナットー"),
    ("醤油", "ショーユ"),
    ("砂糖", "サトー"),
    ("塩", "シオ"),
    ("天ぷら", "テンプラ"),
    ("寿司", "スシ"),
    ("蕎麦", "ソバ"),
    ("うどん", "ウドン"),
    ("ラーメン", "ラーメン"),
    ("カレー", "カレー"),
    ("林檎", "リンゴ"),
    ("蜜柑", "ミカン"),
    ("葡萄", "ブドー"),
    ("苺", "イチゴ"),
    ("桃", "モモ"),
    ("梨", "ナシ"),
    ("西瓜", "スイカ"),
    ("柿", "カキ"),
    ("牛肉", "ギューニク"),
    # === 動物 ===
    ("犬", "イヌ"),
    ("猫", "ネコ"),
    ("馬", "ウマ"),
    ("牛", "ウシ"),
    ("豚", "ブタ"),
    ("鶏", "ニワトリ"),
    ("羊", "ヒツジ"),
    ("兎", "ウサギ"),
    ("鹿", "シカ"),
    ("熊", "クマ"),
    ("鷹", "タカ"),
    ("烏", "カラス"),
    ("雀", "スズメ"),
    ("蛙", "カエル"),
    ("蛇", "ヘビ"),
    ("蟹", "カニ"),
    ("鯛", "タイ"),
    ("鮭", "サケ"),
    ("蝶", "チョー"),
    ("蜂", "ハチ"),
    # === 色 ===
    ("赤", "アカ"),
    ("青", "アオ"),
    ("緑", "ミドリ"),
    ("紫", "ムラサキ"),
    ("白", "シロ"),
    ("黒", "クロ"),
    ("橙", "ダイダイ"),
    ("茶色", "チャイロ"),
    ("灰色", "ハイイロ"),
    ("水色", "ミズイロ"),
    ("桃色", "モモイロ"),
    ("金色", "キンイロ"),
    ("銀色", "ギンイロ"),
    # === 天気・自然 ===
    ("晴れ", "ハレ"),
    ("曇り", "クモリ"),
    ("雨", "アメ"),
    ("雪", "ユキ"),
    ("嵐", "アラシ"),
    ("台風", "タイフー"),
    ("雷", "カミナリ"),
    ("虹", "ニジ"),
    ("霧", "キリ"),
    ("霜", "シモ"),
    ("山", "ヤマ"),
    ("川", "カワ"),
    ("海", "ウミ"),
    ("湖", "ミズウミ"),
    ("森", "モリ"),
    ("丘", "オカ"),
    ("谷", "タニ"),
    ("滝", "タキ"),
    ("島", "シマ"),
    ("半島", "ハントー"),
    # === 身体 ===
    ("頭", "アタマ"),
    ("顔", "カオ"),
    ("目", "メ"),
    ("耳", "ミミ"),
    ("鼻", "ハナ"),
    ("口", "クチ"),
    ("歯", "ハ"),
    ("首", "クビ"),
    ("肩", "カタ"),
    ("腕", "ウデ"),
    ("手", "テ"),
    ("指", "ユビ"),
    ("胸", "ムネ"),
    ("腹", "ハラ"),
    ("背中", "セナカ"),
    ("腰", "コシ"),
    ("膝", "ヒザ"),
    ("足", "アシ"),
    ("爪", "ツメ"),
    # === 感情 ===
    ("嬉しい", "ウレシイ"),
    ("悲しい", "カナシイ"),
    ("楽しい", "タノシイ"),
    ("恐怖", "キョーフ"),
    ("驚き", "オドロキ"),
    ("寂しい", "サビシイ"),
    # === 職業 ===
    ("教師", "キョーシ"),
    ("医師", "イシ"),
    ("弁護士", "ベンゴシ"),
    ("警察官", "ケーサツカン"),
    ("消防士", "ショーボーシ"),
    ("運転手", "ウンテンシュ"),
    ("料理人", "リョーリニン"),
    ("農家", "ノーカ"),
    ("漁師", "リョーシ"),
    ("大工", "ダイク"),
    ("画家", "ガカ"),
    ("作家", "サッカ"),
    ("歌手", "カシュ"),
    ("俳優", "ハイユー"),
    # === 数詞 ===
    ("一", "イチ"),
    ("二", "ニ"),
    ("三", "サン"),
    ("四", "ヨン"),
    ("五", "ゴ"),
    ("六", "ロク"),
    ("七", "ナナ"),
    ("八", "ハチ"),
    ("九", "キュー"),
    ("十", "ジュー"),
    ("百", "ヒャク"),
    ("千", "セン"),
    ("万", "マン"),
    ("億", "オク"),
    # === 時間 ===
    ("今日", "キョー"),
    ("明日", "アシタ"),
    ("昨日", "キノー"),
    ("今朝", "ケサ"),
    ("今晩", "コンバン"),
    ("来年", "ライネン"),
    ("去年", "キョネン"),
    ("今年", "コトシ"),
    ("月曜日", "ゲツヨービ"),
    ("火曜日", "カヨービ"),
    ("水曜日", "スイヨービ"),
    ("木曜日", "モクヨービ"),
    ("金曜日", "キンヨービ"),
    ("土曜日", "ドヨービ"),
    ("日曜日", "ニチヨービ"),
    # === 地名 ===
    ("東京", "トーキョー"),
    ("大阪", "オーサカ"),
    ("京都", "キョート"),
    ("北海道", "ホッカイドー"),
    ("沖縄", "オキナワ"),
    ("名古屋", "ナゴヤ"),
    ("横浜", "ヨコハマ"),
    ("富士山", "フジサン"),
    ("日本海", "ニホンカイ"),
    ("太平洋", "タイヘーヨー"),
    # === 文 ===
    ("今日はいい天気ですね", "キョーワイイテンキデスネ"),
    ("明日は雨が降るでしょう", "アシタワアメガフルデショー"),
    ("東京タワーに行きました", "トーキョータワーニイキマシタ"),
    ("電車が遅れています", "デンシャガオクレテイマス"),
    ("新幹線で大阪に行く", "シンカンセンデオーサカニイク"),
    ("桜の花が咲いている", "サクラノハナガサイテイル"),
    ("夏休みに海に行った", "ナツヤスミニウミニイッタ"),
    ("宿題を忘れました", "シュクダイヲワスレマシタ"),
    ("来週の月曜日に会議がある", "ライシューノゲツヨービニカイギガアル"),
    ("最近忙しくて大変です", "サイキンイソガシクテタイヘンデス"),
    # === 熟語 ===
    ("経済", "ケーザイ"),
    ("政治", "セージ"),
    ("文化", "ブンカ"),
    ("歴史", "レキシ"),
    ("科学", "カガク"),
    ("技術", "ギジュツ"),
    ("芸術", "ゲージュツ"),
    ("哲学", "テツガク"),
    ("自由", "ジユー"),
    ("平和", "ヘーワ"),
    ("幸福", "コーフク"),
    ("努力", "ドリョク"),
    ("根性", "コンジョー"),
    ("忍耐", "ニンタイ"),
    ("勇気", "ユーキ"),
    # === 形容詞 ===
    ("大きい", "オーキイ"),
    ("小さい", "チーサイ"),
    ("長い", "ナガイ"),
    ("短い", "ミジカイ"),
    ("高い", "タカイ"),
    ("低い", "ヒクイ"),
    ("広い", "ヒロイ"),
    ("狭い", "セマイ"),
    ("速い", "ハヤイ"),
    ("遅い", "オソイ"),
    ("強い", "ツヨイ"),
    ("弱い", "ヨワイ"),
    ("明るい", "アカルイ"),
    ("暗い", "クライ"),
    ("重い", "オモイ"),
    ("軽い", "カルイ"),
    # === 動詞 ===
    ("食べる", "タベル"),
    ("飲む", "ノム"),
    ("走る", "ハシル"),
    ("歩く", "アルク"),
    ("読む", "ヨム"),
    ("書く", "カク"),
    ("聞く", "キク"),
    ("話す", "ハナス"),
    ("見る", "ミル"),
    ("買う", "カウ"),
    ("売る", "ウル"),
    ("作る", "ツクル"),
    ("使う", "ツカウ"),
    ("送る", "オクル"),
    ("届く", "トドク"),
    ("届ける", "トドケル"),
    # === 複合語 ===
    ("新聞紙", "シンブンシ"),
    ("図書館", "トショカン"),
    ("美術館", "ビジュツカン"),
    ("博物館", "ハクブツカン"),
    ("動物園", "ドーブツエン"),
    ("水族館", "スイゾクカン"),
    ("運動会", "ウンドーカイ"),
    ("修学旅行", "シューガクリョコー"),
    ("卒業式", "ソツギョーシキ"),
    ("入学式", "ニューガクシキ"),
    # === 外来語 ===
    ("コンピューター", "コンピューター"),
    ("インターネット", "インターネット"),
    ("テレビ", "テレビ"),
    ("ラジオ", "ラジオ"),
    # === 辞書修正関連の周辺語デグレ確認 ===
    ("柵を設ける", "サクヲモーケル"),
    ("柵越え", "サクゴエ"),
    ("殿様", "トノサマ"),
    ("殿堂", "デンドー"),
    ("宮殿", "キューデン"),
    ("蔵に入る", "クラニハイル"),
    ("蔵書", "ゾーショ"),
    ("冷蔵", "レーゾー"),
    ("冷蔵庫", "レーゾーコ"),
    ("蔵元", "クラモト"),
    ("蓮の花", "ハスノハナ"),
    ("蓮根", "レンコン"),
    ("百花繚乱", "ヒャッカリョーラン"),
    ("百合", "ユリ"),
    ("百万", "ヒャクマン"),
    ("百年", "ヒャクネン"),
    ("百科事典", "ヒャッカジテン"),
    ("百日", "ヒャクニチ"),
    ("百人", "ヒャクニン"),
    ("一枚", "イチマイ"),
    ("一回", "イッカイ"),
    ("一人", "ヒトリ"),
    ("一度", "イチド"),
    ("一人で歩く", "ヒトリデアルク"),
    ("七月", "シチガツ"),
    ("七人", "シチニン"),
    ("七回", "ナナカイ"),
    ("七五三のお祝い", "シチゴサンノオイワイ"),
    ("七五三参り", "シチゴサンマイリ"),
    ("七五三掛", "シメカケ"),
    ("七五三掛さん", "シメカケサン"),
    ("火傷した", "ヤケドシタ"),
    ("長江を渡る", "チョーコーヲワタル"),
    ("紫蘇の葉", "シソノハ"),
    ("殺陣師", "タテシ"),
    ("不治の病", "フジノヤマイ"),
    ("一昨年のこと", "オトトシノコト"),
    ("一昨日", "オトトイ"),
    ("頭数を揃える", "アタマカズヲソロエル"),
    ("私事で恐縮ですが", "ワタクシゴトデキョーシュクデスガ"),
    ("目前に迫る", "モクゼンニセマル"),
    ("目の前", "メノマエ"),
    ("施術を受ける", "セジュツヲウケル"),
    ("施設", "シセツ"),
    ("施行", "シコー"),
    ("茸を採る", "キノコヲトル"),
    ("松茸", "マツタケ"),
    ("三百円", "サンビャクエン"),
    ("仇名", "アダナ"),
    ("敵討ち", "カタキウチ"),
    ("家庭", "カテー"),
    ("家族", "カゾク"),
    ("家具", "カグ"),
    ("家賃", "ヤチン"),
    ("馬車", "バシャ"),
    ("馬力", "バリキ"),
    ("長男", "チョーナン"),
    ("長期", "チョーキ"),
    ("紫外線", "シガイセン"),
    ("日本語", "ニホンゴ"),
    # === 「蓮」のコスト調整関連（敬称が続く文脈での人名読みへの切り替え） ===
    ("蓮", "ハス"),
    ("蓮くん", "レンクン"),
    ("蓮さん", "レンサン"),
    ("蓮ちゃん", "レンチャン"),
    ("山田蓮", "ヤマダレン"),
    ("蓮が咲いた", "ハスガサイタ"),
    ("亮さん", "リョーサン"),
    ("百花さん", "モモカサン"),
    ("長江さん", "ナガエサン"),
    ("洪さん", "ヒロシサン"),
    ("山田洪", "ヤマダヒロシ"),
    ("山田一人", "ヤマダカズト"),
    ("山田三百", "ヤマダミツオ"),
    ("蓮華", "レンゲ"),
    ("蓮池", "ハスイケ"),
    # === 芳し* 活用形統一 ===
    ("芳しい", "カンバシイ"),
    ("芳しく", "カンバシク"),
    ("芳しかった", "カンバシカッタ"),
    ("芳しけれ", "カンバシケレ"),
    ("芳しさ", "カンバシサ"),
    # === 擦り 周辺語 ===
    ("擦り切れ", "スリキレ"),
    ("擦り合わせ", "スリアワセ"),
    ("擦り寄る", "スリヨル"),
    ("擦り傷", "スリキズ"),
    ("擦り減る", "スリヘル"),
    # === 施術 複合語 ===
    ("施術台", "セジュツダイ"),
    ("施術者", "セジュツシャ"),
    # === 醸酒 ===
    ("醸酒", "ジョーシュ"),
    ("醸造", "ジョーゾー"),
    # === 数詞の組み合わせ ===
    ("三十", "サンジュー"),
    ("五百", "ゴヒャク"),
    ("千二百", "センニヒャク"),
    ("二千", "ニセン"),
    ("三万", "サンマン"),
    # === 地名の読み分け ===
    ("日向", "ヒナタ"),
    ("日向ぼっこ", "ヒナタボッコ"),
    ("日向市", "ヒューガシ"),
    # === 蓮 追加文脈 ===
    ("蓮田", "ハスダ"),
    # === 百花 人名文脈 ===
    ("百花さん", "モモカサン"),
    # === 長江 人名文脈 ===
    ("長江さん", "ナガエサン"),
    # === 一人前 ===
    ("一人前", "イチニンマエ"),
    # === 戯れ 活用形 ===
    ("戯れる", "タワムレル"),
    ("戯れた", "タワムレタ"),
    # === 寝惚け 活用形 ===
    ("寝惚ける", "ネボケル"),
    ("寝惚けた", "ネボケタ"),
    # === 画策・愁える・敗着 ===
    ("画策する", "カクサクスル"),
    ("愁える", "ウレエル"),
    ("敗着", "ハイチャク"),
    # === 百の周辺数詞（コスト調整で百・三百等を変更しているため） ===
    ("百日紅", "サルスベリ"),
    ("百人一首", "ヒャクニンイッシュ"),
    ("百姓", "ヒャクショー"),
    ("百貨店", "ヒャッカテン"),
    ("百発百中", "ヒャッパツヒャクチュー"),
    ("二百", "ニヒャク"),
    ("八百屋", "ヤオヤ"),
    # === 火傷 活用形・複合語 ===
    ("火傷する", "ヤケドスル"),
    ("火傷を負う", "ヤケドヲオウ"),
    # === 柵 複合語 ===
    ("鉄柵", "テッサク"),
    ("防護柵", "ボーゴサク"),
    # === 施術 活用形・複合語 ===
    ("施術する", "セジュツスル"),
    ("施術室", "セジュツシツ"),
    # === 抱き 活用形 ===
    ("抱きしめる", "ダキシメル"),
    ("抱き上げる", "ダキアゲル"),
    ("抱きつく", "ダキツク"),
    # === 暇 複合語 ===
    ("暇つぶし", "ヒマツブシ"),
    ("お暇", "オヒマ"),
    # === 柄 複合語 ===
    ("柄が悪い", "ガラガワルイ"),
    ("花柄", "ハナガラ"),
    # === 早急 活用形 ===
    ("早急に", "ソーキューニ"),
    # === 一分 複合語 ===
    ("一分", "イップン"),
    ("一分間", "イップンカン"),
    ("十一分", "ジューイップン"),
    # === 殿 複合語 ===
    ("殿方", "トノガタ"),
    ("お殿様", "オトノサマ"),
    # === 蓮根 複合語 ===
    ("蓮根を食べる", "レンコンヲタベル"),
    # === 奉る 活用形 ===
    ("奉る", "タテマツル"),
    # === 最高値 複合語 ===
    ("統計上の最高値", "トーケージョーノサイコーチ"),
    # === 艶やか 活用形 ===
    ("艶やかな光沢", "ツヤヤカナコータク"),
    # === 擦 活用形 ===
    ("擦る", "スル"),
    ("擦れ", "スレ"),
    ("擦った", "スッタ"),
    # === 戯れ 追加活用形 ===
    ("戯れに", "タワムレニ"),
    # === 「温」の活用形の統一（「ヌル」を優先し、「アタタカ」系と「ヌクト」系は変えない） ===
    ("温い", "ヌルイ"),
    ("温かった", "ヌルカッタ"),
    ("温い風呂", "ヌルイフロ"),
    ("温くなる", "ヌルクナル"),
    ("温かい", "アタタカイ"),
    ("温かく", "アタタカク"),
    ("温とい", "ヌクトイ"),
]


@pytest.mark.parametrize(
    "text, expected",
    DEGRESSION_CHECKS,
    ids=[t[0] for t in DEGRESSION_CHECKS],
)
def test_degression_checks(text: str, expected: str) -> None:
    """辞書修正が一般的な単語の読みに悪影響を与えていないことを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("黄色人種", "オーショクジンシュ"),
        ("黄色ブドウ球菌", "オーショクブドーキューキン"),
        ("黄色ぶどう球菌", "オーショクブドーキューキン"),
        ("黄色葡萄球菌", "オーショクブドーキューキン"),
        ("黄色新聞", "オーショクシンブン"),
        ("黄色組合", "オーショククミアイ"),
        ("黄色火薬", "オーショクカヤク"),
        ("黄色植物", "オーショクショクブツ"),
        ("黄色矮星", "オーショクワイセー"),
        ("黄色わい星", "オーショクワイセー"),
        ("黄色血滷塩", "オーショクケツロエン"),
        ("黄色靭帯骨化症", "オーショクジンタイコッカショー"),
        ("黄色靱帯骨化症", "オーショクジンタイコッカショー"),
        ("黄色じん帯骨化症", "オーショクジンタイコッカショー"),
        ("黄色骨髄", "オーショクコツズイ"),
        ("黄色脂肪症", "オーショクシボーショー"),
        ("灰黄色", "カイコーショク"),
        ("橙黄色", "トーコーショク"),
        ("退黄色", "タイコーショク"),
        ("褪黄色", "タイコーショク"),
        ("黄色調", "オーショクチョー"),
        ("黄色爪症候群", "オーショクソーショーコーグン"),
        ("黄色巨星", "オーショクキョセー"),
        ("黄色酸化鉄", "オーショクサンカテツ"),
        ("微黄色", "ビオーショク"),
        ("黄色顔料", "オーショクガンリョー"),
        ("黄色蛍光灯", "キイロケーコートー"),
        ("黄色酵素", "オーショクコーソ"),
        ("黄色素胞", "オーシキソホー"),
        ("黄色素胞がない。", "オーシキソホーガナイ。"),
        (
            "黒色素胞がなく黄色素胞が発達していないため、体は白い。",
            "クロシキソホーガナクオーシキソホーガハッタツシテイナイタメ、カラダワシロイ。",
        ),
        ("黄色人参", "キイロニンジン"),
        ("食用黄色4号", "ショクヨーキイロヨンゴー"),
        ("黄色4号", "キイロヨンゴー"),
        ("黄色五号", "キイロゴゴー"),
        ("黄色警報", "キイロケーホー"),
        ("黄色作戦", "オーショクサクセン"),
        ("黄色回転灯", "キイロカイテントー"),
        ("黄色点滅灯", "キイロテンメツトー"),
        ("黄色点滅", "キイロテンメツ"),
        ("緑黄色野菜", "リョクオーショクヤサイ"),
        ("黄色信号", "キーロシンゴー"),
    ],
)
def test_yellow_compound_readings(text: str, expected: str) -> None:
    """「黄色人種」の「オーショク」など、複合語ごとの読みを保つ。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("黄色", "k i i r o"),
        ("黄色の花", "k i i r o n o h a n a"),
        ("黄色が好き", "k i i r o g a s U k i"),
        ("真っ黄色", "m a cl k i i r o"),
        ("（上図黄色部）", "j o o z u k i i r o b u"),
        ("黄色勢", "k i i r o z e e"),
        ("黄色箱", "k i i r o b a k o"),
        ("黄色玉", "k i i r o d a m a"),
    ],
)
def test_yellow_standalone_phonemes(text: str, expected: str) -> None:
    """「黄色の花」は「キイロノハナ」と読み、音素列でも音読みを選ばない。"""

    assert pyopenjtalk.g2p(text) == expected


@pytest.mark.parametrize(
    "text, surface",
    [
        (
            "スィッチを左にすると、ベッドサイドランプが暖かい黄色光点灯します、スィッチを右にすると、ベッドサイドランプが自然な白光点灯します、スィッチを真ん中にすると、OFFにします。",
            "点灯",
        ),
        (
            "思春期にニキビが出来ることは皮脂分泌が活発になりアクネ桿菌の餌が多くなり、バランスがくずれ黄色ブドウ球菌も多くなり化膿する機会が増えます。",
            "黄色ブドウ球菌",
        ),
    ],
)
def test_yellow_compound_phrase_boundaries(text: str, surface: str) -> None:
    """「黄色光」の後の「点灯」や、「くずれ」の後の「黄色ブドウ球菌」の句を保つ。"""

    features = pyopenjtalk.run_frontend(text)
    target = next(feature for feature in features if feature["string"] == surface)
    assert target["chain_flag"] == 0
    if surface == "黄色ブドウ球菌":
        previous = next(feature for feature in features if feature["string"] == "くずれ")
        assert previous["pos"] == "動詞"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("宗教色の強い団体", "シューキョーショクノツヨイダンタイ"),
        ("同系色の服を選ぶ", "ドーケーショクノフクヲエラブ"),
        ("類似色を選ぶ", "ルイジショクヲエラブ"),
        ("蛍光色のペン", "ケーコーショクノペン"),
        ("隠蔽色を持つ動物", "インペーショクヲモツドーブツ"),
        ("セピア色の街並み", "セピアイロノマチナミ"),
        ("ベージュ色のコート", "ベージュイロノコート"),
        ("夕焼け色の雲", "ユーヤケイロノクモ"),
    ],
)
def test_color_suffix_compounds(text: str, expected: str) -> None:
    """
    傾向を表す「宗教色」や色彩の用語の「同系色」は「ショク」と読み、「セピア色」のような色の名前に付く接尾辞の「色」は「イロ」のまま読むことを確認する。
    接尾辞の「色」は色の名前に付く「イロ」の用例が多いので、「ショク」と読む語は1語の辞書の行で選ばせている。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("右腕を高く上げる", "ミギウデヲタカクアゲル"),
        ("社長の右腕として働く", "シャチョーノミギウデトシテハタラク"),
        ("右腕投手が先発する", "ウワントーシュガセンパツスル"),
        ("左腕投手が先発する", "サワントーシュガセンパツスル"),
    ],
)
def test_right_arm_reading(text: str, expected: str) -> None:
    """
    体の部位や信頼する補佐役を指す単独の「右腕」は「ミギウデ」と読み、野球の「右腕投手」は「ウワントーシュ」と読むことを確認する。
    単独の「右腕」は「ミギウデ」の用例が野球の投手を指す「ウワン」より大幅に多いので、単独の語の既定を「ミギウデ」にしている。
    「右腕投手」は「ウワン」と読む複合語として1語の辞書の行で選ばせ、対になる「左腕投手」の「サワントーシュ」と同じ読み方にする。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("米大統領が来日した", "ベーダイトーリョーガライニチシタ"),
        ("米ドルで支払う", "ベードルデシハラウ"),
        ("日米の首脳が会談した", "ニチベーノシュノーガカイダンシタ"),
        ("米国の大統領", "ベーコクノダイトーリョー"),
        ("米軍の基地", "ベーグンノキチ"),
        ("米を炊く", "コメヲタク"),
    ],
)
def test_us_abbreviation_compounds(text: str, expected: str) -> None:
    """
    アメリカを表す「米」の複合語の「米大統領」「米ドル」を「ベーダイトーリョー」「ベードル」と読み、「日米」を「ニチベー」と読むことを確認する。
    「米大統領」と「米ドル」は「アメリカダイトーリョー」「アメリカドル」という読みで辞書に登録されていたので、行の読みを書き換えている。
    「日米」は辞書の行のコストが高すぎて「日」と接尾辞の「米」に分かれ、「ニチマイ」と読まれていたので、コストを下げて1語で選ばせている。
    同じ「米」でも、国名の略の「米国」「米軍」の「ベー」と、穀物の「コメ」の読みは変わらないことも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("出生の秘密", "シュッショーノヒミツ"),
        ("出生数の減少", "シュッショースーノゲンショー"),
        ("出生率と死亡率", "シュッショーリツトシボーリツ"),
        ("出生地は東京", "シュッショーチワトーキョー"),
        ("出生届を出す", "シュッショートドケヲダス"),
        ("正式な出生証明書", "セーシキナシュッショーショーメーショ"),
        ("出生前診断を受ける", "シュッショーマエシンダンヲウケル"),
        ("出生前の検査", "シュッショーマエノケンサ"),
        ("出世した", "シュッセシタ"),
        ("摘出生検を行う", "テキシュツセーケンヲオコナウ"),
        ("学生生活", "ガクセーセーカツ"),
    ],
)
def test_birth_compound_readings(text: str, expected: str) -> None:
    """
    「出生」とその複合語の「出生数」「出生率」「出生届」などを、「シュッセー」でなく「シュッショー」と読むことを確認する。
    「シュッセー」と読む人も多いが、本来の読みは「シュッショー」なので既定の読みにしている。
    「出生前診断」は「前」を「ゼン」と読む「シュッショーゼンシンダン」に分かれていたので、「マエ」と読む1語の辞書の行で選ばせている。
    「出世」の「シュッセ」、「摘出」と「生検」に分かれる「摘出生検」、「学生生活」のように「出生」でない「生」の読みは変わらないことも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected_prosody",
    [
        ("出生", "^ sh u [ cl sh o o $"),
        ("出生率", "^ sh u [ cl sh o ] o r i ts u $"),
        ("出生地", "^ sh u [ cl sh o ] o ch i $"),
        ("出生前診断", "^ sh u [ cl sh o o m a e sh i ] N d a N $"),
    ],
)
def test_birth_compound_accent_nuclei(text: str, expected_prosody: str) -> None:
    """
    「出生」は平板型、「出生率」「出生地」は「ショ」の後で下がる「シュッショ＼ーリツ」の型で読むことを確認する。
    「出生前診断」は1つのアクセント句にまとめ、後部の「診断」の「シ」の後で下がる「シュッショーマエシ＼ンダン」の型で読む。
    """

    assert pyopenjtalk.g2p_prosody(text) == expected_prosody.split()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("領収書に金拾万円と書いた", "リョーシューショニキンジューマンエントカイタ"),
        ("小切手に金参拾万円と記す", "コギッテニキンサンジューマンエントシルス"),
        ("拾円玉を見つけた", "ジューエンダマヲミツケタ"),
        ("五拾銭を貰った", "ゴジッセンヲモラッタ"),
        ("道で財布を拾う", "ミチデサイフヲヒロウ"),
        ("タクシーを拾った", "タクシーヲヒロッタ"),
        ("小石を拾い上げる", "コイシヲヒロイアゲル"),
        ("ゴミが拾えない", "ゴミガヒロエナイ"),
        ("事態の収拾を図る", "ジタイノシューシューヲハカル"),
    ],
)
def test_formal_numeral_ten_reading(text: str, expected: str) -> None:
    """
    証書の金額に使う大字の「拾」は、「十」と同じく「ジュー」と読むことを確認する。
    既定辞書には「拾」を数として読む行がなく、「金拾万円」は動詞の「ヒロエ」、「拾円玉」は地名の「ジツ」と読まれていた。
    NJD の数詞処理は「拾」を数字として扱わないので、「五拾銭」は「拾銭」の1語の行で「ジッセン」と促音にする。
    動詞の「拾う」「拾った」「拾い上げる」「拾えない」と、複合語の「収拾」の読みは変わらないことも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("土地は別の会社に譲渡された", "トチワベツノカイシャニジョートサレタ"),
        ("練習の時間を捻出したい", "レンシューノジカンヲネンシュツシタイ"),
        ("被害者は絞殺された", "ヒガイシャワコーサツサレタ"),
        ("汽笛が吹鳴された", "キテキガスイメーサレタ"),
        ("式を導出される値に代入する", "シキヲドーシュツサレルアタイニダイニュースル"),
        ("人質が解放された", "ヒトジチガカイホーサレタ"),
        ("権利を譲り渡す", "ケンリヲユズリワタス"),
        ("知恵を捻り出す", "チエヲヒネリダス"),
        ("関数を呼出す", "カンスーヲヨビダス"),
        ("来月に引越す予定だ", "ライゲツニヒッコスヨテーダ"),
        ("権利を譲渡せずに済む", "ケンリヲジョートセズニスム"),
        ("刺殺すと脅した", "サシコロストオドシタ"),
        ("競売で競落す", "キョーバイデセリオトス"),
        ("鋼材の焼鈍しを行う", "コーザイノヤキナマシヲオコナウ"),
    ],
)
def test_sino_japanese_verb_over_unmarked_okurigana(text: str, expected: str) -> None:
    """
    「譲渡された」「捻出したい」「譲渡せず」のように、サ変名詞に「さ」「し」「せ」が続く表層は、サ変動詞として音読みすることを確認する。
    既定辞書には、和語の複合動詞の送り仮名を省いた「譲渡す」(「ユズリワタス」) や「捻出す」(「ヒネリダス」) の行があり、同じ表層を訓読みしていた。
    サ変動詞と同じ表層になる未然形・連用形・仮定形の行だけを選ばれにくくしたので、基本形の「刺殺す」(「サシコロス」) や「競落す」(「セリオトス」) は訓読みのまま読む。
    送り仮名を付けた「譲り渡す」「捻り出す」、送り仮名を省いた表記が定着している「呼出す」「引越す」、名詞の「焼鈍し」も読みが変わらない。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("誰でも理解し得る説明", "ダレデモリカイシウルセツメー"),
        ("誰もが認め得る結果だ", "ダレモガミトメウルケッカダ"),
        ("考え得る限りの手を打つ", "カンガエウルカギリノテヲウツ"),
        ("十分に起こり得る事故", "ジューブンニオコリウルジコ"),
        ("それはあり得る話だ", "ソレワアリウルハナシダ"),
        ("できることはすべてなし得た", "デキルコトワスベテナシエタ"),
        ("得る", "エル"),
        ("知識を得る", "チシキヲエル"),
        ("得るものが多い", "エルモノガオーイ"),
        ("そんなことはあり得ない", "ソンナコトワアリエナイ"),
        ("協力を得られる", "キョーリョクヲエラレル"),
    ],
)
def test_auxiliary_uru_reading(text: str, expected: str) -> None:
    """
    動詞の連用形に続いて可能を表す「得る」は、終止形と連体形で「ウル」と読むことを確認する。
    非自立の「得る」の「ウル」の行のコストを、単独の「得る」が「エル」のまま保たれる範囲で下げている。
    単独の「得る」と「知識を得る」の「エル」、未然形や連用形の「あり得ない」「なし得た」「得られる」の「エ」は変わらない。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("川で投網を打つ", "カワデトアミヲウツ"),
        ("村の童が集まる", "ムラノワラベガアツマル"),
        ("童話を読む", "ドーワヲヨム"),
        ("技は口伝で受け継がれた", "ワザワクデンデウケツガレタ"),
        ("噂が口伝えに広まる", "ウワサガクチズタエニヒロマル"),
        ("香典袋に黒白の水引を掛ける", "コーデンブクロニクロシロノミズヒキヲカケル"),
        ("白黒写真を撮る", "シロクロシャシンヲトル"),
        ("春に梅園を訪れる", "ハルニバイエンヲオトズレル"),
        ("梅園さんが来た", "ウメゾノサンガキタ"),
        ("社長の旅行にお供します", "シャチョーノリョコーニオトモシマス"),
        ("上司の出張にお供する", "ジョーシノシュッチョーニオトモスル"),
        ("喜んでお供いたします", "ヨロコンデオトモイタシマス"),
        ("受付でお供物料を渡す", "ウケツケデオクモツリョーヲワタス"),
        ("お供え物を置く", "オソナエモノヲオク"),
        ("空前の人気を集める", "クーゼンノニンキヲアツメル"),
        ("先帝の太后が亡くなった", "センテーノタイコーガナクナッタ"),
        ("山の湧水を汲む", "ヤマノユースイヲクム"),
        ("湧き水を汲む", "ワキミズヲクム"),
        ("トラックの登坂能力", "トラックノトーハンノーリョク"),
        ("登坂さんが来た", "トサカサンガキタ"),
        ("登坂アナウンサー", "トサカアナウンサー"),
        ("庭の囲いを直す", "ニワノカコイヲナオス"),
        ("雪囲いをする", "ユキガコイヲスル"),
        ("才能が埋もれる", "サイノーガウモレル"),
        ("グラスを傾けよう", "グラスヲカタムケヨー"),
        ("耳を傾ける", "ミミヲカタムケル"),
        ("仏像に魂が宿される", "ブツゾーニタマシーガヤドサレル"),
        ("命を宿す", "イノチヲヤドス"),
        ("ほんの戯れで言った", "ホンノタワムレデイッタ"),
        ("子猫が戯れる", "コネコガタワムレル"),
        ("狐が憑いている", "キツネガツイテイル"),
        ("狐に憑かれる", "キツネニツカレル"),
        ("工事現場の仮囲い", "コージゲンバノカリガコイ"),
    ],
)
def test_common_reading_over_rare_rows(text: str, expected: str) -> None:
    """
    辞典の見出しの読みが一般的な語で、まれな読みの行が同じかより低いコストで選ばれていた誤りを直したことを確認する。
    「投網」「童」「口伝」「お供」「空前」「埋もれる」「傾ける」「宿す」は、行のコストを調整して一般的な読みを選ばせる。
    「黒白」は弔事の水引の色を指す「クロシロ」の用法があるので、是非を指す「コクビャク」へは寄せずに「クロシロ」を保つ。
    「梅園」「湧水」「憑く」は一般語の行がなかったので加え、「太后」は誤った読みの行を直す。
    「登坂」は単独では姓の用例が多いので、「登坂車線」と同じく複合語の「登坂能力」で「トーハン」と読ませる。
    単独の「囲い」を「カコイ」にした後も、複合語の「雪囲い」「仮囲い」は連濁した「ガコイ」と読む。
    「童話」「口伝え」「お供え物」「お供物料」「湧き水」と、姓の「梅園さん」「登坂さん」「登坂アナウンサー」は読みが変わらないことも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("会議の資料を五部刷る", "カイギノシリョーヲゴブスル"),
        ("小説の第五部", "ショーセツノダイゴブ"),
        ("高さ三丈の滝", "タカササンジョーノタキ"),
        ("着物の丈を直す", "キモノノタケヲナオス"),
        ("背丈が伸びる", "セタケガノビル"),
        ("袖・丈・襟を直す", "ソデ・タケ・エリヲナオス"),
        ("手紙を三つ折りにする", "テガミヲミツオリニスル"),
        ("地図を四つ折りにする", "チズヲヨツオリニスル"),
        ("紙を六つ折りにする", "カミヲムツオリニスル"),
        ("布を八つ折りにする", "ヌノヲヤツオリニスル"),
        ("財布を二つ折りにする", "サイフヲフタツオリニスル"),
    ],
)
def test_numeral_counter_readings(text: str, expected: str) -> None:
    """
    数詞に続く助数詞と、和語の数詞で読む複合語の読みを確認する。
    「五部」は地名の「ゴヘ」でなく「ゴブ」、長さの単位の「三丈」は「サンジョー」と読み、名詞の「丈」の「タケ」は変えない。
    折る回数の「三つ折り」「四つ折り」「六つ折り」「八つ折り」は、「サンツオリ」のような漢語の数詞でなく「ミツオリ」のように和語の数詞で読む。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("十一日午前十時に開く", "ジューイチニチゴゼンジュージニヒラク"),
        ("五日午前九時に集まる", "イツカゴゼンクジニアツマル"),
        ("二十日午前に届く", "ハツカゴゼンニトドク"),
        ("十一日午後の会議", "ジューイチニチゴゴノカイギ"),
        ("午前中に終える", "ゴゼンチューニオエル"),
        ("正午前に着く", "ショーゴマエニツク"),
        ("正午前後に混む", "ショーゴゼンゴニコム"),
        ("丙午の年", "ヒノエウマノトシ"),
        ("午の刻に起きる", "ウマノコクニオキル"),
    ],
)
def test_gozen_after_date(text: str, expected: str) -> None:
    """
    「十一日」「五日」のような日付に続く「午前」を、「午」(「ウマ」) と「前」(「マエ」) に分けずに「ゴゼン」と1語で読むことを確認する。
    日付の行の一部は地名の品詞を持っていて、その右の連接では、副詞可能名詞の「午前」より、一般名詞の「午」と接尾辞の「前」に分ける経路の方がコストが低かった。
    干支を表す「午」(「ウマ」) を含む「丙午」「午の刻」と、「正午」に続く「前」「前後」は変わらないことも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("十間先の家", "ジッケンサキノイエ"),
        ("十間の距離", "ジッケンノキョリ"),
        ("五間の道", "ゴケンノミチ"),
        ("十軒の家", "ジッケンノイエ"),
    ],
)
def test_jikken_length_reading(text: str, expected: str) -> None:
    """
    長さの単位の「十間」を「ジッケン」と読むことを確認する。
    「十間」は地名の1語の行が選ばれ、その読みが促音を欠いた「ジツケン」になっていたので、行の読みを直している。
    地名の「十間川」「十間橋」も「ジッケン」と読むので、行を地名のまま残しても読みは変わらない。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("仏語", "ブツゴ"),
        ("経典の仏語を学ぶ", "キョーテンノブツゴヲマナブ"),
        ("フランス語を学ぶ", "フランスゴヲマナブ"),
    ],
)
def test_buddhist_term_reading(text: str, expected: str) -> None:
    """
    「仏語」は仏教の言葉を指す「ブツゴ」を既定にすることを確認する。
    フランス語の意味で「仏語」と書くことは今日ではまれなので、「フツゴ」の行より「ブツゴ」の行のコストを低くしている。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text, expected_pronunciation, expected_prosody",
    [
        ("いそぎんちゃくが", "イソギンチャクガ", "^ i [ s o g i ] N ch a k u g a $"),
        ("キャパシティーが", "キャパシティーガ", "^ ky a [ p a ] sh I t i i g a $"),
        ("ありじごくが", "アリジゴクガ", "^ a [ r i j i ] g o k u g a $"),
        ("ヤマが", "ヤマガ", "^ y a [ m a ] g a $"),
    ],
)
def test_common_noun_accent_nuclei(
    text: str, expected_pronunciation: str, expected_prosody: str
) -> None:
    """「いそぎんちゃく」の3型などを NHK アクセント辞典に合わせ、尾高型の「ヤマ」は後続の助詞で下げる。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected_pronunciation
    assert pyopenjtalk.g2p_prosody(text) == expected_prosody.split()


@pytest.mark.parametrize(
    "text, expected_prosody",
    [
        ("ほうぼうが", "^ h o ] o b o o g a $"),
        ("れんちゅうが", "^ r e [ N ch u u g a $"),
        ("一年生が", "^ i [ ch i n e ] N s e e g a $"),
    ],
)
def test_ambiguous_noun_accent_nuclei(text: str, expected_prosody: str) -> None:
    """「ほうぼう」は方々の1型、「れんちゅう」は平板型、「一年生」は学年を表す3型を保つ。"""

    assert pyopenjtalk.g2p_prosody(text) == expected_prosody.split()
