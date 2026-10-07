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
from itertools import pairwise
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
    ("堤体", "テータイ"),
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("札の表と裏を見比べる。", "サツノオモテトウラヲミクラベル。"),
        ("カードの表に印を付けた。", "カードノオモテニシルシヲツケタ。"),
        ("表彰される。", "ヒョーショーサレル。"),
        ("表計算を使う。", "ヒョウケーサンヲツカウ。"),
        ("カードの表面に印を付けた。", "カードノヒョーメンニシルシヲツケタ。"),
        ("統計の表に名前を書く。", "トーケーノヒョウニナマエヲカク。"),
        ("カードの表彰式だ。", "カードノヒョーショーシキダ。"),
        ("表と裏付け資料を整理する。", "ヒョウトウラズケシリョーヲセーリスル。"),
        ("表と裏面を比べる。", "ヒョウトリメンヲクラベル。"),
    ],
)
def test_front_side_phrases_preserve_table_readings(text: str, expected: str) -> None:
    """
    裏面と対になる「表と裏」と「カードの表に」で、単独の「表」の「ヒョウ」と競合しても、「表」が「オモテ」と読まれることを確認する。
    「表彰」「表計算」「カードの表面」や、一覧表を指す「統計の表に名前を書く」は、連語の前後に同じ字があっても元の読みで読まれることを確認する。
    「表と裏付け資料」「表と裏面」では、連語の行で後続語の一部を取り込まず、「裏付け」「裏面」が元の読みで読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("昔の縁を断った。", "ムカシノエンヲタッタ。"),
        ("縁を断って旅立つ。", "エンヲタッテタビダツ。"),
        ("悪い関係を断つ。", "ワルイカンケーヲタツ。"),
        ("申込みを断る。", "モーシコミヲコトワル。"),
        ("関係者の依頼を断った。", "カンケーシャノイライヲコトワッタ。"),
        ("縁側で休む。", "エンガワデヤスム。"),
    ],
)
def test_ties_cutting_phrase_preserves_refusal(text: str, expected: str) -> None:
    """
    関係を切る「縁を断った」「縁を断って」で、「断る」の活用形の「コトワッ」と競合しても、「断っ」が「タッ」と読まれることを確認する。
    「申込みを断る」「依頼を断った」は拒絶を表す読みを保ち、「関係を断つ」や「縁側」も元の読みで読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("姫をお守りする。", "ヒメヲオマモリスル。"),
        ("私がお守りします。", "ワタシガオマモリシマス。"),
        ("王をお守りした。", "オーヲオマモリシタ。"),
        ("皆様をお守りしたい。", "ミナサマヲオマモリシタイ。"),
        ("お守りして帰る。", "オマモリシテカエル。"),
        ("お守りしないと誓った。", "オマモリシナイトチカッタ。"),
        ("お守りすれば安心だ。", "オマモリスレバアンシンダ。"),
        ("お守りしよう。", "オマモリシヨー。"),
        ("お守りを買う。", "オマモリヲカウ。"),
        ("お守りを渡す。", "オマモリヲワタス。"),
        ("子守りを頼む。", "コモリヲタノム。"),
        ("お守り袋を縫う。", "オマモリブクロヲヌウ。"),
        ("お守り仕様の袋だ。", "オマモリシヨーノフクロダ。"),
    ],
)
def test_humble_guarding_phrases_preserve_amulet_readings(text: str, expected: str) -> None:
    """
    人を守る「お守りする」「お守りします」「お守りした」「お守りしたい」などで、子供の世話を表す「お守り」の「オモリ」と競合しても、「オマモリ」と読まれることを確認する。
    「お守りを買う」「お守り袋」はお札としての「オマモリ」を保ち、「子守り」は「コモリ」と読まれることを確認する。
    「お守り仕様」では、「仕様」の一部が連語の活用形に取り込まれずに読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.xfail(
    strict=True,
    reason="守る意味の連語「お守りし」を優先するため、子供の世話を表す「お守り」も「オマモリ」と読まれる",
)
def test_childcare_omori_suru_reading() -> None:
    """
    子供の世話を表す「幼い妹を、夕方までお守りします」で、守る意味の連語「お守りし」の行と競合しても、「お守り」が「オモリ」と読まれることを確認する。
    """

    assert (
        pyopenjtalk.g2p("幼い妹を、夕方までお守りします。", kana=True)
        == "オサナイイモートヲ、ユーガタマデオモリシマス。"
    )


@pytest.mark.parametrize("text", ["私がお守りする。", "私がお守りします。"])
def test_humble_guarding_phrases_keep_subject_accent_separate(text: str) -> None:
    """
    「私がお守りする」「私がお守りします」で、単独の「する」を格助詞の後でも前のアクセント句に結合する規則があっても、「私が」と「お守りする」のアクセント句が分かれ、「私」の平板型が保たれることを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert features[0]["string"] == "私"
    assert features[0]["acc"] == 0
    assert features[2]["read"].startswith("オマモリ")
    assert features[2]["chain_flag"] == 0


@pytest.mark.parametrize(
    ("text", "reference"),
    [
        ("お守りする", "お届けする"),
        ("お守りした", "お届けした"),
        ("お守りします", "お待ちします"),
        ("お守りします", "お送りします"),
    ],
)
def test_humble_guarding_phrases_follow_polite_accent(text: str, reference: str) -> None:
    """
    「お守りする」「お守りした」で、「お守り」を名詞として解析する経路と競合しても、謙譲表現の「お届けする」「お届けした」と同じ平板型で読まれることを確認する。
    「お守りします」では、平板型の謙譲表現に助動詞「ます」が結合し、「お待ちします」「お送りします」と同じく「マ」の直後で下がることを確認する。
    """

    prosody = pyopenjtalk.g2p_prosody(text)
    reference_prosody = pyopenjtalk.g2p_prosody(reference)
    markers = {"^", "$", "?", "[", "]", "#", "_"}
    assert [token for token in prosody if token in markers] == [
        token for token in reference_prosody if token in markers
    ]
    if text.endswith("します"):
        assert prosody[-6:] == reference_prosody[-6:] == ["m", "a", "]", "s", "U", "$"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("窓から梅の香が入る。", "マドカラウメノカガハイル。"),
        ("梅の香を楽しむ。", "ウメノカヲタノシム。"),
        ("梅の香りを楽しむ。", "ウメノカオリヲタノシム。"),
        ("梅の香る庭だ。", "ウメノカオルニワダ。"),
        ("梅の香水を買う。", "ウメノコースイヲカウ。"),
        ("梅の香料を使う。", "ウメノコーリョーヲツカウ。"),
        ("香を焚く。", "コーヲタク。"),
    ],
)
def test_plum_fragrance_phrase_preserves_longer_words(text: str, expected: str) -> None:
    """
    梅の花のにおいを表す「梅の香」で、香料を表す「香」の「コウ」と競合しても、「香」が「カ」と読まれることを確認する。
    同じ字で始まる「梅の香り」「梅の香る」「梅の香水」「梅の香料」や、香料を焚く「香を焚く」は、連語の登録後も元の読みで読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("本を正せば誤解だった。", "モトヲタダセバゴカイダッタ。"),
        ("本を正しく並べる。", "ホンヲタダシクナラベル。"),
        ("本日出発する。", "ホンジツシュッパツスル。"),
    ],
)
def test_origin_phrase_preserves_book_readings(text: str, expected: str) -> None:
    """
    物事の起こりを調べる「本を正せば」で、書籍を指す「本」の「ホン」と競合しても、「本」が「モト」と読まれることを確認する。
    後ろに形容詞が続く「本を正しく並べる」や、同じ字で始まる「本日」は、書籍や日付を表す元の読みで読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("商いで身上を築く。", "アキナイデシンショーヲキズク。"),
        ("商いで身上を築いた。", "アキナイデシンショーヲキズイタ。"),
        ("賭け事で身上をつぶした。", "カケゴトデシンショーヲツブシタ。"),
        ("身上をつぶす。", "シンショーヲツブス。"),
        ("身上書を書く。", "シンジョーショヲカク。"),
        ("正直さが身上だ。", "ショージキサガシンジョーダ。"),
    ],
)
def test_wealth_phrases_preserve_personal_attribute_readings(text: str, expected: str) -> None:
    """
    財産を作る「身上を築く」と失う「身上をつぶす」の活用形で、身の上や取り柄を表す「身上」の「シンジョウ」と競合しても、「身上」が「シンショー」と発音されることを確認する。
    「身上書」や取り柄を指す「正直さが身上だ」は、財産の連語を登録した後も「シンジョー」と発音されることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("二十の誕生日に旅に出る。", "ハタチノタンジョービニタビニデル。"),
        ("二十日まで休む。", "ハツカマデヤスム。"),
        ("二十歳になった。", "ハタチニナッタ。"),
        ("二十四歳です。", "ニジューヨンサイデス。"),
        ("二十人が集まる。", "ニジューニンガアツマル。"),
        ("二十を数える。", "ニジューヲカゾエル。"),
        ("二十あまり残る。", "ニジューアマリノコル。"),
        ("二十年前に会った。", "ニジューネンマエニアッタ。"),
        ("二十代の青年だ。", "ニジューダイノセーネンダ。"),
    ],
)
def test_twentieth_birthday_preserves_numeric_readings(text: str, expected: str) -> None:
    """
    年齢が二十歳になる「二十の誕生日」で、数詞の「二十」の「ニジュウ」と競合しても、「二十」が「ハタチ」と読まれることを確認する。
    日付の「二十日」、既存の「二十歳」、後ろに数詞や助数詞が続く「二十四」「二十人」「二十年前」「二十代」と、数量を表す「二十を数える」「二十あまり」は元の読みで読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("市場でネギを一束買う。", "シジョーデネギヲヒトタバカウ。"),
        ("花を一束飾った。", "ハナヲヒトタバカザッタ。"),
        ("花を一束ずつ渡す。", "ハナヲヒトタバズツワタス。"),
        ("一束ずつ渡す。", "ヒトタバズツワタス。"),
        ("一束一本と数える。", "イッソクイッポントカゾエル。"),
        ("値一束と記される。", "アタイイッソクトシルサレル。"),
        ("花束を渡す。", "ハナタバヲワタス。"),
        ("ネギの束を買う。", "ネギノタバヲカウ。"),
    ],
)
def test_bundle_default_preserves_traditional_phrases(text: str, expected: str) -> None:
    """
    花やネギをまとめた「花を一束」「ネギを一束」で、数詞と助数詞に分かれる経路と競合しても、既定の「一束」が「ヒトタバ」と読まれることを確認する。
    「一束ずつ」の「ずつ」を残して読み、「花束」「ネギの束」と、伝統的な表現の「一束一本」「値一束」の「イッソク」が元の読みを保つことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_bundle_dictionary_keeps_both_readings() -> None:
    """
    「一束」の既定の読みを「ヒトタバ」にしても、百単位を表す「イッソク」の候補が読みの書き換えで失われず、異なる読みの2行として辞書に残ることを確認する。
    """

    dictionary_path = Path(pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8")) / "naist-jdic.csv"
    with dictionary_path.open(encoding="utf-8", newline="") as dictionary_file:
        rows = [row for row in csv.reader(dictionary_file) if row[0] == "一束"]

    assert sorted(row[11] for row in rows) == ["イッソク", "ヒトタバ"]
    costs = {row[11]: int(row[3]) for row in rows}
    assert costs["ヒトタバ"] < costs["イッソク"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一束", [("一束", "ヒトタバ", 2, 4, -1)]),
        ("カードの表", [("カードの表", "カードノオモテ", 1, 7, -1)]),
        ("本を正せば", [("本を", "モトヲ", 2, 3, -1), ("正せば", "タダセバ", 2, 4, 0)]),
    ],
)
def test_dictionary_phrases_keep_accent_nuclei(
    text: str, expected: list[tuple[str, str, int, int, int]]
) -> None:
    """
    辞書に登録した「一束」の核が2になり、「カードの表」は「カード」の核を保つ1句の「1/7」、「本を正せば」は別々の句の「2/3:2/4」になることを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert [
        (
            feature["string"],
            feature["read"],
            feature["acc"],
            feature["mora_size"],
            feature["chain_flag"],
        )
        for feature in features
    ] == expected


@pytest.mark.parametrize("text", ["静音", "静音性に優れる。", "静音モードで動かす。"])
def test_quiet_operation_compound_keeps_sound_inside_one_word(text: str) -> None:
    """
    「静音」「静音性」「静音モード」で、「静」「音」に分かれる経路と競合しても、「静音」が「セイオン」と読まれる1語として解析され、熟語の内側の「音」が読み分けの対象にならないことを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert [
        (feature["string"], feature["read"]) for feature in features if "静" in feature["string"]
    ] == [("静音", "セイオン")]
    assert all(feature["string"] != "音" for feature in features)
    result = pyopenjtalk.g2p(text, kana=True)
    assert isinstance(result, str)
    assert result.startswith("セーオン")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("高音", "コーオン"),
        ("高音を出す。", "コーオンヲダス。"),
        ("高音域を広げる。", "コウオンイキヲヒロゲル。"),
        ("高音質で録音する。", "コウオンシツデロクオンスル。"),
        ("高音部を歌う。", "コーオンブヲウタウ。"),
        ("東高音楽部の演奏だ。", "トーコーオンガクブノエンソーダ。"),
    ],
)
def test_high_pitch_default_preserves_sound_compounds(text: str, expected: str) -> None:
    """
    音の高さを表す「高音」と「高音を出す」で、同じ表記の「タカネ」の行と競合しても、「高音」が「コーオン」と発音されることを確認する。
    「高音域」「高音質」は正しい既存の「コウオン」を保ち、「高音部」は「コーオン」と発音されることを確認する。
    高校の略称「東高」に「音楽部」が続く入力では、「高音」の行と競合しても「音楽部」を途中で分割しないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("消音", "ショーオン"),
        ("防音", "ボーオン"),
        ("騒音", "ソーオン"),
        ("足音", "アシオト"),
        ("物音", "モノオト"),
        ("本音", "ホンネ"),
        ("弱音", "ヨワネ"),
        ("音色", "ネイロ"),
    ],
)
def test_sound_compounds_preserve_word_boundaries(text: str, expected: str) -> None:
    """
    「静音」「高音」のコスト調整と同じ「音」を含む「消音」「防音」「騒音」「足音」「物音」「本音」「弱音」「音色」で、既存の読みを保ち、熟語の内側の「音」が別の語に分かれないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected
    assert [feature["string"] for feature in pyopenjtalk.run_frontend(text)] == [text]


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("千年前の道具が見つかった。", "センネンマエノドーグガミツカッタ。"),
        ("千年の歴史を調べる。", "センネンノレキシヲシラベル。"),
        ("千年以上前の建物だ。", "センネンイジョーマエノタテモノダ。"),
        ("千年王国について学ぶ。", "センネンオーコクニツイテマナブ。"),
        ("千年に一度の祭りを開く。", "センネンニイチドノマツリヲヒラク。"),
        ("川崎市高津区千年に住む。", "カワサキシタカツクチトセニスム。"),
        ("川崎市高津区千年五百番地へ行く。", "カワサキシタカツクチトセゴヒャクバンチエイク。"),
        ("千歳空港に着いた。", "チトセクーコーニツイタ。"),
        ("千歳の街を歩く。", "チトセノマチヲアルク。"),
        ("一千年前の記録だ。", "イッセンネンマエノキロクダ。"),
        ("二千年の歴史がある。", "ニセンネンノレキシガアル。"),
        ("千年紀の始まりを祝う。", "センネンキノハジマリヲイワウ。"),
    ],
)
def test_millennium_reading_preserves_chitose_addresses(text: str, expected: str) -> None:
    """
    年数を表す「千年前」「千年以上」「千年の歴史」「千年王国」で、地名や人名の「チトセ」と競合しても、「千年」が「センネン」と読まれることを確認する。
    川崎市高津区の住所の「千年」は「チトセ」と読み、別の表記の「千歳」と、「一千年」「二千年」「千年紀」の読みも保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.xfail(
    strict=True,
    reason="年数の「千年」と施設名の「千年」は後続が一般名詞で、コストだけでは読み分けられない",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("千年温泉へ行く。", "チトセオンセンエイク。"),
        ("千年くすのき公園で遊ぶ。", "チトセクスノキコーエンデアソブ。"),
    ],
)
def test_chitose_facility_names(text: str, expected: str) -> None:
    """
    施設名の「千年温泉」「千年くすのき公園」で、年数の「センネン」の行と競合する「千年」が、地名の「チトセ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("売上が十倍になった。", "ウリアゲガジューバイニナッタ。"),
        ("倍率を十倍に設定した。", "バイリツヲジューバイニセッテーシタ。"),
        ("十倍に薄めた液を使う。", "ジューバイニウスメタエキヲツカウ。"),
        ("倍率は十倍だ。", "バイリツワジューバイダ。"),
        ("一日で十倍も増えた。", "イチニチデジューバイモフエタ。"),
        ("十倍速で再生する。", "ジューバイソクデサイセースル。"),
        ("十倍以上の重さだ。", "ジューバイイジョーノオモサダ。"),
        ("二十倍に拡大する。", "ニジューバイニカクダイスル。"),
        ("百倍に拡大する。", "ヒャクバイニカクダイスル。"),
        ("数十倍の差がある。", "スージューバイノサガアル。"),
        ("十一倍の値段だ。", "ジューイチバイノネダンダ。"),
    ],
)
def test_tenfold_reading_preserves_other_multipliers(text: str, expected: str) -> None:
    """
    倍率の「十倍」が姓の「トベ」の行と競合しても、数詞の「十」と助数詞の「倍」に分かれて「ジューバイ」と発音されることを確認する。
    「十倍速」「十倍以上」でも同じ発音になり、「二十倍」「百倍」「数十倍」「十一倍」の発音も保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十倍さんに会う。", "トベサンニアウ。"),
        pytest.param(
            "十倍という姓を持つ。",
            "トベトイウセーヲモツ。",
            marks=pytest.mark.xfail(
                strict=True,
                reason="姓を指す「十倍という」でも、倍率の「十」と「倍」の経路が選ばれる",
            ),
        ),
    ],
)
def test_tobe_surname_reading(text: str, expected: str) -> None:
    """
    姓の「十倍」が数詞の「十」と助数詞の「倍」の経路と競合しても、「トベ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("八名", "ハチメー"),
        ("八名様の席を用意した。", "ハチメーサマノセキヲヨーイシタ。"),
        ("三名で荷物を運ぶ。", "サンメーデニモツヲハコブ。"),
        ("三名様の席を用意した。", "サンメーサマノセキヲヨーイシタ。"),
        ("二名ずつ乗車する。", "ニメーズツジョーシャスル。"),
        ("二名で資料をまとめる。", "ニメーデシリョーヲマトメル。"),
        ("五名の候補者を選んだ。", "ゴメーノコーホシャヲエランダ。"),
        ("五名様の予約を受けた。", "ゴメーサマノヨヤクヲウケタ。"),
        ("六名で交代する。", "ロクメーデコータイスル。"),
        ("六名の係員を配置する。", "ロクメーノカカリインヲハイチスル。"),
        ("八名の選手が集まった。", "ハチメーノセンシュガアツマッタ。"),
        ("八名で作業する。", "ハチメーデサギョースル。"),
        ("一名だけ残った。", "イチメーダケノコッタ。"),
        ("四名の係員がいる。", "ヨンメーノカカリインガイル。"),
        ("七名で並ぶ。", "ナナメーデナラブ。"),
        ("九名で出発する。", "キューメーデシュッパツスル。"),
        ("十名の席を確保する。", "ジューメーノセキヲカクホスル。"),
        ("十三名の生徒がいる。", "ジューサンメーノセートガイル。"),
        ("五十名を募集する。", "ゴジューメーヲボシュースル。"),
    ],
)
def test_person_counts_preserve_other_counts(text: str, expected: str) -> None:
    """
    人数の「二名」「三名」「五名」「六名」「八名」が地名や姓の行と競合しても、数詞と助数詞に分かれて「ニメー」「サンメー」「ゴメー」「ロクメー」「ハチメー」と発音されることを確認する。
    「様」や「ずつ」が続く場合も人数として読み、「一名」「四名」「七名」「九名」「十名」「十三名」「五十名」の発音も保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一名様の予約を確認する。", "イチメーサマノヨヤクヲカクニンスル。"),
        ("二名様の予約を確認する。", "ニメーサマノヨヤクヲカクニンスル。"),
        ("三名様の予約を確認する。", "サンメーサマノヨヤクヲカクニンスル。"),
        ("四名様の予約を確認する。", "ヨンメーサマノヨヤクヲカクニンスル。"),
        ("五名様の予約を確認する。", "ゴメーサマノヨヤクヲカクニンスル。"),
        ("六名様の予約を確認する。", "ロクメーサマノヨヤクヲカクニンスル。"),
        ("七名様の予約を確認する。", "ナナメーサマノヨヤクヲカクニンスル。"),
        ("八名様の予約を確認する。", "ハチメーサマノヨヤクヲカクニンスル。"),
        ("九名様の予約を確認する。", "キューメーサマノヨヤクヲカクニンスル。"),
        ("十名様の予約を確認する。", "ジューメーサマノヨヤクヲカクニンスル。"),
    ],
)
def test_person_counts_with_sama(text: str, expected: str) -> None:
    """
    予約の人数を示す「二名様」が姓の「二名」の行と競合しても、数詞の「二」と助数詞の「名」に分かれて「ニメーサマ」と発音されることを確認する。
    同じ「様」が続く「一名」「三名」「四名」「五名」「六名」「七名」「八名」「九名」「十名」も、人数としての発音が保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.xfail(
    strict=True,
    reason="「二名様」を人数として読むコストでは、姓の「二名」に「さん」や「様」が続く場合も数詞と「名」の経路が選ばれる",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("二名さんに会う。", "ニミョーサンニアウ。"),
        ("二名様にお会いする。", "ニミョーサマニオアイスル。"),
    ],
)
def test_nimyo_surname_reading(text: str, expected: str) -> None:
    """
    姓の「二名」に「さん」や「様」が続く場合は、数詞の「二」と助数詞の「名」の経路と競合しても、「二名」が「ニミョー」と発音されることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.xfail(
    strict=True,
    reason="人数の読みを優先するコストでは、市名や「地名」が含まれていても数詞と「名」の経路が選ばれる",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("三名という地名を調べる。", "サンミョートイウチメーヲシラベル。"),
        ("奈良市二名に住む。", "ナラシニミョーニスム。"),
        ("東かがわ市五名に住む。", "ヒガシカガワシゴミョーニスム。"),
        ("岡崎市六名に住む。", "オカザキシムツナニスム。"),
    ],
)
def test_person_count_homographs_in_place_names(text: str, expected: str) -> None:
    """
    地名の「三名」「二名」「五名」「六名」が、人数を表す数詞と助数詞の経路と競合しても、「サンミョー」「ニミョー」「ゴミョー」「ムツナ」と発音されることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.xfail(
    strict=True,
    reason="「八名様」を人数として読むコストでは、姓を指す「八名さん」も数詞と「名」の経路が選ばれる",
)
def test_yana_surname_reading() -> None:
    """
    姓の「八名」に「さん」が続く場合は、数詞の「八」と助数詞の「名」の経路と競合しても、「八名」が「ヤナ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p("八名さんが話した。", kana=True) == "ヤナサンガハナシタ。"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("聾学校", "ローガッコー"),
        ("盲聾者", "モーローシャ"),
        ("盲ろう者", "モーローシャ"),
        ("聾話学校", "ローワガッコー"),
        ("盲聾唖", "モーローア"),
        ("盲学校", "モーガッコー"),
        ("盲人", "モージン"),
        ("盲導犬", "モードーケン"),
        ("聾者", "ローシャ"),
        ("聾唖", "ローア"),
        ("全盲", "ゼンモー"),
        ("養護学校", "ヨーゴガッコー"),
        ("盲目的", "モーモクテキ"),
    ],
)
def test_disability_compounds_preserve_existing_readings(text: str, expected: str) -> None:
    """
    福祉・教育の語「聾学校」「盲聾者」「盲ろう者」「聾話学校」「盲聾唖」で、単漢字の「盲」「聾」の訓読みの行と競合しても、「モウ」「ロウ」を使って読まれることを確認する。
    すでに音読みで読める「盲学校」「盲人」「盲導犬」「聾者」「聾唖」「全盲」「養護学校」「盲目的」の読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("持ち手の長さを測る。", "モチテノナガサヲハカル。"),
        ("傘の持ち手が折れた。", "カサノモチテガオレタ。"),
        ("鞄の持ち手を握った。", "カバンノモチテヲニギッタ。"),
        ("持ち主に返す。", "モチヌシニカエス。"),
        ("掛け持ち手当を申請する。", "カケモチテアテヲシンセースル。"),
        ("手当を受ける。", "テアテヲウケル。"),
        ("持ち出しは禁じる。", "モチダシワキンジル。"),
        ("持ち歩く。", "モチアルク。"),
        ("手持ちの金で買う。", "テモチノカネデカウ。"),
    ],
)
def test_handle_compound_preserves_neighboring_words(text: str, expected: str) -> None:
    """
    鞄や傘の「持ち手」で、「持ち」と「手」を分けて「モチシュ」と読む経路と競合しても、1語として「モチテ」と読まれることを確認する。
    「持ち主」「掛け持ち手当」「持ち出し」「持ち歩く」「手持ち」は、表記の一部が「持ち手」に取り込まれず、元の読みが保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("掃除しない人を注意した。", "ソージシナイヒトヲチューイシタ。"),
        ("無理をしない人が長生きする。", "ムリヲシナイヒトガナガイキスル。"),
        ("料理しない人も参加できる。", "リョーリシナイヒトモサンカデキル。"),
        ("歌える人いますか。", "ウタエルヒトイマスカ。"),
        ("詳しい人いませんか。", "クワシイヒトイマセンカ。"),
        ("知っている人いました。", "シッテイルヒトイマシタ。"),
        ("今日は掃除しない。", "キョーワソージシナイ。"),
        ("宿題をしない。", "シュクダイヲシナイ。"),
        ("しないを構える。", "シナイヲカマエル。"),
        ("釣り竿のしないを調べる。", "ツリザオノシナイヲシラベル。"),
        ("人い的な原因を探す。", "ジンイテキナゲンインヲサガス。"),
        ("人いによる災害だ。", "ジンイニヨルサイガイダ。"),
        ("人為的な原因を探す。", "ジンイテキナゲンインヲサガス。"),
        ("学生がいる。", "ガクセーガイル。"),
        ("日本人がいる。", "ニホンジンガイル。"),
    ],
)
def test_negative_verbs_and_person_predicates_preserve_noun_readings(
    text: str, expected: str
) -> None:
    """
    「掃除しない人」「料理しない人」で、名詞「しない」の行と競合しても、動詞の否定形に続く「人」が「ヒト」と読まれることを確認する。
    「歌える人いますか」「詳しい人いませんか」「知っている人いました」で、名詞「人い」の行と競合しても、「人」と動詞「いる」の活用形に分かれて読まれることを確認する。
    名詞の「しない」「人い」と、文末の「しない」「日本人」の読みが保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize("text", ["しないを構える。", "釣り竿のしないを調べる。"])
@pytest.mark.xfail(
    strict=True,
    reason="動詞の否定形を優先するため、名詞の「しない」も「し」「ない」に分かれてアクセントが変わる",
)
def test_shinai_noun_remains_one_word(text: str) -> None:
    """
    竹刀を指す「しないを構える」と、竿のしなりを指す「釣り竿のしないを調べる」で、動詞「し」と助動詞「ない」の経路と競合しても、「しない」が1語の名詞として解析され、「シナイ」と読まれることを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert [
        (feature["pos"], feature["read"], feature["mora_size"])
        for feature in features
        if feature["string"] == "しない"
    ] == [("名詞", "シナイ", 3)]


@pytest.mark.parametrize(
    "text",
    [
        "勉強しないで寝た。",
        "しない方がいい。",
        "何もしない。",
        "しないと困る。",
        "気にしないでください。",
        "失敗しないように。",
        "しないわけにはいかない。",
        "しないこと。",
        "しない理由。",
        "しない人が多い。",
        "参加するしないは自由だ。",
        "印刷する／しないを選べる。",
    ],
)
def test_shinai_negative_form_is_not_a_noun(text: str) -> None:
    """
    「しない方」「しない理由」「しない人」などの否定表現で、名詞「しない」の行と競合しても、動詞「し」と助動詞「ない」に分かれて解析されることを確認する。
    「参加するしない」「印刷する／しない」のように肯定と否定を並べた形でも、否定形が名詞「しない」に取り込まれないことを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert any(
        previous["string"] == "し"
        and previous["pos"] == "動詞"
        and feature["string"] == "ない"
        and feature["pos"] == "助動詞"
        for previous, feature in pairwise(features)
    )
    assert not any(
        feature["string"] == "しない" and feature["pos"] == "名詞" for feature in features
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("会議で目上の人と向き合う。", "カイギデメウエノヒトトムキアウ。"),
        ("目上の人の助言を忘れない。", "メウエノヒトノジョゲンヲワスレナイ。"),
        ("上の人に相談した。", "ウエノヒトニソーダンシタ。"),
        ("二歳年上の人が隣に住む。", "ニサイトシウエノヒトガトナリニスム。"),
        ("年上の人から誘われた。", "トシウエノヒトカラサソワレタ。"),
        ("手紙を目上の人に送る。", "テガミヲメウエノヒトニオクル。"),
        ("棚の上の本を取る。", "タナノウエノホンヲトル。"),
        ("上の方に置いた。", "ウエノホーニオイタ。"),
        ("上野公園を歩く。", "ウエノコーエンヲアルク。"),
        ("上野駅で降りる。", "ウエノエキデオリル。"),
        ("上の階で会おう。", "ウエノカイデアオー。"),
        ("年下の人も参加する。", "トシシタノヒトモサンカスル。"),
        ("坂の上の家に住む。", "サカノウエノイエニスム。"),
    ],
)
def test_seniority_phrases_preserve_people_and_place_readings(text: str, expected: str) -> None:
    """
    「目上の人」「二歳年上の人」「上の人」で、地名「上の」の行と競合しても、「目上」「年上」「上」と助詞「の」が正しく分かれ、後ろの「人」が「ヒト」と読まれることを確認する。
    「棚の上の本」「上の方」「上野公園」「上野駅」「上の階」「年下の人」「坂の上の家」の読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("七色の糸で模様を織る。", "ナナイロノイトデモヨーヲオル。"),
        ("七色に光る石を拾った。", "ナナイロニヒカルイシヲヒロッタ。"),
        ("七色の声を使い分ける。", "ナナイロノコエヲツカイワケル。"),
        ("七色の帯を縫い付ける。", "ナナイロノオビヲヌイツケル。"),
        ("七色の旗を揚げる。", "ナナイロノハタヲアゲル。"),
        ("七色を数えた。", "ナナイロヲカゾエタ。"),
        ("七色刷りのポスターを貼る。", "ナナイロズリノポスターヲハル。"),
        ("七色鉛筆を買った。", "ナナイロエンピツヲカッタ。"),
        ("三色のボールペンを買った。", "サンショクノボールペンヲカッタ。"),
        ("十七色から選べる。", "ジューナナショクカラエラベル。"),
        ("二十七色の見本がある。", "ニジューナナショクノミホンガアル。"),
        ("十津川村の七色を訪れた。", "トツカワムラノナナイロヲオトズレタ。"),
        ("一色ずつ塗る。", "イッショクズツヌル。"),
        ("配色を決める。", "ハイショクヲキメル。"),
    ],
)
def test_seven_colors_reading(text: str, expected: str) -> None:
    """
    「七色の糸」「七色に光る」「七色の声」などで、数詞「七」と助数詞「色」に分かれる経路と競合しても、一般名詞「七色」が「ナナイロ」と読まれることを確認する。
    地名の「十津川村の七色」も「ナナイロ」と読まれ、「七色刷り」「七色鉛筆」「三色」「十七色」「二十七色」「一色」「配色」の読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize("text", ["十七色から選べる。", "二十七色の見本がある。"])
def test_color_counts_keep_numeral_and_counter_boundaries(text: str) -> None:
    """
    数を表す「十七色」「二十七色」では、一般名詞「七色」の行が数詞の末尾を取り込まず、「色」が独立した助数詞として解析されることを確認する。
    """

    _, morphs = pyopenjtalk.run_mecab_detailed(text)
    assert not any(morph["surface"] == "七色" for morph in morphs)
    counter = next(morph for morph in morphs if morph["surface"] == "色")
    assert counter["features"][1:4] == ["名詞", "接尾", "助数詞"]


def test_seven_colors_accent() -> None:
    """
    「七色」が数詞と助数詞に分かれるとアクセントの結合も変わるため、一般名詞「七色」の読み「ナナイロ」が4モーラで、アクセント核の位置が2であることを確認する。
    """

    feature = pyopenjtalk.run_frontend("七色")[0]
    assert (feature["string"], feature["read"], feature["acc"], feature["mora_size"]) == (
        "七色",
        "ナナイロ",
        2,
        4,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十二色入りのペンを並べる。", "ジューニショクイリノペンヲナラベル。"),
        ("十二色セットを使う。", "ジューニショクセットヲツカウ。"),
        ("十二色の色鉛筆で描く。", "ジューニショクノイロエンピツデエガク。"),
        ("十二人分の席を用意する。", "ジューニニンブンノセキヲヨーイスル。"),
        pytest.param(
            "一万人当たりの割合は十二人六分だった。",
            "イチマンニンアタリノワリアイワジューニニンロクブダッタ。",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="人数の端数を表す「六分」を時間の「ロップン」と読む",
            ),
        ),
        ("箱に十二個入れる。", "ハコニジューニコイレル。"),
        ("十二時に集合する。", "ジューニジニシューゴースル。"),
        ("十二月に帰省する。", "ジューニガツニキセースル。"),
        ("十二台の車を並べる。", "ジューニダイノクルマヲナラベル。"),
        ("十二日間休む。", "ジューニニチカンヤスム。"),
        ("十二万円を支払う。", "ジューニマンエンヲシハラウ。"),
        ("十二年前の写真を見る。", "ジューニネンマエノシャシンヲミル。"),
        ("十二歳の誕生日を祝う。", "ジューニサイノタンジョービヲイワウ。"),
        ("十二番目の席に座る。", "ジューニバンメノセキニスワル。"),
        ("二百十二個を数えた。", "ニヒャクジューニコヲカゾエタ。"),
        ("第十二章を開く。", "ダイジューニショーヲヒラク。"),
        ("十二分に話し合った。", "ジューニブンニハナシアッタ。"),
        ("十二支を覚える。", "ジューニシヲオボエル。"),
        ("十二単を展示する。", "ジューニヒトエヲテンジスル。"),
        ("福知山市十二の住所を調べる。", "フクチヤマシジューニノジューショヲシラベル。"),
        ("新潟市北区十二に住む。", "ニーガタシキタクジューニニスム。"),
        ("七色の糸で模様を織る。", "ナナイロノイトデモヨーヲオル。"),
        ("十七色から選べる。", "ジューナナショクカラエラベル。"),
    ],
)
def test_twelve_counts_preserve_compound_and_place_readings(text: str, expected: str) -> None:
    """
    「十二色入り」「十二色セット」では、地名「十二」の行と競合しても数詞と助数詞に分かれ、「ジューニショク」と読まれることを確認する。
    「十二人分」は名詞「二人」の行と競合しても「ジューニニンブン」と読まれ、割合の「十二人六分」は時間の「六分」と競合しても「ジューニニンロクブ」と読まれることを確認する。
    「十二個」「十二時」「十二月」などの数量や暦の読みと、「十二分」「十二支」「十二単」、地名の「福知山市十二」「新潟市北区十二」、「七色」「十七色」の読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    "text",
    ["十二色セットを使う。", "箱に十二個入れる。", "十二時に集合する。"],
)
def test_twelve_counts_use_numeral_entries(text: str) -> None:
    """
    「十二色」「十二個」「十二時」では、地名「十二」として解析される経路と競合しても、助数詞の前にある「十」「二」が数詞として解析されることを確認する。
    """

    _, morphs = pyopenjtalk.run_mecab_detailed(text)
    assert not any(morph["surface"] == "十二" for morph in morphs)
    for surface in ("十", "二"):
        numeral = next(morph for morph in morphs if morph["surface"] == surface)
        assert numeral["features"][1:3] == ["名詞", "数"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="人数の末尾を名詞「二人」の行が取り込み、「ニニン」を「フタリ」と読む",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十二人で輪になった。", "ジューニニンデワニナッタ。"),
        ("十二人ほど集まった。", "ジューニニンホドアツマッタ。"),
        ("二十二人で輪になった。", "ニジューニニンデワニナッタ。"),
        ("百十二人で輪になった。", "ヒャクジューニニンデワニナッタ。"),
    ],
)
def test_twelve_people_known_compound_reading(text: str, expected: str) -> None:
    """
    「十二人」「二十二人」「百十二人」では、名詞「二人」の行が数詞の末尾を取り込む経路と競合しても、末尾が「ニニン」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("二人で歩く。", "フタリデアルク。"),
        ("二人ほど集まった。", "フタリホドアツマッタ。"),
        ("二人きりで話した。", "フタリキリデハナシタ。"),
        ("二人とも帰った。", "フタリトモカエッタ。"),
        ("二人目の客が来た。", "フタリメノキャクガキタ。"),
        pytest.param(
            "二人前を注文した。",
            "ニニンマエヲチューモンシタ。",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="名詞「二人」の行が選ばれ、「ニニンマエ」を「フタリマエ」と読む",
            ),
        ),
        ("二人分を買う。", "フタリブンヲカウ。"),
        ("二人組の選手が来た。", "フタリグミノセンシュガキタ。"),
        ("二人連れの客が来た。", "フタリズレノキャクガキタ。"),
        ("二人部屋を予約した。", "フタリベヤヲヨヤクシタ。"),
        ("二人っきりで話した。", "フタリッキリデハナシタ。"),
        ("二人称の代名詞を使う。", "ニニンショーノダイメーシヲツカウ。"),
        ("２．二人とも予約した。", "ニ．フタリトモヨヤクシタ。"),
        pytest.param(
            "二人掛けの椅子を買う。",
            "フタリガケノイスヲカウ。",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="「二人」と「掛け」に分かれ、後ろの「掛け」が連濁しない",
            ),
        ),
        ("二人三脚で走る。", "ニニンサンキャクデハシル。"),
        pytest.param(
            "二人羽織を披露する。",
            "ニニンバオリヲヒロースル。",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="「二人」と「羽織」に分かれ、「ニニンバオリ」を「フタリハオリ」と読む",
            ),
        ),
        ("一人で帰った。", "ヒトリデカエッタ。"),
        ("三人で歌う。", "サンニンデウタウ。"),
    ],
)
def test_two_people_preserve_standalone_and_compound_readings(text: str, expected: str) -> None:
    """
    「二人組の選手」「二人連れ」「二人分」「二人部屋」「二人っきり」は、数詞「二」と助数詞の行に分かれる経路と競合しても、「二人」が「フタリ」と読まれることを確認する。
    箇条書きの「２．二人とも予約した。」では、「２．二」を小数と扱う経路と競合しても、項目番号が独立して読まれ、後ろの「二人」が「フタリ」と読まれることを確認する。
    「二人称」「二人三脚」では「二人」が「ニニン」と読まれ、同じ助数詞「人」を使う「一人」「三人」も、それぞれ「ヒトリ」「サンニン」と読まれることを確認する。
    「二人前」は名詞「二人」の行と競合しても「ニニンマエ」と読まれ、「二人掛け」は「二人」「掛け」に分かれる経路と競合しても「フタリガケ」と読まれることを確認する。
    「二人羽織」は「二人」と「羽織」に分かれる経路と競合しても、複合語として「ニニンバオリ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("在家", "ザイケ"),
        ("在家者が集まって読経する。", "ザイケシャガアツマッテドキョースル。"),
        ("在家信者の修行を紹介する。", "ザイケシンジャノシュギョーヲショーカイスル。"),
        ("在家仏教の歴史を調べる。", "ザイケブッキョーノレキシヲシラベル。"),
        ("在家のまま戒を受ける。", "ザイケノママカイヲウケル。"),
        ("在家として活動する。", "ザイケトシテカツドースル。"),
        ("在家で修行を続ける。", "ザイケデシュギョーヲツズケル。"),
        ("在家佛教協会の会報を読む。", "ザイケブッキョーキョーカイノカイホーヲヨム。"),
        ("さいたま市桜区在家の地図を見る。", "サイタマシサクラクザイケノチズヲミル。"),
        ("川口市安行領在家に住む。", "カワグチシアンギョーリョーザイケニスム。"),
        ("新在家駅で待ち合わせる。", "シンザイケエキデマチアワセル。"),
        ("在家塚の住所を調べる。", "ザイケツカノジューショヲシラベル。"),
        ("在宅で仕事をする。", "ザイタクデシゴトヲスル。"),
        ("現在家族と暮らしている。", "ゲンザイカゾクトクラシテイル。"),
        ("家に帰る。", "イエニカエル。"),
    ],
)
def test_lay_buddhist_reading_preserves_place_and_organization_names(
    text: str, expected: str
) -> None:
    """
    「在家」「在家者」「在家信者」「在家仏教」では、組織名として読む「アリイエ」と競合しても、一般名詞「在家」が「ザイケ」と読まれることを確認する。
    「在家信者の修行」では「在」「家信」「者」に分かれる経路とも競合するため、「在家」「信者」として「ザイケシンジャ」と読まれることを確認する。
    「在家佛教協会」「さいたま市桜区在家」「安行領在家」「新在家駅」「在家塚」や、別の語である「在宅」「家」でも、それぞれの読みが保たれることを確認する。
    「現在家族と暮らしている。」のように単語境界をまたいで「在家」を含む場合でも、「現在」と「家族」がそれぞれ「ゲンザイ」「カゾク」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_lay_buddhist_accent() -> None:
    """
    単独の「在家」では、組織名として読む「アリイエ」と競合しても一般名詞の「ザイケ」が選ばれ、3モーラの平板型として読まれることを確認する。
    """

    feature = pyopenjtalk.run_frontend("在家")[0]
    assert (feature["read"], feature["acc"], feature["mora_size"]) == ("ザイケ", 0, 3)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="一般名詞「在家」の行が選ばれ、人名「在家」が「ザイケ」と読まれるため",
)
def test_ariie_person_name_known_reading() -> None:
    """
    公卿の人名を含む「唐橋在家の記録を調べる。」では、一般名詞「在家」の行と競合しても、人名「在家」が「アリイエ」と読まれることを確認する。
    """

    assert (
        pyopenjtalk.g2p("唐橋在家の記録を調べる。", kana=True)
        == "カラハシアリイエノキロクヲシラベル。"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("神前", "シンゼン"),
        ("神前に玉串を置く。", "シンゼンニタマグシヲオク。"),
        ("神前で鈴を鳴らした。", "シンゼンデスズヲナラシタ。"),
        ("神前への供物を用意した。", "シンゼンエノクモツヲヨーイシタ。"),
        ("神前結婚式の費用を調べる。", "シンゼンケッコンシキノヒヨーヲシラベル。"),
    ],
)
def test_shrine_front_readings(text: str, expected: str) -> None:
    """
    単独の「神前」と「神前に玉串を置く」「神前で鈴を鳴らした」「神前への供物」「神前結婚式」では、「神」「前」に分かれる経路と競合しても、神の前を表す一般名詞「神前」が「シンゼン」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_shrine_front_accent() -> None:
    """
    単独の「神前」では、「神」「前」に分かれる経路と競合しても一般名詞の「シンゼン」が選ばれ、4モーラの平板型として読まれることを確認する。
    """

    feature = pyopenjtalk.run_frontend("神前")[0]
    assert (feature["read"], feature["acc"], feature["mora_size"]) == ("シンゼン", 0, 4)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("神前式を挙げる。", "シンゼンシキヲアゲル。"),
        ("神前駅で降りる。", "カンザキエキデオリル。"),
        ("神前郡の地名を調べる。", "カンザキグンノチメーヲシラベル。"),
        ("神前俊彦監督の経歴を調べる。", "カミマエトシヒコカントクノケーレキヲシラベル。"),
        ("神社の前で待つ。", "ジンジャノマエデマツ。"),
        ("神の前に立つ。", "カミノマエニタツ。"),
        ("午前中に出発する。", "ゴゼンチューニシュッパツスル。"),
        ("名前を書き直した。", "ナマエヲカキナオシタ。"),
    ],
)
def test_shrine_front_reading_preserves_place_names(text: str, expected: str) -> None:
    """
    「神前式」は「シンゼン」の読みを保ち、地名を含む「神前駅」「神前郡」は一般名詞の行と競合しても「神前」が「カンザキ」と読まれることを確認する。
    姓に名が続く「神前俊彦監督」は人名の行が選ばれ、「神前」が「カミマエ」と読まれることを確認する。
    「神社の前」「神の前」「午前中」「名前」は別の語なので、それぞれの読みが保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="一般名詞「神前」の行が選ばれ、姓や学校名の「神前」が「シンゼン」と読まれるため",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("神前監督の講演を聞いた。", "カミマエカントクノコーエンヲキイタ。"),
        ("神前監督が試合後に語った。", "カミマエカントクガシアイゴニカタッタ。"),
        ("四日市市立神前小学校で学ぶ。", "ヨッカイチシリツカンザキショーガッコーデマナブ。"),
    ],
)
def test_shrine_front_person_and_school_names_known_readings(text: str, expected: str) -> None:
    """
    神前俊彦の姓と役職だけを記した「神前監督の講演」「神前監督が試合後に語った」と「四日市市立神前小学校」では、一般名詞「神前」の行と競合しても、姓が「カミマエ」、学校名が「カンザキ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("改札口手前で切符を確認する。", "カイサツグチテマエデキップヲカクニンスル。"),
        ("交差点手前に標識がある。", "コーサテンテマエニヒョーシキガアル。"),
        ("カーブ手前で減速する。", "カーブテマエデゲンソクスル。"),
        ("バス停手前で待つ。", "バステーテマエデマツ。"),
        ("入り口手前の照明を交換した。", "イリグチテマエノショーメーヲコーカンシタ。"),
    ],
)
def test_position_before_landmark_readings(text: str, expected: str) -> None:
    """
    「改札口手前」「交差点手前」「カーブ手前」「バス停手前」「入り口手前」では、接尾辞「手」（「シュ」）と「前」に分かれる経路と競合しても、地点のこちら側を表す「手前」が「テマエ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("橋の手前で右に曲がる。", "ハシノテマエデミギニマガル。"),
        ("手前味噌ですが、おすすめです。", "テマエミソデスガ、オススメデス。"),
        ("手前勝手な話をする。", "テマエガッテナハナシヲスル。"),
        ("お手前を拝見する。", "オテマエヲハイケンスル。"),
        ("投手が交代する。", "トーシュガコータイスル。"),
        ("助手と相談した。", "ジョシュトソーダンシタ。"),
        ("選手を紹介する。", "センシュヲショーカイスル。"),
        ("運転手前田さんが案内する。", "ウンテンシュマエダサンガアンナイスル。"),
        ("投手前田の成績を調べた。", "トーシュマエダノセーセキヲシラベタ。"),
    ],
)
def test_position_before_landmark_preserves_compounds(text: str, expected: str) -> None:
    """
    「手前」を1語として読む場合も、「手前味噌」「手前勝手」「お手前」や助詞「の」に続く「手前」の読みが保たれ、特に「手前勝手」が連濁して「テマエガッテ」と読まれることを確認する。
    人を指す「投手」「助手」「選手」や、姓が続く「運転手前田」「投手前田」では、「手前」の行と競合しても「手」が「シュ」、姓の「前田」が「マエダ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_position_before_landmark_accent() -> None:
    """
    単独の「手前」では、「手」と「前」に分かれる経路と競合しても語全体の行が選ばれ、「テマエ」が3モーラの平板型として読まれることを確認する。
    """

    feature = pyopenjtalk.run_frontend("手前")[0]
    assert (feature["string"], feature["read"], feature["acc"], feature["mora_size"]) == (
        "手前",
        "テマエ",
        0,
        3,
    )
    assert pyopenjtalk.g2p_prosody("手前") == "^ t e [ m a e $".split()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="「駅手」「三塁手」の行が選ばれ、地点を表す「手前」が「シュマエ」と読まれるため",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("駅手前の踏切を撤去する。", "エキテマエノフミキリヲテッキョスル。"),
        ("三塁手前の白線を引き直す。", "サンルイテマエノハクセンヲヒキナオス。"),
    ],
)
def test_position_before_landmark_after_compound_nouns_known_readings(
    text: str, expected: str
) -> None:
    """
    「駅手前の踏切」「三塁手前の白線」では、「駅手」「三塁手」の行と競合しても、地点のこちら側を表す「手前」が「テマエ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("笑い者になる。", "ワライモノニナル。"),
        ("笑い者にする。", "ワライモノニスル。"),
        ("笑い者扱いはひどい。", "ワライモノアツカイワヒドイ。"),
        ("笑い者にされたくない。", "ワライモノニサレタクナイ。"),
        ("笑い者（笑）と呼ばれた。", "ワライモノ（ワライ）トヨバレタ。"),
    ],
)
def test_laughingstock_readings(text: str, expected: str) -> None:
    """
    「笑い者になる」「笑い者にする」「笑い者扱い」「笑い者にされたくない」や括弧が続く「笑い者（笑）と呼ばれた」では、「笑い」と接尾辞「者」（「シャ」）に分かれる経路と競合しても、嘲笑の対象を表す「笑い者」が「ワライモノ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("笑いものになる。", "ワライモノニナル。"),
        ("笑い物にする。", "ワライモノニスル。"),
        ("笑い声が聞こえる。", "ワライゴエガキコエル。"),
        ("笑い話をする。", "ワライバナシヲスル。"),
        ("笑う者が勝つ。", "ワラウモノガカツ。"),
        ("研究者を紹介する。", "ケンキュウシャヲショーカイスル。"),
    ],
)
def test_laughingstock_preserves_related_readings(text: str, expected: str) -> None:
    """
    「笑い者」の行を優先しても、表記違いの「笑いもの」「笑い物」は「ワライモノ」、同じ「笑い」で始まる「笑い声」「笑い話」は「ワライゴエ」「ワライバナシ」と読まれることを確認する。
    「笑う者が勝つ」「研究者を紹介する」では、「笑い者」の調整が単独の「者」や接尾辞「者」に波及せず、それぞれ「モノ」「シャ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("九九表をノートに写した。", "ククヒョーヲノートニウツシタ。"),
        ("かけ算九九の宿題を終えた。", "カケザンククノシュクダイヲオエタ。"),
        ("かけざん九九でつまずいた。", "カケザンククデツマズイタ。"),
        ("かけざん九九表で積を調べる。", "カケザンククヒョーデセキヲシラベル。"),
    ],
)
def test_multiplication_table_compound_readings(text: str, expected: str) -> None:
    """
    「九九表をノートに写した」「かけ算九九の宿題を終えた」「かけざん九九でつまずいた」「かけざん九九表で積を調べる」では、数詞の「九」「九」に分かれる経路と競合しても、掛け算の「九九」が「クク」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("九九を暗記する。", "ククヲアンキスル。"),
        ("九十四．九九", "キュージューヨンテンキューキュー"),
        ("一九九九年に発売された。", "センキューヒャクキュージューキューネンニハツバイサレタ。"),
        ("一九九式の機器を選んだ。", "ヒャクキュージューキューシキノキキヲエランダ。"),
        ("九九パーセント", "キュージューキューパーセント"),
        ("九九個", "キュージューキューコ"),
        ("九九番", "キュージューキューバン"),
        ("九九円", "キュージューキューエン"),
        ("かけ算を学ぶ。", "カケザンヲマナブ。"),
        ("表計算で集計する。", "ヒョウケーサンデシューケースル。"),
    ],
)
def test_multiplication_table_preserves_numeral_readings(text: str, expected: str) -> None:
    """
    「九九表」「かけ算九九」「かけざん九九」の行を足しても、「九九を暗記する」の「クク」と、小数の「九十四．九九」、年・型式・数量・番号を表す「一九九九年」「一九九式」「九九パーセント」「九九個」「九九番」「九九円」の数詞としての読みが保たれることを確認する。
    「かけ算を学ぶ」「表計算で集計する」では、複合語の一部と同じ字を使う別の語が「カケザン」「ヒョウケーサン」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize("text", ["かけ算九九", "かけざん九九"])
def test_multiplication_table_keeps_separate_accent_phrases(text: str) -> None:
    """
    「かけ算九九」「かけざん九九」では、数詞の「九」「九」の経路より複合語の行が優先され、「かけ算」「九九」が別々のアクセント句としてアクセント核を2と1に保つことを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert [(feature["read"], feature["acc"], feature["mora_size"]) for feature in features] == [
        ("カケザン", 2, 4),
        ("クク", 1, 2),
    ]
    assert features[1]["chain_flag"] == 0
    assert pyopenjtalk.g2p_prosody(text) == "^ k a [ k e ] z a N # k u ] k u $".split()


def test_multiplication_table_chart_accent() -> None:
    """
    「九九表」では、数詞の「九」「九」と名詞の「表」に分かれる経路より複合語の行が優先され、接尾辞「表」の平板型に合わせた4モーラの「ククヒョー」が1つのアクセント句になることを確認する。
    """

    features = pyopenjtalk.run_frontend("九九表")
    assert [(feature["read"], feature["acc"], feature["mora_size"]) for feature in features] == [
        ("ククヒョウ", 0, 4),
    ]
    assert pyopenjtalk.g2p_prosody("九九表") == "^ k U [ k u hy o o $".split()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="小数の桁を表す「九九」が、掛け算の「九九」の行に取られるため",
)
def test_multiplication_table_decimal_digits_known_reading() -> None:
    """
    「零点九九をかける」では、掛け算の「九九」の行と競合しても、小数点の後の「九九」が桁ごとの「キューキュー」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p("零点九九をかける。", kana=True) == "レーテンキューキューヲカケル。"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "九十九年春に新校舎が完成した。",
            "キュージューキューネンハルニシンコーシャガカンセーシタ。",
        ),
        ("九十九年春の写真を整理する。", "キュージューキューネンハルノシャシンヲセーリスル。"),
        ("本年春に苗木を植える。", "ホンネンハルニナエギヲウエル。"),
        ("文化元年春の記録を読む。", "ブンカガンネンハルノキロクヲヨム。"),
    ],
)
def test_year_followed_by_spring_readings(text: str, expected: str) -> None:
    """
    「九十九年春に新校舎が完成した」「九十九年春の写真を整理する」では、人名の「九十九」「年春」の行と競合しても、年数の「九十九年」と季節の「春」に分かれて「キュージューキューネンハル」と読まれることを確認する。
    「本年春に苗木を植える」「文化元年春の記録を読む」でも、名の「年春」に引かれて「本」「元」が「モト」と読まれず、「ホンネンハル」「ガンネンハル」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("九九年春に出会った。", "キュージューキューネンハルニデアッタ。"),
        ("二千年春に開店した。", "ニセンネンハルニカイテンシタ。"),
        ("二〇二六年春に卒業する。", "ニセンニジューロクネンハルニソツギョースル。"),
        ("平成九年春に植えた。", "ヘーセーキューネンハルニウエタ。"),
        ("昨年春に引っ越した。", "サクネンハルニヒッコシタ。"),
        ("来年春に出発する。", "ライネンハルニシュッパツスル。"),
        ("年春さんが到着した。", "トシハルサンガトーチャクシタ。"),
        ("中村年春さんの論文を読んだ。", "ナカムラトシハルサンノロンブンヲヨンダ。"),
        ("九十九さんが来た。", "ツクモサンガキタ。"),
        ("九十九里浜を歩いた。", "クジュークリハマヲアルイタ。"),
    ],
)
def test_year_followed_by_spring_preserves_related_readings(text: str, expected: str) -> None:
    """
    名の「年春」のコストを上げても、年数や元号に続く「年春」と「昨年春」「来年春」は年と季節に分かれ、「年」「春」が「ネン」「ハル」と読まれることを確認する。
    「年春さん」「中村年春さん」では名の「トシハル」、「九十九さん」「九十九里浜」では姓や地名の「ツクモ」「クジュークリ」と読まれ、年数と季節の経路を優先する調整で固有名詞の読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("大黒様に手を合わせる。", "ダイコクサマニテヲアワセル。"),
        ("大黒様の木像を磨く。", "ダイコクサマノモクゾーヲミガク。"),
        ("七福神の大黒様を描く。", "シチフクジンノダイコクサマヲエガク。"),
        ("恵比寿様と大黒様を祀る。", "エビスサマトダイコクサマヲマツル。"),
    ],
)
def test_daikokusama_readings(text: str, expected: str) -> None:
    """
    「大黒様に手を合わせる」「大黒様の木像を磨く」「七福神の大黒様を描く」「恵比寿様と大黒様を祀る」では、姓の「大黒」（「オオクロ」）と敬称の「様」に分かれる経路と競合しても、神を表す「大黒様」が「ダイコクサマ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("大黒様", "ダイコクサマ"),
        ("大黒天を祀る。", "ダイコクテンヲマツル。"),
        ("大黒柱を磨く。", "ダイコクバシラヲミガク。"),
        ("大黒ふ頭へ向かう。", "ダイコクフトーエムカウ。"),
        ("大黒屋で買い物をした。", "ダイコクヤデカイモノヲシタ。"),
        ("大黒さんに電話する。", "オークロサンニデンワスル。"),
        ("一様な色に塗る。", "イチヨーナイロニヌル。"),
        ("様子を見に行く。", "ヨースヲミニイク。"),
    ],
)
def test_daikokusama_preserves_related_readings(text: str, expected: str) -> None:
    """
    「大黒様」の行を優先しても、単独の「大黒様」と複合語の「大黒天」「大黒柱」「大黒ふ頭」「大黒屋」は「ダイコク」で始まる読みを保ち、敬称「さん」が続く姓の「大黒」は「オークロ」と発音されることを確認する。
    「一様な色に塗る」「様子を見に行く」では、調整が「様」を含む別の語へ波及せず、「イチヨー」「ヨース」と発音されることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize("text", ["大黒様", "大黒様に手を合わせる。"])
def test_daikokusama_keeps_compound_accent(text: str) -> None:
    """
    「大黒様」と「大黒様に手を合わせる」では、姓の「大黒」と敬称「様」の経路より神を表す1語の行が優先され、6モーラの「ダイコクサマ」の最終モーラにアクセント核が置かれることを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert features[0]["string"] == "大黒様"
    assert features[0]["read"] == "ダイコクサマ"
    assert features[0]["mora_size"] == 6
    assert features[0]["acc"] == 6


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="神を表す「大黒様」の行が優先され、姓の「大黒」に敬称「様」が続く文も「ダイコクサマ」と読まれるため",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("受付の大黒様をお呼びします。", "ウケツケノオークロサマヲオヨビシマス。"),
        ("大黒様宛ての請求書です。", "オークロサマアテノセーキューショデス。"),
    ],
)
def test_daikokusama_surname_honorific_known_readings(text: str, expected: str) -> None:
    """
    「受付の大黒様をお呼びします」「大黒様宛ての請求書です」では、神を表す「大黒様」の行と競合しても、呼び出す人や請求書の宛先を表す姓「大黒」が「オークロ」と発音されることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("辞退の申出が届いた。", "ジタイノモーシデガトドイタ。"),
        ("援助の申出を歓迎する。", "エンジョノモーシデヲカンゲースル。"),
        ("登録内容を変える申出を受けた。", "トーロクナイヨーヲカエルモーシデヲウケタ。"),
        ("この申出について話し合う。", "コノモーシデニツイテハナシアウ。"),
        ("申出書に住所を書く。", "モーシデショニジューショヲカク。"),
        ("申出人の話を聞く。", "モーシデニンノハナシヲキク。"),
        ("「申出人」という語を調べた。", "「モーシデニン」トイウゴヲシラベタ。"),
        ("苦情の申出人に連絡する。", "クジョーノモーシデニンニレンラクスル。"),
        (
            "委員会は、申出人、担当者に話を聞く。",
            "イインカイワ、モーシデニン、タントーシャニハナシヲキク。",
        ),
    ],
)
def test_moushide_readings(text: str, expected: str) -> None:
    """
    「辞退の申出が届いた」「援助の申出を歓迎する」「登録内容を変える申出を受けた」では、干支の「申」（「サル」）と「出」に分かれる経路と競合しても、名詞「申出」が「モウシデ」と読まれることを確認する。
    「この申出について話し合う」「申出書に住所を書く」「申出人の話を聞く」では、助詞や接尾辞が続いても「モウシデ」と読まれることを確認する。
    引用符で囲んだ「申出人」や「苦情の申出人に連絡する」、「委員会は、申出人、担当者に話を聞く」では、名詞の行と接尾辞「人」（「ジン」）の経路と競合しても、「申出人」が「モウシデニン」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("申年に生まれた。", "サルネンニウマレタ。"),
        ("申し出を受けた。", "モーシデヲウケタ。"),
        ("申請書を提出する。", "シンセーショヲテーシュツスル。"),
        ("申告を済ませる。", "シンコクヲスマセル。"),
        ("外出の予定を伝えた。", "ガイシュツノヨテーヲツタエタ。"),
        ("申し入れに応じる。", "モーシイレニオージル。"),
    ],
)
def test_moushide_preserves_related_readings(text: str, expected: str) -> None:
    """
    名詞「申出」の行を優先しても、「申年に生まれた」では干支の「申」が「サル」と読まれ、「申し出を受けた」では名詞「申し出」が「モウシデ」と読まれることを確認する。
    「申請書」「申告」「外出」「申し入れ」では、同じ漢字を含む別の語がそれぞれ「シンセイショ」「シンコク」「ガイシュツ」「モウシイレ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_moushide_keeps_noun_accent() -> None:
    """
    「辞退の申出が届いた」では、干支の「申」と接尾辞「出」の経路と競合しても、名詞「申出」が選ばれ、4モーラの「モーシデ」が平板型で読まれることを確認する。
    """

    feature = pyopenjtalk.run_frontend("辞退の申出が届いた。")[2]
    assert feature["string"] == "申出"
    assert feature["pos"] == "名詞"
    assert feature["pron"] == "モーシデ"
    assert feature["acc"] == 0
    assert feature["mora_size"] == 4


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="「この申出」の「申出」が動詞「申出る」の連用形として解析され、名詞の平板型にならないため",
)
def test_moushide_after_demonstrative_known_noun_accent() -> None:
    """
    「この申出について話し合う」では、同じ表記の動詞「申出る」の連用形と競合しても、名詞「申出」が選ばれ、4モーラの「モーシデ」が平板型で読まれることを確認する。
    """

    feature = pyopenjtalk.run_frontend("この申出について話し合う。")[1]
    assert feature["string"] == "申出"
    assert feature["pos"] == "名詞"
    assert feature["pron"] == "モーシデ"
    assert feature["acc"] == 0
    assert feature["mora_size"] == 4


@pytest.mark.parametrize("text", ["申出る。", "申出た。", "申出てください。", "申出ればよい。"])
def test_moushide_preserves_verb_conjugations(text: str) -> None:
    """
    「申出る」「申出た」「申出てください」「申出ればよい」では、同じ表記の名詞「申出」の行と競合しても、動詞の活用形が選ばれ、「モウシデ」で始まる読みが保たれることを確認する。
    """

    feature = pyopenjtalk.run_frontend(text)[0]
    assert feature["pos"] == "動詞"
    assert feature["read"].startswith("モウシデ")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="名詞「申出」の行が優先されると、前の接頭辞「御」が「ゴ」と読まれるため",
)
def test_moushide_honorific_prefix_known_reading() -> None:
    """
    「この際御申出をお願いします」では、接頭辞「御」（「ゴ」）と名詞「申出」の経路と競合しても、和語の「申し出」に付く「御」が「オ」と読まれることを確認する。
    """

    assert (
        pyopenjtalk.g2p("この際御申出をお願いします。", kana=True)
        == "コノサイオモーシデヲオネガイシマス。"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="後ろに「ごと」「先」が続くと、名詞「申出」の行より干支の「申」と接尾辞「出」の経路が優先されるため",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("個別の申出ごとに番号を付ける。", "コベツノモーシデゴトニバンゴーヲツケル。"),
        ("苦情の申出先を一覧にした。", "クジョーノモーシデサキヲイチランニシタ。"),
    ],
)
def test_moushide_with_suffixes_known_readings(text: str, expected: str) -> None:
    """
    「個別の申出ごとに番号を付ける」「苦情の申出先を一覧にした」では、干支の「申」と接尾辞「出」に分かれる経路と競合しても、「ごと」「先」が続く名詞「申出」が「モウシデ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("相同", "ソードー"),
        ("相同性を比較した。", "ソードーセーヲヒカクシタ。"),
        ("相同染色体の対を数える。", "ソードーセンショクタイノタイヲカゾエル。"),
        ("相同組換えを利用する。", "ソードークミカエヲリヨースル。"),
        ("相同組み換えの仕組みを学ぶ。", "ソードークミカエノシクミヲマナブ。"),
        ("この構造は相同である。", "コノコーゾーワソードーデアル。"),
        ("相同な三角形を描いた。", "ソードーナサンカッケーヲエガイタ。"),
    ],
)
def test_homology_readings(text: str, expected: str) -> None:
    """
    「相同性」「相同染色体」「相同組換え」「相同組み換え」と単独・連体用法の「相同」では、「相」「同」「同性」に分かれる経路と競合しても、「相同」が「ソウドウ」と読まれることを確認する。
    「相同組換えを利用する」では、接尾辞「組」の行と競合しても、「組換え」が「クミカエ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("位相同型を調べる。", "イソードーケーヲシラベル。"),
        ("同性愛の歴史を学ぶ。", "ドーセーアイノレキシヲマナブ。"),
        ("同性の友人と話す。", "ドーセーノユージントハナス。"),
        ("相場が動いた。", "ソーバガウゴイタ。"),
        ("同窓会に出席する。", "ドーソーカイニシュッセキスル。"),
        ("同意書を読む。", "ドーイショヲヨム。"),
        ("「相同じ」という古語を調べた。", "「アイオナジ」トイウコゴヲシラベタ。"),
        ("古い記述は相同じくしている。", "フルイキジュツワアイオナジクシテイル。"),
    ],
)
def test_homology_readings_preserve_neighboring_words(text: str, expected: str) -> None:
    """
    「位相同型」「相同じ」「相同じく」では、「相同」の行と競合しても「位相」と「同型」、「相」と「同じ」の区切りを保ち、それぞれ「イソウドウケイ」「アイオナジ」「アイオナジク」と読まれることを確認する。
    「同性愛」「同性」「相場」「同窓会」「同意書」は別の語なので、それぞれの読みが保たれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_homology_accent() -> None:
    """
    単独の「相同」では、単漢字の行と競合しても語全体の行が選ばれ、4モーラの平板型として読まれることを確認する。
    """

    feature = pyopenjtalk.run_frontend("相同")[0]
    assert (feature["string"], feature["read"], feature["acc"], feature["mora_size"]) == (
        "相同",
        "ソウドウ",
        0,
        4,
    )


def test_homologous_recombination_accent_phrases() -> None:
    """
    「相同組換え」では、接尾辞「組」の行と競合しても語全体の辞書行が選ばれ、「:」区切りに従って「相同」「組換え」が別々のアクセント句となり、それぞれ4モーラの平板型として読まれることを確認する。
    """

    features = pyopenjtalk.run_frontend("相同組換え")
    assert [
        (feature["orig"], feature["read"], feature["pron"], feature["acc"], feature["mora_size"])
        for feature in features
    ] == [
        ("相同", "ソウドウ", "ソードー", 0, 4),
        ("組換え", "クミカエ", "クミカエ", 0, 4),
    ]
    assert features[1]["chain_flag"] == 0
    assert pyopenjtalk.g2p_prosody("相同組換え") == "^ s o [ o d o o # k u [ m i k a e $".split()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Sudachi との形態素境界の不一致で読み補正が戻され、一覧表の「表」が「オモテ」と読まれるため",
)
def test_homology_table_reference_known_reading() -> None:
    """
    「相同性の測定結果をまとめた（表）。」では、「相同」「性」の区切りが Sudachi の「相同性」と一致せず読み補正が戻されても、括弧内の一覧表を示す「表」が「ヒョウ」と読まれることを確認する。
    """

    assert (
        pyopenjtalk.g2p("相同性の測定結果をまとめた（表）。", kana=True)
        == "ソードーセーノソクテーケッカヲマトメタ（ヒョウ）。"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("光量が不足した。", "コーリョーガフソクシタ。"),
        ("測定した光量を記録する。", "ソクテーシタコーリョーヲキロクスル。"),
        ("周辺光量が落ちる。", "シューヘンコーリョーガオチル。"),
        ("光量不足で撮影を中止した。", "コーリョーブソクデサツエーヲチューシシタ。"),
        ("光の量を比べる。", "ヒカリノリョーヲクラベル。"),
        ("光子を検出した。", "コーシヲケンシュツシタ。"),
        ("光沢がある。", "コータクガアル。"),
    ],
)
def test_light_quantity_reading(text: str, expected: str) -> None:
    """
    「光量が不足した」「周辺光量」「光量不足」などで、辞書の「ヒカリリョウ」という誤った読みを使わず、「光量」が「コウリョウ」と読まれることを確認する。
    「光の量」「光子」「光沢」は別の語なので、それぞれの読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("堤体に亀裂が見つかった。", "テータイニキレツガミツカッタ。"),
        ("堤体内を点検した。", "テータイナイヲテンケンシタ。"),
        ("堤体の補修工事を行う。", "テータイノホシューコージヲオコナウ。"),
        ("ダムの堤体を調べた。", "ダムノテータイヲシラベタ。"),
        ("堤防を補強する。", "テーボーヲホキョースル。"),
        ("堤さんに会う。", "ツツミサンニアウ。"),
        ("本体を修理する。", "ホンタイヲシューリスル。"),
    ],
)
def test_dam_body_reading(text: str, expected: str) -> None:
    """
    「堤体に亀裂」「堤体内」「堤体の補修工事」などで、追加行の「ツツミタイ」という誤った読みを使わず、「堤体」が「テイタイ」と読まれることを確認する。
    「堤防」、姓の「堤」、「本体」は別の語なので、それぞれの読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_light_quantity_accent_and_mora_count() -> None:
    """
    「光量」の読みを「ヒカリリョウ」から「コウリョウ」に直すと5モーラから4モーラに変わるため、発音が「コーリョー」、アクセント核が3、モーラ数が4になることを確認する。
    """

    feature = pyopenjtalk.run_frontend("光量")[0]
    assert (feature["read"], feature["pron"], feature["acc"], feature["mora_size"]) == (
        "コウリョウ",
        "コーリョー",
        3,
        4,
    )


def test_dam_body_accent_and_mora_count() -> None:
    """
    「堤体」の読みを「ツツミタイ」から「テイタイ」に直すと5モーラから4モーラに変わるため、発音が「テータイ」、アクセント核が0、モーラ数が4になることを確認する。
    """

    feature = pyopenjtalk.run_frontend("堤体")[0]
    assert (feature["read"], feature["pron"], feature["acc"], feature["mora_size"]) == (
        "テイタイ",
        "テータイ",
        0,
        4,
    )


def test_handle_entry_keeps_allowance_word_boundary() -> None:
    """
    「持ち手」と途中まで表記が一致する「掛け持ち手当」で、「手当」の先頭が「持ち手」の行に取り込まれず、「手当」が1語として解析されることを確認する。
    「持ち手の長さ」では、「持ち手」が1語として解析され、3モーラの尾高型で発音されることを確認する。
    """

    allowance = pyopenjtalk.run_frontend("掛け持ち手当を申請する。")
    assert "手当" in [feature["string"] for feature in allowance]
    handle = pyopenjtalk.run_frontend("持ち手の長さを測る。")[0]
    assert (handle["string"], handle["read"], handle["acc"], handle["mora_size"]) == (
        "持ち手",
        "モチテ",
        3,
        3,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("本節では実験方法を説明する。", "ホンセツデワジッケンホーホーヲセツメースル。"),
        ("本節の図を参照してください。", "ホンセツノズヲサンショーシテクダサイ。"),
        ("結果を本節にまとめた。", "ケッカヲホンセツニマトメタ。"),
        ("本節を読み終えた。", "ホンセツヲヨミオエタ。"),
        ("本節で使う記号を定義する。", "ホンセツデツカウキゴーヲテーギスル。"),
        ("本節を削除する。", "ホンセツヲサクジョスル。"),
    ],
)
def test_current_section_reading(text: str, expected: str) -> None:
    """
    文書の節を指す「本節」が鰹節の「ホンブシ」の行と競合しても、「ホンセツ」と読まれることを確認する。
    「本節を削除する」も、鰹節の連語「本節を削る」と途中まで表記が一致していても、文書の節として読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("本節削りを料理に使う。", "ホンブシケズリヲリョーリニツカウ。"),
        ("鰹本節でだしを取る。", "カツオホンブシデダシヲトル。"),
        ("かつお本節を使う。", "カツオホンブシヲツカウ。"),
        ("枯本節を使う。", "カレホンブシヲツカウ。"),
        ("本節を削る。", "ホンブシヲケズル。"),
        ("本節を削った。", "ホンブシヲケズッタ。"),
        ("本節を削らない。", "ホンブシヲケズラナイ。"),
        ("本節を削ります。", "ホンブシヲケズリマス。"),
        ("本節を削ればよい。", "ホンブシヲケズレバヨイ。"),
        ("本節を削れ。", "ホンブシヲケズレ。"),
        ("本節を削ろう。", "ホンブシヲケズロー。"),
        ("本節を削りゃいい。", "ホンブシヲケズリャイイ。"),
    ],
)
def test_bonito_section_phrases(text: str, expected: str) -> None:
    """
    鰹節を表す「本節削り」「鰹本節」「かつお本節」「枯本節」と「本節を削る」の活用形が、文書の節を指す「ホンセツ」の行と競合しても、複合語や連語の行によって「ホンブシ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("資本節約型の設備を導入する。", "シホンセツヤクガタノセツビヲドーニュースル。"),
        ("日本の節約術を学ぶ。", "ニホンノセツヤクジュツヲマナブ。"),
        ("本の節目に印を付ける。", "ホンノフシメニシルシヲツケル。"),
        ("鰹節を削ってだしを取る。", "カツオブシヲケズッテダシヲトル。"),
        ("節分に豆をまく。", "セツブンニマメヲマク。"),
        ("本編の続きを読む。", "ホンペンノツズキヲヨム。"),
        ("本枯節でだしを取る。", "ホンカレブシデダシヲトル。"),
    ],
)
def test_current_section_preserves_neighboring_words(text: str, expected: str) -> None:
    """
    「資本節約型」の途中にある「本節」が1語として選ばれず、「資本」と「節約」の読みを保つことを確認する。
    表記に「本」や「節」を含む「日本」「節目」「鰹節」「節分」「本編」「本枯節」も、文書の節や鰹節の連語の行を追加しても読みが変わらないことを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "accent", "mora_size"),
    [("本節", 0, 4), ("本節削り", 0, 7), ("鰹本節", 0, 7), ("かつお本節", 0, 7), ("枯本節", 0, 6)],
)
def test_current_section_compound_accents(text: str, accent: int, mora_size: int) -> None:
    """
    「本節」「本節削り」「鰹本節」「かつお本節」「枯本節」が語全体で解析され、平板型の「本節」や「鰹節」と同じくアクセント核が0になることを確認する。
    「本節」と「削り」などの別の語へ分割されず、それぞれのモーラ数が保たれることを確認する。
    """

    features = pyopenjtalk.run_frontend(text)
    assert len(features) == 1
    assert features[0]["acc"] == accent
    assert features[0]["mora_size"] == mora_size


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("本節を削る", "^ h o [ N b u sh i o # k e [ z u r u $"),
        ("本節を削ります", "^ h o [ N b u sh i o # k e [ z u r i m a ] s U $"),
        ("本節を削ればよい", "^ h o [ N b u sh i o # k e [ z u r e ] b a # y o ] i $"),
        ("本節を削れ。", "^ h o [ N b u sh i o # k e [ z u r e _ $"),
    ],
)
def test_bonito_section_phrase_accents(text: str, expected: str) -> None:
    """
    鰹節の連語「本節を削る」のエントリが選ばれても、「本節を」と「削る」が別の平板型のアクセント句になることを確認する。
    助動詞「ます」や助詞「ば」が続く場合は動詞のアクセント変化が保たれ、命令文の「本節を削れ」でも名詞句と動詞の区切りが保たれることを確認する。
    """

    assert pyopenjtalk.g2p_prosody(text) == expected.split()


@pytest.mark.xfail(
    strict=True,
    reason="複合語や連語として登録していない鰹節の用法は、文書の節と同じ「ホンセツ」が選ばれる",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("本節は鰹節の一種だ。", "ホンブシワカツオブシノイッシュダ。"),
        ("本節を割って味見する。", "ホンブシヲワッテアジミスル。"),
        ("本節を削り取る。", "ホンブシヲケズリトル。"),
    ],
)
def test_bonito_section_without_registered_phrase(text: str, expected: str) -> None:
    """
    「鰹節の一種」として説明される「本節」や、料理のために割る「本節」が、文書の「ホンセツ」の行と競合しても「ホンブシ」と読まれることを確認する。
    「削り取る」が1語として選ばれて連語「本節を削り」のエントリが選ばれない場合も、食品の「本節」が「ホンブシ」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


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
