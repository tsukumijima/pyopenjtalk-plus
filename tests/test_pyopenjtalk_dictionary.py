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


# ============================================================
# 辞書未登録語の追加テスト
# report 2.1, 4.1, 9.7, Appendix の ✅ 判定エントリ
# ============================================================

UNREGISTERED_WORDS = [
    # 一般語
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
    "text, expected",
    UNREGISTERED_WORDS,
    ids=[t[0] for t in UNREGISTERED_WORDS],
)
def test_unregistered_words(text: str, expected: str) -> None:
    """辞書に未登録だった語が正しく読めることを検証する。"""

    result = pyopenjtalk.g2p(text, kana=True)
    assert result == expected, f"{text}: got {result!r}, expected {expected!r}"


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
    ("text", "expected_features", "expected_pron"),
    [
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
    ("皇道派", "コードーハ"),
    ("居城", "キョジョー"),
    ("城内", "ジョーナイ"),
    ("欠損歯", "ケッソンシ"),
    ("構造相転移", "コーゾーソーテンイ"),
    ("十二面体", "ジューニメンタイ"),
    ("彫刻刀", "チョーコクトー"),
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
