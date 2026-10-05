"""Python 側の読み・アクセント後処理を検証する。"""

import copy

import pytest

import pyopenjtalk
import pyopenjtalk.utils as pyopenjtalk_utils
from pyopenjtalk import NJDFeature
from pyopenjtalk.types import IuPronunciation
from pyopenjtalk.utils import modify_acc_after_chaining, restore_loanword_kana


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("黒﨑さん", "クロサキサン"),
        ("﨔の木", "ケヤキノキ"),
        ("𠮷野家", "ヨシノヤ"),
        ("𡈽井さん", "ドイサン"),
        ("醫學部に進む", "イガクブニススム"),
    ],
)
def test_normalize_itaiji_before_frontend(text: str, expected: str) -> None:
    """異体字を通用字体へ正規化することで、既存辞書に登録された語単位の読みが正しく利用されることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_itaiji_normalization_ignores_mixed_spaces() -> None:
    """半角と全角の空白が混ざっていても、未知語の異体字を正しい位置で置き換え、空白がない場合と同じ読みになることを確認する。"""

    assert pyopenjtalk.g2p(" 𠮷\u3000", kana=True) == pyopenjtalk.g2p("𠮷", kana=True) == "ヨシ"


def test_use_vanilla_keeps_itaiji_unchanged() -> None:
    """素の OpenJTalk 経路では異体字の表層を変更しない。"""

    assert pyopenjtalk.g2p("𠮷野家", kana=True, use_vanilla=True) == "𠮷ノヤ"


def test_itaiji_normalization_preserves_mapping_surface() -> None:
    """異体字で辞書を引いてもマッピングは呼び出し元の表層と文字位置を返す。"""

    mapping = pyopenjtalk.g2p_mapping("𠮷野家")

    assert "".join(entry["surface"] for entry in mapping) == "𠮷野家"
    assert mapping[0]["char_span"] == (0, 3)


@pytest.mark.parametrize(
    ("text", "expected_phonemes", "expected_prosody"),
    [
        ("地質", "ch i sh I ts u", "^chi[shItsu$"),
        ("地表", "ch i hy o o", "^chi[hyoo$"),
        ("地裁", "ch i s a i", "^chi[sai$"),
        ("地平", "ch i h e e", "^chi[hee$"),
        ("地層", "ch i s o o", "^chi[soo$"),
    ],
)
def test_affricate_before_fricative_keeps_vowel_voiced(
    text: str, expected_phonemes: str, expected_prosody: str
) -> None:
    """
    「地質」「地表」などでは、破擦音から摩擦音へ続く「チ」の母音を有声に保つ。
    「地質」の次の「シ」は、連続する無声化を抑える条件から外れて無声化する。
    """

    assert pyopenjtalk.g2p(text) == expected_phonemes
    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize(
    ("pronunciation", "expected"),
    [
        ("チサ", "チサ"),
        ("チシャ", "チシャ"),
        ("チフ", "チフ"),
        ("チハ", "チハ"),
        ("チヒョ", "チヒョ"),
        ("ツフュ", "ツフュ"),
        ("チュサ", "チュサ"),
        ("ツィヒ", "ツィヒ"),
        ("チキ", "チ’キ"),
        ("シチ", "シ’チ"),
        ("ヒキ", "ヒ’キ"),
    ],
)
def test_unvoicing_manner_exception_with_flat_accent(pronunciation: str, expected: str) -> None:
    """
    平板型の発音を NJD に渡し、アクセント核による有声化と子音の組み合わせによる例外を区別する。
    ch/ts から sh/s/f/fy/h/hy へ続く母音は有声に保ち、破裂音へ続く場合や逆向きの並びは無声化する。
    """

    jtalk = pyopenjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)
    mora_size = len(pyopenjtalk_utils.split_kana_mora(pronunciation))
    features = jtalk.run_njd_from_mecab(
        [f"発音,名詞,一般,*,*,*,*,発音,{pronunciation},{pronunciation},0/{mora_size},C1"]
    )

    assert features[0]["pron"] == expected


def test_g2p_nani_model():
    """「何」の文脈依存読みがモデル有無で切り替わる。"""

    test_cases = [
        {
            "text": "何か問題があれば何でも言ってください、どんな些細なことでも何とかします。",
            "pron_without_nani": "ナニカモンダイガアレバナニデモイッテクダサイ、ドンナササイナコトデモナニトカシマス。",
            "pron_with_nani": "ナニカモンダイガアレバナンデモイッテクダサイ、ドンナササイナコトデモナントカシマス。",
        },
        {
            "text": "何か特別なことをしたわけではありませんが、何故か周りの人々が何かと気にかけてくれます。何と言えばいいのか分かりません。",
            "pron_without_nani": "ナニカトクベツナコトヲシタワケデワアリマセンガ、ナゼカマワリノヒトビトガナニカトキニカケテクレマス。ナニトイエバイイノカワカリマセン。",
            "pron_with_nani": "ナニカトクベツナコトヲシタワケデワアリマセンガ、ナゼカマワリノヒトビトガナニカトキニカケテクレマス。ナントイエバイイノカワカリマセン。",
        },
        {
            "text": "私も何とかしたいですが、何でも行くリソースはありません。",
            "pron_without_nani": "ワタシモナニトカシタイデスガ、ナニデモイクリソースワアリマセン。",
            "pron_with_nani": "ワタシモナントカシタイデスガ、ナンデモイクリソースワアリマセン。",
        },
        {
            "text": "何を言っても何の問題もありません。",
            "pron_without_nani": "ナニヲイッテモナニノモンダイモアリマセン。",
            "pron_with_nani": "ナニヲイッテモナンノモンダイモアリマセン。",
        },
        {
            "text": "これは何ですか？何の情報？",
            "pron_without_nani": "コレワナニデスカ？ナンノジョーホー？",
            "pron_with_nani": "コレワナンデスカ？ナンノジョーホー？",
        },
        {
            "text": "何だろう、何でも嘘つくのやめてもらっていいですか？",
            "pron_without_nani": "ナニダロー、ナニデモウソツクノヤメテモラッテイイデスカ？",
            "pron_with_nani": "ナンダロー、ナンデモウソツクノヤメテモラッテイイデスカ？",
        },
        {
            "text": "質問は何のことかな？",
            "pron_without_nani": "シツモンワナンノコトカナ？",
            "pron_with_nani": "シツモンワナンノコトカナ？",
        },
    ]

    # without nani model
    for case in test_cases:
        p = pyopenjtalk.g2p(case["text"], kana=True, use_vanilla=True)
        assert p == case["pron_without_nani"]

    # with nani model
    for case in test_cases:
        p = pyopenjtalk.g2p(case["text"], kana=True, use_vanilla=False)
        assert p == case["pron_with_nani"]


@pytest.mark.parametrize(
    "text",
    [
        "何を選ぶ",
        "何が必要だ",
        "何に使う",
        "何もない",
        "何するつもりだ",
        "何であるか",
    ],
)
def test_predict_nani_reading_keeps_high_confidence_nani_rules(
    text: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """後続形態素だけで確定する「何」はモデル誤判定より確実なナニ規則を優先する。"""

    def fail_predict(_features: list[NJDFeature | None]) -> int:
        """高確信ナニ規則でモデル推論が呼ばれた場合は失敗させる。"""

        raise AssertionError("predict() must not be called for a high-confidence ナニ context")

    monkeypatch.setattr(pyopenjtalk_utils, "predict", fail_predict)
    njd_features = pyopenjtalk.run_frontend(text, predict_nani=False)

    corrected_features = pyopenjtalk_utils.predict_nani_reading(njd_features)

    nani_feature = next(feature for feature in corrected_features if feature["orig"] == "何")
    assert nani_feature["read"] == "ナニ"
    assert nani_feature["pron"] == "ナニ"


def test_predict_nani_reading_keeps_product_default_before_case_particle_de(
    monkeypatch: pytest.MonkeyPatch,
):
    """格助詞「で」の前にある「何」は、製品既定のナンを維持する。"""

    def fail_predict(_features: list[NJDFeature | None]) -> int:
        """格助詞「で」の確定規則からモデル推論へ進んだ場合は失敗させる。"""

        raise AssertionError("predict() must not be called before the case particle で")

    monkeypatch.setattr(pyopenjtalk_utils, "predict", fail_predict)
    njd_features = pyopenjtalk.run_frontend("何で塗る", predict_nani=False)

    corrected_features = pyopenjtalk_utils.predict_nani_reading(njd_features)

    nani_feature = next(feature for feature in corrected_features if feature["orig"] == "何")
    assert nani_feature["read"] == "ナン"
    assert nani_feature["pron"] == "ナン"


@pytest.mark.parametrize(
    ("text", "expected_next_orig"),
    [
        ("答えは何", None),
        ("何かを選ぶ", "か"),
    ],
)
def test_predict_nani_reading_uses_model_outside_high_confidence_rules(
    text: str,
    expected_next_orig: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    """文末と曖昧な後続語では「何」の読みをモデルへ問い合わせる。"""

    received_features: list[list[NJDFeature | None]] = []

    def predict_nan(features: list[NJDFeature | None]) -> int:
        """モデルへ渡された後続形態素を記録してナン判定を返す。"""

        received_features.append(features)
        return 1

    monkeypatch.setattr(pyopenjtalk_utils, "predict", predict_nan)
    njd_features = pyopenjtalk.run_frontend(text, predict_nani=False)

    corrected_features = pyopenjtalk_utils.predict_nani_reading(njd_features)

    assert len(received_features) == 1
    next_feature = received_features[0][0]
    assert (next_feature["orig"] if next_feature is not None else None) == expected_next_orig
    nani_feature = next(feature for feature in corrected_features if feature["orig"] == "何")
    assert nani_feature["read"] == "ナン"
    assert nani_feature["pron"] == "ナン"


def test_modify_kanji_yomi_does_not_partially_mutate_on_alignment_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sudachi と NJD の途中不一致では照合済み形態素も変更しない。"""

    njd_features = pyopenjtalk.run_frontend(
        "外国人と数百人",
        use_sudachi_kanji_yomi=False,
    )
    original_features = copy.deepcopy(njd_features)

    def return_partial_sudachi_result(_text: str, _targets: frozenset[str]) -> list[list[str]]:
        """NJD の途中までしか対応しない Sudachi 解析結果を返す。"""

        return [["人", "ジン"], ["テスト"]]

    monkeypatch.setattr(
        pyopenjtalk_utils,
        "sudachi_analyze",
        return_partial_sudachi_result,
    )

    corrected_features = pyopenjtalk_utils.modify_kanji_yomi(
        "外国人と数百人",
        njd_features,
        frozenset({"人"}),
    )

    assert corrected_features == original_features
    assert njd_features == original_features


def test_modify_kanji_yomi_separates_hou_reading_and_pronunciation(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sudachi の「方」（「ホウ」）を正書法の読みと長音発音へ分ける。"""

    njd_features = pyopenjtalk.run_frontend(
        "その方",
        use_sudachi_kanji_yomi=False,
    )

    def return_hou(_text: str, _targets: frozenset[str]) -> list[list[str]]:
        """特殊変換の入力となる Sudachi の読みを返す。"""

        return [["方", "ホウ"]]

    monkeypatch.setattr(pyopenjtalk_utils, "sudachi_analyze", return_hou)

    corrected_features = pyopenjtalk_utils.modify_kanji_yomi(
        "その方",
        njd_features,
        frozenset({"方"}),
    )

    hou_feature = next(feature for feature in corrected_features if feature["orig"] == "方")
    assert hou_feature["read"] == "ホウ"
    assert hou_feature["pron"] == "ホー"


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("下の方", "シタノホー"),
        ("右の方", "ミギノホー"),
        ("左の方", "ヒダリノホー"),
        ("その方", "ソノカタ"),
    ),
)
def test_g2p_directional_hou_keeps_long_vowel_pronunciation(
    text: str,
    expected_kana: str,
) -> None:
    """方向を表す「方」は「ホー」、人を表す連体詞後の「方」は「カタ」と読む。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected_kana


@pytest.mark.parametrize(
    ("text", "surface", "expected_pron", "expected_mora_count"),
    (
        ("表と裏", "表", "ヒョウ", 2),
        ("下の方", "方", "ホー", 2),
    ),
)
def test_modify_kanji_yomi_updates_mora_size(
    text: str,
    surface: str,
    expected_pron: str,
    expected_mora_count: int,
) -> None:
    """Sudachi 補正で読みを差し替えた形態素は、モーラ数も差し替え後の発音に揃える。"""

    mapping = pyopenjtalk.g2p_mapping(text)
    corrected = next(entry for entry in mapping if entry["surface"] == surface)

    assert corrected["pron"] == expected_pron
    assert corrected["mora_count"] == expected_mora_count


def test_g2p_nani_model_does_not_require_sudachi_when_only_nani(monkeypatch: pytest.MonkeyPatch):
    """「何」の読み推定だけなら Sudachi を読み込まない。"""

    def fail_sudachi_analyze(_text: str, _targets: frozenset[str]) -> list[list[str]]:
        """「何」だけの補正で Sudachi 解析が呼ばれた場合は失敗させる。"""

        raise AssertionError("sudachi_analyze should not be called for '何'-only correction")

    monkeypatch.setattr(pyopenjtalk_utils, "sudachi_analyze", fail_sudachi_analyze)

    assert pyopenjtalk.g2p("これは何ですか？", kana=True) == "コレワナンデスカ？"


def test_g2p_predict_nani_can_be_disabled():
    """「何」の読み推定を個別に無効化できる。"""

    assert pyopenjtalk.g2p("何ですか", kana=True, predict_nani=True) == "ナンデスカ"
    assert pyopenjtalk.g2p("何ですか", kana=True, predict_nani=False) == "ナニデスカ"


@pytest.mark.parametrize(
    ("text", "expected_pronunciation", "expected_surfaces"),
    [
        ("何でそんなことを言うの？", "ナンデソンナコトヲイウノ？", ["何", "で"]),
        ("何でかな？", "ナンデカナ？", ["何", "で"]),
        ("何でもできる", "ナンデモデキル", ["何", "でも"]),
        ("何では駄目ですか？", "ナンデワダメデスカ？", ["何", "で", "は"]),
        ("何でしょうか？", "ナンデショーカ？", ["何", "でしょ", "う", "か", "？"]),
        ("何であるかを説明する。", "ナニデアルカヲセツメースル。", ["何", "で", "ある"]),
        ("何を使いますか？", "ナニヲツカイマスカ？", ["何", "を"]),
    ],
)
def test_nande_cost_keeps_neighboring_expression_boundaries(
    text: str,
    expected_pronunciation: str,
    expected_surfaces: list[str],
):
    """「何で」の分割を安定させても、隣接表現の形態素境界と読みを維持できる。"""

    morphs = pyopenjtalk.run_frontend_detailed(text, predict_nani=False)[1]
    surfaces = [morph["surface"] for morph in morphs if morph["is_ignored"] is False]

    assert pyopenjtalk.g2p(text, kana=True) == expected_pronunciation
    assert surfaces[: len(expected_surfaces)] == expected_surfaces


def test_g2p_can_disable_sudachi_kanji_yomi_and_keep_nani_enabled():
    """Sudachi の漢字読み補正を無効化しても「何」の読み推定を維持する。"""

    text = "風がこんな風に吹く。これは何ですか？"

    assert (
        pyopenjtalk.g2p(
            text,
            kana=True,
            use_sudachi_kanji_yomi=False,
            predict_nani=True,
        )
        == "カゼガコンナカゼニフク。コレワナンデスカ？"
    )


@pytest.mark.parametrize(
    ("text", "expected_phonemes", "expected_kana"),
    [
        ("しなじう", ["sh", "i", "n", "a", "j", "i", "u"], "シナジウ"),
        ("いみじう", ["i", "m", "i", "j", "i", "u"], "イミジウ"),
        ("買わう", ["k", "a", "w", "a", "u"], "カワウ"),
        ("捨てう", ["s", "U", "t", "e", "u"], "ステウ"),
        ("行こう", ["i", "k", "o", "o"], "イコー"),
        ("言おう", ["i", "o", "o"], "イオー"),
    ],
)
def test_g2p_auxiliary_u_long_vowel_revert(
    text: str,
    expected_phonemes: list[str],
    expected_kana: str,
):
    """助動詞「う」の長音を読み表記へ復元する。"""

    assert pyopenjtalk.g2p(text, join=False) == expected_phonemes
    assert pyopenjtalk.g2p(text, kana=True) == expected_kana


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一見さんです", "イチゲンサンデス"),
        ("一声かける", "ヒトコエカケル"),
        ("一行ごとに", "イチギョーゴトニ"),
        ("兵ども", "ツワモノドモ"),
        ("阿弥陀仏", "アミダブツ"),
        ("先生方", "センセーガタ"),
        ("支配の下", "シハイノモト"),
        ("紀元前", "キゲンゼン"),
        ("同じ様に", "オナジヨーニ"),
        ("如何ですか", "イカガデスカ"),
        ("笠間焼", "カサマヤキ"),
        ("匹見峡", "ヒキミキョー"),
        ("秋田分屯基地", "アキタブントンキチ"),
        ("九角形の図形", "キューカクケーノズケー"),
        ("室町通を歩く", "ムロマチドーリヲアルク"),
        ("四条通", "シジョードーリ"),
        ("軍艦旗", "グンカンキ"),
        ("可動橋", "カドーキョー"),
        ("山寺", "ヤマデラ"),
        ("霊山寺を訪ねる", "リョーゼンジヲタズネル"),
        ("防衛記念章", "ボーエーキネンショー"),
        ("徳川公", "トクガワコー"),
        ("歯茎硬口蓋音", "ハグキコーコーガイオン"),
        ("諏訪洞を探検する", "スワドーヲタンケンスル"),
        ("宮沢湖のほとり", "ミヤザワコノホトリ"),
        ("風景印", "フーケーイン"),
        ("山小屋", "ヤマゴヤ"),
        ("貴乃花部屋", "タカノハナベヤ"),
        ("三角形の和", "サンカッケーノワ"),
        ("資金不足", "シキンブソク"),
        ("機器等", "キキトー"),
        ("それ等", "ソレラ"),
        ("時が経つ", "トキガタツ"),
        ("時は金なり", "トキワカネナリ"),
        ("修道尼の生活", "シュードーニノセーカツ"),
        ("料理茶屋に入る", "リョーリジャヤニハイル"),
        ("東京に於て開催する", "トーキョーニオイテカイサイスル"),
        ("彼を識っている", "カレヲシッテイル"),
        ("芭蕉翁の句", "バショーオーノク"),
        ("先生が仰しゃる", "センセーガオッシャル"),
        ("忘れて了った", "ワスレテシマッタ"),
        ("落ち込んでいた処に声をかけた", "オチコンデイタトコロニコエヲカケタ"),
        ("傘の柄を握る", "カサノエヲニギル"),
        ("剣の柄を握る", "ケンノツカヲニギル"),
        ("刀の柄を握る", "カタナノツカヲニギル"),
        ("そう思うより外にない", "ソーオモウヨリホカニナイ"),
        ("大津波に備える", "オーツナミニソナエル"),
        ("御言葉をいただく", "オコトバヲイタダク"),
        ("医学博士になる", "イガクハクシニナル"),
        ("寺小屋で学ぶ", "テラコヤデマナブ"),
        ("いつか公になる", "イツカオーヤケニナル"),
        ("道路橋を渡る", "ドーロキョーヲワタル"),
        ("就学前の児童", "シューガクマエノジドー"),
        ("前会長の話です。", "ゼンカイチョーノハナシデス。"),
        ("前首相が来日した", "ゼンシュショーガライニチシタ"),
        ("この間の話です。", "コノアイダノハナシデス。"),
        ("領収書に金五万円と書いた", "リョーシューショニキンゴマンエントカイタ"),
        ("小切手に金１０万円と記す", "コギッテニキンジューマンエントシルス"),
        ("彼は部下に服従を強いた", "カレワブカニフクジューヲシイタ"),
        ("番号の下四桁を入力する", "バンゴーノシモヨンケタヲニューリョクスル"),
        ("番号の上二桁を確認する", "バンゴーノカミフタケタヲカクニンスル"),
        ("七つ、八つ、九つ、十と数える", "ナナツ、ヤッツ、ココノツ、トートカゾエル"),
        ("謹んで申し上げ候", "ツツシンデモーシアゲソーロー"),
    ],
)
def test_modify_context_reading(text: str, expected: str) -> None:
    """前後の文脈（隣接する形態素）によって読みが一意に決まる複合語や定型表現が、意図通り正しく読み分けられることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("白老町史に残る", "シラオイチョーシニノコル"),
        ("おばさんも娘も一しょに大声で笑った", "オバサンモムスメモイッショニオーゴエデワラッタ"),
        ("青の背景と金の縁がある", "アオノハイケートカネノフチガアル"),
        ("ペトリ皿の縁", "ペトリサラノフチ"),
        ("成り上がり者は", "ナリアガリモノワ"),
        ("繁みの中に走り入ったとき", "シゲミノナカニハシリイッタトキ"),
        ("朝鮮、台湾に攻め入った", "チョーセン、タイワンニセメイッタ"),
        ("折り入って話がある", "オリイッテハナシガアル"),
        (
            "マングローブ林を数十メートル分け入った場所",
            "マングローブリンヲスージューメートルワケイッタバショ",
        ),
        ("プレス金型を使用して", "プレスカナガタヲシヨーシテ"),
        ("高圧金型鋳造の導入", "コーアツカナガタチューゾーノドーニュー"),
    ],
)
def test_context_reading_additional_compounds(text: str, expected: str) -> None:
    """
    「町史」「一しょ」と器物の「縁」は、後続語や「の」の前の語から読みを決める。
    活用語に続く「者」、確認できた複合動詞の「入っ」、鋳型を表す「金型」も文脈に応じた読みへ補正する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("何にも知らない", "ナンニモシラナイ"),
        ("何にも知りません", "ナンニモシリマセン"),
        ("何にも知らず", "ナンニモシラズ"),
        ("何にも知らん", "ナンニモシラン"),
        ("何にも知らぬ", "ナンニモシラヌ"),
        ("何にも分からなかった", "ナンニモワカラナカッタ"),
        ("何にも知らなければ", "ナンニモシラナケレバ"),
        ("何にも答えてもらっていない", "ナンニモコタエテモラッテイナイ"),
        ("何にも返事がなく", "ナンニモヘンジガナク"),
        ("何にもありません", "ナンニモアリマセン"),
        ("何にもすることができず", "ナンニモスルコトガデキズ"),
        ("何にも面白くない", "ナンニモオモシロクナイ"),
        ("社会のこと何にも知らず", "シャカイノコトナンニモシラズ"),
        ("何にもしないで宣伝ばかり", "ナンニモシナイデセンデンバカリ"),
        ("何にも似ていない", "ナニニモニテイナイ"),
        ("何にも全然似ていない", "ナニニモゼンゼンニテイナイ"),
        ("何にも代えがたい", "ナニニモカエガタイ"),
        ("何にも依存していない", "ナニニモイゾンシテイナイ"),
        ("何にも属さない", "ナニニモゾクサナイ"),
        ("何にも頼らない", "ナニニモタヨラナイ"),
        ("何にも答えたが教えない", "ナニニモコタエタガオシエナイ"),
        ("何にも。知らない", "ナニニモ。シラナイ"),
        ("「何にも」という語を知らない", "「ナニニモ」トイウゴヲシラナイ"),
        ("何にも似ることができない", "ナニニモニルコトガデキナイ"),
        ("何にもとづいて決めた", "ナニニモトズイテキメタ"),
        ("如何にも知らない", "イカニモシラナイ"),
    ],
)
def test_context_reading_nannimo(text: str, expected: str) -> None:
    """
    打ち消しの「何にも知らない」は「ナンニモ」と読み、格助詞の「何にも似ていない」は「ナニニモ」を保つ。
    「ません」「ず」や補助動詞を含む否定も扱い、別の節・引用の後ろにある否定は元の読みを保つ。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_context_reading_nannimo_preserves_other_features() -> None:
    """
    「何にも知らない」は「何」の読みと発音だけを変え、アクセント核と句の区切りを保つ。
    保護された「何」の読みと、use_vanilla=True の出力は元のままにする。
    """

    features = pyopenjtalk.run_frontend("何にも知らない", use_vanilla=True)
    original = copy.deepcopy(features)
    corrected = pyopenjtalk_utils.modify_context_reading(features)
    original[0]["read"] = original[0]["pron"] = "ナン"
    assert corrected == original
    assert pyopenjtalk.g2p("何にも知らない") == "n a N n i m o sh i r a n a i"
    assert pyopenjtalk.g2p("何にも知らない", kana=True, use_vanilla=True) == "ナニニモシラナイ"

    protected = pyopenjtalk.run_frontend("何にも知らない", use_vanilla=True)
    protected[0]["is_reading_protected"] = True
    original = copy.deepcopy(protected)
    assert pyopenjtalk_utils.modify_context_reading(protected) == original


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一見して分かる", "イッケンシテワカル"),
        ("一声も出ない", "イッセーモデナイ"),
        ("兵の数", "ヘーノカズ"),
        ("一行が", "イッコーガ"),
        ("この方", "コノカタ"),
        ("夕方", "ユーガタ"),
        ("読み方", "ヨミカタ"),
        ("念仏", "ネンブツ"),
        ("机の下", "ツクエノシタ"),
        ("橋の下を", "ハシノシタヲ"),
        ("情報が不足する", "ジョーホーガフソクスル"),
        ("情報の不足", "ジョーホーノフソク"),
        ("不足を補う", "フソクヲオギナウ"),
        ("屯する若者", "タムロスルワカモノ"),
        ("旗を振る", "ハタヲフル"),
        ("山が焼ける", "ヤマガヤケル"),
        ("公園前", "コーエンマエ"),
        ("縁切寺", "エンキリデラ"),
        ("駆け込み寺", "カケコミデラ"),
        ("田舎寺", "イナカデラ"),
        ("お寺の鐘", "オテラノカネ"),
        ("洞に隠れる", "ホラニカクレル"),
        ("湖を眺める", "ミズウミヲナガメル"),
        ("部屋に入る", "ヘヤニハイル"),
        ("公の場", "オーヤケノバ"),
        ("十八角形", "ジューハチカクケー"),
        ("一等", "イットー"),
        ("三等賞", "サントーショー"),
        ("等分", "トーブン"),
        ("高等学校", "コートーガッコー"),
        ("平等", "ビョードー"),
        ("等しい", "ヒトシイ"),
        ("如何にも", "イカニモ"),
        ("中国", "チューゴク"),
        ("外国", "ガイコク"),
        ("帝国", "テーコク"),
        ("韓国", "カンコク"),
        ("開催時", "カイサイジ"),
        ("15時が過ぎた", "ジューゴジガスギタ"),
        ("梅雨時", "ツユドキ"),
        ("時を待つ", "トキヲマツ"),
        ("この時がきた", "コノトキガキタ"),
        ("100年の時が経つ", "ヒャクネンノトキガタツ"),
        ("翌日尼が来た", "ヨクジツアマガキタ"),
        ("明日翁が来る", "アシタオキナガクル"),
        ("二ツ茶屋に行く", "フタツチャヤニイク"),
        ("広瀬博士は話した", "ヒロセハカセワハナシタ"),
        ("地球より外に惑星はない", "チキューヨリソトニワクセーワナイ"),
        ("就学前教育", "シューガクゼンキョーイク"),
        ("前の会長", "マエノカイチョー"),
        ("この間抜けな娘", "コノマヌケナムスメ"),
        ("賞金百万円", "ショーキンヒャクマンエン"),
        ("お金を払う", "オカネヲハラウ"),
        ("主として学生が使う", "シュトシテガクセーガツカウ"),
        ("利用者は主として高齢者だ", "リヨーシャワシュトシテコーレーシャダ"),
        ("彼の主として研究した分野", "カレノシュトシテケンキューシタブンヤ"),
        ("野外での主として鳥類の観察", "ヤガイデノシュトシテチョールイノカンサツ"),
        ("強い風が吹いた", "ツヨイカゼガフイタ"),
        ("風が強いため中止した", "カゼガツヨイタメチューシシタ"),
        ("強い国が勝つ", "ツヨイクニガカツ"),
        ("机の下に置く", "ツクエノシタニオク"),
        ("九つ、十人いる", "ココノツ、ジューニンイル"),
        ("天候が悪い", "テンコーガワルイ"),
    ],
)
def test_modify_context_reading_keeps_negative_examples(text: str, expected: str) -> None:
    """文脈補正規則の対象外となる通常の文脈では、誤って補正が発動せず辞書本来の読みが維持されることを確認する（ネガティブテスト）。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("受者の責任", "ウケシャノセキニン"),
        ("ちょっと下の人間のひがみ入ってます", "チョットシタノニンゲンノヒガミハイッテマス"),
        ("低解約返戻金型終身保険", "テーカイヤクヘンレーキンガタシューシンホケン"),
        ("縁を切る", "エンヲキル"),
        ("手に入った", "テニハイッタ"),
    ],
)
def test_context_reading_keeps_short_stems_and_monetary_compounds(text: str, expected: str) -> None:
    """
    動詞として解析される「受」「ひがみ」を規則の対象から外し、「シャ」「ハイッ」を保つ。
    「返戻金型」は「返戻金」に「型」が付く表現なので「キン」を保ち、「縁を切る」「手に入った」も辞書の読みを維持する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


@pytest.mark.parametrize(("accent", "expected"), [(0, 0), (1, 1), (2, 1), (4, 3)])
def test_context_reading_keeps_nucleus_when_mora_count_changes(accent: int, expected: int) -> None:
    """
    「識っている」の「シキ」を「シ」へ縮めても、後続するモーラにあったアクセント核の位置を保つ。
    消えるモーラに核があった場合は「シ」に置き、平板型の核は0のままにする。
    """

    features = pyopenjtalk.run_frontend("識っている", use_vanilla=True)
    features[0]["acc"] = accent
    corrected = pyopenjtalk_utils.modify_context_reading(features)

    assert corrected[0]["pron"] == "シ"
    assert corrected[0]["mora_size"] == 1
    assert corrected[0]["acc"] == expected


def test_context_reading_preserves_protected_reading_and_devoicing() -> None:
    """
    保護された「識」は、登録された読みとアクセント句の核をそのまま保つ。
    既に「ハクシ」と読める「博士」は、NJD が付けた無声化の記号もそのまま残す。
    """

    protected = pyopenjtalk.run_frontend("識っている", use_vanilla=True)
    protected[0]["is_reading_protected"] = True
    original = copy.deepcopy(protected)
    assert pyopenjtalk_utils.modify_context_reading(protected) == original

    features = pyopenjtalk.run_frontend("博士論文", use_vanilla=True)
    features[0]["read"] = "ハクシ"
    features[0]["pron"] = "ハク’シ"
    assert pyopenjtalk_utils.modify_context_reading(features)[0]["pron"] == "ハク’シ"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("兵どもが", "ヘードモガ"),
        ("一声かけて", "イッセーカケテ"),
        ("皆様方", "ミナサマカタ"),
    ],
)
def test_context_reading_can_be_disabled(text: str, expected: str) -> None:
    """use_vanilla=True を指定した場合に、文脈による読み補正が無効化され辞書本来の読みが出力されることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("石見国", "イワミノクニ"),
        ("越後国", "エチゴノクニ"),
        ("阿波国", "アワノクニ"),
        ("岩代国", "イワシロノクニ"),
        ("大和国", "ヤマトノクニ"),
        ("中国に行く", "チューゴクニイク"),
        ("外国の文化", "ガイコクノブンカ"),
    ],
)
def test_modify_old_province_yomi(text: str, expected: str) -> None:
    """旧国名に続く接尾辞「国」だけが「ノクニ」と読まれ、1語の「中国」「外国」は変わらないことを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_modify_old_province_yomi_can_be_disabled() -> None:
    """use_vanilla=True を指定した場合は、旧国名に続く「国」が辞書の読みのままになることを確認する。"""

    assert pyopenjtalk.g2p("石見国", kana=True, use_vanilla=True) == "イワミコク"


@pytest.mark.parametrize(
    ("surface", "pronunciation", "expected", "expected_mora_size"),
    [
        ("ヴィクトリーヌ", "ビク’トリーヌ", "ヴィク’トリーヌ", 6),
        ("アイシュヴァルヤ", "アイシュバルヤ", "アイシュヴァルヤ", 6),
        ("テュルク", "チュルク", "テュルク", 3),
        ("アクスィス", "アクシス", "アクスィス", 4),
    ],
)
def test_restore_loanword_kana(
    surface: str,
    pronunciation: str,
    expected: str,
    expected_mora_size: int,
) -> None:
    """「ヴィ」や「テュ」など、辞書で一般的な仮名に置き換えられた外来語の読み・発音表記が、元の表層形に合わせて正しく復元されることを確認する。"""

    feature = NJDFeature(
        string=surface,
        pos="名詞",
        pos_group1="固有名詞",
        pos_group2="*",
        pos_group3="*",
        ctype="*",
        cform="*",
        orig=surface,
        read=pronunciation,
        pron=pronunciation,
        acc=0,
        mora_size=0,
        chain_rule="*",
        chain_flag=-1,
    )

    restore_loanword_kana([feature])

    assert feature["read"] == expected
    assert feature["pron"] == expected
    # 復元後の発音に合わせて mora_size が正しく再計算されていることを検証する
    assert feature["mora_size"] == expected_mora_size


@pytest.mark.parametrize(
    ("surface", "pronunciation"),
    [
        ("ホンデュラス", "ホンジュラス"),
        ("バースディ", "バースデイ"),
        ("キウィ", "キウイ"),
        ("エヌ・エイチ・ヴィ", "エヌエイチブイ"),
    ],
)
def test_restore_loanword_kana_keeps_unmatched_pronunciation(
    surface: str,
    pronunciation: str,
) -> None:
    """「ホンデュラス」のように表層と発音の対応が一対一にならない語では、無理に復元せず辞書本来の発音を維持することを確認する。"""

    feature = NJDFeature(
        string=surface,
        pos="名詞",
        pos_group1="固有名詞",
        pos_group2="*",
        pos_group3="*",
        ctype="*",
        cform="*",
        orig=surface,
        read=pronunciation,
        pron=pronunciation,
        acc=0,
        mora_size=0,
        chain_rule="*",
        chain_flag=-1,
    )

    restore_loanword_kana([feature])

    assert feature["read"] == pronunciation
    assert feature["pron"] == pronunciation


def test_restore_loanword_kana_in_g2p() -> None:
    """既定の後処理で「ヴィクトリーヌ」の表記が戻り、use_vanilla=True では辞書の「ビクトリーヌ」のままになることを確認する。"""

    assert pyopenjtalk.g2p("ヴィクトリーヌ", kana=True) == "ヴィクトリーヌ"
    assert pyopenjtalk.g2p("ヴィクトリーヌ", kana=True, use_vanilla=True) == "ビクトリーヌ"


def test_restore_loanword_kana_moves_accent_nucleus() -> None:
    """「イエ」を「イェ」に戻してモーラが減っても、アクセント核が戻す前と同じ「テ」のモーラに置かれることを確認する。"""

    vanilla_features = pyopenjtalk.run_frontend("イェテボリ", use_vanilla=True)
    assert [(feature["pron"], feature["acc"]) for feature in vanilla_features] == [
        ("イエテボリ", 3)
    ]
    features = pyopenjtalk.run_frontend("イェテボリ")
    assert [(feature["pron"], feature["acc"], feature["mora_size"]) for feature in features] == [
        ("イェテボリ", 2, 4)
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("騸馬", "センバ"),
        ("嚳", "コク"),
        ("多禰国", "タネコク"),
        ("嗅神経", "キューシンケー"),
        ("痘", "トー"),
        ("こんにちは、世界。", "コンニチワ、セカイ。"),
    ],
)
def test_read_unknown_kanji(text: str, expected: str) -> None:
    """デフォルト辞書に登録されていない未知漢字についてのみ、Unihan の読みデータで補完されることを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True) == expected


def test_read_unknown_kanji_is_enabled_by_default() -> None:
    """未知漢字がデフォルトで補完され、送り仮名を伴う語については Sudachi による語単位の訓読みが適用されることを確認する。"""

    assert pyopenjtalk.g2p("騸馬", kana=True) == "センバ"
    assert pyopenjtalk.g2p("悪魔憑き", kana=True) == "アクマツキ"
    assert pyopenjtalk.g2p("取り憑く", kana=True) == "トリツク"


def test_read_unknown_kanji_does_not_load_sudachi_without_unknown_kanji(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """読めない漢字がない文では、Sudachi の読み補正を切っていれば未知漢字の処理でも Sudachi を読み込まないことを確認する。"""

    def fail_get_sudachi_tokenizer() -> None:
        """未知の漢字がないのに Sudachi が読み込まれた場合は失敗させる。"""

        raise AssertionError("Sudachi should not be loaded without unknown kanji")

    monkeypatch.setattr(pyopenjtalk_utils, "_get_sudachi_tokenizer", fail_get_sudachi_tokenizer)

    assert (
        pyopenjtalk.g2p("今日はいい天気ですね", kana=True, use_sudachi_kanji_yomi=False)
        == "キョーワイイテンキデスネ"
    )


@pytest.mark.parametrize(
    ("text", "following_mora_size"),
    [("本論文", 4), ("当ホテル", 3), ("同制度", 3), ("全項目", 4)],
)
def test_independent_prefix_starts_new_accent_phrase(
    text: str,
    following_mora_size: int,
) -> None:
    """指示的な漢語接頭辞の後続語は独立したアクセント句として扱う。"""

    features = pyopenjtalk.run_frontend(text)

    assert features[0]["pos"] == "接頭詞"
    assert features[1]["chain_flag"] == 0

    # フルコンテキストラベルでも、接頭辞2モーラと後続語の独立した句長・核位置を保持する
    labels = pyopenjtalk.make_label(features)
    assert any("/F:2_1#" in label for label in labels)
    assert any(f"/F:{following_mora_size}_" in label for label in labels)


def test_fused_prefix_keeps_same_accent_phrase() -> None:
    """語彙的に後続語と融合する接頭辞は既定のアクセント句を維持する。"""

    features = pyopenjtalk.run_frontend("新製品")

    assert features[0]["pos"] == "接頭詞"
    assert features[1]["chain_flag"] == 1


def test_odoriji():
    """踊り字を直前の漢字と読みに従って展開する。"""

    # 一の字点（ゝ、ゞ、ヽ、ヾ）の処理テスト
    # 濁点なしの一の字点
    njd_features = pyopenjtalk.run_frontend("なゝ樹")
    assert njd_features[0]["read"] == "ナ"
    assert njd_features[0]["pron"] == "ナ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ナ"
    assert njd_features[1]["pron"] == "ナ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "キ"
    assert njd_features[2]["pron"] == "キ"
    assert njd_features[2]["mora_size"] == 1

    # 濁点ありの一の字点
    njd_features = pyopenjtalk.run_frontend("金子みすゞ")
    assert njd_features[0]["read"] == "カネコ"
    assert njd_features[0]["pron"] == "カネコ"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "ミス"
    assert njd_features[1]["pron"] == "ミス"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ズ"
    assert njd_features[2]["pron"] == "ズ"
    assert njd_features[2]["mora_size"] == 1

    # 濁点なしの一の字点（づゝ）
    njd_features = pyopenjtalk.run_frontend("づゝ")
    assert njd_features[0]["read"] == "ヅ"
    assert njd_features[0]["pron"] == "ヅ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ツ"
    assert njd_features[1]["pron"] == "ツ"
    assert njd_features[1]["mora_size"] == 1

    # 濁点ありの一の字点（ぶゞ漬け）
    njd_features = pyopenjtalk.run_frontend("ぶゞ漬け")
    assert njd_features[0]["read"] == "ブ"
    assert njd_features[0]["pron"] == "ブ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ブ"
    assert njd_features[1]["pron"] == "ブ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "ヅケ"
    assert njd_features[2]["pron"] == "ヅケ"
    assert njd_features[2]["mora_size"] == 2

    # 片仮名の一の字点（バナヽ）
    njd_features = pyopenjtalk.run_frontend("バナヽ")
    assert njd_features[0]["read"] == "バナ"
    assert njd_features[0]["pron"] == "バナ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ナ"
    assert njd_features[1]["pron"] == "ナ"
    assert njd_features[1]["mora_size"] == 1

    # use_vanilla=True の場合は処理されない
    njd_features = pyopenjtalk.run_frontend("なゝ樹", use_vanilla=True)
    assert njd_features[1]["read"] == "、"
    assert njd_features[1]["pron"] == "、"

    # 単一の踊り字（辞書に登録されていないパターン）
    njd_features = pyopenjtalk.run_frontend("愛々")
    assert njd_features[0]["read"] == "アイ"
    assert njd_features[0]["pron"] == "アイ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "アイ"
    assert njd_features[1]["pron"] == "アイ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("咲々")
    assert njd_features[0]["read"] == "サキ"
    assert njd_features[0]["pron"] == "サキ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "サキ"
    assert njd_features[1]["pron"] == "サキ"
    assert njd_features[1]["mora_size"] == 2

    # 単一の踊り字だが、形態素解析で展開しないと正しい読みを取得できないケース
    # 実装上漢字1字だけで再解析した際に読みが間違ってしまうことがあるが、改善するのが面倒なのでテストケースには含めていない
    njd_features = pyopenjtalk.run_frontend("結婚式々場")
    assert njd_features[0]["read"] == "ケッコンシキ"
    assert njd_features[0]["pron"] == "ケッコンシ’キ"
    assert njd_features[0]["mora_size"] == 6
    assert njd_features[1]["read"] == "シキジョウ"
    assert njd_features[1]["pron"] == "シ’キジョー"
    assert njd_features[1]["mora_size"] == 4
    njd_features = pyopenjtalk.run_frontend("学生々活")
    assert njd_features[0]["read"] == "ガクセイ"
    assert njd_features[0]["pron"] == "ガク’セー"
    assert njd_features[0]["mora_size"] == 4
    assert njd_features[1]["read"] == "セイカツ"
    assert njd_features[1]["pron"] == "セーカツ"
    assert njd_features[1]["mora_size"] == 4
    njd_features = pyopenjtalk.run_frontend("民主々義")
    assert njd_features[0]["read"] == "ミンシュ"
    assert njd_features[0]["pron"] == "ミンシュ"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "シュギ"
    assert njd_features[1]["pron"] == "シュギ"
    assert njd_features[1]["mora_size"] == 2

    # 連続する踊り字
    njd_features = pyopenjtalk.run_frontend("叙々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("叙々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ジョ"
    assert njd_features[2]["pron"] == "ジョ"
    assert njd_features[2]["mora_size"] == 1
    njd_features = pyopenjtalk.run_frontend("叙々々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ジョジョ"
    assert njd_features[2]["pron"] == "ジョジョ"
    assert njd_features[2]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("叙々々々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョジョジョジョ"
    assert njd_features[1]["pron"] == "ジョジョジョジョジョ"
    assert njd_features[1]["mora_size"] == 5
    njd_features = pyopenjtalk.run_frontend("複々々線")
    assert njd_features[0]["read"] == "フク"
    assert njd_features[0]["pron"] == "フ’ク"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "フクフク"
    assert njd_features[1]["pron"] == "フ’クフ’ク"
    assert njd_features[1]["mora_size"] == 4
    njd_features = pyopenjtalk.run_frontend("複々々々線")
    assert njd_features[0]["read"] == "フク"
    assert njd_features[0]["pron"] == "フ’ク"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "フクフク"
    assert njd_features[1]["pron"] == "フ’クフ’ク"
    assert njd_features[1]["mora_size"] == 4
    assert njd_features[2]["read"] == "フク"
    assert njd_features[2]["pron"] == "フ’ク"
    assert njd_features[2]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("今日も前進々々")
    assert njd_features[0]["read"] == "キョウ"
    assert njd_features[0]["pron"] == "キョー"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "モ"
    assert njd_features[1]["pron"] == "モ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "ゼンシン"
    assert njd_features[2]["pron"] == "ゼンシン"
    assert njd_features[2]["mora_size"] == 4
    assert njd_features[3]["read"] == "ゼンシン"
    assert njd_features[3]["pron"] == "ゼンシン"
    assert njd_features[3]["mora_size"] == 4

    # 2文字以上の漢字の後の踊り字
    njd_features = pyopenjtalk.run_frontend("部分々々")
    assert njd_features[0]["read"] == "ブブン"
    assert njd_features[0]["pron"] == "ブブン"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "ブブン"
    assert njd_features[1]["pron"] == "ブブン"
    assert njd_features[1]["mora_size"] == 3
    njd_features = pyopenjtalk.run_frontend("後手々々")
    assert njd_features[0]["read"] == "ゴテ"
    assert njd_features[0]["pron"] == "ゴテ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ゴテ"
    assert njd_features[1]["pron"] == "ゴテ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("其他々々")
    assert njd_features[0]["read"] == "ソノ"
    assert njd_features[0]["pron"] == "ソノ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ホカ"
    assert njd_features[1]["pron"] == "ホカ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ソノホカ"
    assert njd_features[2]["pron"] == "ソノホカ"
    assert njd_features[2]["mora_size"] == 4

    # 踊り字の前に漢字がない場合
    # 絵文字除去はこのライブラリの範囲外とし、とりあえず ? という記号を繰り返すことがないようにする
    njd_features = pyopenjtalk.run_frontend("やっほー！元気かな？ヾ(≧▽≦)ﾉ")
    assert njd_features[0]["read"] == "ヤッホー"
    assert njd_features[0]["pron"] == "ヤッホー"
    assert njd_features[0]["mora_size"] == 4
    assert njd_features[1]["read"] == "！"
    assert njd_features[1]["pron"] == "！"
    assert njd_features[1]["mora_size"] == 0
    assert njd_features[2]["read"] == "ゲンキ"
    assert njd_features[2]["pron"] == "ゲンキ’"
    assert njd_features[2]["mora_size"] == 3
    assert njd_features[3]["read"] == "カ"
    assert njd_features[3]["pron"] == "カ"
    assert njd_features[3]["mora_size"] == 1
    assert njd_features[4]["read"] == "ナ"
    assert njd_features[4]["pron"] == "ナ"
    assert njd_features[4]["mora_size"] == 1
    assert njd_features[5]["read"] == "？"
    assert njd_features[5]["pron"] == "？"
    assert njd_features[5]["mora_size"] == 0
    assert njd_features[6]["read"] == "、"
    assert njd_features[6]["pron"] == "、"
    assert njd_features[6]["mora_size"] == 0
    assert njd_features[7]["read"] == "、"
    assert njd_features[7]["pron"] == "、"
    assert njd_features[7]["mora_size"] == 0
    assert njd_features[8]["read"] == "ノ"
    assert njd_features[8]["pron"] == "ノ"
    assert njd_features[8]["mora_size"] == 1

    # use_vanilla=True の場合は処理されない
    njd_features = pyopenjtalk.run_frontend("愛々", use_vanilla=True)
    assert njd_features[1]["read"] == "、"
    assert njd_features[1]["pron"] == "、"


@pytest.mark.parametrize("text", ["学生々活", "民主々義", "結婚式々場"])
def test_apply_postprocessing_matches_run_frontend_when_jtalk_is_provided(text: str):
    """分割実行でも jtalk を渡せば踊り字の再解析結果が通常実行と一致することを確認。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    mecab_features = jtalk.run_mecab(text)
    njd_features = jtalk.run_njd_from_mecab(mecab_features)

    assert pyopenjtalk.apply_postprocessing(
        text, njd_features, jtalk=jtalk
    ) == pyopenjtalk.run_frontend(
        text,
        jtalk=jtalk,
    )


def test_modify_acc_after_chaining_unit():
    """modify_acc_after_chaining() が「参ります」のアクセント核を正しく移動することを確認。"""

    features: list[NJDFeature] = [
        {
            "string": "参り",
            "pos": "動詞",
            "pos_group1": "自立",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "五段・ラ行",
            "cform": "連用形",
            "orig": "参る",
            "read": "マイリ",
            "pron": "マイリ",
            "acc": 1,
            "mora_size": 3,
            "chain_rule": "*",
            "chain_flag": -1,
        },
        {
            "string": "ます",
            "pos": "助動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "特殊・マス",
            "cform": "基本形",
            "orig": "ます",
            "read": "マス",
            "pron": "マス'",
            "acc": 1,
            "mora_size": 2,
            "chain_rule": "動詞%F2@1/助詞%F2@1",
            "chain_flag": 1,
        },
    ]
    result = modify_acc_after_chaining(features)
    # 「参ります」→ ま[いりま]す: アクセント核が「ま」(4 モーラ目) に移動する
    assert result[0]["acc"] == 4


def test_revert_long_vowels():
    """revert_long_vowels=True で辞書が自動的に長音化した発音が元に復元されることを確認。"""

    text = "人生は効果的。"

    # デフォルト: 長音化された pron
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "セー" in kana_default
    assert "コーカ" in kana_default
    assert "ワ" in kana_default  # 助詞は「ワ」

    # revert_long_vowels=True: pron が read に復元される
    kana_revert = pyopenjtalk.g2p(text, kana=True, revert_long_vowels=True)
    assert "セイ" in kana_revert
    assert "コウカ" in kana_revert
    assert "ワ" in kana_revert  # 助詞の「ワ」は維持されること


def test_revert_yotsugana():
    """revert_yotsugana=True で四つ仮名の発音統合が元に復元されることを確認。"""

    text = "鼻血に気づかず。"

    # デフォルト: 「ヅ」→「ズ」、「ヂ」→「ジ」に統合された pron
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "ハナジ" in kana_default
    assert "キズカズ" in kana_default

    # revert_yotsugana=True: 「ヅ」/「ヂ」が復元される
    kana_revert = pyopenjtalk.g2p(text, kana=True, revert_yotsugana=True)
    assert "ハナヂ" in kana_revert
    assert "キヅカズ" in kana_revert


def test_use_read_as_pron():
    """use_read_as_pron=True で全ての pron が read に置き換わることを確認。"""

    text = "こんにちは、人生。"

    # デフォルト: 助詞「は」は「ワ」
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "コンニチワ" in kana_default

    # use_read_as_pron=True: 助詞「は」も「ハ」になる
    kana_revert = pyopenjtalk.g2p(text, kana=True, use_read_as_pron=True)
    assert "コンニチハ" in kana_revert


def test_revert_pron_combined():
    """revert_long_vowels + revert_yotsugana の複合ケースが同時に動作することを確認。"""

    text = "人生は、鼻血に気づかず。"
    kana = pyopenjtalk.g2p(
        text,
        kana=True,
        revert_long_vowels=True,
        revert_yotsugana=True,
    )
    assert "ジンセイ" in kana  # 長音復元
    assert "ワ" in kana  # 助詞は維持
    assert "ハナヂ" in kana  # 四つ仮名復元
    assert "キヅカズ" in kana  # 四つ仮名復元


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("本当は嬉しい", "ホントウワウレシイ"),
        ("百票", "ヒャッピョウ"),
        ("ゲリラ豪雨", "ゲリラゴウウ"),
        ("ニャーと鳴く", "ニャートナク"),
    ],
)
def test_revert_long_vowels_keeps_other_pronunciation_differences(text: str, expected: str) -> None:
    """長音の復元が長音だけを戻し、同じ語の助詞の「ワ」や連濁を読みで上書きせず、元の表記の「ー」も残すことを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True, revert_long_vowels=True) == expected


def test_revert_long_vowels_keeps_devoicing_mark() -> None:
    """長音の復元が、同じ語の無声化の記号「’」を消さないことを確認する。"""

    njd_features = pyopenjtalk.run_frontend("ありがとうございました", revert_long_vowels=True)

    assert [feature["pron"] for feature in njd_features] == ["アリガトウ", "ゴザイマシ’タ"]


def test_revert_pron_with_use_vanilla():
    """use_vanilla=True でも発音復元オプションは独立して適用されることを確認。"""

    text = "人生は効果的。"

    # use_vanilla=True + revert_long_vowels=True: 後処理は省略されるが発音復元は適用
    njd = pyopenjtalk.run_frontend(
        text,
        use_vanilla=True,
        revert_long_vowels=True,
    )
    jinsei = next(f for f in njd if f["orig"] == "人生")
    assert jinsei["pron"] == "ジンセイ"  # 長音復元が適用されている

    kouka = next(f for f in njd if f["orig"] == "効果")
    assert kouka["pron"] == "コウカ"  # 長音復元が適用されている

    wa = next(f for f in njd if f["orig"] == "は")
    assert wa["pron"] == "ワ"  # 助詞の「ワ」は維持


def test_use_vanilla_keeps_marine_and_iu_pronunciation(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_vanilla=True でも、明示的に指定した marine と「言う」の発音の方式は適用されることを確認する。"""

    call_count = 0

    def estimate_accent(njd_features: list[NJDFeature]) -> list[NJDFeature]:
        """marine の代わりに、アクセント核を変更した予測結果を返す。"""

        nonlocal call_count
        call_count += 1
        predicted_features = copy.deepcopy(njd_features)
        next(feature for feature in predicted_features if feature["orig"] == "言う")["acc"] = 1
        return predicted_features

    monkeypatch.setattr(pyopenjtalk, "estimate_accent", estimate_accent)
    njd_features = pyopenjtalk.run_frontend(
        "言う",
        run_marine=True,
        use_vanilla=True,
        iu_pronunciation="Yuu",
    )

    iu_feature = next(feature for feature in njd_features if feature["orig"] == "言う")
    assert call_count == 1
    assert iu_feature["acc"] == 1
    assert iu_feature["pron"] == "ユウ"


def test_revert_pron_default_no_change():
    """発音復元オプションを指定しない場合は pron が変更されないことを確認。"""

    text = "人生は効果的。"
    njd = pyopenjtalk.run_frontend(text)
    jinsei = next(f for f in njd if f["orig"] == "人生")
    assert "ー" in jinsei["pron"]  # デフォルトでは長音化された pron


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("言う", "ユー"),
        ("そういう事", "ソーユーコト"),
        ("物言う株主", "モノユーカブヌシ"),
        ("という", "トユー"),
        ("ていう", "テユー"),
        ("っていう", "ッテユー"),
        ("とかいう", "トカユー"),
        ("あっという", "アットユー"),
        ("言わない", "イワナイ"),
        ("言えば", "イエバ"),
        ("言った", "イッタ"),
    ],
)
def test_normalize_iu_uses_standard_base_form_pronunciation(text: str, expected: str) -> None:
    """iu_pronunciation="YuuBase" を指定した際、「言う」の終止形・連体形だけが「ユー」と発音され、活用形のイ段（言わない、言えば等）は変化しないことを確認する。"""

    assert pyopenjtalk.g2p(text, kana=True, iu_pronunciation="YuuBase") == expected

    # 発音復元との複合時も明示した発音方式を最終結果に反映する
    if text == "言う":
        assert (
            pyopenjtalk.g2p(
                text,
                kana=True,
                iu_pronunciation="YuuBase",
                use_read_as_pron=True,
            )
            == "ユー"
        )
        assert (
            pyopenjtalk.g2p(
                text,
                kana=True,
                iu_pronunciation="YuuBase",
                revert_long_vowels=True,
            )
            == "ユー"
        )


def test_normalize_iu_keeps_dictionary_pronunciation_by_default() -> None:
    """iu_pronunciation を指定しないデフォルト状態では、辞書本来の「イウ」発音が維持されることを確認する。"""

    assert pyopenjtalk.g2p("言う", kana=True) == "イウ"


@pytest.mark.parametrize(
    ("mode", "text", "expected_fragment"),
    [
        ("Iu", "言う", "イウ"),
        ("Iu", "こういう事", "コーイウ"),
        ("Iu", "あっという間に", "アットイウ"),
        ("Yuu", "言って", "ユッテ"),
        ("Yuu", "言えば", "ユエバ"),
        ("Yuu", "言おう", "ユオー"),
        ("Yuu", "言わない", "ユワナイ"),
        ("Yuu", "君ていう人は", "テユウ"),
        ("Yuu", "誰っていうの", "ッテユウ"),
        ("Yuu", "誰とかいう", "トカユウ"),
        ("KanjiIu", "アッと言う間に", "アットイウ"),
        ("KanjiYuu", "アッと言う間に", "アットユウ"),
        ("KanjiYuu", "物言う株主", "モノユウ"),
        ("KanjiYuuBase", "言う", "ユー"),
        ("KanjiYuuBase", "言わない", "イワナイ"),
    ],
)
def test_normalize_iu_modes(
    mode: IuPronunciation,
    text: str,
    expected_fragment: str,
) -> None:
    """「言う」の発音の6つの方式が、活用形・定型表現・漢字だけに絞る範囲をそれぞれ守ることを確認する。"""

    assert expected_fragment in pyopenjtalk.g2p(
        text,
        kana=True,
        iu_pronunciation=mode,
    )


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("Yuu", "正当な理由"),
        ("Yuu", "髪を結う"),
        ("KanjiYuu", "そういう事"),
        ("KanjiYuu", "君ていう人は"),
        ("KanjiYuu", "ものいう株主"),
        ("KanjiYuuBase", "そういう事"),
    ],
)
def test_normalize_iu_modes_keep_excluded_words(
    mode: IuPronunciation,
    text: str,
) -> None:
    """同音の別語と漢字限定外の平仮名表記は辞書の発音を維持する。"""

    assert pyopenjtalk.g2p(text, kana=True, iu_pronunciation=mode) == pyopenjtalk.g2p(
        text,
        kana=True,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("琵琶湖", "^bi[wako$"),
        ("西湖", "^sa]iko$"),
        ("河口湖", "^ka[waguchi]ko$"),
        ("田沢湖", "^ta[zawa]ko$"),
        ("富士河口湖", "^fu]ji#ka[waguchi]ko$"),
        ("余呉湖", "^yo[go]ko$"),
        ("塩湖", "^shi[o]ko$"),
        ("塩湖の水", "^shi[o]kono#mi[zu$"),
        ("津久井湖", "^tsU[kui]ko$"),
        ("山中湖の水", "^ya[manaka]kono#mi[zu$"),
        ("宮沢湖", "^mi[yazawa]ko$"),
        ("阿寒湖", "^a[ka]Nko$"),
        ("摩周湖", "^ma[shu]uko$"),
        ("仙台市", "^se[Nda]ishi$"),
        ("チャイ缶", "^cha]ikaN$"),
        ("葛西駅", "^ka[sa]ieki$"),
        ("津久井町", "^tsU[kui]machi$"),
        ("福井県", "^fU[kui]keN$"),
        ("山梨県", "^ya[manashi]keN$"),
        ("東京駅", "^to[okyo]oeki$"),
        ("住民税", "^ju[umi]Nzee$"),
        ("青梅線", "^o[omeseN$"),
    ],
)
def test_suffix_accent_keeps_lexical_exceptions(text: str, expected: str) -> None:
    """
    前部末型の特殊拍・二重母音を補正し、短い湖名と平板型の路線名は辞書のアクセントを保つ。
    「津久井」の「クイ」は母音連続だけを根拠に核を後退させず、「仙台」の「ダイ」と区別する。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected


def test_c3_devoiced_final_mora_retreats_once() -> None:
    """
    前部末の拍に無声化記号が残る場合は核を1拍前へ移し、繰り返し適用してもさらに移動しない。
    NHK アクセント辞典の「特別市」にある「トクベ＼ツシ」の型を検証する。
    """

    features = pyopenjtalk.run_frontend("特別市")
    features[0]["pron"] = "トクベツ’"
    features = pyopenjtalk_utils.retreat_acc_nuc(features)

    assert features[0]["acc"] == 3
    assert pyopenjtalk_utils.retreat_acc_nuc(features)[0]["acc"] == 3


def test_odori_hard_boundary():
    """踊り字が境界より前の無関係な漢字を参照しないことを確認する。"""

    # 記号がハード境界として機能するケース
    # 「人。々」では「。」がハード境界となり、「人」の読みを「々」に引き継がない
    njd = pyopenjtalk.run_frontend("人。々")
    assert len(njd) >= 1
    # 踊り字トークンに「人」の読み（「ヒト」/「ジン」）が引き継がれていないことを確認
    odori_tokens = [f for f in njd if "々" in f["orig"]]
    assert len(odori_tokens) >= 1, "踊り字トークンが存在すること"
    for token in odori_tokens:
        assert "ヒト" not in token["read"]
        assert "ジン" not in token["read"]

    # 非漢字トークンが境界として機能するケース
    # 「人は々」では助詞「は」が境界となり、「人」の読みを引き継がない
    njd2 = pyopenjtalk.run_frontend("人は々")
    odori_tokens2 = [f for f in njd2 if "々" in f["orig"]]
    assert len(odori_tokens2) >= 1, "踊り字トークンが存在すること"
    for token in odori_tokens2:
        assert "ヒト" not in token["read"]
        assert "ジン" not in token["read"]


def test_odoriji_voiced_and_voiceless_conversion():
    """一の字点の清音化・濁音化が期待どおりに動作することを確認。"""

    assert pyopenjtalk.g2p("がゝ", kana=True) == "ガカ"
    assert pyopenjtalk.g2p("バヽ", kana=True) == "バハ"
    assert pyopenjtalk.g2p("かゞ", kana=True) == "カガ"
    assert pyopenjtalk.g2p("ハヾ", kana=True) == "ハバ"


def test_odoriji_small_kana_handling():
    """拗音を含むモーラに対する一の字点処理が安定していることを確認。"""

    assert pyopenjtalk.g2p("じょゝ", kana=True) == "ジョジョ"
    assert pyopenjtalk.g2p("ちゅゞ", kana=True) == "チュヂュ"


def test_odoriji_invalid_cases():
    """不正または孤立した一の字点を与えても安全に処理されることを確認。"""

    assert pyopenjtalk.g2p("ゝ", kana=True) == "ゝ"
    assert pyopenjtalk.g2p("かゝ゜", kana=True) == "カカ゜"


def test_odoriji_basic_expansion():
    """一の字点 (ゝ/ゞ/ヽ/ヾ) の基本展開が正しく行われることを確認。"""

    assert pyopenjtalk.g2p("さゝみ", kana=True) == "ササミ"
    assert pyopenjtalk.g2p("いすゞ", kana=True) == "イスズ"
    assert pyopenjtalk.g2p("カヽ", kana=True) == "カカ"
    assert pyopenjtalk.g2p("ガヾ", kana=True) == "ガガ"


def test_odoriji_mapping_known_word():
    """辞書登録済みの一の字点語でも mapping の音素列が崩れないことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("いすゞ")
    assert len(mapping) == 1
    assert mapping[0]["surface"] == "いすゞ"
    assert mapping[0]["phonemes"] == ["i", "s", "u", "z", "u"]
