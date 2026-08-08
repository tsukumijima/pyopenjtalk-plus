"""Python 側の読み・アクセント後処理を検証する。"""

import copy

import pytest

import pyopenjtalk
import pyopenjtalk.utils as pyopenjtalk_utils
from pyopenjtalk import NJDFeature
from pyopenjtalk.utils import modify_acc_after_chaining


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
    ],
)
def test_predict_nani_reading_keeps_high_confidence_nani_rules(
    text: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """助詞と「する」が後続する「何」はモデル誤判定より確実なナニ規則を優先する。"""

    def fail_predict(_features: list[NJDFeature | None]) -> int:
        """高確信ナニ規則でモデル推論が呼ばれた場合は失敗させる。"""

        raise AssertionError("predict() must not be called for a high-confidence ナニ context")

    monkeypatch.setattr(pyopenjtalk_utils, "predict", fail_predict)
    njd_features = pyopenjtalk.run_frontend(text, predict_nani=False)

    corrected_features = pyopenjtalk_utils.predict_nani_reading(njd_features)

    nani_feature = next(feature for feature in corrected_features if feature["orig"] == "何")
    assert nani_feature["read"] == "ナニ"
    assert nani_feature["pron"] == "ナニ"


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


def test_modify_kanji_yomi_converts_hou_to_hoo(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sudachi が「方」をホウと返した場合は OpenJTalk の長音表記へ変換する。"""

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
    assert hou_feature["read"] == "ホオ"
    assert hou_feature["pron"] == "ホオ"


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
    """modify_acc_after_chaining が「参ります」のアクセント核を正しく移動することを確認。"""

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

    # デフォルト: ヅ→ズ, ヂ→ジ に統合された pron
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "ハナジ" in kana_default
    assert "キズカズ" in kana_default

    # revert_yotsugana=True: ヅ/ヂ が復元される
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


def test_revert_pron_default_no_change():
    """発音復元オプションを指定しない場合は pron が変更されないことを確認。"""

    text = "人生は効果的。"
    njd = pyopenjtalk.run_frontend(text)
    jinsei = next(f for f in njd if f["orig"] == "人生")
    assert "ー" in jinsei["pron"]  # デフォルトでは長音化された pron


def test_odori_hard_boundary():
    """踊り字が境界より前の無関係な漢字を参照しないことを確認する。"""

    # 記号がハード境界として機能するケース
    # 「人。々」では「。」がハード境界となり、「人」の読みを「々」に引き継がない
    njd = pyopenjtalk.run_frontend("人。々")
    assert len(njd) >= 1
    # 踊り字トークンに「人」の読み (ヒト/ジン) が引き継がれていないことを確認
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
