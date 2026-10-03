"""
r9y9/pyopenjtalk に由来する最小テストを、フォーク後も公開 API の基本的な利用方法を維持する基準点として残す。
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import pyopenjtalk
from pyopenjtalk import NJDFeature
from pyopenjtalk.utils import merge_njd_marine_features


def _print_results(njd_features: list[NJDFeature], labels: list[str]):
    for f in njd_features:
        s, p = f["string"], f["pron"]
        print(s, p)

    for label in labels:
        print(label)


def test_hello():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    labels = pyopenjtalk.make_label(njd_features)
    _print_results(njd_features, labels)


def test_hello_marine():
    """marine のアクセント推定を通して基本的なフロントエンド処理を実行できる。"""
    pytest.importorskip("marine")
    njd_features = pyopenjtalk.run_frontend("こんにちは", run_marine=True)
    labels = pyopenjtalk.make_label(njd_features)
    _print_results(njd_features, labels)


def test_njd_features():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "-1",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_njd_features_marine():
    """marine のアクセント推定後も期待する NJD feature を返す。"""
    pytest.importorskip("marine")
    njd_features = pyopenjtalk.run_frontend("こんにちは", run_marine=True)
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "-1",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_marine_result_length_mismatch_is_rejected() -> None:
    """marine の結果数が NJD feature 数と異なる場合は明示的に拒否する。"""

    njd_features = pyopenjtalk.run_frontend("こんにちは")
    with pytest.raises(ValueError, match="Invalid sequence sizes"):
        merge_njd_marine_features(
            njd_features,
            {
                "accent_status": [],
                "accent_phrase_boundary": [],
            },
        )


def test_fullcontext():
    features = pyopenjtalk.run_frontend("こんにちは")
    labels = pyopenjtalk.make_label(features)
    labels2 = pyopenjtalk.extract_fullcontext("こんにちは")
    assert labels == labels2


def test_fullcontext_marine():
    """marine のアクセント推定を通した一括処理と分割処理のラベルが一致する。"""
    pytest.importorskip("marine")
    features = pyopenjtalk.run_frontend("こんにちは", run_marine=True)
    labels = pyopenjtalk.make_label(features)
    labels2 = pyopenjtalk.extract_fullcontext("こんにちは", run_marine=True)
    assert labels == labels2


@pytest.mark.parametrize(
    ("text", "accent", "mora_size"), [("宮沢湖", 4, 5), ("山中湖", 4, 5), ("津久井湖", 3, 4)]
)
def test_marine_accent_stays_within_shortened_context_reading(
    text: str, accent: int, mora_size: int
) -> None:
    """
    「宮沢湖」と「山中湖」は4モーラ目、「津久井湖」は3モーラ目の、前の要素の末尾に核を置く。
    「宮沢湖」で一般名詞の「ミズウミ」が選ばれても、結合規則が前の要素の末尾を指すので核が句の範囲内に収まる。
    marine を使う場合も、この辞書の核を保つ。
    """

    pytest.importorskip("marine")
    features = pyopenjtalk.run_frontend(text, run_marine=True)

    assert features[0]["acc"] == accent
    assert sum(feature["mora_size"] for feature in features) == mora_size
    assert pyopenjtalk.make_label(features)


@pytest.mark.parametrize(
    ("place", "expected_rule"), [("余呉", "C1"), ("津久井", "C3"), ("宮沢", "C3")]
)
def test_lake_chaining_uses_shortened_mora_count(place: str, expected_rule: str) -> None:
    """
    一般名詞「湖」を接尾辞として読む場合、縮めた「コ」を含む4モーラ以上の語だけ前部末型にする。
    語全体の候補が選ばれる湖名も、MeCab の分割結果を渡して結合規則の適用を検証する。
    """

    jtalk = pyopenjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)
    mecab_features = jtalk.run_mecab(place) + jtalk.run_mecab("湖")
    features = jtalk.run_njd_from_mecab(mecab_features)

    assert features[-1]["chain_rule"] == expected_rule
    assert features[-1]["pron"] == "ミズウミ"


def test_jtalk():
    for text in [
        "今日も良い天気ですね",
        "こんにちは。",
        "どんまい！",
        "パソコンのとりあえず知っておきたい使い方",
    ]:
        njd_features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(njd_features)
        _print_results(njd_features, labels)

        surface = "".join(map(lambda f: f["string"], njd_features))
        assert surface == text


def test_jtalk_marine():
    """marine のアクセント推定を通して代表文の表層を保持する。"""
    pytest.importorskip("marine")
    for text in [
        "今日も良い天気ですね",
        "こんにちは。",
        "どんまい！",
        "パソコンのとりあえず知っておきたい使い方",
    ]:
        njd_features = pyopenjtalk.run_frontend(text, run_marine=True)
        labels = pyopenjtalk.make_label(njd_features)
        _print_results(njd_features, labels)

        surface = "".join(map(lambda f: f["string"], njd_features))
        assert surface == text


def test_g2p_kana():
    for text, pron in [
        ("", ""),  # empty string
        ("今日もこんにちは", "キョーモコンニチワ"),
        ("いやあん", "イヤーン"),
        (
            "パソコンのとりあえず知っておきたい使い方",
            "パソコンノトリアエズシッテオキタイツカイカタ",
        ),
    ]:
        p = pyopenjtalk.g2p(text, kana=True)
        assert p == pron


def test_g2p_phone():
    for text, pron in [
        ("", ""),  # empty string
        ("こんにちは", "k o N n i ch i w a"),
        ("ななみんです", "n a n a m i N d e s U"),
        ("ハローユーチューブ", "h a r o o y u u ch u u b u"),
    ]:
        p = pyopenjtalk.g2p(text, kana=False)
        assert p == pron


def test_userdic():
    for text, expected in [
        ("nnmn", "n a n a m i N"),
        ("GNU", "g u n u u"),
    ]:
        p = pyopenjtalk.g2p(text)
        assert p != expected

    user_csv = str(Path(__file__).parent / "test_data" / "user.csv")
    user_dic = str(Path(__file__).parent / "test_data" / "user.dic")

    with open(user_csv, "w", encoding="utf-8") as f:
        f.write("ｎｎｍｎ,,,1,名詞,一般,*,*,*,*,ｎｎｍｎ,ナナミン,ナナミン,1/4,*\n")
        f.write("ＧＮＵ,,,1,名詞,一般,*,*,*,*,ＧＮＵ,グヌー,グヌー,2/3,*\n")

    try:
        pyopenjtalk.mecab_dict_index(f.name, user_dic)
        pyopenjtalk.update_global_jtalk_with_user_dict(user_dic)

        for text, expected in [
            ("nnmn", "n a n a m i N"),
            ("GNU", "g u n u u"),
        ]:
            p = pyopenjtalk.g2p(text)
            assert p == expected
    finally:
        pyopenjtalk.unset_user_dict()


def test_multithreading():
    ojt = pyopenjtalk.openjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)
    texts = [
        "今日もいい天気ですね",
        "こんにちは",
        "マルチスレッドプログラミング",
        "テストです",
        "Pythonはプログラミング言語です",
        "日本語テキストを音声合成します",
    ] * 4

    # Test consistency between single and multi-threaded runs
    # make sure no corruptions happen in OJT internal
    results_s = [ojt.run_frontend(text) for text in texts]
    results_m = []
    with ThreadPoolExecutor() as e:
        results_m = [i for i in e.map(ojt.run_frontend, texts)]
    for s, m in zip(results_s, results_m):
        assert len(s) == len(m)
        for s_, m_ in zip(s, m):
            # full context must exactly match
            assert s_ == m_
