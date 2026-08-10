"""同形異音語辞書が語彙素と活用形に沿った候補構造を持つことを検証する。"""

from types import SimpleNamespace

import pytest

import pyopenjtalk
from pyopenjtalk import tsqyomi
from pyopenjtalk.tsqyomi import inference as tsqyomi_inference


@pytest.mark.parametrize(
    ("text", "surface", "expected_candidates"),
    (
        (
            "勝った",
            "勝っ",
            {
                "カッ": ("勝つ", "五段・タ行", "1/2"),
                "マサッ": ("勝る", "五段・ラ行", "2/3"),
            },
        ),
        (
            "断った",
            "断っ",
            {
                "コトワッ": ("断る", "五段・ラ行", "3/4"),
                "タッ": ("断つ", "五段・タ行", "1/2"),
            },
        ),
        (
            "通った",
            "通っ",
            {
                "カヨッ": ("通う", "五段・ワ行促音便", "0/3"),
                "トーッ": ("通る", "五段・ラ行", "1/3"),
            },
        ),
    ),
)
def test_inflected_candidates_use_lemma_nodes(
    text: str,
    surface: str,
    expected_candidates: dict[str, tuple[str, str, str]],
) -> None:
    """過去形の候補を活用語幹と助動詞へ分けて列挙する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    target_span = (0, len(surface))
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))

    # 全表層の名詞ノードに戻らず、語彙素を持つ活用語幹と助動詞へ分ける
    assert tuple(morph["surface"] for morph in analysis["morphs"]) == (surface, "た")
    assert all(morph["surface"] != text for morph in analysis["morphs"])

    # 同じ活用語幹の範囲で、語彙素・活用型・アクセントが異なる候補を保持する
    actual_candidates = {
        path["pronunciation"]: (
            path["features"][0].split(",")[7],
            path["features"][0].split(",")[5],
            path["features"][0].split(",")[10],
        )
        for path in analysis["paths"]
        if path["char_span"] == target_span and path["pronunciation"] in expected_candidates
    }
    assert actual_candidates == expected_candidates


@pytest.mark.parametrize(
    (
        "text",
        "surface",
        "allowed_pronunciations",
        "selected_pronunciation",
        "expected_orig",
        "expected_ctype",
        "expected_read",
        "expected_accent_nucleus",
        "expected_kana",
    ),
    (
        (
            "勝った",
            "勝っ",
            ("カッ", "マサッ"),
            "カッ",
            "勝つ",
            "五段・タ行",
            "カッ",
            1,
            "カッタ",
        ),
        (
            "勝った",
            "勝っ",
            ("カッ", "マサッ"),
            "マサッ",
            "勝る",
            "五段・ラ行",
            "マサッ",
            2,
            "マサッタ",
        ),
        (
            "断った",
            "断っ",
            ("コトワッ", "タッ"),
            "コトワッ",
            "断る",
            "五段・ラ行",
            "コトワッ",
            3,
            "コトワッタ",
        ),
        (
            "断った",
            "断っ",
            ("コトワッ", "タッ"),
            "タッ",
            "断つ",
            "五段・タ行",
            "タッ",
            1,
            "タッタ",
        ),
        (
            "通った",
            "通っ",
            ("カヨッ", "トーッ"),
            "カヨッ",
            "通う",
            "五段・ワ行促音便",
            "カヨッ",
            4,
            "カヨッタ",
        ),
        (
            "通った",
            "通っ",
            ("カヨッ", "トーッ"),
            "トーッ",
            "通る",
            "五段・ラ行",
            "トオッ",
            1,
            "トーッタ",
        ),
    ),
)
def test_tsqyomi_selects_inflected_lemma_and_keeps_auxiliary(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    surface: str,
    allowed_pronunciations: tuple[str, str],
    selected_pronunciation: str,
    expected_orig: str,
    expected_ctype: str,
    expected_read: str,
    expected_accent_nucleus: int,
    expected_kana: str,
) -> None:
    """選択した活用語幹だけを交換し、助動詞の NJD 特徴とアクセントを維持する。"""

    def predict_selected(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """指定された候補読みを選んだモデル結果を返す。"""

        return (
            tsqyomi.ReadingPrediction(
                pronunciation=selected_pronunciation,
                scores=(1.0, 0.0),
            ),
        )

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={surface[0]: (surface,)},
            reading_class_ids_by_surface_and_pronunciation={
                surface: {
                    pronunciation: (f"rc_{index}",)
                    for index, pronunciation in enumerate(allowed_pronunciations)
                }
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_selected,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    # 選ばれなかった語彙素を混ぜず、中心の活用語幹だけをモデル選択へ差し替える
    selected_features, selected_morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(
        text,
        jtalk,
    )
    stem_fields = selected_features[0].split(",")
    auxiliary_fields = selected_features[1].split(",")
    assert stem_fields[0:2] == [surface, "動詞"]
    assert stem_fields[5:10] == [
        expected_ctype,
        "連用タ接続",
        expected_orig,
        expected_read,
        selected_pronunciation,
    ]
    assert selected_morphs[0]["features"] == stem_fields

    # 後続の「た」は語幹の交換対象から外し、助動詞として NJD へ渡す
    assert auxiliary_fields[0:10] == [
        "た",
        "助動詞",
        "*",
        "*",
        "*",
        "特殊・タ",
        "基本形",
        "た",
        "タ",
        "タ",
    ]
    assert selected_morphs[1]["features"] == auxiliary_fields
    njd_features = jtalk.run_njd_from_mecab(selected_features)
    assert [(feature["string"], feature["pos"]) for feature in njd_features] == [
        (surface, "動詞"),
        ("た", "助動詞"),
    ]
    assert njd_features[0]["orig"] == expected_orig
    assert njd_features[0]["ctype"] == expected_ctype
    assert njd_features[1]["ctype"] == "特殊・タ"
    assert njd_features[1]["chain_flag"] == 1

    # 公開 API でも語幹と助動詞が連結し、語彙素に対応するアクセント核を返す
    mapping = pyopenjtalk.g2p_mapping(
        text,
        use_tsqyomi=True,
        use_vanilla=True,
        jtalk=jtalk,
    )
    assert tuple(entry["surface"] for entry in mapping) == (surface, "た")
    assert mapping[0]["accent_nucleus"] == expected_accent_nucleus
    assert mapping[1]["accent_nucleus"] == 0
    assert (
        pyopenjtalk.g2p(
            text,
            kana=True,
            use_tsqyomi=True,
            use_vanilla=True,
            jtalk=jtalk,
        )
        == expected_kana
    )


def test_kuku_has_one_identical_candidate() -> None:
    """九九のクク候補を1行だけ保持する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates("九九", ((0, 2),))
    kuku_paths = [path for path in analysis["paths"] if path["pronunciation"] == "クク"]
    kuku_nodes = [node for node in analysis["nodes"] if node["pronunciation"] == "クク"]

    # 完全に同じ辞書特徴列の候補を重ねず、低コスト側を既定経路として残す
    assert len(kuku_paths) == 1
    assert len(kuku_nodes) == 1
    assert kuku_nodes[0]["word_cost"] == 5620
    assert analysis["morphs"][0]["word_cost"] == 5620
    assert pyopenjtalk.g2p("九九", kana=True, use_vanilla=True, jtalk=jtalk) == "クク"


def test_sotozura_keeps_orthographic_read_and_standard_pronunciation() -> None:
    """外面のソトヅラ表記を現代標準発音のソトズラへ対応させる。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates("外面", ((0, 2),))
    sotozura_paths = [path for path in analysis["paths"] if path["pronunciation"] == "ソトズラ"]

    # 読み表記のヅと発音のズを区別し、発音だけ異なる重複候補を作らない
    assert len(sotozura_paths) == 1
    fields = sotozura_paths[0]["features"][0].split(",")
    assert fields[8:11] == ["ソトヅラ", "ソトズラ", "0/4"]
    assert all(path["pronunciation"] != "ソトヅラ" for path in analysis["paths"])


@pytest.mark.parametrize(
    ("text", "expected_pos", "expected_pronunciation", "expected_kana"),
    (
        ("然る人を訪ねた。", "連体詞", "サル", "サルヒトヲタズネタ。"),
        ("部下を然る。", "動詞", "シカル", "ブカヲシカル。"),
    ),
)
def test_saru_and_shikaru_use_contextual_parts_of_speech(
    text: str,
    expected_pos: str,
    expected_pronunciation: str,
    expected_kana: str,
) -> None:
    """然るの2読みを連体詞と動詞として同じ裸表層へ保持する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    start = text.index("然る")
    target_span = (start, start + len("然る"))
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    target_paths = [path for path in analysis["paths"] if path["char_span"] == target_span]

    # 前後の接続費用で既定品詞を選びつつ、候補グラフには両方の読みを残す
    assert {
        (path["pronunciation"], path["features"][0].split(",")[1]) for path in target_paths
    } >= {("サル", "連体詞"), ("シカル", "動詞")}
    target_morph = next(morph for morph in analysis["morphs"] if morph["char_span"] == target_span)
    assert target_morph["features"][1] == expected_pos
    assert target_morph["features"][9] == expected_pronunciation
    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True, jtalk=jtalk) == expected_kana
