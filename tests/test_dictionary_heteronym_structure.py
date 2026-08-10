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
    ("text", "expected_kana"),
    (
        ("然る人を訪ねた。", "シカルヒトヲタズネタ。"),
        ("部下を然る。", "ブカヲシカル。"),
    ),
)
def test_shikaru_uses_adjudicated_verb_reading(text: str, expected_kana: str) -> None:
    """然るを現代の漢字表記で使うシカルに固定する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    start = text.index("然る")
    target_span = (start, start + len("然る"))
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    target_paths = [path for path in analysis["paths"] if path["char_span"] == target_span]

    # 仮名書きが通例の文語読みを候補から外し、前後の文脈による誤選択を防ぐ
    assert {path["pronunciation"] for path in target_paths} == {"シカル"}
    target_morph = next(morph for morph in analysis["morphs"] if morph["char_span"] == target_span)
    assert target_morph["features"][1] == "動詞"
    assert target_morph["features"][9] == "シカル"
    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True, jtalk=jtalk) == expected_kana


@pytest.mark.parametrize(
    ("surface", "removed_pronunciations", "text", "expected_kana"),
    (
        ("主筋", ("シュースジ", "シュスジ"), "主筋を組む", "シュキンヲクム"),
        ("作法", ("サクホウ",), "作法を学ぶ", "サホーヲマナブ"),
        ("古本", ("コホン",), "古本を買う", "フルホンヲカウ"),
        ("地方", ("ジカタ",), "地方へ行く", "チホーエイク"),
        ("彼の", ("カノ",), "彼の本", "カレノホン"),
        ("悪気", ("アッキ",), "悪気はない", "ワルギワナイ"),
        ("正面", ("マトモ",), "正面を向く", "ショーメンヲムク"),
        ("海馬", ("ウミウマ",), "海馬を調べる", "カイバヲシラベル"),
        ("漢書", ("カラブミ",), "漢書を読む", "カンショヲヨム"),
        ("盛る", ("サカル",), "料理を盛る", "リョーリヲモル"),
    ),
)
def test_adjudicated_fixed_readings_drop_removed_candidates(
    surface: str,
    removed_pronunciations: tuple[str, ...],
    text: str,
    expected_kana: str,
) -> None:
    """現代の標準的な読みへ固定した表層から撤回済み候補を外す。"""

    start = text.index(surface)
    target_span = (start, start + len(surface))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    target_pronunciations = {
        path["pronunciation"] for path in analysis["paths"] if path["char_span"] == target_span
    }

    # 実在しても裸表層では使わない読みを消し、製品の既定読みも同時に固定する
    assert target_pronunciations.isdisjoint(removed_pronunciations)
    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True, jtalk=jtalk) == expected_kana


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("人前式を挙げる", "ジンゼンシキヲアゲル"),
        ("人前結婚式を選ぶ", "ジンゼンケッコンシキヲエラブ"),
        ("仏前式を挙げる", "ブツゼンシキヲアゲル"),
        ("仏前結婚式を選ぶ", "ブツゼンケッコンシキヲエラブ"),
        ("人前に出る", "ヒトマエニデル"),
        ("仏を拝む", "ホトケヲオガム"),
        ("仏教を学ぶ", "ブッキョーヲマナブ"),
        ("神前式を挙げる", "シンゼンシキヲアゲル"),
        ("教会式を選ぶ", "キョーカイシキヲエラブ"),
    ),
)
def test_wedding_style_compounds_preserve_independent_word_readings(
    text: str,
    expected_kana: str,
) -> None:
    """婚礼複合語の読みと裸表層の多数派読みを両立する。"""

    # ジンゼンとブツゼンを閉じた複合語で供給し、裸の人前と仏や既存の婚礼語を変えない
    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True) == expected_kana


@pytest.mark.parametrize(
    ("surface", "pronunciation"),
    (
        ("一分", "イチブン"),
        ("上下", "アゲサゲ"),
        ("主筋", "シュキン"),
        ("然る", "シカル"),
        ("米粉", "コメコ"),
        ("行", "クダリ"),
        ("行", "オコナイ"),
        ("経緯", "タテヌキ"),
        ("経緯", "タテヨコ"),
        ("経緯", "ユクタテ"),
        ("八戸", "ハチコ"),
        ("放出", "ハナテン"),
    ),
)
def test_reclassified_candidates_preserve_structural_and_limited_readings(
    surface: str,
    pronunciation: str,
) -> None:
    """数詞構造や限定用法で必要な辞書候補を維持する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    target_span = (0, len(surface))
    analysis = jtalk.analyze_mecab_candidates(surface, (target_span,))

    # モデルの出力対象から外れても、辞書が担う実在読みの候補は残す
    assert any(
        path["char_span"] == target_span and path["pronunciation"] == pronunciation
        for path in analysis["paths"]
    )


@pytest.mark.parametrize(
    (
        "text",
        "target_span",
        "expected_surface",
        "expected_orig",
        "expected_cform",
        "expected_read",
        "expected_pronunciation",
        "expected_word_cost",
        "expected_kana",
    ),
    (
        (
            "大手を振って歩く",
            (0, 5),
            "大手を振っ",
            "大手を振る",
            "連用タ接続",
            "オオデヲフッ",
            "オーデオフッ",
            4000,
            "オーデオフッテアルク",
        ),
        (
            "堂に入った演技",
            (0, 4),
            "堂に入っ",
            "堂に入る",
            "連用タ接続",
            "ドウニイッ",
            "ドーニイッ",
            6628,
            "ドーニイッタエンギ",
        ),
        (
            "念が入る",
            (0, 4),
            "念が入る",
            "念が入る",
            "基本形",
            "ネンガイル",
            "ネンガイル",
            4000,
            "ネンガイル",
        ),
        (
            "労わりの言葉",
            (0, 3),
            "労わり",
            "労わる",
            "連用形",
            "イタワリ",
            "イタワリ",
            7948,
            "イタワリノコトバ",
        ),
    ),
)
def test_general_idioms_and_inflections_use_verb_nodes(
    text: str,
    target_span: tuple[int, int],
    expected_surface: str,
    expected_orig: str,
    expected_cform: str,
    expected_read: str,
    expected_pronunciation: str,
    expected_word_cost: int,
    expected_kana: str,
) -> None:
    """一般的な慣用句と表記違いを語彙素に対応する動詞候補として供給する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    target_paths = [path for path in analysis["paths"] if path["char_span"] == target_span]

    # 助詞を含む慣用句も全表層の名詞へ潰さず、正しい活用型と活用形を持つ1つの動詞にする
    matching_path = next(
        path for path in target_paths if path["pronunciation"] == expected_pronunciation
    )
    fields = matching_path["features"][0].split(",")
    assert fields[0:2] == [expected_surface, "動詞"]
    assert fields[5:10] == [
        "五段・ラ行",
        expected_cform,
        expected_orig,
        expected_read,
        expected_pronunciation,
    ]

    # 候補供給だけで終わらず、前後の接続費用を含む既定経路でも対象の動詞を選ぶ
    selected_morph = next(
        morph for morph in analysis["morphs"] if morph["char_span"] == target_span
    )
    assert selected_morph["surface"] == expected_surface
    assert selected_morph["word_cost"] == expected_word_cost
    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True, jtalk=jtalk) == expected_kana


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("大手企業で働く", "オーテキギョーデハタラク"),
        ("本堂に入った", "ホンドーニハイッタ"),
        ("念が頭に入る", "ネンガアタマニハイル"),
        ("労う言葉", "ネギラウコトバ"),
    ),
)
def test_general_idiom_entries_keep_competing_usages(text: str, expected_kana: str) -> None:
    """慣用句の動詞候補が一致しない一般用法の解析を維持する。"""

    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=True) == expected_kana


@pytest.mark.parametrize(
    ("text", "expected_pronunciation"),
    (
        ("千里さん", "チサト"),
        ("森高千里さん", "チサト"),
        ("千里の道", "センリ"),
        ("千里駅", "センリ"),
    ),
)
def test_chisato_uses_name_and_place_connections(
    text: str,
    expected_pronunciation: str,
) -> None:
    """千里の人名と一般語・地名を品詞接続と費用で読み分ける。"""

    start = text.index("千里")
    target_span = (start, start + len("千里"))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    selected_morph = next(
        morph for morph in analysis["morphs"] if morph["char_span"] == target_span
    )

    # 人名の名だけを費用6000まで下げ、一般語と駅名では専用品詞のセンリを選ぶ
    if expected_pronunciation == "チサト":
        assert selected_morph["features"][1:4] == ["名詞", "固有名詞", "人名"]
        assert selected_morph["features"][4] == "名"
        assert selected_morph["word_cost"] == 6000
    else:
        assert selected_morph["features"][2:4] != ["固有名詞", "人名"]
    assert selected_morph["features"][9] == expected_pronunciation
