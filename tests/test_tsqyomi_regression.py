"""tsqyomi v3 実推論回帰。期待値は pin 済み revision の v3 モデルで実測した値。"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
import pyopenjtalk.tsqyomi.diagnostics as tsqyomi_diagnostics
import pyopenjtalk.tsqyomi.inference as tsqyomi_inference
import pyopenjtalk.tsqyomi.model as tsqyomi_model
from pyopenjtalk.tsqyomi.inference import select_mecab_features_with_tsqyomi
from pyopenjtalk.types import MeCabMorph


@dataclass(frozen=True)
class _TargetExpectation:
    """1対象について期待する診断と発音。"""

    surface: str
    char_span: tuple[int, int]
    expected_outcome: str
    expected_pronunciation: str | None = None
    expected_segment_text: str | None = None
    was_preserved: bool = False


@dataclass(frozen=True)
class _Case:
    """1文の読み推論回帰症例。"""

    text: str
    expected_kana: str
    targets: tuple[_TargetExpectation, ...] = ()
    expect_no_diagnostics: bool = False


# revision 1157e36e (v3/model.onnx) で CPU 推論した期待値
_CASES: tuple[_Case, ...] = (
    _Case(
        text="人気の店です。",
        expected_kana="ニンキノミセデス。",
        targets=(
            _TargetExpectation(
                surface="人気",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="ニンキ",
                expected_segment_text="人気の店です。",
            ),
        ),
    ),
    _Case(
        text="人気のない店",
        expected_kana="ヒトケノナイミセ",
        targets=(
            _TargetExpectation(
                surface="人気",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="ヒトケ",
                expected_segment_text="人気のない店",
            ),
        ),
    ),
    _Case(
        text="誰もいないはずの茶室に人気を感じたが、座卓には最中が置かれていた。",
        expected_kana="ダレモイナイハズノチャシツニニンキヲカンジタガ、ザタクニワモナカガオカレテイタ。",
        targets=(
            _TargetExpectation(
                surface="人気",
                char_span=(11, 13),
                expected_outcome="applied",
                expected_pronunciation="ニンキ",
            ),
            _TargetExpectation(
                surface="最中",
                char_span=(23, 25),
                expected_outcome="applied",
                expected_pronunciation="モナカ",
            ),
        ),
    ),
    _Case(
        text="十分です",
        expected_kana="ジューブンデス",
        targets=(
            _TargetExpectation(
                surface="十分",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="ジューブン",
            ),
        ),
    ),
    _Case(
        text="この踊りは私の一番の十八番です",
        expected_kana="コノオドリワワタシノイチバンノオハコデス",
        targets=(
            _TargetExpectation(
                surface="十八番",
                char_span=(10, 13),
                expected_outcome="applied",
                expected_pronunciation="オハコ",
            ),
        ),
    ),
    _Case(
        text="何人いますか",
        expected_kana="ナンニンイマスカ",
        targets=(
            _TargetExpectation(
                surface="何人",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="ナンニン",
            ),
        ),
    ),
    _Case(
        text="会議を行った",
        expected_kana="カイギヲオコナッタ",
        targets=(
            _TargetExpectation(
                surface="行っ",
                char_span=(3, 5),
                expected_outcome="applied",
                expected_pronunciation="オコナッ",
            ),
        ),
    ),
    _Case(
        text="駅へ行った",
        expected_kana="エキエイッタ",
        targets=(
            _TargetExpectation(
                surface="行っ",
                char_span=(2, 4),
                expected_outcome="applied",
                expected_pronunciation="イッ",
            ),
        ),
    ),
    _Case(
        text="学校に通っている",
        expected_kana="ガッコーニカヨッテイル",
        targets=(
            _TargetExpectation(
                surface="通っ",
                char_span=(3, 5),
                expected_outcome="applied",
                expected_pronunciation="カヨッ",
            ),
        ),
    ),
    _Case(
        text="門を通って入る",
        expected_kana="モンヲトーッテハイル",
        targets=(
            _TargetExpectation(
                surface="通っ",
                char_span=(2, 4),
                expected_outcome="applied",
                expected_pronunciation="トーッ",
            ),
        ),
    ),
    _Case(
        text="この通りで待つ",
        expected_kana="コノトーリデマツ",
        targets=(
            _TargetExpectation(
                surface="通り",
                char_span=(2, 4),
                expected_outcome="applied",
                expected_pronunciation="トーリ",
            ),
        ),
    ),
    _Case(
        text="予想通りで驚いた",
        expected_kana="ヨソードーリデオドロイタ",
        targets=(
            _TargetExpectation(
                surface="通り",
                char_span=(2, 4),
                expected_outcome="applied",
                expected_pronunciation="ドーリ",
            ),
        ),
    ),
    _Case(
        text="商売上",
        expected_kana="ショーバイジョー",
        targets=(
            _TargetExpectation(
                surface="上",
                char_span=(2, 3),
                expected_outcome="applied",
                expected_pronunciation="ジョー",
            ),
        ),
    ),
    _Case(
        text="÷÷÷÷人気",
        expected_kana="÷÷÷÷ニンキ",
        targets=(
            _TargetExpectation(
                surface="人気",
                char_span=(4, 6),
                expected_outcome="applied",
                expected_pronunciation="ニンキ",
            ),
        ),
    ),
    _Case(
        text="一寸です",
        expected_kana="チョットデス",
        targets=(
            _TargetExpectation(
                surface="一寸",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="チョット",
            ),
        ),
    ),
    _Case(
        text="いじけるなんて大人気ないな君は。",
        expected_kana="イジケルナンテオトナゲナイナキミワ。",
        targets=(
            _TargetExpectation(
                surface="大人気",
                char_span=(7, 10),
                expected_outcome="applied",
                expected_pronunciation="オトナゲ",
            ),
        ),
    ),
    _Case(
        text="この中で何曲歌える？",
        expected_kana="コノナカデナンキョクウタエル？",
        targets=(
            _TargetExpectation(
                surface="何",
                char_span=(4, 5),
                expected_outcome="applied",
                expected_pronunciation="ナン",
            ),
        ),
    ),
    _Case(
        text="人の金で食う飯は美味い。",
        expected_kana="ヒトノカネデクウメシワウマイ。",
        targets=(
            _TargetExpectation(
                surface="金",
                char_span=(2, 3),
                expected_outcome="applied",
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _Case(
        text="仕事の最中に最中を食べるな！",
        expected_kana="シゴトノサイチューニモナカヲタベルナ！",
        targets=(
            _TargetExpectation(
                surface="最中",
                char_span=(3, 5),
                expected_outcome="applied",
                expected_pronunciation="サイチュー",
            ),
            _TargetExpectation(
                surface="最中",
                char_span=(6, 8),
                expected_outcome="applied",
                expected_pronunciation="モナカ",
            ),
        ),
    ),
    _Case(
        text="大分県にもう大分長いこと住んでいるな。",
        expected_kana="オーイタケンニモーダイブナガイコトスンデイルナ。",
        targets=(
            _TargetExpectation(
                surface="大分",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="オーイタ",
            ),
            _TargetExpectation(
                surface="大分",
                char_span=(6, 8),
                expected_outcome="applied",
                expected_pronunciation="ダイブ",
            ),
        ),
    ),
    _Case(
        text="彼に敬意を表します。",
        expected_kana="カレニケーイヲヒョーシマス。",
        targets=(
            _TargetExpectation(
                surface="表し",
                char_span=(5, 7),
                expected_outcome="applied",
                expected_pronunciation="ヒョーシ",
            ),
        ),
    ),
    _Case(
        text="新しく金が発見された地に赴くにも金がかかる。",
        expected_kana="アタラシクキンガハッケンサレタチニオモムクニモカネガカカル。",
        targets=(
            _TargetExpectation(
                surface="金",
                char_span=(3, 4),
                expected_outcome="applied",
                expected_pronunciation="キン",
            ),
            _TargetExpectation(
                surface="金",
                char_span=(16, 17),
                expected_outcome="applied",
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _Case(
        text="泥を被るという被害を被った。",
        expected_kana="ドロヲカブルトイウヒガイヲカブッタ。",
        targets=(
            _TargetExpectation(
                surface="被る",
                char_span=(2, 4),
                expected_outcome="applied",
                expected_pronunciation="カブル",
            ),
        ),
    ),
    _Case(
        text="竹田はかつて岡藩の城下町であった。",
        expected_kana="タケタワカツテオカハンノジョーカマチデアッタ。",
        targets=(
            _TargetExpectation(
                surface="竹田",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="タケタ",
            ),
        ),
    ),
    _Case(
        text="素振りをする素振りを見せた。",
        expected_kana="スブリヲスルソブリヲミセタ。",
        targets=(
            _TargetExpectation(
                surface="素振り",
                char_span=(0, 3),
                expected_outcome="applied",
                expected_pronunciation="スブリ",
            ),
            _TargetExpectation(
                surface="素振り",
                char_span=(6, 9),
                expected_outcome="applied",
                expected_pronunciation="ソブリ",
            ),
        ),
    ),
    _Case(
        text="角の生えた鬼に向かって角が立たない言い回し。",
        expected_kana="ツノノハエタオニニムカッテカドガタタナイイーマワシ。",
        targets=(
            _TargetExpectation(
                surface="角",
                char_span=(0, 1),
                expected_outcome="applied",
                expected_pronunciation="ツノ",
            ),
            _TargetExpectation(
                surface="角",
                char_span=(11, 12),
                expected_outcome="applied",
                expected_pronunciation="カド",
            ),
        ),
    ),
    _Case(
        text="辛いことだが仕方がない。",
        expected_kana="ツライコトダガシカタガナイ。",
        targets=(
            _TargetExpectation(
                surface="辛い",
                char_span=(0, 2),
                expected_outcome="applied",
                expected_pronunciation="ツライ",
            ),
        ),
    ),
    _Case(
        text="深夜の路地は人気が無くて怖い。",
        expected_kana="シンヤノロジワヒトケガナクテコワイ。",
        targets=(
            _TargetExpectation(
                surface="人気",
                char_span=(6, 8),
                expected_outcome="applied",
                expected_pronunciation="ヒトケ",
            ),
        ),
    ),
    _Case(
        text="金の時計を買うために、一生懸命に金を貯めた。",
        expected_kana="キンノトケーヲカウタメニ、イッショーケンメーニカネヲタメタ。",
        targets=(
            _TargetExpectation(
                surface="金",
                char_span=(0, 1),
                expected_outcome="applied",
                expected_pronunciation="キン",
            ),
            _TargetExpectation(
                surface="金",
                char_span=(16, 17),
                expected_outcome="applied",
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _Case(
        text="カブトムシの立派な角に止まった小さな虫を、指で軽く弾く。",
        expected_kana="カブトムシノリッパナツノニトマッタチーサナムシヲ、ユビデカルクハジク。",
        targets=(
            _TargetExpectation(
                surface="角",
                char_span=(9, 10),
                expected_outcome="applied",
                expected_pronunciation="ツノ",
            ),
            _TargetExpectation(
                surface="弾く",
                char_span=(25, 27),
                expected_outcome="applied",
                expected_pronunciation="ハジク",
            ),
        ),
    ),
    _Case(
        text="庭に植えた紅葉の木が立派に育ってきた。",
        expected_kana="ニワニウエタモミジノキガリッパニソダッテキタ。",
        targets=(
            _TargetExpectation(
                surface="紅葉",
                char_span=(5, 7),
                expected_outcome="applied",
                expected_pronunciation="モミジ",
            ),
        ),
    ),
    _Case(
        text="ひらがなだけ",
        expected_kana="ヒラガナダケ",
        expect_no_diagnostics=True,
    ),
)


def _load_tsqyomi_v3() -> None:
    """pin 済み v3 モデルをロードする。"""

    pytest.importorskip("onnxruntime")
    if tsqyomi.is_model_loaded() is False:
        tsqyomi.load_model(["CPUExecutionProvider"])


@pytest.fixture(scope="session")
def tsqyomi_v3() -> Iterator[None]:
    """セッション全体で v3 モデルを1回ロードする。"""

    _load_tsqyomi_v3()
    yield
    if tsqyomi.is_model_loaded():
        tsqyomi.unload_model()


def _run_with_diagnostics(text: str) -> tuple[str, list[tsqyomi_diagnostics.TargetDiagnostic]]:
    """g2p() の結果と診断記録を同時に返す。"""

    tsqyomi_diagnostics.start_recording()
    try:
        kana_result = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True)
    except Exception:
        tsqyomi_diagnostics.stop_recording()
        raise
    assert isinstance(kana_result, str)
    return kana_result, tsqyomi_diagnostics.stop_recording()


def _find_diagnostic(
    diagnostics: list[tsqyomi_diagnostics.TargetDiagnostic],
    expectation: _TargetExpectation,
) -> tsqyomi_diagnostics.TargetDiagnostic:
    """表層と位置から診断1件を特定する。"""

    matched = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.surface == expectation.surface
        and diagnostic.char_span == expectation.char_span
    ]
    assert len(matched) == 1
    return matched[0]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.text)
def test_reading_regression(case: _Case, tsqyomi_v3: None) -> None:
    """v3 モデルの読み選択とカタカナ出力が pin 済み期待値と一致する。"""

    kana, diagnostics = _run_with_diagnostics(case.text)

    assert kana == case.expected_kana
    if case.expect_no_diagnostics:
        assert diagnostics == []
        return

    for expectation in case.targets:
        diagnostic = _find_diagnostic(diagnostics, expectation)
        assert diagnostic.outcome == expectation.expected_outcome
        assert diagnostic.selected_pronunciation == expectation.expected_pronunciation
        assert diagnostic.was_preserved is expectation.was_preserved
        if expectation.expected_segment_text is not None:
            assert diagnostic.segment_text == expectation.expected_segment_text


def test_load_model_is_idempotent(tsqyomi_v3: None) -> None:
    """ロード済みのモデルを繰り返し取得しない。"""

    loaded_model = tsqyomi.get_loaded_model()
    tsqyomi.load_model(["CPUExecutionProvider"])
    assert tsqyomi.get_loaded_model() is loaded_model


def test_unload_model_can_reload(tsqyomi_v3: None) -> None:
    """unload_model() 後に再ロードできる。"""

    assert tsqyomi.is_model_loaded() is True
    tsqyomi.unload_model()
    assert tsqyomi.is_model_loaded() is False
    _load_tsqyomi_v3()
    assert tsqyomi.is_model_loaded() is True


def test_long_text_passes_only_target_sentence_to_model(tsqyomi_v3: None) -> None:
    """長い前置きでは対象を含む末尾文だけをモデルへ渡す。"""

    prefix = "これはひらがなだけのぶんしょうです。" * 50
    target_sentence = "人気のない店"
    text = prefix + target_sentence

    _kana, diagnostics = _run_with_diagnostics(text)

    ninki = _find_diagnostic(
        diagnostics,
        _TargetExpectation(
            surface="人気",
            char_span=(900, 902),
            expected_outcome="applied",
            expected_pronunciation="ヒトケ",
            expected_segment_text=target_sentence,
        ),
    )
    assert ninki.segment_text == target_sentence


def test_adjacent_targets_use_candidate_connection_cost(tsqyomi_v3: None) -> None:
    """隣接する2対象では後側形態素の link_cost に候補間接続辺を反映する。"""

    text = "人気最中です"
    target_spans = ((0, 2), (2, 4))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, target_spans)
    paths_by_span = {
        target_span: tuple(path for path in analysis["paths"] if path["char_span"] == target_span)
        for target_span in target_spans
    }
    connection_costs = {
        (connection["left_node_id"], connection["right_node_id"]): connection["cost"]
        for connection in analysis["connections"]
    }

    _features, morphs = select_mecab_features_with_tsqyomi(text, jtalk)

    left_pronunciation = morphs[0]["features"][9]
    right_pronunciation = morphs[1]["features"][9]
    left_path = next(
        path
        for path in paths_by_span[target_spans[0]]
        if path["pronunciation"] == left_pronunciation
    )
    right_path = next(
        path
        for path in paths_by_span[target_spans[1]]
        if path["pronunciation"] == right_pronunciation
    )
    expected_link_cost = connection_costs[(left_path["node_ids"][-1], right_path["node_ids"][0])]
    assert morphs[1]["link_cost"] == expected_link_cost


def test_high_level_dictionary_protection_skips_model_inference(
    tmp_path: Path,
    tsqyomi_v3: None,
) -> None:
    """読み保護ユーザー辞書ではモデル推論を止める。"""

    unprotected_csv = tmp_path / "unprotected.csv"
    protected_csv = tmp_path / "protected.csv"
    unprotected_dic = tmp_path / "unprotected.dic"
    protected_dic = tmp_path / "protected.dic"
    unprotected_csv.write_text(
        "人気,,,1,名詞,一般,*,*,*,*,人気,ニンキ,ニンキ,0/3,*\n",
        encoding="utf-8",
    )
    protected_csv.write_text(
        "人気,,,1,名詞,一般,*,*,*,*,人気,ヒトケ,ヒトケ,0/3,*\n",
        encoding="utf-8",
    )
    pyopenjtalk.mecab_dict_index(str(unprotected_csv), str(unprotected_dic))
    pyopenjtalk.mecab_dict_index(str(protected_csv), str(protected_dic))

    try:
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                {
                    "dic_path": str(unprotected_dic),
                    "is_reading_protected": False,
                },
                {
                    "dic_path": str(protected_dic),
                    "is_reading_protected": True,
                },
            ]
        )
        kana, diagnostics = _run_with_diagnostics("人気の店です。")
        assert kana == "ニンキノミセデス。"
        assert len(diagnostics) == 1
        assert diagnostics[0].outcome == "reading_protected"
        assert diagnostics[0].selected_pronunciation is None
    finally:
        pyopenjtalk.unset_user_dict()


def test_include_morphs_false_skips_morph_rebuild(tsqyomi_v3: None) -> None:
    """include_morphs=False では形態素差し替えを省略し feature だけ更新する。"""

    replace_calls = 0
    original_replace = tsqyomi_inference._replace_morph

    def counting_replace(*args: Any, **kwargs: Any) -> MeCabMorph:
        """形態素差し替えの呼び出し回数を記録する。"""

        nonlocal replace_calls
        replace_calls += 1
        return original_replace(*args, **kwargs)

    text = "一寸です"
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    tsqyomi_inference._replace_morph = counting_replace
    try:
        features, morphs = select_mecab_features_with_tsqyomi(
            text,
            jtalk,
            include_morphs=False,
        )
    finally:
        tsqyomi_inference._replace_morph = original_replace

    assert morphs == []
    assert replace_calls == 0
    assert any("チョット" in feature for feature in features)


def test_v3_onnx_contract_matches_loaded_model(tsqyomi_v3: None) -> None:
    """ロード済み v3 セッションがメタデータ契約を満たす。"""

    model = tsqyomi.get_loaded_model()
    tsqyomi.TsqyomiModel.validate_onnx_contract(model.session, model.metadata)


def test_model_revision_is_pinned() -> None:
    """テストが参照するモデル revision が実装 pin と一致する。"""

    assert tsqyomi_model._MODEL_REVISION == "1157e36e1bf81a4cc01ed911b7dc691106c1ccdb"
    assert tsqyomi_model._MODEL_FILES["model"] == "v3/model.onnx"


def test_g2p_mapping_aligns_tsqyomi_reading_to_morph_char_span(tsqyomi_v3: None) -> None:
    """g2p_mapping() の char_span と phoneme 列が、tsqyomi の選択結果と一致する。"""

    text = "深夜の路地は人気が無くて怖い。"
    mapping = pyopenjtalk.g2p_mapping(text, use_tsqyomi=True, use_vanilla=True)
    ninki = next(entry for entry in mapping if entry["surface"] == "人気")
    assert ninki["char_span"] == (6, 8)
    assert ninki["phonemes"] == ["h", "I", "t", "o", "k", "e"]


def test_run_frontend_detailed_reflects_tsqyomi_pronunciation(tsqyomi_v3: None) -> None:
    """run_frontend_detailed() の NJD feature が tsqyomi による選択発音を反映する。"""

    text = "大分県にもう大分長いこと住んでいるな。"
    _features, morphs = pyopenjtalk.run_frontend_detailed(text, use_tsqyomi=True, use_vanilla=True)
    oita_first = next(morph for morph in morphs if morph["char_span"] == (0, 2))
    oita_second = next(morph for morph in morphs if morph["char_span"] == (6, 8))
    assert oita_first["features"][9] == "オーイタ"
    assert oita_second["features"][9] == "ダイブ"


def test_extract_fullcontext_succeeds_with_tsqyomi(tsqyomi_v3: None) -> None:
    """extract_fullcontext() が tsqyomi 有効時でもラベル列を返す。"""

    text = "竹田はかつて岡藩の城下町であった。"
    labels = pyopenjtalk.extract_fullcontext(text, use_tsqyomi=True, use_vanilla=True)
    assert len(labels) >= 1
    assert all(isinstance(label, str) for label in labels)


@pytest.mark.parametrize(
    ("text", "expected_surfaces"),
    [
        (
            "もし明朝体が重いようならまたおいで。",
            ("明朝", "体"),
        ),
        (
            "将棋において玉の扱いは重要。",
            ("玉", "要"),
        ),
    ],
)
def test_compound_scored_surface_reports_no_exact_morph_range(
    tsqyomi_v3: None,
    text: str,
    expected_surfaces: tuple[str, ...],
) -> None:
    """辞書形態素境界と target span が一致しない複合表層は差し替えを行わない。"""

    _kana, diagnostics = _run_with_diagnostics(text)
    assert tuple(diagnostic.surface for diagnostic in diagnostics) == expected_surfaces
    assert all(diagnostic.outcome == "no_exact_morph_range" for diagnostic in diagnostics)
    assert all(diagnostic.selected_pronunciation is None for diagnostic in diagnostics)


def test_enabled_tsqyomi_changes_g2p_output_from_baseline(tsqyomi_v3: None) -> None:
    """tsqyomi 有効時は無効時と異なるカタカナ出力になる対象文を通す。"""

    text = "深夜の路地は人気が無くて怖い。"
    without_tsqyomi = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)
    assert without_tsqyomi != with_tsqyomi
    assert with_tsqyomi == "シンヤノロジワヒトケガナクテコワイ。"
