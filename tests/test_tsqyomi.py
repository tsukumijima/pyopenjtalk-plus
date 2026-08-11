"""tsqyomi の契約検証、診断、MeCab 候補解析、Execution Provider 選択を確認する。"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
import pyopenjtalk.tsqyomi.diagnostics as tsqyomi_diagnostics
import pyopenjtalk.tsqyomi.inference as tsqyomi_inference
import pyopenjtalk.tsqyomi.model as tsqyomi_model
from pyopenjtalk.types import MeCabMorph


@pytest.fixture(scope="module")
def default_jtalk() -> pyopenjtalk.OpenJTalk:
    """数量表現テストで共有するデフォルト辞書の OpenJTalk を返す。

    Returns:
        pyopenjtalk.OpenJTalk: デフォルト辞書を読み込んだ共有インスタンス
    """

    return pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)


def test_load_model_initializes_once_without_blocking_status_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """モデル初期化中も状態照会を止めず、重複ロードを1回へまとめる。"""

    for asset_name in ("model.onnx", "tokenizer.json", "metadata.json"):
        (tmp_path / asset_name).write_bytes(b"test")
    is_loading_started = Event()
    can_finish_loading = Event()
    loaded_model = cast(tsqyomi.TsqyomiModel, SimpleNamespace())
    load_count = 0

    def slow_load_model_from_paths(*_args: Any, **_kwargs: Any) -> tsqyomi.TsqyomiModel:
        """初期化待ちを再現し、同時呼び出し回数を記録する。"""

        nonlocal load_count
        load_count += 1
        is_loading_started.set()
        assert can_finish_loading.wait(timeout=5.0) is True
        return loaded_model

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", None)
    monkeypatch.setattr(tsqyomi_model, "_is_model_loading", False)
    monkeypatch.setattr(tsqyomi_model, "_load_model_from_paths", slow_load_model_from_paths)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            first_load = executor.submit(tsqyomi.load_model, model_dir=tmp_path)
            assert is_loading_started.wait(timeout=5.0) is True
            second_load = executor.submit(tsqyomi.load_model, model_dir=tmp_path)
            status_query = executor.submit(tsqyomi.is_model_loaded)
            model_query = executor.submit(tsqyomi.get_loaded_model)

            assert status_query.result(timeout=5.0) is False
            with pytest.raises(RuntimeError, match="load_model"):
                model_query.result(timeout=5.0)
            can_finish_loading.set()
            first_load.result(timeout=5.0)
            second_load.result(timeout=5.0)
    finally:
        can_finish_loading.set()
        tsqyomi.unload_model()

    assert load_count == 1


def _minimal_v3_metadata_payload(**overrides: Any) -> dict[str, Any]:
    """テスト用の最小 v3 metadata 辞書を返す。"""

    payload: dict[str, Any] = {
        "schema_version": "v3",
        "model_max_length": 256,
        "class_index_by_surface_and_pronunciation": {
            "人気": {"ニンキ": 0, "ヒトケ": 1},
        },
    }
    payload.update(overrides)
    return payload


def test_diagnostics_rebase_recording_char_spans() -> None:
    """分割片で記録した診断位置を元の本文位置へ戻す。"""

    tsqyomi_diagnostics.start_recording()
    tsqyomi_diagnostics.record(
        tsqyomi_diagnostics.TargetDiagnostic(
            segment_text="対象を含む分割片",
            char_span=(3, 5),
            surface="対象",
            outcome="applied",
            score_margin=1.5,
        )
    )
    tsqyomi_diagnostics.rebase_recording_char_spans(0, 20)

    diagnostics = tsqyomi_diagnostics.stop_recording()
    assert diagnostics[0].char_span == (23, 25)


def test_diagnostics_separates_concurrent_recordings() -> None:
    """並行する実行コンテキストごとに診断記録を分離する。"""

    barrier = Barrier(2)

    def collect(surface: str) -> list[tsqyomi_diagnostics.TargetDiagnostic]:
        """独立した診断記録を1件収集する。"""

        tsqyomi_diagnostics.start_recording()
        tsqyomi_diagnostics.record(
            tsqyomi_diagnostics.TargetDiagnostic(
                segment_text=surface,
                char_span=(0, len(surface)),
                surface=surface,
                outcome="applied",
            )
        )
        barrier.wait(timeout=1.0)
        return tsqyomi_diagnostics.stop_recording()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(collect, "人気")
        second_future = executor.submit(collect, "最中")

    assert [record.surface for record in first_future.result()] == ["人気"]
    assert [record.surface for record in second_future.result()] == ["最中"]


def test_onnx_contract_accepts_v3_model_shape() -> None:
    """出力列を直接参照する v3 ONNX とメタデータの組を受理する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v3_metadata_payload())
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="target_mask", type="tensor(bool)"),
        ],
        get_outputs=lambda: [
            SimpleNamespace(
                name="reading_class_logits",
                type="tensor(float)",
                shape=["batch", "target", 2],
            )
        ],
    )

    tsqyomi.TsqyomiModel.validate_onnx_contract(session, metadata)


def test_metadata_rejects_obsolete_schema_version() -> None:
    """旧 schema_version のメタデータを v3 契約へ誤接続しない。"""

    with pytest.raises(ValueError, match="schema_version"):
        tsqyomi.TsqyomiMetadata.model_validate(
            _minimal_v3_metadata_payload(schema_version="v2")
        )


@pytest.mark.parametrize(
    ("leading_token_id", "trailing_token_id"),
    [(1, None), (None, 2)],
)
def test_metadata_rejects_one_sided_boundary_token_id(
    leading_token_id: int | None,
    trailing_token_id: int | None,
) -> None:
    """学習時の前後特殊トークン ID は片側だけの指定を拒否する。"""

    with pytest.raises(ValueError, match="must be specified together"):
        tsqyomi.TsqyomiMetadata.model_validate(
            _minimal_v3_metadata_payload(
                leading_token_id=leading_token_id,
                trailing_token_id=trailing_token_id,
            )
        )


def test_onnx_contract_rejects_wrong_target_mask_type() -> None:
    """対象マスクが bool でない旧世代または破損した ONNX を拒否する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v3_metadata_payload())
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="target_mask", type="tensor(int64)"),
        ],
        get_outputs=lambda: cast(list[Any], []),
    )

    with pytest.raises(ValueError, match="inputs do not match the v3 contract"):
        tsqyomi.TsqyomiModel.validate_onnx_contract(session, metadata)


def test_onnx_contract_rejects_different_reading_class_count() -> None:
    """メタデータと異なる読みクラス数の ONNX をモデル初期化前に拒否する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v3_metadata_payload())
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="target_mask", type="tensor(bool)"),
        ],
        get_outputs=lambda: [
            SimpleNamespace(
                name="reading_class_logits",
                type="tensor(float)",
                shape=["batch", "target", 3],
            )
        ],
    )

    with pytest.raises(ValueError, match="reference every ONNX output class"):
        tsqyomi.TsqyomiModel.validate_onnx_contract(session, metadata)


def test_metadata_rejects_negative_class_index() -> None:
    """表層別対応表に負の ONNX 出力列があればロード前に拒否する。"""

    with pytest.raises(ValueError, match="must not be negative"):
        tsqyomi.TsqyomiMetadata.model_validate(
            _minimal_v3_metadata_payload(
                class_index_by_surface_and_pronunciation={
                    "人気": {"ニンキ": -1, "ヒトケ": 1},
                }
            )
        )


def test_onnx_contract_rejects_wrong_output_rank() -> None:
    """読みクラス出力が3次元でない ONNX をモデル初期化前に拒否する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v3_metadata_payload())
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="target_mask", type="tensor(bool)"),
        ],
        get_outputs=lambda: [
            SimpleNamespace(
                name="reading_class_logits",
                type="tensor(float)",
                shape=["target", 2],
            )
        ],
    )

    with pytest.raises(ValueError, match="batch, target, and class dimensions"):
        tsqyomi.TsqyomiModel.validate_onnx_contract(session, metadata)


def test_default_provider_selection_prefers_cuda_then_cpu() -> None:
    """既定設定は利用可能な CUDA を先頭に置き、CPU をフォールバックとして続ける。"""

    class FakeONNXRuntime:
        """CUDA と CPU を利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    resolved = tsqyomi_model._resolve_onnx_providers(FakeONNXRuntime, None)
    assert resolved == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_explicit_provider_selection_rejects_unavailable_entries() -> None:
    """明示された実行プロバイダが利用できない場合は失敗させる。"""

    class FakeONNXRuntime:
        """DirectML と CPU を利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        tsqyomi_model._resolve_onnx_providers(
            FakeONNXRuntime,
            [
                "CUDAExecutionProvider",
                ("DmlExecutionProvider", {"device_id": 1}),
                "CPUExecutionProvider",
            ],
        )


def test_provider_selection_rejects_missing_provider() -> None:
    """利用できない実行プロバイダを明示した場合は拒否する。"""

    class CPUOnlyONNXRuntime:
        """CPU だけを利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="unavailable"):
        tsqyomi_model._resolve_onnx_providers(
            CPUOnlyONNXRuntime,
            ["CUDAExecutionProvider"],
        )


def test_session_provider_verification_rejects_silent_fallback() -> None:
    """allow_provider_fallback=False のとき、CUDA 要求で CPU だけ有効なセッションを拒否する。"""

    class FakeSession:
        """CUDA 初期化に失敗して CPU だけが有効になったセッションの代用品。"""

        @staticmethod
        def get_providers() -> list[str]:
            """実際に有効化された実行プロバイダを返す。"""

            return ["CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="did not activate"):
        tsqyomi_model._verify_session_providers(
            FakeSession,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            False,
        )


def test_session_provider_verification_accepts_activated_head_and_explicit_fallback() -> None:
    """最優先プロバイダが有効なら通し、allow_provider_fallback=True なら CPU フォールバックも通す。"""

    class CUDASession:
        """CUDA が先頭で有効になったセッションの代用品。"""

        @staticmethod
        def get_providers() -> list[str]:
            """実際に有効化された実行プロバイダを返す。"""

            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class CPUFallbackSession:
        """CPU へフォールバックしたセッションの代用品。"""

        @staticmethod
        def get_providers() -> list[str]:
            """実際に有効化された実行プロバイダを返す。"""

            return ["CPUExecutionProvider"]

    tsqyomi_model._verify_session_providers(
        CUDASession,
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        False,
    )
    tsqyomi_model._verify_session_providers(
        CPUFallbackSession,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        True,
    )


def test_explicitly_disabled_tsqyomi_preserves_all_high_level_api_results() -> None:
    """全高レベル API で tsqyomi の省略時と明示無効時の結果が一致することを確認する。"""

    text = "人気の店です。"

    assert pyopenjtalk.g2p(text, kana=True) == pyopenjtalk.g2p(
        text,
        kana=True,
        use_tsqyomi=False,
    )
    assert pyopenjtalk.g2p_mapping(text) == pyopenjtalk.g2p_mapping(
        text,
        use_tsqyomi=False,
    )
    assert pyopenjtalk.extract_fullcontext(text) == pyopenjtalk.extract_fullcontext(
        text,
        use_tsqyomi=False,
    )
    assert pyopenjtalk.run_frontend(text) == pyopenjtalk.run_frontend(
        text,
        use_tsqyomi=False,
    )
    assert pyopenjtalk.run_frontend_detailed(text) == pyopenjtalk.run_frontend_detailed(
        text,
        use_tsqyomi=False,
    )

    default_waveform, default_sampling_rate = pyopenjtalk.tts(text)
    disabled_waveform, disabled_sampling_rate = pyopenjtalk.tts(text, use_tsqyomi=False)
    assert disabled_sampling_rate == default_sampling_rate
    assert np.array_equal(disabled_waveform, default_waveform)


def test_enabled_tsqyomi_requires_explicit_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """高レベル API もモデルの暗黙ロードや CPU 推論への切り替えを行わない。"""

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", None)
    with pytest.raises(RuntimeError, match="load_model"):
        pyopenjtalk.g2p("人気の店です。", use_tsqyomi=True)


def test_sentence_segmentation_keeps_period_inside_matched_quote() -> None:
    """対応する引用符の内側にある句点で対象文を分断しない。"""

    text = "前置きです。彼は「仕事の最中だ。」と言った。後続です。"
    target_start = text.index("最中")

    segments = tsqyomi_inference._split_target_processing_segments(
        text,
        ((target_start, target_start + len("最中")),),
    )

    assert tuple(text[start:end] for start, end in segments) == (
        "前置きです。",
        "彼は「仕事の最中だ。」と言った。",
        "後続です。",
    )


def test_sentence_segmentation_tracks_nested_delimiters() -> None:
    """入れ子の括弧と引用符が閉じるまで対象文を保つ。"""

    text = "前置きです。彼は「（仕事の最中だ。）と記した。」と語った。後続です。"
    target_start = text.index("最中")

    segments = tsqyomi_inference._split_target_processing_segments(
        text,
        ((target_start, target_start + len("最中")),),
    )

    assert tuple(text[start:end] for start, end in segments) == (
        "前置きです。",
        "彼は「（仕事の最中だ。）と記した。」と語った。",
        "後続です。",
    )


def test_sentence_segmentation_uses_period_after_closing_parenthesis() -> None:
    """閉じ括弧直後の句点で対象文を分断する。"""

    text = "前置きです。彼は（仕事の最中だ。）と書いた。後続です。"
    target_start = text.index("最中")

    segments = tsqyomi_inference._split_target_processing_segments(
        text,
        ((target_start, target_start + len("最中")),),
    )

    assert tuple(text[start:end] for start, end in segments) == (
        "前置きです。",
        "彼は（仕事の最中だ。）と書いた。",
        "後続です。",
    )


def test_sentence_segmentation_does_not_extend_unmatched_quote() -> None:
    """対応しない引用符の内側へ対象文を広げない。"""

    text = "前置きです。彼は「仕事の最中だ。後続です。"
    target_start = text.index("最中")

    segments = tsqyomi_inference._split_target_processing_segments(
        text,
        ((target_start, target_start + len("最中")),),
    )

    assert tuple(text[start:end] for start, end in segments) == (
        "前置きです。",
        "彼は「仕事の最中だ。",
        "後続です。",
    )


def test_analyze_mecab_candidates_expands_symbol_morphs_like_detailed() -> None:
    """候補解析の最良経路の morphs が、run_mecab_detailed() と同じ記号分割になることを確認。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    text = "人気÷÷÷÷"
    _, detailed_morphs = jtalk.run_mecab_detailed(text)
    analysis = jtalk.analyze_mecab_candidates(text, ((0, 2),))

    assert [morph["surface"] for morph in analysis["morphs"]] == [
        morph["surface"] for morph in detailed_morphs
    ]
    assert all(
        morph["is_unknown"] is False for morph in analysis["morphs"] if morph["surface"] == "÷"
    )
    assert analysis["feature_index_by_morph"] == (0, 1, 1, 1, 1)


def test_analyze_mecab_candidates_filters_public_connections() -> None:
    """公開候補ノード同士の辺だけを connections へ載せることを確認。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates("人気の店です。", ((0, 2),))
    public_node_ids = {node["node_id"] for node in analysis["nodes"]}

    assert len(public_node_ids) >= 1
    assert len(analysis["connections"]) < 100
    for connection in analysis["connections"]:
        assert connection["left_node_id"] in public_node_ids
        assert connection["right_node_id"] in public_node_ids


def test_analyze_mecab_candidates_exposes_dual_readings() -> None:
    """候補グラフ上に複数読みが到達可能であることを確認する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    text = "素振りをする素振りを見せた。"
    analysis = jtalk.analyze_mecab_candidates(text, ((0, 3), (6, 9)))
    pronunciations_by_span = {
        target_span: {
            path["pronunciation"] for path in analysis["paths"] if path["char_span"] == target_span
        }
        for target_span in ((0, 3), (6, 9))
    }

    assert pronunciations_by_span[(0, 3)] >= {"スブリ", "ソブリ"}
    assert pronunciations_by_span[(6, 9)] >= {"スブリ", "ソブリ"}


@pytest.mark.parametrize(
    "text",
    (
        "パーティー日は会場を貸し切ります。",
        "サービス日はポイントが二倍になります。",
        "定休日は木・金となります。",
        "外来日は休みです。",
        "誕生日は休みです。",
    ),
)
def test_tsqyomi_preserves_out_of_class_suffix_default(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    """モデル候補外の接尾辞読みは、前接語を含む辞書解析の既定値を維持する。"""

    def predict_nichi(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """同じ接尾用法の候補読みをモデル選択結果として返す。"""

        return (tsqyomi.ReadingPrediction(pronunciation="ニチ", scores=(0.0, 1.0)),)

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"日": ("日",)},
            class_index_by_surface_and_pronunciation={
                "日": {"ヒ": 0, "ニチ": 1},
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_nichi,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    tsqyomi_diagnostics.start_recording()
    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(
        text,
        jtalk,
    )
    diagnostics = tsqyomi_diagnostics.stop_recording()

    assert any(feature.split(",")[0:3] == ["日", "名詞", "接尾"] for feature in features)
    assert any(feature.split(",")[9] == "ビ" for feature in features if feature.startswith("日,"))
    assert len(diagnostics) == 1
    assert diagnostics[0].outcome == "dictionary_default_protected"
    assert diagnostics[0].selected_pronunciation == "ビ"
    assert diagnostics[0].was_preserved is True


@pytest.mark.parametrize(
    ("text", "expected_pronunciation"),
    (
        ("漫画家です。", "カ"),
        ("専門家です。", "カ"),
        ("山田家です。", "ケ"),
        ("絵を描く漫画家です。", "カ"),
    ),
)
def test_tsqyomi_preserves_productive_compound_suffix_default(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    expected_pronunciation: str,
) -> None:
    """前接名詞と結合した「家」は、用法に応じた辞書既定読みを維持する。"""

    def predict_ie(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """一般名詞の「家」としてイエを選んだモデル結果を返す。"""

        return (tsqyomi.ReadingPrediction(pronunciation="イエ", scores=(1.0, 0.0)),)

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"家": ("家",)},
            class_index_by_surface_and_pronunciation={
                "家": {"イエ": 0, "ウチ": 1},
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_ie,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    tsqyomi_diagnostics.start_recording()
    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(text, jtalk)
    diagnostics = tsqyomi_diagnostics.stop_recording()

    assert any(
        feature.split(",")[9] == expected_pronunciation
        for feature in features
        if feature.startswith("家,")
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].outcome == "dictionary_default_protected"
    assert diagnostics[0].selected_pronunciation == expected_pronunciation
    assert diagnostics[0].was_preserved is True


@pytest.mark.parametrize(
    ("text", "expected_ranges"),
    (
        ("一時間かかります。", ((0, 3),)),
        ("二十四時間営業です。", ((2, 5),)),
        ("二時間", ((0, 3),)),
        ("三時間かかります。", ((0, 3),)),
        ("四時間かかります。", ((0, 3),)),
        ("九時間かかります。", ((0, 3),)),
        ("十時間かかります。", ((0, 3),)),
        ("四十分後に戻ります。", ((3, 4),)),
        ("五分後に戻ります。", ((2, 3),)),
        ("十分後に戻ります。", ((2, 3),)),
        ("三秒後です。", ((2, 3),)),
        ("三日後です。", ((2, 3),)),
        ("三週間後です。", ((3, 4),)),
        ("三か月後です。", ((3, 4),)),
        ("三年後です。", ((2, 3),)),
        ("三回後です。", ((2, 3),)),
        ("三枚後です。", ((2, 3),)),
        ("百二十分です。", ((0, 4),)),
        ("百四十分です。", ((0, 4),)),
        ("百五十分です。", ((0, 4),)),
        ("あと三十分。", ((2, 5),)),
        ("数分後に届きます。", ((2, 3),)),
        ("何分", ((0, 2),)),
        ("何時間", ((0, 3),)),
        ("何時間何分", ((0, 3), (3, 5))),
        ("何時間何分かかります。", ((0, 3), (3, 5))),
        ("何分後に届きます。", ((0, 2), (2, 3))),
        ("何時何分", ((0, 4),)),
        ("何時後に届きます。", ((0, 3),)),
        ("何時まで後", ()),
        ("あと二時間です。", ((2, 5),)),
        ("あと十時間後です。", ((2, 5), (5, 6))),
        ("何人と。", ()),
        ("何軒か。", ((0, 2),)),
        ("何個か。", ((0, 2),)),
        ("この中で何曲歌える？", ((4, 6),)),
        ("一月前かかります。", ((0, 2),)),
        ("あと一月程度です。", ((2, 4),)),
        ("一月号を読みます。", ()),
        ("来年の一月前に準備します。", ()),
        ("五分の一を使います。", ()),
        ("三人後に並びます。", ((2, 3),)),
        ("その後どうする。", ()),
        ("作業の後で。", ()),
        ("晴れた後。", ()),
        ("何時まで営業しますか。", ()),
        ("何時まで後ろに並んでください。", ()),
        ("数分", ()),
        ("数分後", ()),
        ("門を通った時に止める間もなく進んだ。", ()),
        ("体中が痛い。", ()),
    ),
)
def test_dictionary_owned_quantity_ranges_only_cover_deterministic_expressions(
    text: str,
    expected_ranges: tuple[tuple[int, int], ...],
    default_jtalk: pyopenjtalk.OpenJTalk,
) -> None:
    """読みが確定した数量表現と、直後に続く接尾辞「後」(ゴ) だけを辞書所有範囲として検出する。"""

    _features, morphs = default_jtalk.run_mecab_detailed(text)

    assert (
        tsqyomi_inference._find_dictionary_owned_quantity_ranges(tuple(morphs)) == expected_ranges
    )


def test_dictionary_owned_quantity_ranges_accept_compound_nanji_morph() -> None:
    """辞書差分で「何時 + 間」へ分かれる場合も、時間量の全体を保護する。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    _features, hour_morphs = jtalk.run_mecab_detailed("一時間")
    compound_nanji_morph = cast(
        MeCabMorph,
        dict(next(morph for morph in hour_morphs if morph["surface"] == "一")),
    )
    compound_nanji_morph["surface"] = "何時"
    compound_nanji_morph["char_span"] = (0, 2)
    duration_morph = cast(
        MeCabMorph,
        dict(next(morph for morph in hour_morphs if morph["surface"] == "間")),
    )
    duration_morph["char_span"] = (2, 3)

    assert tsqyomi_inference._find_dictionary_owned_quantity_ranges(
        (compound_nanji_morph, duration_morph)
    ) == ((0, 3),)


@pytest.mark.parametrize(
    ("text", "surface", "allowed_readings", "selected_pronunciation"),
    (
        (
            "みづからを虐ぐる日は声に唱ふ乳房なき女の乾物はいかが？",
            "日",
            ("ヒ", "ニチ"),
            "ヒ",
        ),
        (
            "子宝に恵まれ、代々家が栄えるように",
            "家",
            ("イエ", "ウチ"),
            "イエ",
        ),
        ("家では猫を飼っています。", "家", ("イエ", "ウチ"), "ウチ"),
        ("主です。", "主", ("シュ", "ヌシ"), "ヌシ"),
    ),
)
def test_tsqyomi_changes_out_of_class_default_for_independent_reading(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    surface: str,
    allowed_readings: tuple[str, str],
    selected_pronunciation: str,
) -> None:
    """接尾辞の誤解析を含む独立語は、モデルが選んだ候補へ差し替える。"""

    def predict_independent_reading(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """候補外の辞書既定読みを直すモデル選択結果を返す。"""

        return (
            tsqyomi.ReadingPrediction(
                pronunciation=selected_pronunciation,
                scores=(1.0, 0.0),
            ),
        )

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={surface[0]: (surface,)},
            class_index_by_surface_and_pronunciation={
                surface: {
                    allowed_readings[0]: 0,
                    allowed_readings[1]: 1,
                },
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_independent_reading,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(text, jtalk)

    assert any(
        feature.split(",")[9] == selected_pronunciation
        for feature in features
        if feature.startswith(f"{surface},")
    )


def test_tsqyomi_preserves_inflection_pronunciation_within_same_reading_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    同じ語義クラスに属する活用発音は、形態素解析が確定した辞書既定値を維持する。
    """

    def reject_unnecessary_prediction(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """
        活用形だけが異なる対象をモデルへ渡した場合は失敗させる。
        """

        raise AssertionError("same-class inflection must not reach model inference")

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"来": ("来",)},
            class_index_by_surface_and_pronunciation={
                "来": {
                    "キ": 0,
                    "ク": 0,
                    "コ": 0,
                },
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=reject_unnecessary_prediction,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    tsqyomi_diagnostics.start_recording()
    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(
        "来ない場合は連絡してください。",
        jtalk,
    )
    diagnostics = tsqyomi_diagnostics.stop_recording()

    assert any(feature.split(",")[9] == "コ" for feature in features if feature.startswith("来,"))
    assert len(diagnostics) == 1
    assert diagnostics[0].outcome == "dictionary_default_protected"
    assert diagnostics[0].selected_pronunciation == "コ"
    assert diagnostics[0].was_preserved is True


@pytest.mark.parametrize(
    ("text", "expected_pronunciation"),
    (
        ("前代表者の後を継ぐ。", "アト"),
        ("後を追って歩き出す。", "アト"),
        ("選挙後を見越して準備する。", "ゴ"),
    ),
)
def test_tsqyomi_resolves_case_marked_independent_noun_from_morphology(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    expected_pronunciation: str,
) -> None:
    """
    格助詞を受ける独立名詞と、直前名詞へ付く接尾辞を形態素構造から分ける。
    """

    def predict_go(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """
        接尾用法の読みを常に選ぶモデル結果を返す。
        """

        return (tsqyomi.ReadingPrediction(pronunciation="ゴ", scores=(0.0, 0.0, 1.0, 0.0)),)

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"後": ("後",)},
            class_index_by_surface_and_pronunciation={
                "後": {
                    "アト": 0,
                    "ウシロ": 1,
                    "ゴ": 2,
                    "ノチ": 3,
                },
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_go,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(text, jtalk)

    assert any(
        feature.split(",")[9] == expected_pronunciation
        for feature in features
        if feature.startswith("後,")
    )


@pytest.mark.parametrize("text", ("二回表に逆転した。", "九回表を抑えた。"))
def test_tsqyomi_resolves_baseball_inning_half_from_morphology(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    """
    数詞と助数詞「回」に続く「表」は、野球のイニング前半を表す「オモテ」と読む。
    """

    def predict_hyou(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """
        表や図表を表す読みを常に選ぶモデル結果を返す。
        """

        return (tsqyomi.ReadingPrediction(pronunciation="ヒョー", scores=(0.0, 1.0)),)

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"表": ("表",)},
            class_index_by_surface_and_pronunciation={
                "表": {
                    "オモテ": 0,
                    "ヒョー": 1,
                },
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_hyou,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(text, jtalk)

    assert any(
        feature.split(",")[9] == "オモテ" for feature in features if feature.startswith("表,")
    )


@pytest.mark.parametrize("text", ("時間は十分にある。", "十分な量を用意する。"))
def test_tsqyomi_preserves_juubun_before_adjectival_continuation(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    """
    形容動詞として「に」「な」へ続く「十分」は、辞書が確定した「ジューブン」を維持する。
    """

    def predict_duration(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """
        時間量の読みを常に選ぶモデル結果を返す。
        """

        return (tsqyomi.ReadingPrediction(pronunciation="ジュップン", scores=(0.0, 1.0)),)

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"十": ("十分",)},
            class_index_by_surface_and_pronunciation={
                "十分": {
                    "ジューブン": 0,
                    "ジュップン": 1,
                },
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_duration,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(text, jtalk)

    assert any(
        feature.split(",")[9] == "ジューブン" for feature in features if feature.startswith("十分,")
    )


@pytest.mark.parametrize("text", ("復活の時を待つ。", "時あたかも春だった。"))
def test_tsqyomi_resolves_toki_from_fixed_morphology(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    """
    連体助詞の後ろと固定句「時あたかも」では、名詞の「トキ」を選ぶ。
    """

    def predict_ji(
        _text: str,
        _targets: tuple[tsqyomi.ReadingTarget, ...],
    ) -> tuple[tsqyomi.ReadingPrediction, ...]:
        """
        時刻を表す読みを常に選ぶモデル結果を返す。
        """

        return (tsqyomi.ReadingPrediction(pronunciation="ジ", scores=(0.0, 1.0)),)

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            surfaces_by_first_character={"時": ("時",)},
            class_index_by_surface_and_pronunciation={
                "時": {
                    "ジ": 0,
                    "トキ": 1,
                },
            },
            preserve_dictionary_default_pronunciations=(),
        ),
        predict=predict_ji,
    )
    monkeypatch.setattr(tsqyomi_inference, "get_loaded_model", lambda: model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    features, _morphs = tsqyomi_inference.select_mecab_features_with_tsqyomi(text, jtalk)

    assert any(feature.split(",")[9] == "トキ" for feature in features if feature.startswith("時,"))
