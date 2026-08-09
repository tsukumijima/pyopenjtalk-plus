"""tsqyomi の契約検証、診断、MeCab 候補解析、Execution Provider 選択を確認する。"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
import pyopenjtalk.tsqyomi.diagnostics as tsqyomi_diagnostics
import pyopenjtalk.tsqyomi.inference as tsqyomi_inference
import pyopenjtalk.tsqyomi.model as tsqyomi_model


def _minimal_v2_metadata_payload(**overrides: Any) -> dict[str, Any]:
    """テスト用の最小 v2 metadata 辞書を返す。"""

    payload: dict[str, Any] = {
        "schema_version": "v2",
        "model_max_length": 256,
        "output_class_order": ["rc_1", "rc_2"],
        "reading_class_ids_by_surface_and_pronunciation": {
            "人気": {"ニンキ": ["rc_1"], "ヒトケ": ["rc_2"]},
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


def test_onnx_contract_accepts_v2_model_shape() -> None:
    """読みクラス列を共有する v2 ONNX とメタデータの組を受理する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v2_metadata_payload())
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
    """旧 schema_version のメタデータを v2 契約へ誤接続しない。"""

    with pytest.raises(ValueError, match="schema_version"):
        tsqyomi.TsqyomiMetadata.model_validate(
            _minimal_v2_metadata_payload(schema_version="modernbert_reading_class_v2")
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
            _minimal_v2_metadata_payload(
                leading_token_id=leading_token_id,
                trailing_token_id=trailing_token_id,
            )
        )


def test_onnx_contract_rejects_wrong_target_mask_type() -> None:
    """対象マスクが bool でない旧世代または破損した ONNX を拒否する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v2_metadata_payload())
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="target_mask", type="tensor(int64)"),
        ],
        get_outputs=lambda: cast(list[Any], []),
    )

    with pytest.raises(ValueError, match="inputs do not match the v2 contract"):
        tsqyomi.TsqyomiModel.validate_onnx_contract(session, metadata)


def test_onnx_contract_rejects_different_reading_class_count() -> None:
    """メタデータと異なる読みクラス数の ONNX をモデル初期化前に拒否する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v2_metadata_payload())
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

    with pytest.raises(ValueError, match="output class count"):
        tsqyomi.TsqyomiModel.validate_onnx_contract(session, metadata)


def test_onnx_contract_rejects_wrong_output_rank() -> None:
    """読みクラス出力が3次元でない ONNX をモデル初期化前に拒否する。"""

    metadata = tsqyomi.TsqyomiMetadata.model_validate(_minimal_v2_metadata_payload())
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
