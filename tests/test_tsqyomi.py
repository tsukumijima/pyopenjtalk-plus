"""tsqyomi のモデル管理、読み推論、MeCab feature 差し替えを確認する。"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from pyopenjtalk.tsqyomi.inference import select_mecab_features_with_tsqyomi
from pyopenjtalk.tsqyomi.types import CandidateNode, CandidatePath, ReadingAnalysis
from pyopenjtalk.types import MeCabMorph


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


class _FakeMetadata:
    """`surfaces_by_first_character` を持つテスト用メタデータ。"""

    def __init__(
        self,
        scored_surfaces: frozenset[str],
        reading_class_ids_by_surface_and_pronunciation: dict[str, dict[str, tuple[str, ...]]],
    ) -> None:
        """
        推論対象表層と読みクラスのテスト用メタデータを構築する。

        Args:
            scored_surfaces (frozenset[str]): 推論対象表層の集合
            reading_class_ids_by_surface_and_pronunciation (dict[str, dict[str, tuple[str, ...]]]):
                表層と発音に対応する読みクラス ID
        """

        self.model_scored_surfaces = scored_surfaces
        self.reading_class_ids_by_surface_and_pronunciation = (
            reading_class_ids_by_surface_and_pronunciation
        )
        self.preserve_dictionary_default_pronunciations: tuple[tuple[str, str], ...] = ()
        self._surfaces_by_first_character = (
            tsqyomi.TsqyomiMetadata._index_surfaces_by_first_character(scored_surfaces)
        )

    @property
    def surfaces_by_first_character(self) -> dict[str, tuple[str, ...]]:
        """
        先頭文字ごとの推論対象表層を返す。

        Returns:
            dict[str, tuple[str, ...]]: 先頭文字をキーとする表層の索引
        """

        return self._surfaces_by_first_character


def test_diagnostics_rebase_recording_char_spans() -> None:
    """
    分割片で記録した診断位置を元の本文位置へ戻す。
    """

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


class _FakeModel:
    """
    モデル管理と MeCab feature 差し替えを ONNX Runtime から分離して検証するためのテスト用スタブ。
    """

    def __init__(self) -> None:
        """推論対象表層と推論呼び出し履歴を初期化する。"""

        scored_surfaces = frozenset({"人気"})
        self.metadata = _FakeMetadata(
            scored_surfaces,
            {
                "人気": {"ニンキ": ("rc_1",), "ヒトケ": ("rc_2",)},
            },
        )
        self.predict_calls: list[tuple[str, tuple[int, int], tuple[str, ...]]] = []

    def predict(self, text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
        """本文1回の呼び出しを記録し、各対象でヒトケを選ぶ。"""

        predictions = []
        for target in targets:
            self.predict_calls.append((text, target.char_span, target.pronunciations))
            predictions.append(
                tsqyomi.ReadingPrediction(
                    pronunciation="ヒトケ",
                    scores=tuple(float(index) for index in range(len(target.pronunciations))),
                )
            )
        return tuple(predictions)


def test_load_model_is_idempotent_when_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ロード済みのモデルを繰り返し取得しない。"""

    model_module = tsqyomi_model
    fake_model = _FakeModel()
    monkeypatch.setattr(model_module, "_loaded_model", fake_model)
    tsqyomi.load_model(["CPUExecutionProvider"], "/cache")
    assert model_module._loaded_model is fake_model


def test_load_model_passes_each_downloaded_asset_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """個別に取得した3ファイルの実パスをモデル構築へ渡す。"""

    import huggingface_hub

    model_module = tsqyomi_model
    downloaded_paths = {
        "v2/model.onnx": Path("/cache/model/model.onnx"),
        "v2/tokenizer.json": Path("/cache/tokenizer/tokenizer.json"),
        "v2/metadata.json": Path("/cache/metadata/metadata.json"),
    }
    loaded_paths: tuple[Path, Path, Path] | None = None
    loaded_allow_provider_fallback: bool | None = None
    fake_model = _FakeModel()

    def fake_download(*, filename: str, **_kwargs: Any) -> str:
        """ファイルごとに異なるディレクトリのパスを返す。"""

        return str(downloaded_paths[filename])

    def fake_load(
        model_path: Path,
        tokenizer_path: Path,
        metadata_path: Path,
        _onnx_providers: Sequence[Any] | None,
        allow_provider_fallback: bool,
    ) -> _FakeModel:
        """モデル構築へ渡された3ファイルのパスとフォールバック設定を記録する。"""

        nonlocal loaded_paths, loaded_allow_provider_fallback
        loaded_paths = (model_path, tokenizer_path, metadata_path)
        loaded_allow_provider_fallback = allow_provider_fallback
        return fake_model

    monkeypatch.setattr(model_module, "_loaded_model", None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(model_module, "_load_model_from_paths", fake_load)

    tsqyomi.load_model(["CPUExecutionProvider"], "/cache")

    assert loaded_paths == (
        downloaded_paths["v2/model.onnx"],
        downloaded_paths["v2/tokenizer.json"],
        downloaded_paths["v2/metadata.json"],
    )
    assert loaded_allow_provider_fallback is True
    assert model_module._loaded_model is fake_model


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


def test_model_tokenizes_all_targets_at_mecab_boundaries() -> None:
    """同一文の対象を共有し、直後の助詞を対象部分語へ混ぜない。"""

    class FakeTokenizer:
        """入力片を1トークンにして境界分割を観測するトークナイザー。"""

        def __init__(self) -> None:
            """入力片とトークン ID の対応を初期化する。"""

            self.encoded_segments: list[str] = []
            self.token_id_by_segment: dict[str, int] = {}

        def encode(self, text: str, *, add_special_tokens: bool = True) -> SimpleNamespace:
            """空文字には特殊トークン、通常入力には1つの内容トークンを返す。"""

            if text == "" and add_special_tokens is True:
                return SimpleNamespace(
                    ids=[1, 2],
                    offsets=[(0, 0), (0, 0)],
                    special_tokens_mask=[1, 1],
                )
            self.encoded_segments.append(text)
            token_id = self.token_id_by_segment.setdefault(text, len(self.token_id_by_segment) + 10)
            return SimpleNamespace(
                ids=[token_id],
                offsets=[(0, len(text))],
                special_tokens_mask=[0],
            )

    class FakeSession:
        """モデル入力を保存し、2対象で異なる読みを選ぶ ONNX セッション。"""

        def __init__(self) -> None:
            """最後に受け取ったモデル入力を未設定で初期化する。"""

            self.model_inputs: dict[str, np.ndarray] | None = None

        def run(
            self, _output_names: list[str], model_inputs: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            """対象数に合わせた固定ロジットを返す。"""

            self.model_inputs = model_inputs
            return [np.asarray([[[2.0, 1.0], [1.0, 2.0]]], dtype=np.float32)]

        @staticmethod
        def get_providers() -> list[str]:
            """CPU 利用中の ONNX セッション相当の EP 列を返す。"""

            return ["CPUExecutionProvider"]

    metadata = tsqyomi.TsqyomiMetadata.model_validate(
        _minimal_v2_metadata_payload(
            model_max_length=64,
            reading_class_ids_by_surface_and_pronunciation={
                "最中": {"サイチュー": ["rc_1"], "モナカ": ["rc_2"]},
            },
        )
    )
    tokenizer = FakeTokenizer()
    session = FakeSession()
    model = tsqyomi.TsqyomiModel(tokenizer, session, metadata)
    text = "仕事の最中に最中を食べる。"
    first_start = text.index("最中")
    second_start = text.rindex("最中")
    targets = (
        tsqyomi.ReadingTarget(
            char_span=(first_start, first_start + len("最中")),
            surface="最中",
            pronunciations=("サイチュー", "モナカ"),
        ),
        tsqyomi.ReadingTarget(
            char_span=(second_start, second_start + len("最中")),
            surface="最中",
            pronunciations=("サイチュー", "モナカ"),
        ),
    )

    predictions = model.predict(text, targets)

    assert tokenizer.encoded_segments == ["仕事の", "最中", "に", "最中", "を食べる。"]
    assert tuple(prediction.pronunciation for prediction in predictions) == ("サイチュー", "モナカ")
    assert session.model_inputs is not None
    target_mask = session.model_inputs["target_mask"]
    assert target_mask.shape == (1, 2, 7)
    assert target_mask.sum(axis=2).tolist() == [[1, 1]]
    assert np.flatnonzero(target_mask[0, 0]).tolist() == [2]
    assert np.flatnonzero(target_mask[0, 1]).tolist() == [4]

    unknown_surface_targets = (
        tsqyomi.ReadingTarget(
            char_span=(0, len("仕事")),
            surface="仕事",
            pronunciations=("シゴト",),
        ),
        targets[1],
    )
    with pytest.raises(ValueError, match="surface: 仕事"):
        model.predict(text, unknown_surface_targets)

    unknown_pronunciation_targets = (
        tsqyomi.ReadingTarget(
            char_span=targets[0].char_span,
            surface="最中",
            pronunciations=("サナカ",),
        ),
        targets[1],
    )
    with pytest.raises(ValueError, match="最中/サナカ"):
        model.predict(text, unknown_pronunciation_targets)


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

    # 要求先頭 (タプル形式の設定つき指定を含む) が有効ならそのまま通る
    tsqyomi_model._verify_session_providers(
        CUDASession,
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        False,
    )
    # 既定の allow_provider_fallback=True 相当で CPU フォールバックを受け入れる
    tsqyomi_model._verify_session_providers(
        CPUFallbackSession,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        True,
    )


def test_explicitly_disabled_tsqyomi_preserves_all_high_level_api_results() -> None:
    """全高レベル API で tsqyomi の省略時と明示無効時の結果が一致することを確認する。"""

    text = "人気の店です。"

    # 解析結果を直接返す5つの API は、引数省略時と明示的な `False` を値で比較する
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

    # 音声波形は `ndarray` なので、サンプリング周波数と全サンプルを分けて比較する
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


def test_unload_model_clears_loaded_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """unload_model() はロード済みフラグを落とす。"""

    model_module = tsqyomi_model
    monkeypatch.setattr(model_module, "_loaded_model", _FakeModel())
    assert tsqyomi.is_model_loaded() is True
    tsqyomi.unload_model()
    assert tsqyomi.is_model_loaded() is False


def test_target_free_frontend_skips_detailed_morphology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """形態素対応を返さない対象なし本文では詳細形態素列の構築を省く。"""

    class FeatureOnlyOpenJTalk:
        """通常特徴列の取得だけを許す OpenJTalk の代用品。"""

        @staticmethod
        def normalize_for_mecab(text: str) -> str:
            """入力を変更せずに返す。"""

            return text

        @staticmethod
        def run_mecab(_text: str) -> list[str]:
            """対象なし高速処理の番兵 feature 列を返す。"""

            return ["名詞,一般,*,*,*,*,天気,テンキ,テンキ,1/3,*"]

        @staticmethod
        def run_mecab_detailed(_text: str) -> list[Any]:
            """不要な詳細処理へ到達した場合に試験を失敗させる。"""

            raise AssertionError("target-free feature-only inference must skip detailed morphology")

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", _FakeModel())
    stub_jtalk: Any = FeatureOnlyOpenJTalk()
    features, morphs = select_mecab_features_with_tsqyomi(
        "明日の天気です。",
        stub_jtalk,
        include_morphs=False,
    )

    assert features == ["名詞,一般,*,*,*,*,天気,テンキ,テンキ,1/3,*"]
    assert morphs == []


def test_single_reachable_reading_skips_model_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候補グラフに1読みしかない対象ではモデルを呼ばない。"""

    class Model(_FakeModel):
        """呼び出された時点で試験を失敗させるテスト用スタブ。"""

        def predict(self, text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """不要なモデル推論へ到達したことを通知する。"""

            raise AssertionError("single reachable reading must skip model inference")

    class SingleReadingOpenJTalk:
        """メタデータ上は2読みでも解析時には1読みだけ到達可能な候補解析を返す。"""

        @staticmethod
        def normalize_for_mecab(text: str) -> str:
            """入力を変更せずに返す。"""

            return text

        @staticmethod
        def analyze_mecab_candidates(
            text: str,
            _target_spans: tuple[tuple[int, int], ...],
        ) -> ReadingAnalysis:
            """ニンキだけを実現できる候補グラフを返す。"""

            morph: Any = {
                "surface": "人気",
                "features": ["名詞"] * 9 + ["ニンキ"],
                "char_span": (0, 2),
                "is_ignored": False,
            }
            node = CandidateNode(
                node_id=1,
                surface="人気",
                feature="名詞,一般,*,*,*,*,人気,ニンキ,ニンキ,0/3,*",
                pronunciation="ニンキ",
                char_span=(0, 2),
                pos_id=38,
                left_id=1,
                right_id=1,
                word_cost=1,
                dictionary_index=0,
                is_unknown=False,
                is_ignored=False,
                is_reading_protected=False,
            )
            path = CandidatePath(
                path_id=1,
                node_ids=(1,),
                char_span=(0, 2),
                surface="人気",
                pronunciation="ニンキ",
                features=(node["feature"],),
                left_boundary_cost=1,
                right_boundary_cost=1,
                right_link_cost=2,
                boundary_cost=2,
            )
            return ReadingAnalysis(
                normalized_text=text,
                features=(node["feature"],),
                morphs=(morph,),
                feature_index_by_morph=(0,),
                nodes=(node,),
                paths=(path,),
                connections=(),
            )

        @staticmethod
        def run_njd_from_mecab(_features: list[str]) -> list[Any]:
            """既定特徴列を維持したことが分かる番兵結果を返す。"""

            return [{"pron": "ニンキ"}]

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())

    stub_jtalk: Any = SingleReadingOpenJTalk()
    features, morphs = pyopenjtalk._run_frontend_with_tsqyomi(
        "人気",
        jtalk=stub_jtalk,
    )

    assert features == [{"pron": "ニンキ"}]
    assert morphs[0]["surface"] == "人気"


def test_long_text_analyzes_only_sentence_containing_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """長い前置きに対象がなければ対象を含む末尾文だけをモデルへ渡す。"""

    fake_model = _FakeModel()
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", fake_model)
    prefix = "これはひらがなだけのぶんしょうです。" * 50
    target_sentence = "人気のない店"
    text = prefix + target_sentence

    features, morphs = pyopenjtalk.run_frontend_detailed(
        text,
        use_tsqyomi=True,
        use_vanilla=True,
    )

    assert (
        next(feature for feature in features if feature["string"] == "人気")["pron"].replace(
            "’", ""
        )
        == "ヒトケ"
    )
    target_morph = next(morph for morph in morphs if morph["surface"] == "人気")
    assert target_morph["char_span"] == (len(prefix), len(prefix) + 2)
    assert fake_model.predict_calls == [(target_sentence, (0, 2), ("ヒトケ", "ニンキ"))]


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
    """括弧内の句点を保ち、閉じ括弧直後の句点で対象文を切る。"""

    text = "前置きです。（仕事の最中だ。）後続です。"
    target_start = text.index("最中")

    segments = tsqyomi_inference._split_target_processing_segments(
        text,
        ((target_start, target_start + len("最中")),),
    )

    assert tuple(text[start:end] for start, end in segments) == (
        "前置きです。",
        "（仕事の最中だ。）後続です。",
    )


def test_sentence_segmentation_does_not_extend_unmatched_quote() -> None:
    """閉じられていない引用符の後ろも通常の句点で分割する。"""

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


def test_enabled_tsqyomi_replaces_feature_without_viterbi_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """モデルが選んだ発音の辞書特徴列を固定した形態素範囲へ交換する。"""

    class PreferHitokeModel(_FakeModel):
        """常に「ヒトケ」を選ぶ製品推論の代用品。"""

    fake_model = PreferHitokeModel()
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", fake_model)

    features, morphs = pyopenjtalk.run_frontend_detailed(
        "人気のない店",
        use_tsqyomi=True,
        use_vanilla=True,
    )

    assert features[0]["pron"].replace("’", "") == "ヒトケ"
    assert morphs[0]["surface"] == "人気"
    assert morphs[0]["features"][9] == "ヒトケ"
    assert fake_model.predict_calls == [("人気のない店", (0, 2), ("ヒトケ", "ニンキ"))]


def test_enabled_tsqyomi_replaces_different_surfaces_in_one_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じ文にある異なる2表層を1回のモデル推論で個別に読み分ける。"""

    class PreferContextualReadingsModel(_FakeModel):
        """表層ごとに指定した読みを選ぶ製品推論の代用品。"""

        def __init__(self) -> None:
            """2表層の読みクラスと推論呼び出し履歴を初期化する。"""

            super().__init__()
            scored_surfaces = frozenset({"人気", "最中"})
            self.metadata.model_scored_surfaces = scored_surfaces
            self.metadata.reading_class_ids_by_surface_and_pronunciation = {
                "人気": {"ニンキ": ("rc_1",), "ヒトケ": ("rc_2",)},
                "最中": {"サイチュー": ("rc_3",), "モナカ": ("rc_4",)},
            }
            self.metadata._surfaces_by_first_character = (
                tsqyomi.TsqyomiMetadata._index_surfaces_by_first_character(scored_surfaces)
            )

        def predict(self, text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """共有本文の対象を記録し、表層ごとの文脈読みを返す。"""

            selected_readings = {"人気": "ヒトケ", "最中": "モナカ"}
            predictions = []
            for target in targets:
                # 2表層が別々の推論呼び出しへ分かれていないことを predict_calls から検査する
                self.predict_calls.append((text, target.char_span, target.pronunciations))
                predictions.append(
                    tsqyomi.ReadingPrediction(
                        pronunciation=selected_readings[target.surface],
                        scores=tuple(float(index) for index in range(len(target.pronunciations))),
                    )
                )
            return tuple(predictions)

    fake_model = PreferContextualReadingsModel()
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", fake_model)
    text = "誰もいないはずの茶室に人気を感じたが、座卓には最中が置かれていた。"

    tsqyomi_diagnostics.start_recording()
    try:
        features, morphs = pyopenjtalk.run_frontend_detailed(
            text,
            use_tsqyomi=True,
            use_vanilla=True,
        )
    finally:
        diagnostics = tsqyomi_diagnostics.stop_recording()

    pronunciations = {
        feature["string"]: feature["pron"].replace("’", "")
        for feature in features
        if feature["string"] in {"人気", "最中"}
    }
    assert pronunciations == {"人気": "ヒトケ", "最中": "モナカ"}
    assert [morph["surface"] for morph in morphs if morph["surface"] in {"人気", "最中"}] == [
        "人気",
        "最中",
    ]
    assert len(fake_model.predict_calls) == 2
    assert {call[0] for call in fake_model.predict_calls} == {text}
    assert {(item.surface, item.score_margin) for item in diagnostics} == {
        ("人気", 1.0),
        ("最中", 1.0),
    }


@pytest.mark.parametrize(
    (
        "text",
        "surface",
        "selected_pronunciation",
        "baseline_morph_count",
        "expected_accent_nucleus",
        "expected_mora_count",
    ),
    [
        ("十分です", "十分", "ジップン", 1, 1, 4),
        ("十八番です", "十八番", "オハコ", 3, 4, 3),
        ("何人いますか", "何人", "ナンニン", 2, 0, 4),
    ],
)
def test_v2_replaces_exact_morph_range_with_one_dictionary_node(
    text: str,
    surface: str,
    selected_pronunciation: str,
    baseline_morph_count: int,
    expected_accent_nucleus: int,
    expected_mora_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """限定した3表層では1辞書ノードへ交換して発音とアクセントを維持する。"""

    class Model:
        """指定した発音を返す本文共有型のテスト用スタブ。"""

        metadata = _FakeMetadata(frozenset({surface}), {})

        @staticmethod
        def predict(_text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """到達可能な候補からテスト指定の発音を選ぶ。"""

            assert len(targets) == 1
            assert selected_pronunciation in targets[0].pronunciations
            return (
                tsqyomi.ReadingPrediction(
                    pronunciation=selected_pronunciation,
                    scores=tuple(0.0 for _ in targets[0].pronunciations),
                ),
            )

    # 実行時の辞書で到達可能な候補を使い、メタデータと候補グラフの読み候補を一致させる
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    _, baseline_morphs = jtalk.run_mecab_detailed(text)
    analysis = jtalk.analyze_mecab_candidates(text, ((0, len(surface)),))
    pronunciations = tuple(dict.fromkeys(path["pronunciation"] for path in analysis["paths"]))
    if selected_pronunciation not in pronunciations:
        pytest.skip(
            f"selected pronunciation {selected_pronunciation!r} is unavailable; "
            f"candidates: {list(pronunciations)}"
        )
    Model.metadata.reading_class_ids_by_surface_and_pronunciation[surface] = {
        pronunciation: (f"rc_{index}",) for index, pronunciation in enumerate(pronunciations)
    }
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())

    features, morphs = pyopenjtalk.run_frontend_detailed(
        text,
        use_tsqyomi=True,
        use_vanilla=True,
    )

    assert (
        sum(morph["char_span"][1] <= len(surface) for morph in baseline_morphs)
        == baseline_morph_count
    )
    assert morphs[0]["surface"] == surface
    assert morphs[0]["features"][9] == selected_pronunciation
    target_feature = next(feature for feature in features if feature["string"] == surface)
    assert target_feature["pron"] == selected_pronunciation
    assert target_feature["acc"] == expected_accent_nucleus
    assert target_feature["mora_size"] == expected_mora_count
    assert target_feature["chain_rule"] == "C2"
    assert target_feature["chain_flag"] == -1


@pytest.mark.parametrize(
    (
        "text",
        "surface",
        "selected_pronunciation",
        "expected_original",
        "expected_conjugation_type",
        "expected_accent_nucleus",
        "expected_chain_flag",
    ),
    [
        ("会議を行った", "行っ", "オコナッ", "行う", "五段・ワ行促音便", 5, 0),
        ("駅へ行った", "行っ", "イッ", "行く", "五段・カ行促音便", 3, 0),
        ("学校に通っている", "通っ", "カヨッ", "通う", "五段・ワ行促音便", 6, 0),
        ("門を通って入る", "通っ", "トーッ", "通る", "五段・ラ行", 1, 0),
        ("この通りで待つ", "通り", "トーリ", "通り", "*", 3, 0),
        ("予想通りで驚いた", "通り", "ドーリ", "通り", "*", 3, 1),
    ],
)
def test_v2_replaces_inflected_meaning_node_with_complete_dictionary_features(
    text: str,
    surface: str,
    selected_pronunciation: str,
    expected_original: str,
    expected_conjugation_type: str,
    expected_accent_nucleus: int,
    expected_chain_flag: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """活用語の意味中心ノードを選び、品詞・活用・アクセントまで候補側へ交換する。"""

    class Model:
        """指定した発音を選ぶ本文共有型のテスト用スタブ。"""

        metadata = _FakeMetadata(frozenset({surface}), {})

        @staticmethod
        def predict(_text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """試験対象の発音を候補内から選ぶ。"""

            assert len(targets) == 1
            assert selected_pronunciation in targets[0].pronunciations
            return (
                tsqyomi.ReadingPrediction(
                    pronunciation=selected_pronunciation,
                    scores=tuple(0.0 for _ in targets[0].pronunciations),
                ),
            )

    # 実辞書の到達可能読みからメタデータを作り、候補供給と差し替えの両方を同じ条件で検査する
    target_start = text.index(surface)
    target_span = (target_start, target_start + len(surface))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    pronunciations = tuple(
        dict.fromkeys(
            path["pronunciation"] for path in analysis["paths"] if path["char_span"] == target_span
        )
    )
    Model.metadata.reading_class_ids_by_surface_and_pronunciation[surface] = {
        pronunciation: (f"rc_{index}",) for index, pronunciation in enumerate(pronunciations)
    }
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())

    features, morphs = pyopenjtalk.run_frontend_detailed(
        text,
        use_tsqyomi=True,
        use_vanilla=True,
    )

    target_feature = next(feature for feature in features if feature["string"] == surface)
    target_morph = next(morph for morph in morphs if morph["char_span"] == target_span)
    assert target_morph["features"][9] == selected_pronunciation
    assert target_feature["pron"] == selected_pronunciation
    assert target_feature["orig"] == expected_original
    assert target_feature["ctype"] == expected_conjugation_type
    assert target_feature["cform"] == ("*" if expected_conjugation_type == "*" else "連用タ接続")
    assert target_feature["acc"] == expected_accent_nucleus
    assert target_feature["chain_flag"] == expected_chain_flag


def test_high_level_dictionary_protection_reaches_tsqyomi_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """高レベル辞書指定の保護フラグを Lattice 候補まで渡してモデル推論を止める。"""

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

    fake_model = _FakeModel()
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", fake_model)
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
        pyopenjtalk.g2p("人気の店です。", kana=True, use_tsqyomi=True)
        assert fake_model.predict_calls == []
    finally:
        pyopenjtalk.unset_user_dict()


def test_analyze_mecab_candidates_expands_symbol_morphs_like_detailed() -> None:
    """候補解析の最良経路 morphs が run_mecab_detailed と同じ記号分割を返すことを確認。"""

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


def test_symbol_expansion_keeps_following_feature_replacement_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """連続記号を形態素へ展開しても後続対象の MeCab feature を正しい位置で差し替える。"""

    fake_model = _FakeModel()
    monkeypatch.setattr(tsqyomi_model, "_loaded_model", fake_model)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    features, _morphs = select_mecab_features_with_tsqyomi(
        "÷÷÷÷人気",
        jtalk,
        include_morphs=False,
    )

    assert any("ヒトケ" in feature for feature in features)
    assert all("ニンキ" not in feature for feature in features)


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


def test_select_mecab_features_without_targets_uses_single_mecab_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """対象表層がない本文では MeCab 詳細解析を1回だけ呼ぶことを確認。"""

    inner = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)

    class SinglePassOpenJTalk:
        """`run_mecab_detailed()` の呼び出し回数だけを数えるラッパ。"""

        detailed_calls = 0

        def normalize_for_mecab(self, text: str) -> str:
            """内部 OpenJTalk と同じ MeCab 正規化結果を返す。"""

            return inner.normalize_for_mecab(text)

        def run_mecab_detailed(self, text: str | bytes | bytearray) -> tuple[list[str], list[Any]]:
            """MeCab 詳細解析の呼び出し回数を記録する。"""

            SinglePassOpenJTalk.detailed_calls += 1
            return inner.run_mecab_detailed(text)

    class Model:
        """対象表層を持たない本文向けのテスト用スタブ。"""

        metadata = _FakeMetadata(frozenset({"人気"}), {})

        @staticmethod
        def predict(_text: str, _targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """対象なし本文で推論が呼ばれた場合は失敗させる。"""

            raise AssertionError("targets must be empty")

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())

    stub_jtalk: Any = SinglePassOpenJTalk()
    features, morphs = select_mecab_features_with_tsqyomi(
        "東京は日本の首都です。",
        stub_jtalk,
    )

    assert SinglePassOpenJTalk.detailed_calls == 1
    assert len(features) >= 1
    assert len(morphs) >= 1


def test_selected_morphs_use_actual_lattice_boundary_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """文脈 ID が異なる候補でも Lattice の両境界コストを詳細形態素へ反映する。"""

    text = "一寸です"
    surface = "一寸"
    selected_pronunciation = "イッスン"
    target_span = (0, len(surface))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))
    selected_path = next(
        path
        for path in analysis["paths"]
        if path["char_span"] == target_span and path["pronunciation"] == selected_pronunciation
    )

    class Model:
        """文脈 ID が既定経路と異なるイッスンを選ぶテスト用スタブ。"""

        metadata = _FakeMetadata(
            frozenset({surface}),
            {
                surface: {
                    path["pronunciation"]: (f"rc_{index}",)
                    for index, path in enumerate(analysis["paths"])
                    if path["char_span"] == target_span
                },
            },
        )

        @staticmethod
        def predict(_text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """接続行列差が生じるイッスンを選ぶ。"""

            assert selected_pronunciation in targets[0].pronunciations
            return (
                tsqyomi.ReadingPrediction(
                    pronunciation=selected_pronunciation,
                    scores=tuple(0.0 for _ in targets[0].pronunciations),
                ),
            )

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())
    _, morphs = select_mecab_features_with_tsqyomi(text, jtalk)

    assert morphs[0]["features"][9] == selected_pronunciation
    assert morphs[0]["link_cost"] == selected_path["left_boundary_cost"]
    assert morphs[1]["link_cost"] == selected_path["right_link_cost"]
    assert [morph["node_cost"] for morph in morphs] == [
        sum(previous["link_cost"] for previous in morphs[: index + 1])
        for index in range(len(morphs))
    ]


def test_adjacent_selected_morphs_use_candidate_connection_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隣接する2対象では後側候補の局所コストに候補間接続辺を使う。"""

    text = "人気最中です"
    surfaces = ("人気", "最中")
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
    # 実辞書で接続可能な読みの組を選び、選択処理と再構築処理に同じ辺を通す
    compatible_pair = next(
        (
            (left_path, right_path)
            for left_path in paths_by_span[target_spans[0]]
            for right_path in paths_by_span[target_spans[1]]
            if (left_path["node_ids"][-1], right_path["node_ids"][0]) in connection_costs
        ),
        None,
    )
    if compatible_pair is None:
        pytest.skip(f"dictionary has no connection between target spans: {target_spans}")
    selected_left_path, selected_right_path = compatible_pair
    selected_readings = {
        surfaces[0]: selected_left_path["pronunciation"],
        surfaces[1]: selected_right_path["pronunciation"],
    }

    class Model:
        """隣接する2表層で接続可能な読みを選ぶテスト用スタブ。"""

        metadata = _FakeMetadata(
            frozenset(surfaces),
            {
                surface: {
                    path["pronunciation"]: (f"rc_{index}",)
                    for index, path in enumerate(paths_by_span[target_span])
                }
                for surface, target_span in zip(surfaces, target_spans, strict=True)
            },
        )

        @staticmethod
        def predict(_text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """表層ごとに選んだ接続可能な読みを返す。"""

            return tuple(
                tsqyomi.ReadingPrediction(
                    pronunciation=selected_readings[target.surface],
                    scores=tuple(0.0 for _ in target.pronunciations),
                )
                for target in targets
            )

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())
    _, morphs = select_mecab_features_with_tsqyomi(text, jtalk)

    assert (
        morphs[1]["link_cost"]
        == connection_costs[
            (selected_left_path["node_ids"][-1], selected_right_path["node_ids"][0])
        ]
    )


def test_tsqyomi_include_morphs_false_skips_morph_rebuild_with_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """対象ありでも include_morphs=False なら形態素コスト再構築を省略する。"""

    replace_calls = 0
    inference_module = tsqyomi_inference
    original_replace = inference_module._replace_morph

    def counting_replace(*args: Any, **kwargs: Any) -> MeCabMorph:
        """
        形態素差し替えの呼び出し回数を記録する。

        Args:
            *args (Any): `_replace_morph()` へ渡す位置引数
            **kwargs (Any): `_replace_morph()` へ渡すキーワード引数

        Returns:
            MeCabMorph: 元の `_replace_morph()` が返した詳細形態素
        """

        nonlocal replace_calls
        replace_calls += 1
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(inference_module, "_replace_morph", counting_replace)

    text = "一寸です"
    surface = "一寸"
    selected_pronunciation = "イッスン"
    target_span = (0, len(surface))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, (target_span,))

    class Model:
        """イッスンを選ぶテスト用スタブ。"""

        metadata = _FakeMetadata(
            frozenset({surface}),
            {
                surface: {
                    path["pronunciation"]: (f"rc_{index}",)
                    for index, path in enumerate(analysis["paths"])
                    if path["char_span"] == target_span
                },
            },
        )

        @staticmethod
        def predict(_text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            assert selected_pronunciation in targets[0].pronunciations
            return (
                tsqyomi.ReadingPrediction(
                    pronunciation=selected_pronunciation,
                    scores=tuple(0.0 for _ in targets[0].pronunciations),
                ),
            )

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", Model())
    features, morphs = select_mecab_features_with_tsqyomi(
        text,
        jtalk,
        include_morphs=False,
    )

    assert morphs == []
    assert replace_calls == 0
    assert any(selected_pronunciation in feature for feature in features)


def test_preserve_dictionary_default_keeps_suffix_joe_when_model_picks_ue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """教師 0 件の接尾辞 上=ジョー では、モデルが ウエ を選んでも辞書既定を維持する。"""

    class PreserveModel(_FakeModel):
        """上 だけを ウエ と誤選択するテスト用スタブ。"""

        def __init__(self) -> None:
            """上 の辞書既定発音を維持するメタデータを構築する。"""

            super().__init__()
            self.metadata = _FakeMetadata(
                frozenset({"上"}),
                {
                    "上": {
                        "ウエ": ("rc_ue",),
                        "ジョー": ("rc_joe",),
                    },
                },
            )
            self.metadata.preserve_dictionary_default_pronunciations = (("上", "ジョー"),)

        def predict(self, text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """
            全対象で辞書既定と異なる ウエ を返す。

            Args:
                text (str): 推論対象本文
                targets (tuple[Any, ...]): 読み選択対象

            Returns:
                tuple[Any, ...]: ウエ を選んだ予測列
            """

            return tuple(
                tsqyomi.ReadingPrediction(
                    pronunciation="ウエ",
                    scores=(0.1, 0.9),
                )
                for _target in targets
            )

    class SuffixJoeOpenJTalk:
        """商売上 で辞書既定 ジョー、候補 ウエ/ジョー の解析を返す。"""

        @staticmethod
        def normalize_for_mecab(text: str) -> str:
            """
            入力を変更せずに返す。

            Args:
                text (str): 入力本文

            Returns:
                str: 入力と同じ本文
            """

            return text

        @staticmethod
        def analyze_mecab_candidates(
            text: str,
            _target_spans: tuple[tuple[int, int], ...],
        ) -> ReadingAnalysis:
            """
            接尾辞 上 の2候補を持つ解析結果を返す。

            Args:
                text (str): 解析対象本文
                _target_spans (tuple[tuple[int, int], ...]): 読み選択対象の範囲

            Returns:
                ReadingAnalysis: ジョーとウエを候補に持つ解析結果
            """

            first_morph: Any = {
                "surface": "商売",
                "features": [
                    "商売",
                    "名詞",
                    "サ変接続",
                    "*",
                    "*",
                    "*",
                    "*",
                    "商売",
                    "ショウバイ",
                    "ショーバイ",
                    "1/4",
                    "C1",
                ],
                "char_span": (0, 2),
                "is_ignored": False,
            }
            second_morph: Any = {
                "surface": "上",
                "features": [
                    "上",
                    "名詞",
                    "接尾",
                    "副詞可能",
                    "*",
                    "*",
                    "*",
                    "上",
                    "ジョウ",
                    "ジョー",
                    "0/2",
                    "C4",
                ],
                "char_span": (2, 3),
                "is_ignored": False,
            }
            morphs = (first_morph, second_morph)
            joe_node = CandidateNode(
                node_id=2,
                surface="上",
                feature=",".join(morphs[1]["features"]),
                pronunciation="ジョー",
                char_span=(2, 3),
                pos_id=38,
                left_id=1,
                right_id=1,
                word_cost=1,
                dictionary_index=0,
                is_unknown=False,
                is_ignored=False,
                is_reading_protected=False,
            )
            ue_node = CandidateNode(
                node_id=3,
                surface="上",
                feature="上,名詞,一般,*,*,*,*,上,ウエ,ウエ,1/2,C4",
                pronunciation="ウエ",
                char_span=(2, 3),
                pos_id=38,
                left_id=1,
                right_id=1,
                word_cost=5000,
                dictionary_index=0,
                is_unknown=False,
                is_ignored=False,
                is_reading_protected=False,
            )
            joe_path = CandidatePath(
                path_id=1,
                node_ids=(2,),
                char_span=(2, 3),
                surface="上",
                pronunciation="ジョー",
                features=(joe_node["feature"],),
                left_boundary_cost=1,
                right_boundary_cost=1,
                right_link_cost=2,
                boundary_cost=2,
            )
            ue_path = CandidatePath(
                path_id=2,
                node_ids=(3,),
                char_span=(2, 3),
                surface="上",
                pronunciation="ウエ",
                features=(ue_node["feature"],),
                left_boundary_cost=1,
                right_boundary_cost=1,
                right_link_cost=2,
                boundary_cost=2,
            )
            return ReadingAnalysis(
                normalized_text=text,
                features=(morphs[0]["features"][0], joe_node["feature"]),
                morphs=morphs,
                feature_index_by_morph=(0, 1),
                nodes=(joe_node, ue_node),
                paths=(joe_path, ue_path),
                connections=(),
            )

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", PreserveModel())
    stub_jtalk: Any = SuffixJoeOpenJTalk()
    features, _morphs = select_mecab_features_with_tsqyomi(
        "商売上",
        stub_jtalk,
        include_morphs=False,
    )

    assert any("ジョー" in feature for feature in features)
    assert all("ウエ" not in feature for feature in features)


def test_preserve_dictionary_default_uses_real_dictionary_pronunciation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """実辞書の読みと発音が異なる接尾辞でも、既定発音のジョーを維持する。"""

    class PreserveModel(_FakeModel):
        """実辞書の接尾辞 上 で ウエ を選ぶテスト用スタブ。"""

        def __init__(self) -> None:
            """上 のジョーを辞書既定発音として設定する。"""

            super().__init__()
            self.metadata = _FakeMetadata(
                frozenset({"上"}),
                {"上": {"ウエ": ("rc_ue",), "ジョー": ("rc_joe",)}},
            )
            self.metadata.preserve_dictionary_default_pronunciations = (("上", "ジョー"),)

        def predict(self, text: str, targets: tuple[Any, ...]) -> tuple[Any, ...]:
            """
            全対象で辞書既定と異なる ウエ を返す。

            Args:
                text (str): 推論対象本文
                targets (tuple[Any, ...]): 読み選択対象

            Returns:
                tuple[Any, ...]: ウエ を選んだ予測列
            """

            return tuple(
                tsqyomi.ReadingPrediction(
                    pronunciation="ウエ",
                    scores=tuple(0.0 for _ in target.pronunciations),
                )
                for target in targets
            )

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    _, baseline_morphs = jtalk.run_mecab_detailed("商売上")
    suffix_morph = next(morph for morph in baseline_morphs if morph["surface"] == "上")
    assert suffix_morph["features"][8] == "ジョウ"
    assert suffix_morph["features"][9] == "ジョー"

    monkeypatch.setattr(tsqyomi_model, "_loaded_model", PreserveModel())
    features, _morphs = select_mecab_features_with_tsqyomi(
        "商売上",
        jtalk,
        include_morphs=False,
    )

    assert any(feature.split(",")[9] == "ジョー" for feature in features)
