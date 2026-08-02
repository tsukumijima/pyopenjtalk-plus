"""tsqyomi のモデル管理、文脈抽出、候補採点を確認する。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from time import sleep
from typing import Any, cast

import numpy as np
import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
import pyopenjtalk.tsqyomi.model as tsqyomi_model
from pyopenjtalk.tsqyomi.context import build_model_context
from pyopenjtalk.tsqyomi.inference import make_cost_adjuster
from pyopenjtalk.types import MeCabCostCandidate


class _FakeModel:
    """
    モデル管理と lattice コスト補正を ONNX Runtime から分離して検証するための偽モデル。
    """

    def __init__(self) -> None:
        """採点対象表層と採点履歴を初期化する。"""

        self.metadata = cast(
            Any,
            type(
                "Metadata",
                (),
                {
                    "model_scored_surfaces": frozenset({"人気"}),
                    "model_scored_readings": None,
                },
            )(),
        )
        self.score_calls: list[tuple[str, tuple[int, int], tuple[str, ...]]] = []

    def score_candidates(
        self,
        text: str,
        char_span: tuple[int, int],
        candidate_pronunciations: Sequence[str],
    ) -> list[dict[str, str | float]]:
        """
        候補順に固定コストを返し、呼び出された文脈と候補を記録する。

        Args:
            text (str): 入力本文
            char_span (tuple[int, int]): 対象語の半開区間
            candidate_pronunciations (Sequence[str]): 比較する候補発音

        Returns:
            list[dict[str, str | float]]: 入力順の固定採点結果
        """

        pronunciations = tuple(candidate_pronunciations)
        self.score_calls.append((text, char_span, pronunciations))
        return [
            {
                "pronunciation": pronunciation,
                "logit": float(-candidate_index),
                "relative_cost": float(candidate_index),
            }
            for candidate_index, pronunciation in enumerate(pronunciations)
        ]


class _FixedEncoding:
    """固定長のトークン ID を保持するトークン化結果。"""

    def __init__(self) -> None:
        """3トークンの固定入力を初期化する。"""

        self.ids = [1, 2, 3]


class _FixedTokenizer:
    """常に固定長のトークン化結果を返すトークナイザー。"""

    @staticmethod
    def encode(_text: str, _pronunciation: str) -> _FixedEncoding:
        """本文と候補発音を固定トークン列へトークン化する。"""

        return _FixedEncoding()


def _inference_metadata(baseline_margin: float = 0.0) -> Any:
    """推論テストで実際に参照するモデル設定だけを返す。

    Args:
        baseline_margin (float): 最良候補との差がこの幅に収まる候補を辞書の判断に任せる保留幅

    Returns:
        Any: モデル推論が参照する属性だけを持つモデル設定
    """

    return type(
        "Metadata",
        (),
        {
            "schema_version": "modernbert_candidate_ranker_v2",
            "model_max_length": 512,
            "cost_weight": 1.0,
            "pad_token_id": 3,
            "baseline_margin": baseline_margin,
        },
    )()


def _candidate(
    pronunciation: str,
    *,
    is_reading_protected: bool = False,
) -> MeCabCostCandidate:
    """
    lattice コスト補正のテストに必要な候補辞書を作る。

    Args:
        pronunciation (str): 候補発音
        is_reading_protected (bool): 読み保護された辞書由来か

    Returns:
        MeCabCostCandidate: `run_mecab_with_cost_adjustments()` と同じ候補辞書
    """

    return {
        "surface": "人気",
        "features": [
            "人気",
            "名詞",
            "一般",
            "*",
            "*",
            "*",
            "*",
            "人気",
            pronunciation,
            pronunciation,
            "0/3",
            "C1",
        ],
        "char_span": (0, 2),
        "pos_id": 1,
        "left_id": 1,
        "right_id": 1,
        "word_cost": 1000,
        "node_cost": 1000,
        "forward_path_cost": 1000,
        "backward_path_cost": 0,
        "complete_path_cost": 1000,
        "is_unknown": False,
        "is_ignored": False,
        "is_reading_protected": is_reading_protected,
        "dictionary_index": 0,
        "node_index": 0,
        "node_id": 1,
    }


def test_context_is_limited_to_target_sentence_and_trailing_symbols() -> None:
    """対象を含む1文だけを残し、連続終端記号と閉じ括弧を文末へ含める"""

    text = "前文です。「人気の店！？！」次の文です。"
    char_start = text.index("人気")
    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        lambda marked_text, pronunciation: len(marked_text) + len(pronunciation),
        512,
    )
    assert context == "「【人気】の店！？！」"


def test_context_does_not_treat_closing_bracket_as_sentence_terminator() -> None:
    """文末記号のない閉じ括弧では後続の本文を切り離さない"""

    text = "前の発話」人気の店\n後の発話"
    char_start = text.index("人気")
    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        lambda marked_text, pronunciation: len(marked_text) + len(pronunciation),
        512,
    )
    assert context == "前の発話」【人気】の店\n"


def test_context_includes_closing_bracket_after_sentence_terminator() -> None:
    """文末記号に続く閉じ括弧は対象文の末尾へ含める"""

    text = "「人気の店です。」次の文です。"
    char_start = text.index("人気")
    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        lambda marked_text, pronunciation: len(marked_text) + len(pronunciation),
        512,
    )
    assert context == "「【人気】の店です。」"


def test_context_replaces_existing_target_markers() -> None:
    """原文の隅付き括弧を置換し、対象を示すマーカーだけを残す"""

    text = "【前】人気【後】"
    char_start = text.index("人気")
    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        lambda marked_text, pronunciation: len(marked_text) + len(pronunciation),
        512,
    )

    assert context == "［前］【人気】［後］"
    assert context.count("【") == 1
    assert context.count("】") == 1


@pytest.mark.parametrize(
    ("terminator", "trailing_closer"),
    [
        ("。", ""),
        ("？", ""),
        ("?", ""),
        ("！", ""),
        ("!", ""),
        ("\n", ""),
        ("！？!", "」"),
    ],
)
def test_context_sentence_boundaries(
    terminator: str,
    trailing_closer: str,
) -> None:
    """句読点、改行、連続記号、閉じ括弧を1文の境界として固定する"""

    text = f"前文{terminator}人気{terminator}{trailing_closer}後文"
    char_start = text.index("人気")
    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        lambda marked_text, pronunciation: len(marked_text) + len(pronunciation),
        512,
    )
    assert context == f"【人気】{terminator}{trailing_closer}"


def test_long_sentence_uses_centered_window_with_target_markers() -> None:
    """長い1文でも対象マーカーを残し、全候補が最大系列長へ収まる"""

    text = "左" * 800 + "人気" + "右" * 800 + "。"
    char_start = text.index("人気")
    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        lambda marked_text, pronunciation: len(marked_text) + len(pronunciation) + 3,
        512,
    )
    assert "【人気】" in context
    assert len(context) + len("ニンキ") + 3 <= 512
    assert abs(context.index("【") - (len(context) - context.index("】") - 1)) <= 2


def test_context_rejects_empty_candidate_pronunciations() -> None:
    """系列長を評価できない空の候補発音を明示的に拒否する"""

    with pytest.raises(ValueError, match="candidate_pronunciations must not be empty"):
        build_model_context(
            "人気の店",
            (0, 2),
            (),
            lambda marked_text, pronunciation: len(marked_text) + len(pronunciation),
            512,
        )


def test_context_window_search_keeps_encoding_count_bounded() -> None:
    """1万文字の長文でも系列長の評価回数を入力文字数に比例させない"""

    text = "前" * 10_000 + "人気です"
    char_start = text.index("人気")
    encoded_call_count = 0

    def encoded_length(marked_text: str, pronunciation: str) -> int:
        """文字数を系列長として返し、呼び出し回数を記録する。

        Args:
            marked_text (str): 対象マーカーを含む本文
            pronunciation (str): 候補発音

        Returns:
            int: テスト用の系列長
        """

        nonlocal encoded_call_count
        encoded_call_count += 1
        return len(marked_text) + len(pronunciation)

    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        encoded_length,
        512,
    )

    assert "【人気】" in context
    assert encoded_call_count <= 100


def test_public_score_requires_explicit_model_and_two_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公開採点 API は自動ロードせず、ロード済みモデルを明示的に使う"""

    tsqyomi.unload_model()
    with pytest.raises(RuntimeError, match="load_model"):
        tsqyomi.score_candidates("人気の店", (0, 2), ["ニンキ", "ヒトケ"])

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    with pytest.raises(TypeError, match="sequence of strings"):
        tsqyomi.score_candidates("人気の店", (0, 2), cast(Any, "ニンキ"))
    with pytest.raises(ValueError, match="at least two"):
        tsqyomi.score_candidates("人気の店", (0, 2), ["ニンキ"])
    scores = tsqyomi.score_candidates("人気の店", (0, 2), ["ニンキ", "ヒトケ"])
    assert scores[0]["pronunciation"] == "ニンキ"
    assert scores[1]["relative_cost"] == 1.0


def test_internal_model_rejects_empty_candidates_before_inference() -> None:
    """内部モデルも空候補をモデル推論の前に拒否する"""

    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        tokenizer=None,
        session=None,
        metadata=None,
        is_inference_serialized=False,
    )

    with pytest.raises(ValueError, match="candidate_pronunciations must not be empty"):
        model.score_candidates("人気の店", (0, 2), ())


def test_load_model_is_idempotent_when_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ロード済みのモデルを繰り返し取得しない"""

    model_module = cast(Any, tsqyomi_model)
    fake_model = _FakeModel()
    monkeypatch.setattr(model_module, "_loaded_model", fake_model)
    tsqyomi.load_model(["CPUExecutionProvider"], "/cache")
    assert model_module._loaded_model is fake_model


def test_load_model_passes_each_downloaded_asset_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """個別に取得した3ファイルの実パスをモデル構築へ渡す"""

    import huggingface_hub

    model_module = cast(Any, tsqyomi_model)
    downloaded_paths = {
        "v1/model.onnx": Path("/cache/model/model.onnx"),
        "v1/tokenizer.json": Path("/cache/tokenizer/tokenizer.json"),
        "v1/metadata.json": Path("/cache/metadata/metadata.json"),
    }
    loaded_paths: tuple[Path, Path, Path] | None = None
    fake_model = _FakeModel()

    def fake_download(*, filename: str, **_kwargs: Any) -> str:
        """ファイルごとに異なるディレクトリのパスを返す。"""

        return str(downloaded_paths[filename])

    def fake_load(
        model_path: Path,
        tokenizer_path: Path,
        metadata_path: Path,
        _onnx_providers: Sequence[Any] | None,
    ) -> _FakeModel:
        """モデル構築へ渡された3ファイルのパスを記録する。"""

        nonlocal loaded_paths
        loaded_paths = (model_path, tokenizer_path, metadata_path)
        return fake_model

    monkeypatch.setattr(model_module, "_loaded_model", None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(model_module, "_load_model_from_paths", fake_load)

    tsqyomi.load_model(["CPUExecutionProvider"], "/cache")

    assert loaded_paths == (
        downloaded_paths["v1/model.onnx"],
        downloaded_paths["v1/tokenizer.json"],
        downloaded_paths["v1/metadata.json"],
    )
    assert model_module._loaded_model is fake_model


def test_metadata_loads_used_fields_and_ignores_other_entries(tmp_path: Path) -> None:
    """メタデータ JSON の未使用項目を無視し、推論用の型へ変換する"""

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        """
        {
            "schema_version": "modernbert_candidate_ranker_v2",
            "model_max_length": 512,
            "pad_token_id": 3,
            "cost_weight": 1.0,
            "model_scored_surfaces": ["人気", "十分"],
            "model_scored_readings": {
                "人気": ["ニンキ", "ヒトケ"],
                "十分": ["ジューブン", "ジップン"]
            },
            "confidence_margin": 0.0
        }
        """,
        encoding="utf-8",
    )

    metadata = cast(Any, tsqyomi_model)._TsqyomiMetadata.load(metadata_path)

    assert metadata.model_max_length == 512
    assert metadata.schema_version == "modernbert_candidate_ranker_v2"
    assert metadata.pad_token_id == 3
    assert metadata.cost_weight == 1.0
    assert metadata.model_scored_surfaces == frozenset({"人気", "十分"})
    assert metadata.model_scored_readings == {
        "人気": frozenset({"ニンキ", "ヒトケ"}),
        "十分": frozenset({"ジューブン", "ジップン"}),
    }
    assert "confidence_margin" not in type(metadata).model_fields


@pytest.mark.parametrize(
    "model_scored_readings",
    (
        None,
        {"人気": ["ニンキ", "ヒトケ"]},
        {"人気": ["ニンキ", "ヒトケ"], "十分": ["ジューブン"]},
        {"人気": ["ニンキ", "ヒトケ"], "十分": ["", "ジューブン"]},
    ),
)
def test_metadata_rejects_incomplete_reading_contract(
    tmp_path: Path,
    model_scored_readings: dict[str, list[str]] | None,
) -> None:
    """表層集合の不一致と競争しない単一読みをメタデータで拒否する"""

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_max_length": 512,
                "schema_version": "modernbert_candidate_ranker_v2",
                "pad_token_id": 3,
                "cost_weight": 1.0,
                "model_scored_surfaces": ["人気", "十分"],
                "model_scored_readings": model_scored_readings,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.load(metadata_path)


def test_metadata_rejects_reading_contract_without_v2_schema(tmp_path: Path) -> None:
    """版番号を付け忘れた新形式を旧 v1 モデルとして受理しない"""

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_max_length": 512,
                "pad_token_id": 3,
                "cost_weight": 1.0,
                "model_scored_surfaces": ["人気"],
                "model_scored_readings": {"人気": ["ニンキ", "ヒトケ"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="v1 metadata must not contain"):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.load(metadata_path)


def test_started_inference_finishes_after_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    """開始済みの採点はモデル参照を保持し、並行した unload_model() 後も完了する"""

    started = Event()
    resume = Event()

    class BlockingModel(_FakeModel):
        """推論開始後にテスト側の合図まで待機する偽モデル。"""

        def score_candidates(
            self,
            text: str,
            char_span: tuple[int, int],
            candidate_pronunciations: Sequence[str],
        ) -> list[dict[str, str | float]]:
            """開始を通知し、再開指示後に固定スコアを返す。"""

            started.set()
            assert resume.wait(timeout=5.0) is True
            return super().score_candidates(text, char_span, candidate_pronunciations)

    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", BlockingModel())
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            tsqyomi.score_candidates,
            "人気の店",
            (0, 2),
            ["ニンキ", "ヒトケ"],
        )
        assert started.wait(timeout=5.0) is True
        tsqyomi.unload_model()
        resume.set()
        assert future.result(timeout=5.0)[0]["pronunciation"] == "ニンキ"
    with pytest.raises(RuntimeError, match="load_model"):
        tsqyomi.score_candidates("人気の店", (0, 2), ["ニンキ", "ヒトケ"])


def test_cost_adjuster_skips_inference_without_target_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """メタデータの採点対象表層がない文では候補群の構築前に採点を省く"""

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    candidates = [_candidate("ニンキ")]
    candidates[0]["surface"] = "店舗"
    assert adjuster(candidates) == [0.0]
    assert fake_model.score_calls == []


def test_cost_adjuster_skips_inference_when_lattice_text_has_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """文字位置を完全に復元できない lattice にはモデルコストを適用しない"""

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    trailing_candidate = _candidate("ミセ")
    trailing_candidate["surface"] = "店"
    trailing_candidate["char_span"] = (3, 4)

    assert adjuster([_candidate("ニンキ"), _candidate("ヒトケ"), trailing_candidate]) == [
        0.0,
        0.0,
        0.0,
    ]
    assert fake_model.score_calls == []


def test_cost_adjuster_skips_whole_group_when_one_dictionary_is_protected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じ候補群に保護辞書由来の候補が1件でもあれば群全体を補正しない"""

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    candidates = [
        _candidate("ニンキ"),
        _candidate("ヒトケ", is_reading_protected=True),
    ]
    assert adjuster(candidates) == [0.0, 0.0]
    assert fake_model.score_calls == []


def test_cost_adjuster_applies_relative_cost_to_unprotected_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未保護の異発音候補群へ候補発音ごとの相対コストを加える"""

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    candidates = [_candidate("ニンキ"), _candidate("ヒトケ")]
    assert adjuster(candidates) == [0.0, 1.0]
    assert fake_model.score_calls == [("人気", (0, 2), ("ニンキ", "ヒトケ"))]


def test_cost_adjuster_keeps_group_unchanged_when_reading_is_outside_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """採点対象読みに含まれない同一表層の固有読みへモデルコストを適用しない"""

    fake_model = _FakeModel()
    fake_model.metadata.model_scored_readings = {
        "人気": frozenset({"ニンキ", "ヒトケ"}),
    }
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    candidates = [_candidate("ニンキ"), _candidate("ヒトケ"), _candidate("ジンキ")]

    assert adjuster(candidates) == [0.0, 0.0, 0.0]
    assert fake_model.score_calls == []


def test_cost_adjuster_keeps_group_unchanged_when_candidate_has_no_pronunciation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v2 では採点対象に登録できない空発音が同居する群をモデル補正から外す"""

    fake_model = _FakeModel()
    fake_model.metadata.model_scored_readings = {
        "人気": frozenset({"ニンキ", "ヒトケ"}),
    }
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    empty_pronunciation_candidate = _candidate("ニンキ")
    empty_pronunciation_candidate["features"] = [*empty_pronunciation_candidate["features"][:9], ""]
    candidates = [_candidate("ニンキ"), _candidate("ヒトケ"), empty_pronunciation_candidate]

    assert adjuster(candidates) == [0.0, 0.0, 0.0]
    assert fake_model.score_calls == []


def test_cost_adjuster_accepts_pronunciation_in_ten_column_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """発音列までを持つ10列の候補も読み選択へ渡す"""

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    candidates = [_candidate("ニンキ"), _candidate("ヒトケ")]
    for candidate in candidates:
        candidate["features"] = candidate["features"][:10]

    assert adjuster(candidates) == [0.0, 1.0]
    assert fake_model.score_calls == [("人気", (0, 2), ("ニンキ", "ヒトケ"))]


def test_default_provider_selection_prefers_cuda_then_cpu() -> None:
    """既定設定は利用可能な CUDA を先頭に置き、CPU をフォールバックとして続ける"""

    class FakeONNXRuntime:
        """CUDA と CPU を利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    resolved = cast(Any, tsqyomi_model)._resolve_onnx_providers(FakeONNXRuntime, None)
    assert resolved == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_explicit_provider_selection_rejects_unavailable_entries() -> None:
    """明示された実行プロバイダが利用できない場合は失敗させる"""

    class FakeONNXRuntime:
        """DirectML と CPU を利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        cast(Any, tsqyomi_model)._resolve_onnx_providers(
            FakeONNXRuntime,
            [
                "CUDAExecutionProvider",
                ("DmlExecutionProvider", {"device_id": 1}),
                "CPUExecutionProvider",
            ],
        )


def test_provider_selection_rejects_missing_provider() -> None:
    """利用できない実行プロバイダを明示した場合は拒否する"""

    class CPUOnlyONNXRuntime:
        """CPU だけを利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="unavailable"):
        cast(Any, tsqyomi_model)._resolve_onnx_providers(
            CPUOnlyONNXRuntime,
            ["CUDAExecutionProvider"],
        )


def test_model_splits_candidates_into_batches_of_four() -> None:
    """候補数が多くても ONNX Runtime へ渡す各バッチを最大4件に制限する"""

    class Session:
        """実行されたバッチサイズを記録する ONNX セッション。"""

        def __init__(self) -> None:
            """バッチサイズの記録領域を初期化する。"""

            self.batch_sizes: list[int] = []

        def run(self, _output_names: list[str], model_inputs: dict[str, Any]) -> list[Any]:
            """バッチサイズを記録し、候補順の固定ロジットを返す。"""

            batch_size = int(model_inputs["input_ids"].shape[0])
            self.batch_sizes.append(batch_size)
            return [list(range(batch_size))]

    session = Session()
    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        _FixedTokenizer(),
        session,
        _inference_metadata(),
        is_inference_serialized=False,
    )
    scores = model.score_candidates(
        "人気の店",
        (0, 2),
        [f"コウホ{candidate_index}" for candidate_index in range(9)],
    )
    assert len(scores) == 9
    assert session.batch_sizes == [4, 4, 1]


def test_baseline_margin_leaves_close_candidates_to_the_dictionary() -> None:
    """保留幅の内側にある候補はコストを動かさず、外側の候補だけコストを受ける"""

    class Session:
        """候補ごとに固定の差を持つロジットを返す ONNX セッション。"""

        def run(self, _output_names: list[str], model_inputs: dict[str, Any]) -> list[Any]:
            """先頭候補を最良とし、以降を1.0ずつ下げたロジットを返す。"""

            batch_size = int(model_inputs["input_ids"].shape[0])
            return [[-float(index) for index in range(batch_size)]]

    candidates = ["コウホ0", "コウホ1", "コウホ2", "コウホ3"]

    # 保留幅なしでは、最良候補との差がそのままコストになる
    model_without_margin = cast(Any, tsqyomi_model)._TsqyomiModel(
        _FixedTokenizer(),
        Session(),
        _inference_metadata(),
        is_inference_serialized=False,
    )
    scores_without_margin = model_without_margin.score_candidates("人気の店", (0, 2), candidates)
    assert [score["relative_cost"] for score in scores_without_margin] == [0.0, 1.0, 2.0, 3.0]

    # 保留幅1.5では、差1.0の候補までが辞書側の判断に任せられる
    model_with_margin = cast(Any, tsqyomi_model)._TsqyomiModel(
        _FixedTokenizer(),
        Session(),
        _inference_metadata(baseline_margin=1.5),
        is_inference_serialized=False,
    )
    scores_with_margin = model_with_margin.score_candidates("人気の店", (0, 2), candidates)
    assert [score["relative_cost"] for score in scores_with_margin] == [0.0, 0.0, 2.0, 3.0]


def test_directml_model_serializes_concurrent_inference() -> None:
    """DirectML 用モデルは複数スレッドの ONNX Run() をモデル側で直列化する"""

    class Session:
        """同時実行数を記録する DirectML 相当の ONNX セッション。"""

        def __init__(self) -> None:
            """同時実行数と排他制御を初期化する。"""

            self.active_count = 0
            self.maximum_active_count = 0
            self.lock = Lock()

        def run(self, _output_names: list[str], model_inputs: dict[str, Any]) -> list[Any]:
            """実行中の同時呼び出し数を記録して固定ロジットを返す。"""

            with self.lock:
                self.active_count += 1
                self.maximum_active_count = max(self.maximum_active_count, self.active_count)
            sleep(0.02)
            with self.lock:
                self.active_count -= 1
            return [[0.0] * int(model_inputs["input_ids"].shape[0])]

    session = Session()
    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        _FixedTokenizer(),
        session,
        _inference_metadata(),
        is_inference_serialized=True,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                model.score_candidates,
                "人気の店",
                (0, 2),
                ["ニンキ", "ヒトケ"],
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=5.0)
    assert session.maximum_active_count == 1


def test_cpu_and_cuda_model_allow_concurrent_inference() -> None:
    """CPU と CUDA 用モデルは複数スレッドの ONNX Run() を並行実行できる"""

    class Session:
        """同時実行数を記録する CPU・CUDA 相当の ONNX セッション。"""

        def __init__(self) -> None:
            """同時実行数と同期用バリアを初期化する。"""

            self.active_count = 0
            self.maximum_active_count = 0
            self.lock = Lock()
            self.barrier = Barrier(2)

        def run(self, _output_names: list[str], model_inputs: dict[str, Any]) -> list[Any]:
            """2スレッドを同期し、並行実行数を記録して固定ロジットを返す。"""

            with self.lock:
                self.active_count += 1
                self.maximum_active_count = max(self.maximum_active_count, self.active_count)
            self.barrier.wait(timeout=5.0)
            with self.lock:
                self.active_count -= 1
            return [[0.0] * int(model_inputs["input_ids"].shape[0])]

    session = Session()
    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        _FixedTokenizer(),
        session,
        _inference_metadata(),
        is_inference_serialized=False,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                model.score_candidates,
                "人気の店",
                (0, 2),
                ["ニンキ", "ヒトケ"],
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=5.0)
    assert session.maximum_active_count == 2


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


def test_enabled_tsqyomi_requires_explicit_model_load() -> None:
    """高レベル API もモデルの暗黙ロードや CPU 推論への切り替えを行わない"""

    tsqyomi.unload_model()
    with pytest.raises(RuntimeError, match="load_model"):
        pyopenjtalk.g2p("人気の店です。", use_tsqyomi=True)


def test_high_level_dictionary_protection_reaches_tsqyomi_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """高レベル辞書指定の保護フラグを lattice 候補まで渡してモデル採点を止める"""

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
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
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
        assert fake_model.score_calls == []
    finally:
        pyopenjtalk.unset_user_dict()


@pytest.mark.parametrize(
    "provider_name",
    [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
    ],
)
def test_real_model_load_and_high_level_inference(
    tmp_path: Path,
    provider_name: str,
) -> None:
    """利用可能な実行プロバイダで実モデルをロードし、ONNX 推論結果を高レベル G2P へ反映する"""

    onnxruntime = pytest.importorskip("onnxruntime")
    if provider_name not in onnxruntime.get_available_providers():
        pytest.skip(f"{provider_name} is not available")

    user_dictionary_csv = tmp_path / "readings.csv"
    user_dictionary_dic = tmp_path / "readings.dic"
    user_dictionary_csv.write_text(
        "人気,,,1,名詞,一般,*,*,*,*,人気,ニンキ,ニンキ,0/3,*\n"
        "人気,,,1,名詞,一般,*,*,*,*,人気,ヒトケ,ヒトケ,0/3,*\n",
        encoding="utf-8",
    )
    pyopenjtalk.mecab_dict_index(
        str(user_dictionary_csv),
        str(user_dictionary_dic),
    )

    # モックでは検出できない固定リビジョンのモデルファイル取得、トークナイザーと ONNX セッションの構築を通す
    tsqyomi.unload_model()
    tsqyomi.load_model([provider_name])
    try:
        assert tsqyomi.is_model_loaded() is True

        # 直接採点 API でモデルが文脈に応じて「ヒトケ」を選ぶことを確認する
        text = "警備員は人気のない倉庫を確認した。"
        char_start = text.index("人気")
        scores = tsqyomi.score_candidates(
            text,
            (char_start, char_start + len("人気")),
            ["ニンキ", "ヒトケ"],
        )
        assert [score["pronunciation"] for score in scores] == ["ニンキ", "ヒトケ"]
        assert scores[0]["relative_cost"] > 0.0
        assert scores[1]["relative_cost"] == 0.0

        # 長い1文でも対象語を残した512トークン窓を構築し、各実行プロバイダで同じ候補採点を完走させる
        long_text = (
            ("倉庫の設備を順番に確認した結果、" * 200) + text + ("周辺の安全も確認した" * 200)
        )
        long_text_char_start = long_text.index("人気")
        long_text_scores = tsqyomi.score_candidates(
            long_text,
            (long_text_char_start, long_text_char_start + len("人気")),
            ["ニンキ", "ヒトケ"],
        )
        assert [score["pronunciation"] for score in long_text_scores] == ["ニンキ", "ヒトケ"]

        # DirectML はモデル内部のロックで直列化され、CPU と CUDA は同じ公開 API から並行推論できる
        with ThreadPoolExecutor(max_workers=2) as executor:
            score_futures = [
                executor.submit(
                    tsqyomi.score_candidates,
                    text,
                    (char_start, char_start + len("人気")),
                    ["ニンキ", "ヒトケ"],
                )
                for _request_index in range(4)
            ]
            concurrent_scores = [score_future.result() for score_future in score_futures]
        assert all(scores[1]["relative_cost"] == 0.0 for scores in concurrent_scores)

        # 同じ表層の2候補を持つ実辞書を使い、モデルのコスト補正が MeCab の one-best まで届くことを確認する
        pyopenjtalk.update_global_jtalk_with_user_dict(str(user_dictionary_dic))
        assert pyopenjtalk.g2p(text, kana=True) == "ケービインワニンキノナイソーコヲカクニンシタ。"
        assert (
            pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True)
            == "ケービインワヒトケノナイソーコヲカクニンシタ。"
        )
    finally:
        pyopenjtalk.unset_user_dict()
        tsqyomi.unload_model()

    assert tsqyomi.is_model_loaded() is False
