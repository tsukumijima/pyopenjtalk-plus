"""tsqyomi のモデル管理、文脈抽出、候補採点を確認する。"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from time import sleep
from typing import Any, cast

import pytest

import pyopenjtalk.tsqyomi as tsqyomi
import pyopenjtalk.tsqyomi.model as tsqyomi_model
from pyopenjtalk.tsqyomi.context import build_model_context
from pyopenjtalk.tsqyomi.inference import make_cost_adjuster
from pyopenjtalk.types import MeCabCostCandidate


class _FakeModel:
    """
    モデル管理と lattice 介入を ONNX Runtime から分離して検証するための偽モデル。
    """

    def __init__(self) -> None:
        """介入表層と採点履歴を初期化する。"""

        self.metadata = cast(
            Any,
            type(
                "Metadata",
                (),
                {"model_scored_surfaces": frozenset({"人気"})},
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
        "is_unknown": False,
        "is_ignored": False,
        "is_reading_protected": is_reading_protected,
        "dictionary_index": 0,
        "node_index": 0,
        "node_id": 1,
    }


def _valid_metadata_payload() -> dict[str, object]:
    """Pydantic スキーマを単独検証するための完全な v1 metadata を作る。

    Returns:
        dict[str, object]: 385件の介入表層を持つ有効な metadata
    """

    return {
        "schema_version": "modernbert_candidate_ranker_v1",
        "tokenizer_type": "tokenizers_json",
        "precision": "mixed_fp16",
        "model_max_length": 512,
        "model_vocabulary_size": 102400,
        "cls_token_id": 1,
        "sep_token_id": 2,
        "unknown_token_id": 0,
        "pad_token_id": 3,
        "cost_weight": 1.0,
        "cost_mode": "relative",
        "confidence_margin": 0.0,
        "rescorer_threshold": 0.0,
        "model_scored_surfaces": tuple(f"表層{index}" for index in range(385)),
        "input_names": ("input_ids", "attention_mask"),
        "output_names": ("candidate_logits",),
    }


def test_metadata_schema_rejects_unknown_and_coerced_values() -> None:
    """配布 metadata の未知フィールドと文字列からの暗黙変換を拒否する"""

    payload = _valid_metadata_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="extra_forbidden"):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.model_validate(payload)

    payload = _valid_metadata_payload()
    payload["model_max_length"] = "512"
    with pytest.raises(ValueError, match="int_type"):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.model_validate(payload)


def test_metadata_schema_accepts_only_complete_mixed_fp16_product_assets() -> None:
    """配布対象外の精度形式と入出力名が欠けた metadata を拒否する"""

    payload = _valid_metadata_payload()
    payload["precision"] = "fp32"
    with pytest.raises(ValueError, match="literal_error"):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.model_validate(payload)

    payload = _valid_metadata_payload()
    del payload["input_names"]
    with pytest.raises(ValueError, match="missing"):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.model_validate(payload)


def test_metadata_schema_rejects_duplicate_scored_surfaces() -> None:
    """件数が385件でも重複を含む介入表層を拒否する"""

    payload = _valid_metadata_payload()
    payload["model_scored_surfaces"] = tuple("重複" for _ in range(385))
    with pytest.raises(ValueError, match="385 unique"):
        cast(Any, tsqyomi_model)._TsqyomiMetadata.model_validate(payload)


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


def test_context_keeps_largest_window_when_encoded_length_is_not_monotonic() -> None:
    """周辺文字でトークン分割が変わっても最大の有効な文脈窓を選ぶ"""

    text = "左" * 6 + "人気" + "右" * 6
    char_start = text.index("人気")

    def encoded_length(marked_text: str, _pronunciation: str) -> int:
        """半径5の窓だけが再び上限内へ収まる非単調な系列長を返す。

        Args:
            marked_text (str): 対象マーカーを含む本文
            _pronunciation (str): 未使用の候補発音

        Returns:
            int: テスト用の系列長
        """

        if marked_text == "左" * 5 + "【人気】" + "右" * 5:
            return 11
        return len(marked_text)

    context = build_model_context(
        text,
        (char_start, char_start + 2),
        ("ニンキ", "ヒトケ"),
        encoded_length,
        11,
    )

    assert context == "左" * 5 + "【人気】" + "右" * 5


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
    with pytest.raises(TypeError, match="sequence of strings"):
        tsqyomi.score_candidates(
            "人気の店",
            (0, 2),
            cast(Any, ["ニンキ", 1]),
        )
    with pytest.raises(ValueError, match="at least two"):
        tsqyomi.score_candidates("人気の店", (0, 2), ["ニンキ"])
    scores = tsqyomi.score_candidates("人気の店", (0, 2), ["ニンキ", "ヒトケ"])
    assert scores[0]["pronunciation"] == "ニンキ"
    assert scores[1]["relative_cost"] == 1.0


def test_load_model_is_idempotent_only_for_same_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じ設定の再ロードは何もせず、異なる設定には明示的な unload_model() を要求する"""

    model_module = cast(Any, tsqyomi_model)
    fake_model = _FakeModel()
    configuration = model_module._configuration_signature(["CPUExecutionProvider"], "/cache")
    monkeypatch.setattr(model_module, "_loaded_model", fake_model)
    monkeypatch.setattr(model_module, "_loaded_configuration", configuration)
    tsqyomi.load_model(["CPUExecutionProvider"], "/cache")
    with pytest.raises(RuntimeError, match="unload_model"):
        tsqyomi.load_model(["CPUExecutionProvider"], "/other-cache")


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
    """metadata の介入表層がない文では候補群の構築前に採点を省く"""

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


def test_cost_adjuster_honors_protected_candidate_without_pronunciation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """読みを抽出できない保護候補も同じ候補群全体のモデル介入を止める"""

    fake_model = _FakeModel()
    monkeypatch.setattr(cast(Any, tsqyomi_model), "_loaded_model", fake_model)
    adjuster = make_cost_adjuster()
    protected_candidate = _candidate("", is_reading_protected=True)
    assert adjuster([_candidate("ニンキ"), _candidate("ヒトケ"), protected_candidate]) == [
        0.0,
        0.0,
        0.0,
    ]
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
    assert resolved == [
        (
            "CUDAExecutionProvider",
            {"use_tf32": "0", "arena_extend_strategy": "kSameAsRequested"},
        ),
        "CPUExecutionProvider",
    ]


def test_explicit_provider_selection_filters_unavailable_entries() -> None:
    """明示された候補列は利用可能な実行プロバイダだけを元の順序で残す"""

    class FakeONNXRuntime:
        """DirectML と CPU を利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    resolved = cast(Any, tsqyomi_model)._resolve_onnx_providers(
        FakeONNXRuntime,
        [
            "CUDAExecutionProvider",
            ("DmlExecutionProvider", {"device_id": 1}),
            "CPUExecutionProvider",
        ],
    )
    assert resolved == [
        ("DmlExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]


def test_provider_selection_rejects_missing_provider_and_tf32() -> None:
    """利用可能な候補がない場合と CUDA の TF32 有効化を明示的に拒否する"""

    class CPUOnlyONNXRuntime:
        """CPU だけを利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["CPUExecutionProvider"]

    with pytest.raises(RuntimeError, match="none of the requested"):
        cast(Any, tsqyomi_model)._resolve_onnx_providers(
            CPUOnlyONNXRuntime,
            ["CUDAExecutionProvider"],
        )

    class CUDAONNXRuntime:
        """CUDA だけを利用可能として返す ONNX Runtime の代用品。"""

        @staticmethod
        def get_available_providers() -> list[str]:
            """利用可能な実行プロバイダを返す。"""

            return ["CUDAExecutionProvider"]

    with pytest.raises(ValueError, match="use_tf32"):
        cast(Any, tsqyomi_model)._resolve_onnx_providers(
            CUDAONNXRuntime,
            [("CUDAExecutionProvider", {"use_tf32": "1"})],
        )


def test_model_loader_configures_product_session(tmp_path: Path) -> None:
    """配布資材から製品用の ONNX セッションを構築する"""

    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    metadata = cast(Any, tsqyomi_model)._TsqyomiMetadata.model_validate(_valid_metadata_payload())
    (tmp_path / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")

    class FakeSessionOptions:
        """モデルローダーが設定した ONNX セッション設定を保持する。"""

        def __init__(self) -> None:
            """検証対象の設定値を未指定状態で初期化する。"""

            self.enable_cpu_mem_arena = False

    class FakeSession:
        """入出力名と利用中の実行プロバイダを返す偽セッション。"""

        def __init__(self, session_options: FakeSessionOptions) -> None:
            """渡されたセッション設定を検証用に保持する。"""

            self.session_options = session_options

        @staticmethod
        def get_inputs() -> list[Any]:
            """metadata と一致する入力名を返す。"""

            return [
                type("Input", (), {"name": "input_ids"})(),
                type("Input", (), {"name": "attention_mask"})(),
            ]

        @staticmethod
        def get_outputs() -> list[Any]:
            """metadata と一致する出力名を返す。"""

            return [type("Output", (), {"name": "candidate_logits"})()]

        @staticmethod
        def get_providers() -> list[str]:
            """CPU 実行中として返す。"""

            return ["CPUExecutionProvider"]

    class FakeONNXRuntime:
        """ONNX Runtime のセッション構築だけを再現する代用品。"""

        SessionOptions = FakeSessionOptions
        created_session: FakeSession | None = None

        @staticmethod
        def get_available_providers() -> list[str]:
            """CPU を利用可能な実行プロバイダとして返す。"""

            return ["CPUExecutionProvider"]

        @classmethod
        def InferenceSession(
            cls,
            _model_path: str,
            *,
            sess_options: FakeSessionOptions,
            providers: list[Any],
        ) -> FakeSession:
            """セッション設定を保持する偽セッションを作る。"""

            assert providers == ["CPUExecutionProvider"]
            cls.created_session = FakeSession(sess_options)
            return cls.created_session

    class FakeTokenizerClass:
        """トークナイザーのファイル読み込みを置き換える代用品。"""

        is_truncation_disabled = False

        @classmethod
        def from_file(cls, _path: str) -> FakeTokenizerClass:
            """ファイル内容に依存しないトークナイザー参照を返す。"""

            return cls()

        def no_truncation(self) -> None:
            """配布トークナイザーの自動切り捨て解除を記録する。"""

            type(self).is_truncation_disabled = True

    runtime_dependencies = cast(Any, tsqyomi_model)._RuntimeDependencies(
        onnxruntime=FakeONNXRuntime,
        tokenizer_class=FakeTokenizerClass,
    )
    cast(Any, tsqyomi_model)._load_model_from_directory(
        tmp_path,
        ["CPUExecutionProvider"],
        runtime_dependencies=runtime_dependencies,
    )

    assert FakeONNXRuntime.created_session is not None
    assert FakeONNXRuntime.created_session.session_options.enable_cpu_mem_arena is True
    assert FakeTokenizerClass.is_truncation_disabled is True


def test_model_splits_candidates_into_batches_of_four() -> None:
    """候補数が多くても ONNX Runtime へ渡す各バッチを最大4件に制限する"""

    class Encoding:
        """固定長のトークン ID を保持する符号化結果。"""

        def __init__(self) -> None:
            """3トークンの固定入力を初期化する。"""

            self.ids = [1, 2, 3]

    class Tokenizer:
        """常に固定長の符号化結果を返すトークナイザー。"""

        @staticmethod
        def encode(_text: str, _pronunciation: str) -> Encoding:
            """本文と候補発音を固定トークン列へ符号化する。"""

            return Encoding()

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

    metadata = cast(
        Any,
        type(
            "Metadata",
            (),
            {
                "model_max_length": 512,
                "cost_weight": 1.0,
                "input_names": ("input_ids", "attention_mask"),
                "output_names": ("candidate_logits",),
                "pad_token_id": 3,
            },
        )(),
    )
    session = Session()
    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        Tokenizer(),
        session,
        metadata,
        is_inference_serialized=False,
    )
    scores = model.score_candidates(
        "人気の店",
        (0, 2),
        [f"コウホ{candidate_index}" for candidate_index in range(9)],
    )
    assert len(scores) == 9
    assert session.batch_sizes == [4, 4, 1]


def test_directml_model_serializes_concurrent_inference() -> None:
    """DirectML 用モデルは複数スレッドの ONNX Run() をモデル側で直列化する"""

    class Encoding:
        """固定長のトークン ID を保持する符号化結果。"""

        def __init__(self) -> None:
            """3トークンの固定入力を初期化する。"""

            self.ids = [1, 2, 3]

    class Tokenizer:
        """常に固定長の符号化結果を返すトークナイザー。"""

        @staticmethod
        def encode(_text: str, _pronunciation: str) -> Encoding:
            """本文と候補発音を固定トークン列へ符号化する。"""

            return Encoding()

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

    metadata = cast(
        Any,
        type(
            "Metadata",
            (),
            {
                "model_max_length": 512,
                "cost_weight": 1.0,
                "input_names": ("input_ids", "attention_mask"),
                "output_names": ("candidate_logits",),
                "pad_token_id": 3,
            },
        )(),
    )
    session = Session()
    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        Tokenizer(),
        session,
        metadata,
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

    class Encoding:
        """固定長のトークン ID を保持する符号化結果。"""

        def __init__(self) -> None:
            """3トークンの固定入力を初期化する。"""

            self.ids = [1, 2, 3]

    class Tokenizer:
        """常に固定長の符号化結果を返すトークナイザー。"""

        @staticmethod
        def encode(_text: str, _pronunciation: str) -> Encoding:
            """本文と候補発音を固定トークン列へ符号化する。"""

            return Encoding()

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

    metadata = cast(
        Any,
        type(
            "Metadata",
            (),
            {
                "model_max_length": 512,
                "cost_weight": 1.0,
                "input_names": ("input_ids", "attention_mask"),
                "output_names": ("candidate_logits",),
                "pad_token_id": 3,
            },
        )(),
    )
    session = Session()
    model = cast(Any, tsqyomi_model)._TsqyomiModel(
        Tokenizer(),
        session,
        metadata,
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
