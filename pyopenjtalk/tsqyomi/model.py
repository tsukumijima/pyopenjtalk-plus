from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, PrivateAttr, model_validator


# モデル、トークナイザー、メタデータの組み合わせを同一スナップショットへ固定する
_MODEL_REPOSITORY = "tsukumijima/tsqyomi-models"
_MODEL_REVISION = "ad4e0693dfd1821acf0a01b9886fc5ebe5b484af"
_MODEL_FILES = {
    "model": "v2/model.onnx",
    "tokenizer": "v2/tokenizer.json",
    "metadata": "v2/metadata.json",
}

ONNXProvider = Union[str, tuple[str, dict[str, Any]]]


class TargetWindowOverflowError(ValueError):
    """全対象が1つのモデル入力窓に同時には収まらない場合に送出する。"""


@dataclass(frozen=True)
class ReadingTarget:
    """
    同じ本文内の1対象と到達可能な発音候補を表す。

    Attributes:
        char_span (tuple[int, int]): 正規化本文上の対象表層の半開区間
        surface (str): 対象表層
        pronunciations (tuple[str, ...]): 候補グラフ上で到達可能な発音 (重複なし)
    """

    char_span: tuple[int, int]
    surface: str
    pronunciations: tuple[str, ...]


@dataclass(frozen=True)
class ReadingPrediction:
    """
    1対象について選択した発音と候補ごとのスコアを保持する。

    Attributes:
        pronunciation (str): 選んだ発音
        scores (tuple[float, ...]): `pronunciations` と同じ順序の対数事後スコア
    """

    pronunciation: str
    scores: tuple[float, ...]


class TsqyomiMetadata(BaseModel):
    """
    tsqyomi v2 ONNX モデルが参照するメタデータ。

    Attributes:
        schema_version (Literal["modernbert_reading_class_v2"]): メタデータ契約の識別子
        target_boundary_contract (Literal["mecab_target_segments_v1"]): 対象境界の契約名
        model_max_length (int): トークナイザー入力の最大系列長
        pad_token_id (int): パディングトークン ID (現行推論では未使用)
        model_scored_surfaces (frozenset[str]): モデルが推論対象とする表層集合
        output_class_order (tuple[str, ...]): ONNX 出力列と対応する読みクラス ID 列
        reading_class_ids_by_surface_and_pronunciation (dict[str, dict[str, tuple[str, ...]]]):
            表層ごとの発音→読みクラス ID 列
    """

    schema_version: Literal["modernbert_reading_class_v2"]
    target_boundary_contract: Literal["mecab_target_segments_v1"]
    model_max_length: int
    pad_token_id: int
    model_scored_surfaces: frozenset[str]
    output_class_order: tuple[str, ...]
    reading_class_ids_by_surface_and_pronunciation: dict[str, dict[str, tuple[str, ...]]]
    _surfaces_by_first_character: dict[str, tuple[str, ...]] = PrivateAttr()

    @property
    def surfaces_by_first_character(self) -> dict[str, tuple[str, ...]]:
        """
        推論対象表層の先頭文字別索引を返す。

        Returns:
            dict[str, tuple[str, ...]]: `validate_reading_classes()` 実行時に構築した索引
        """

        return self._surfaces_by_first_character

    @staticmethod
    def _index_surfaces_by_first_character(
        surfaces: frozenset[str],
    ) -> dict[str, tuple[str, ...]]:
        """
        推論対象表層を先頭文字別・長さ降順へ索引化する。

        Args:
            surfaces (frozenset[str]): モデルが推論対象とする表層の集合

        Returns:
            dict[str, tuple[str, ...]]: 先頭文字から、長い表層を優先した表層列への索引
        """

        indexed: dict[str, list[str]] = {}
        for surface in surfaces:
            indexed.setdefault(surface[0], []).append(surface)
        return {
            character: tuple(sorted(values, key=lambda surface: (-len(surface), surface)))
            for character, values in indexed.items()
        }

    @model_validator(mode="after")
    def validate_reading_classes(self) -> TsqyomiMetadata:
        """
        出力列と表層別の読みクラス定義が完全に対応することを検証する。
        検証成功時に `surfaces_by_first_character` を構築する。

        Returns:
            TsqyomiMetadata: 検証済みの自身

        Raises:
            ValueError: 読みクラス定義・表層集合・バケット内容が v2 仕様と一致しない場合
        """

        if len(self.output_class_order) != len(set(self.output_class_order)):
            raise ValueError("output_class_order must contain unique class IDs")
        if (
            frozenset(self.reading_class_ids_by_surface_and_pronunciation)
            != self.model_scored_surfaces
        ):
            raise ValueError("reading class buckets must cover model_scored_surfaces exactly")
        if any(surface == "" for surface in self.model_scored_surfaces):
            raise ValueError("model-scored surfaces must not be empty")
        class_ids = set(self.output_class_order)
        for buckets in self.reading_class_ids_by_surface_and_pronunciation.values():
            if len(buckets) < 2:
                raise ValueError("each model-scored surface must have at least two pronunciations")
            if any(
                len(ids) == 0 or set(ids).issubset(class_ids) is False for ids in buckets.values()
            ):
                raise ValueError("reading class bucket contains an unknown or empty class set")
        self._surfaces_by_first_character = self._index_surfaces_by_first_character(
            self.model_scored_surfaces
        )
        return self

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> TsqyomiMetadata:
        """
        メタデータ JSON から推論に必要な値を読み込む。

        Args:
            path (Path): メタデータ JSON のパス

        Returns:
            TsqyomiMetadata: モデル設定
        """

        # 未定義の JSON 項目は Pydantic の標準動作で無視し、推論が参照する項目だけを検証する
        return cls.model_validate_json(path.read_bytes())


class TsqyomiModel:
    """
    トークナイザー、ONNX セッション、メタデータを1組で保持する推論エンジン。

    NOTE:
        ONNX Runtime は CPU / CUDA では同一 `InferenceSession` への並行 `run()` をスレッドセーフとしている。
        DirectML EP は同一セッションへの並行 `Run()` をサポートしないため、ロード時に DML が有効なら
        本クラス内で `session.run()` を直列化する。利用側が EP ごとに排他を意識する必要はない。
    """

    def __init__(
        self,
        tokenizer: Any,
        session: Any,
        metadata: TsqyomiMetadata,
    ) -> None:
        """
        トークナイザー、ONNX セッション、メタデータを1つのモデル参照へまとめる。

        Args:
            tokenizer (Any): `tokenizers.Tokenizer` のロード済みインスタンス
            session (Any): `onnxruntime.InferenceSession` のロード済みインスタンス
            metadata (TsqyomiMetadata): 検証済みのモデル設定
        """

        self.tokenizer = tokenizer
        self.session = session
        self.metadata = metadata
        # DirectML だけは ORT 本体が mutex を付けないため、同一セッションの Run() をここで直列化する
        self._inference_lock = Lock() if "DmlExecutionProvider" in session.get_providers() else None
        empty_encoding = tokenizer.encode("")
        if len(empty_encoding.ids) != 2 or empty_encoding.special_tokens_mask != [1, 1]:
            raise ValueError(
                "tsqyomi tokenizer must add one leading and one trailing special token"
            )
        self._leading_token_id = empty_encoding.ids[0]
        self._trailing_token_id = empty_encoding.ids[1]

    @staticmethod
    def validate_onnx_contract(session: Any, metadata: TsqyomiMetadata) -> None:
        """
        ONNX の入出力と読みクラスの列数がメタデータに一致することを検査する。

        Args:
            session (Any): 初期化済みの `onnxruntime.InferenceSession`
            metadata (TsqyomiMetadata): 検証済みのモデル設定

        Raises:
            ValueError: ONNX の入出力名、型、クラス数が v2 仕様と一致しない場合
        """

        # 入力名だけ一致して型が異なる ONNX モデルも、ONNX Runtime の実行時エラーより先に拒否する
        actual_inputs = {value.name: value.type for value in session.get_inputs()}
        expected_inputs = {
            "input_ids": "tensor(int64)",
            "attention_mask": "tensor(int64)",
            "target_mask": "tensor(bool)",
        }
        if actual_inputs != expected_inputs:
            raise ValueError(f"tsqyomi ONNX inputs do not match the v2 contract: {actual_inputs}")

        # 出力の末尾次元を固定し、別世代の読みクラス順を誤って組み合わせない
        outputs = session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "reading_class_logits":
            raise ValueError("tsqyomi ONNX must expose only reading_class_logits")
        output = outputs[0]
        if output.type != "tensor(float)":
            raise ValueError("tsqyomi ONNX reading_class_logits must be float32")
        if len(output.shape) != 3 or output.shape[2] != len(metadata.output_class_order):
            raise ValueError("tsqyomi ONNX output class count does not match output_class_order")

    def _run_onnx_session(self, model_inputs: dict[str, Any]) -> Any:
        """
        ONNX Runtime へ推論を委譲する。
        DirectML 利用時は同一セッションへの並行 Run() を内部ロックで直列化する。

        Args:
            model_inputs (dict[str, Any]): `session.run()` へ渡す入力テンソル

        Returns:
            Any: `reading_class_logits` の生出力
        """

        if self._inference_lock is not None:
            with self._inference_lock:
                return self.session.run(["reading_class_logits"], model_inputs)[0]
        return self.session.run(["reading_class_logits"], model_inputs)[0]

    def predict(
        self,
        text: str,
        targets: tuple[ReadingTarget, ...],
    ) -> tuple[ReadingPrediction, ...]:
        """
        本文内の対象読みを推論する。
        全対象が1入力窓に収まる場合は ONNX を1回だけ実行し、収まらない場合は対象列を分割して再帰する。

        Args:
            text (str): 入力本文
            targets (tuple[ReadingTarget, ...]): 同じ本文にある対象と候補発音

        Returns:
            tuple[ReadingPrediction, ...]: 入力順の対象別予測。`targets` が空なら空タプル

        Raises:
            TargetWindowOverflowError: 単一対象が入力窓に収まらない場合
            ValueError: 候補発音が空、対象が重なる、tokenizer が span を保持できない等
        """

        if len(targets) == 0:
            return ()
        try:
            return self._predict_single_window(text, targets)
        except TargetWindowOverflowError:
            if len(targets) == 1:
                raise
        # 二分した各区間は順次実行し、長文の対象数に比例した巨大バッチを作らない
        middle = len(targets) // 2
        return (
            *self.predict(text, targets[:middle]),
            *self.predict(text, targets[middle:]),
        )

    def _predict_single_window(
        self,
        text: str,
        targets: tuple[ReadingTarget, ...],
    ) -> tuple[ReadingPrediction, ...]:
        """
        全対象を1つのモデル入力窓へ載せて ONNX 推論を1回実行する。

        Args:
            text (str): 入力本文
            targets (tuple[ReadingTarget, ...]): 同じ本文にある対象と候補発音

        Returns:
            tuple[ReadingPrediction, ...]: 入力順の対象別予測

        Raises:
            TargetWindowOverflowError: 全対象が1入力窓に収まらない場合
            ValueError: 候補発音が空、対象が重なる、tokenizer が span を保持できない等
        """

        if len(targets) == 0:
            raise ValueError("targets must not be empty")
        ordered_targets = tuple(sorted(targets, key=lambda target: target.char_span))
        for previous, current in pairwise(ordered_targets):
            if previous.char_span[1] > current.char_span[0]:
                raise ValueError("targets must not overlap")
        boundaries = sorted(
            {
                0,
                len(text),
                *(target.char_span[0] for target in ordered_targets),
                *(target.char_span[1] for target in ordered_targets),
            }
        )
        content_ids: list[int] = []
        content_offsets: list[tuple[int, int]] = []
        for segment_start, segment_end in pairwise(boundaries):
            # MeCab が確定した対象境界で部分語分割も切り、助詞を対象表現へ混入させない
            segment_encoding = self.tokenizer.encode(
                text[segment_start:segment_end],
                add_special_tokens=False,
            )
            content_ids.extend(segment_encoding.ids)
            content_offsets.extend(
                (start + segment_start, end + segment_start)
                for start, end in segment_encoding.offsets
            )
        positions_by_target: list[list[int]] = []
        for target in ordered_targets:
            if text[target.char_span[0] : target.char_span[1]] != target.surface:
                raise ValueError("target span does not match surface")
            positions = [
                index
                for index, (start, end) in enumerate(content_offsets)
                if start < target.char_span[1] and end > target.char_span[0]
            ]
            if len(positions) == 0:
                raise ValueError("tokenizer did not preserve target span")
            positions_by_target.append(positions)

        # 全対象を残す最小窓を求め、余った長さを左右の文脈へ均等に配る
        required_start = min(positions[0] for positions in positions_by_target)
        required_end = max(positions[-1] for positions in positions_by_target) + 1
        if required_end - required_start > self.metadata.model_max_length - 2:
            raise TargetWindowOverflowError("all targets do not fit in one model input window")
        context_capacity = self.metadata.model_max_length - 2 - (required_end - required_start)
        window_start = max(0, required_start - context_capacity // 2)
        window_start = min(
            window_start, max(0, len(content_ids) - self.metadata.model_max_length + 2)
        )
        window_end = min(len(content_ids), window_start + self.metadata.model_max_length - 2)
        input_id_values = [
            self._leading_token_id,
            *content_ids[window_start:window_end],
            self._trailing_token_id,
        ]
        target_mask = np.zeros((1, len(ordered_targets), len(input_id_values)), dtype=np.bool_)
        for target_index, positions in enumerate(positions_by_target):
            shifted_positions = [position - window_start + 1 for position in positions]
            if any(
                position <= 0 or position >= len(input_id_values) - 1
                for position in shifted_positions
            ):
                raise ValueError("target span was truncated from the shared model input")
            target_mask[0, target_index, shifted_positions] = True
        input_ids = np.asarray([input_id_values], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        logits = self._run_onnx_session(model_inputs)
        class_logits = np.asarray(logits, dtype=np.float32)[0]
        class_index_by_id = {
            class_id: index for index, class_id in enumerate(self.metadata.output_class_order)
        }
        predictions_by_span: dict[tuple[int, int], ReadingPrediction] = {}
        for target, target_logits in zip(ordered_targets, class_logits, strict=True):
            buckets = self.metadata.reading_class_ids_by_surface_and_pronunciation[target.surface]
            scores: list[float] = []
            for pronunciation in target.pronunciations:
                indices = [class_index_by_id[class_id] for class_id in buckets[pronunciation]]
                values = target_logits[indices]
                maximum = float(np.max(values))
                scores.append(
                    maximum
                    + math.log(float(np.exp(values - maximum).sum()))
                    - math.log(len(indices))
                )
            selected_index = int(np.argmax(np.asarray(scores)))
            predictions_by_span[target.char_span] = ReadingPrediction(
                target.pronunciations[selected_index],
                tuple(scores),
            )
        # 呼び出し側の targets 順を保つため、内部では文字位置順に並べ替えてから元順で返す
        return tuple(predictions_by_span[target.char_span] for target in targets)


_lifecycle_lock = Lock()
_loaded_model: TsqyomiModel | None = None


def _resolve_onnx_providers(
    onnxruntime: Any,
    onnx_providers: Sequence[ONNXProvider] | None,
) -> list[ONNXProvider]:
    """
    要求された実行プロバイダを検査する。

    Args:
        onnxruntime (Any): import 済みの ONNX Runtime モジュール
        onnx_providers (Sequence[ONNXProvider] | None): 利用者が指定した実行プロバイダ順

    Returns:
        list[ONNXProvider]: ONNX Runtime へ渡す実行プロバイダ設定

    Raises:
        RuntimeError: 指定した実行プロバイダが利用できない場合
    """

    available_providers = set(onnxruntime.get_available_providers())
    # 自動選択では CUDA を優先し、CPU を実行不能時の次候補として残す
    if onnx_providers is None:
        requested_providers: list[ONNXProvider] = []
        if "CUDAExecutionProvider" in available_providers:
            requested_providers.append("CUDAExecutionProvider")
        requested_providers.append("CPUExecutionProvider")
    else:
        requested_providers = list(onnx_providers)

    missing_providers = [
        provider if isinstance(provider, str) else provider[0]
        for provider in requested_providers
        if (provider if isinstance(provider, str) else provider[0]) not in available_providers
    ]
    if len(missing_providers) > 0:
        raise RuntimeError(
            f"requested ONNX Runtime Execution Providers are unavailable: {missing_providers}"
        )
    return requested_providers


def _load_model_from_paths(
    model_path: Path,
    tokenizer_path: Path,
    metadata_path: Path,
    onnx_providers: Sequence[ONNXProvider] | None,
) -> TsqyomiModel:
    """
    ダウンロード済みのモデルファイルからモデルを構築する。

    Args:
        model_path (Path): ONNX モデルのパス
        tokenizer_path (Path): トークナイザー JSON のパス
        metadata_path (Path): メタデータ JSON のパス
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の実行プロバイダ順

    Returns:
        TsqyomiModel: 構築したモデル

    Raises:
        ImportError: tsqyomi または ONNX Runtime の追加依存が導入されていない場合
    """

    # 通常の G2P 利用ではモデル推論用の追加依存を読み込まない
    try:
        import onnxruntime
    except ImportError as ex:
        raise ImportError(
            "tsqyomi requires ONNX Runtime; install `pyopenjtalk-plus[onnxruntime]` for CPU "
            "or install an ONNX Runtime variant suitable for the execution environment"
        ) from ex
    try:
        from tokenizers import Tokenizer
    except ImportError as ex:
        raise ImportError(
            "tsqyomi requires optional dependencies; install `pyopenjtalk-plus[tsqyomi]`"
        ) from ex

    resolved_providers = _resolve_onnx_providers(onnxruntime, onnx_providers)
    metadata = TsqyomiMetadata.load(metadata_path)
    # 固定リビジョンから個別に取得した各ファイルを、その実パスからロードする
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # 学習時の右側切り捨て設定が残っていても、256部分語の対象中央窓を作る前の系列長を測る
    ## 対象を失わない窓は `predict()` が構築するため、トークナイザー側の自動切り捨ては使用しない
    tokenizer.no_truncation()
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=resolved_providers,
    )
    TsqyomiModel.validate_onnx_contract(session, metadata)
    return TsqyomiModel(
        tokenizer,
        session,
        metadata,
    )


def load_model(
    onnx_providers: Sequence[ONNXProvider] | None = None,
    cache_dir: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
) -> None:
    """
    tsqyomi モデルを取得するかローカルディレクトリからプロセス全体へロードする。
    サーバーではリクエスト受付前に呼び出しを完了させ、ダウンロードと ONNX 初期化を起動処理内で済ませる。
    Hugging Face Hub の応答待ちは HF_HUB_ETAG_TIMEOUT と HF_HUB_DOWNLOAD_TIMEOUT で調整できる。

    Args:
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の Execution Provider 順。
            None のときは CUDA が利用可能なら CUDA、続けて CPU を選ぶ
        cache_dir (str | Path | None): Hugging Face Hub のキャッシュディレクトリ
        model_dir (str | Path | None): デバッグと固定評価に使うローカルモデルディレクトリ

    Raises:
        ImportError: tsqyomi / ONNX Runtime / huggingface_hub の追加依存が導入されていない場合
        RuntimeError: 指定した Execution Provider が利用できない場合
        FileNotFoundError: `model_dir` に必須アセットが無い場合
    """

    global _loaded_model

    with _lifecycle_lock:
        # 同じモデルを繰り返し取得しない
        if _loaded_model is not None:
            return

        # ローカル配置のモデルも Hub 配布のモデルも同じ ONNX 検証を通す
        if model_dir is not None:
            directory = Path(model_dir)
            model_path = directory / "model.onnx"
            tokenizer_path = directory / "tokenizer.json"
            metadata_path = directory / "metadata.json"
            for asset_path in (model_path, tokenizer_path, metadata_path):
                if asset_path.is_file() is False:
                    raise FileNotFoundError(f"tsqyomi model asset does not exist: {asset_path}")
            _loaded_model = _load_model_from_paths(
                model_path,
                tokenizer_path,
                metadata_path,
                onnx_providers,
            )
            return

        # オプションの依存関係である huggingface_hub を遅延インポート
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as ex:
            raise ImportError(
                "tsqyomi requires optional dependencies; install `pyopenjtalk-plus[tsqyomi]`"
            ) from ex

        # モデル・トークナイザー・メタデータの3ファイルを同じリビジョンから取得する
        downloaded_assets = {
            asset_name: Path(
                hf_hub_download(
                    repo_id=_MODEL_REPOSITORY,
                    filename=filename,
                    revision=_MODEL_REVISION,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            )
            for asset_name, filename in _MODEL_FILES.items()
        }

        _loaded_model = _load_model_from_paths(
            downloaded_assets["model"],
            downloaded_assets["tokenizer"],
            downloaded_assets["metadata"],
            onnx_providers,
        )


def unload_model() -> None:
    """
    プロセス全体の tsqyomi モデル参照を解除する。
    """

    global _loaded_model

    with _lifecycle_lock:
        _loaded_model = None


def is_model_loaded() -> bool:
    """
    tsqyomi モデルがロード済みか返す。

    Returns:
        bool: モデルがロード済みなら True
    """

    with _lifecycle_lock:
        return _loaded_model is not None


def get_loaded_model() -> TsqyomiModel:
    """
    開始済み推論が保持できるロード済みモデル参照を返す。

    Returns:
        TsqyomiModel: 現在ロードされているモデル

    Raises:
        RuntimeError: モデルが明示的にロードされていない場合
    """

    # ロック内で参照だけを取得し、開始済み推論は unload_model() 後も同じモデルを保持する
    with _lifecycle_lock:
        if _loaded_model is None:
            raise RuntimeError(
                "tsqyomi model is not loaded; call pyopenjtalk.tsqyomi.load_model() first"
            )
        return _loaded_model
