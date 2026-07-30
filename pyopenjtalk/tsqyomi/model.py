"""tsqyomi v1 の配布資材取得、検証、推論ライフサイクルを管理する。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Literal, TypedDict, Union, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tokenizers import Tokenizer

from .context import build_model_context


_MODEL_REPOSITORY = "tsukumijima/tsqyomi-models"
_MODEL_REVISION = "ad4e0693dfd1821acf0a01b9886fc5ebe5b484af"
_MODEL_FILES = {
    "model": (
        "v1/model.onnx",
        "315a62d7c2a9f55f79d782983b531eb694d58ad45b0604838c497f8d13de1ffc",
    ),
    "tokenizer": (
        "v1/tokenizer.json",
        "07f86a623ad7603e734afc3aac6282451cdcbb83ace6c18f6b235f21bd21ec16",
    ),
    "metadata": (
        "v1/metadata.json",
        "7b4a932efdc50d04ae837e24135d1b987078d7396e9a0ba2b010068f62afb64e",
    ),
}
_MAX_CANDIDATES_PER_BATCH = 4

ONNXProvider = Union[str, tuple[str, dict[str, Any]]]


class TsqyomiCandidateScore(TypedDict):
    """
    tsqyomi が候補発音へ付けた採点結果。
    """

    pronunciation: str  # 入力順を維持した候補発音
    logit: float  # ONNX モデルが候補へ付けた未正規化スコア
    relative_cost: float  # 最高ロジットとの差から計算した MeCab への加算コスト


class _TsqyomiMetadata(BaseModel):
    """配布モデルと推論処理の契約を表す metadata スキーマ。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["modernbert_candidate_ranker_v1"]
    tokenizer_type: Literal["tokenizers_json"]
    precision: Literal["mixed_fp16"]
    model_max_length: int = Field(gt=0)
    model_vocabulary_size: int = Field(gt=0)
    cls_token_id: int = Field(ge=0)
    sep_token_id: int = Field(ge=0)
    unknown_token_id: int = Field(ge=0)
    pad_token_id: int = Field(ge=0)
    cost_weight: float = Field(gt=0.0)
    cost_mode: Literal["relative"]
    confidence_margin: float = Field(ge=0.0, le=0.0)
    rescorer_threshold: float = Field(ge=0.0, le=0.0)
    model_scored_surfaces: tuple[str, ...]
    input_names: tuple[Literal["input_ids"], Literal["attention_mask"]]
    output_names: tuple[Literal["candidate_logits"]]

    @model_validator(mode="after")
    def validate_scored_surfaces(self) -> _TsqyomiMetadata:
        """v1 が介入する表層385件の空文字と重複を拒否する。

        Returns:
            _TsqyomiMetadata: 表層集合を検証した metadata
        """

        # 件数だけ合う重複リストも受理すると metadata と早期判定の契約が食い違う
        if len(self.model_scored_surfaces) != 385 or len(set(self.model_scored_surfaces)) != 385:
            raise ValueError("tsqyomi v1 metadata must contain 385 unique scored surfaces")
        if any(surface == "" for surface in self.model_scored_surfaces):
            raise ValueError("tsqyomi model_scored_surfaces must not contain an empty string")
        return self

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> _TsqyomiMetadata:
        """
        配布 metadata を検証して読み込む。

        Args:
            path (Path): metadata JSON のパス

        Returns:
            _TsqyomiMetadata: 検証済みのモデル設定
        """

        return cls.model_validate_json(path.read_bytes())


@dataclass(frozen=True)
class _LoadConfiguration:
    """多重ロード時に同一設定か判定するための正規化済み設定。"""

    onnx_providers: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None
    cache_dir: str | None


@dataclass(frozen=True)
class _RuntimeDependencies:
    """明示ロード時だけ取得する推論依存ライブラリ。"""

    onnxruntime: Any
    tokenizer_class: Any


def _load_runtime_dependencies() -> _RuntimeDependencies:
    """明示ロード時だけ ONNX Runtime を import する。

    Returns:
        _RuntimeDependencies: import 済みの推論依存ライブラリ

    Raises:
        ImportError: ONNX Runtime が導入されていない場合
    """

    # ONNX Runtime の各バリアントは排他的なので、利用者が選んだパッケージ名には依存しない
    try:
        import onnxruntime
    except ImportError as ex:
        raise ImportError(
            "tsqyomi requires ONNX Runtime; install `pyopenjtalk-plus[onnxruntime]` for CPU "
            "or install an ONNX Runtime variant suitable for the execution environment"
        ) from ex
    return _RuntimeDependencies(onnxruntime=onnxruntime, tokenizer_class=Tokenizer)


class _TsqyomiModel:
    """
    トークナイザー、ONNX セッション、モデル設定を1組で保持する。
    """

    def __init__(
        self,
        tokenizer: Any,
        session: Any,
        metadata: _TsqyomiMetadata,
        is_inference_serialized: bool,
    ) -> None:
        """推論に必要な資材を1つのモデル参照へまとめる。

        Args:
            tokenizer (Any): `tokenizers.Tokenizer` のロード済みインスタンス
            session (Any): `onnxruntime.InferenceSession` のロード済みインスタンス
            metadata (_TsqyomiMetadata): 検証済みのモデル設定
            is_inference_serialized (bool): 同一セッションの推論を直列化するか
        """

        self.tokenizer = tokenizer
        self.session = session
        self.metadata = metadata
        self._inference_lock = Lock() if is_inference_serialized is True else None

    def score_candidates(
        self,
        text: str,
        char_span: tuple[int, int],
        candidate_pronunciations: Sequence[str],
    ) -> list[TsqyomiCandidateScore]:
        """
        製品経路と同じ入力処理で候補発音を採点する。

        Args:
            text (str): 入力本文
            char_span (tuple[int, int]): 対象語の半開区間
            candidate_pronunciations (Sequence[str]): 比較する候補発音

        Returns:
            list[TsqyomiCandidateScore]: 入力順の候補採点結果
        """

        # 候補ごとの系列長を実測し、対象語の周辺だけを最大系列長へ収める
        model_context = build_model_context(
            text,
            char_span,
            candidate_pronunciations,
            lambda context, pronunciation: len(self.tokenizer.encode(context, pronunciation).ids),
            self.metadata.model_max_length,
        )
        logits: list[float] = []
        # ONNX の固定バッチ次元に依存せず、検証済みの最大4候補ずつ推論する
        for batch_start in range(0, len(candidate_pronunciations), _MAX_CANDIDATES_PER_BATCH):
            pronunciation_batch = candidate_pronunciations[
                batch_start : batch_start + _MAX_CANDIDATES_PER_BATCH
            ]
            encodings = [
                self.tokenizer.encode(model_context, pronunciation)
                for pronunciation in pronunciation_batch
            ]
            batch_length = max(len(encoding.ids) for encoding in encodings)
            input_ids = np.full(
                (len(encodings), batch_length),
                self.metadata.pad_token_id,
                dtype=np.int64,
            )
            attention_mask = np.zeros((len(encodings), batch_length), dtype=np.int64)
            # バッチ内の最大系列長へ右側をパディングし、実トークンだけを attention 対象にする
            for encoding_index, encoding in enumerate(encodings):
                input_ids[encoding_index, : len(encoding.ids)] = encoding.ids
                attention_mask[encoding_index, : len(encoding.ids)] = 1
            model_inputs = {
                self.metadata.input_names[0]: input_ids,
                self.metadata.input_names[1]: attention_mask,
            }
            # DirectML だけは同一セッションへの並行 Run() を許可しない
            if self._inference_lock is not None:
                with self._inference_lock:
                    batch_logits = self.session.run(list(self.metadata.output_names), model_inputs)[
                        0
                    ]
            else:
                batch_logits = self.session.run(list(self.metadata.output_names), model_inputs)[0]
            logits.extend(float(logit) for logit in np.asarray(batch_logits).reshape(-1))

        maximum_logit = max(logits)
        return [
            {
                "pronunciation": pronunciation,
                "logit": logit,
                "relative_cost": (maximum_logit - logit) * self.metadata.cost_weight,
            }
            for pronunciation, logit in zip(candidate_pronunciations, logits)
        ]


_lifecycle_lock = RLock()
_loaded_model: _TsqyomiModel | None = None
_loaded_configuration: _LoadConfiguration | None = None


def _configuration_signature(
    onnx_providers: Sequence[ONNXProvider] | None,
    cache_dir: str | Path | None,
) -> _LoadConfiguration:
    """ロード要求を順序比較できる不変な設定へ変換する。

    Args:
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の実行プロバイダ順
        cache_dir (str | Path | None): Hugging Face Hub のキャッシュディレクトリ

    Returns:
        _LoadConfiguration: 比較用に正規化したロード設定
    """

    normalized_providers = None
    # 辞書形式の設定値も順序に依存しない文字列表現へ変え、多重ロード時に比較可能にする
    if onnx_providers is not None:
        provider_signatures: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for provider in onnx_providers:
            if isinstance(provider, str):
                provider_signatures.append((provider, ()))
            else:
                provider_name, provider_options = provider
                provider_signatures.append(
                    (
                        provider_name,
                        tuple(
                            sorted((key, repr(value)) for key, value in provider_options.items())
                        ),
                    )
                )
        normalized_providers = tuple(provider_signatures)
    return _LoadConfiguration(
        onnx_providers=normalized_providers,
        cache_dir=None if cache_dir is None else str(Path(cache_dir)),
    )


def _resolve_onnx_providers(
    onnxruntime: Any,
    onnx_providers: Sequence[ONNXProvider] | None,
) -> list[ONNXProvider]:
    """要求された実行プロバイダから利用可能な設定だけを構築する。

    Args:
        onnxruntime (Any): import 済みの ONNX Runtime モジュール
        onnx_providers (Sequence[ONNXProvider] | None): 利用者が指定した実行プロバイダ順

    Returns:
        list[ONNXProvider]: ONNX Runtime へ渡す利用可能な実行プロバイダ設定

    Raises:
        RuntimeError: 利用可能な実行プロバイダが1つもない場合
        ValueError: CUDA の再現性を損なう設定が指定された場合
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

    resolved_providers: list[ONNXProvider] = []
    # 利用者が指定した順序を維持し、現在の ONNX Runtime にない候補だけを除外する
    for provider in requested_providers:
        provider_name = provider if isinstance(provider, str) else provider[0]
        if provider_name not in available_providers:
            continue
        provider_options = {} if isinstance(provider, str) else dict(provider[1])
        if provider_name == "CUDAExecutionProvider":
            configured_tf32 = provider_options.get("use_tf32")
            if configured_tf32 is not None and str(configured_tf32) != "0":
                raise ValueError("CUDAExecutionProvider use_tf32 must be 0 for tsqyomi")
            provider_options["use_tf32"] = "0"
            provider_options["arena_extend_strategy"] = "kSameAsRequested"
        resolved_providers.append(
            provider_name if len(provider_options) == 0 else (provider_name, provider_options)
        )
    if len(resolved_providers) == 0:
        raise RuntimeError("none of the requested ONNX Runtime Execution Providers are available")
    return resolved_providers


def _load_model_from_directory(
    model_dir: Path,
    onnx_providers: Sequence[ONNXProvider] | None,
    runtime_dependencies: _RuntimeDependencies | None = None,
) -> _TsqyomiModel:
    """
    検証済みの取得済み資材からモデルを構築する。

    Args:
        model_dir (Path): `model.onnx`、`tokenizer.json`、`metadata.json` の格納先
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の実行プロバイダ順
        runtime_dependencies (_RuntimeDependencies | None): 取得済みの遅延 import 結果

    Returns:
        _TsqyomiModel: 構築したモデル
    """

    dependencies = (
        _load_runtime_dependencies() if runtime_dependencies is None else runtime_dependencies
    )
    onnxruntime = dependencies.onnxruntime

    resolved_providers = _resolve_onnx_providers(onnxruntime, onnx_providers)
    session_options = onnxruntime.SessionOptions()
    # CPU メモリアリーナは長文の推論後も再利用でき、実測で無効化する根拠がなかったため維持する
    session_options.enable_cpu_mem_arena = True
    model_path = model_dir / "model.onnx"
    metadata_path = model_dir / "metadata.json"
    metadata = _TsqyomiMetadata.load(metadata_path)
    # metadata と同じディレクトリのトークナイザーと ONNX だけを組み合わせる
    tokenizer = dependencies.tokenizer_class.from_file(str(model_dir / "tokenizer.json"))
    # 学習時の右側切り捨てが資材に残っていても、対象中央窓を作る前の系列長を正しく測る
    ## `build_model_context()` が512トークン以内へ縮めるため、推論時の自動切り捨ては使用しない
    tokenizer.no_truncation()
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=resolved_providers,
    )
    session_input_names = tuple(model_input.name for model_input in session.get_inputs())
    session_output_names = tuple(model_output.name for model_output in session.get_outputs())
    # ファイルのハッシュが正しくても入出力契約が違うモデルは推論開始前に拒否する
    if session_input_names != metadata.input_names or session_output_names != metadata.output_names:
        raise ValueError("tsqyomi ONNX inputs or outputs do not match metadata")
    active_providers = session.get_providers()
    return _TsqyomiModel(
        tokenizer,
        session,
        metadata,
        is_inference_serialized="DmlExecutionProvider" in active_providers,
    )


def load_model(
    onnx_providers: Sequence[ONNXProvider] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """
    固定リビジョンの tsqyomi v1 モデルを取得してプロセス全体へロードする。

    Args:
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の実行プロバイダ順
        cache_dir (str | Path | None): Hugging Face Hub のキャッシュディレクトリ

    Raises:
        ImportError: ONNX Runtime が導入されていない場合
        RuntimeError: 異なる設定のモデルがロード済みか、利用可能な実行プロバイダがない場合
        ValueError: 配布 metadata または ONNX の入出力契約が不正な場合
    """

    global _loaded_configuration, _loaded_model

    requested_configuration = _configuration_signature(onnx_providers, cache_dir)
    with _lifecycle_lock:
        # 同じ設定は冪等に扱い、実行プロバイダやキャッシュ先の暗黙変更は拒否する
        if _loaded_model is not None:
            if _loaded_configuration == requested_configuration:
                return
            raise RuntimeError(
                "a tsqyomi model with different settings is already loaded; call unload_model() first"
            )

        # 数十 MB のモデル取得後に実行環境の不足へ気付く無駄を避けるため、実行依存を先に検査する
        runtime_dependencies = _load_runtime_dependencies()
        # Hugging Face Hub は import が重いため、通常の pyopenjtalk import から分離する
        from huggingface_hub import hf_hub_download

        downloaded_assets: dict[str, Path] = {}
        # 固定した3資材を同じリビジョンから取得し、各ファイルを個別に検証する
        for asset_name, (filename, expected_sha256) in _MODEL_FILES.items():
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=_MODEL_REPOSITORY,
                    filename=filename,
                    revision=_MODEL_REVISION,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            )
            # 大容量の ONNX もメモリへ全読み込みせず、固定リビジョンの SHA-256 と照合する
            sha256 = hashlib.sha256()
            with downloaded_path.open("rb") as asset_file:
                while chunk := asset_file.read(1024 * 1024):
                    sha256.update(chunk)
            if sha256.hexdigest() != expected_sha256:
                raise RuntimeError(
                    f"tsqyomi model asset has an invalid SHA-256: {downloaded_path.name}"
                )
            downloaded_assets[asset_name] = downloaded_path

        # Hugging Face のスナップショットでは3資材が同じ v1 ディレクトリへ配置される
        model_dir = downloaded_assets["model"].parent
        if any(asset_path.parent != model_dir for asset_path in downloaded_assets.values()):
            raise RuntimeError("tsqyomi model assets were downloaded from inconsistent snapshots")
        _loaded_model = _load_model_from_directory(
            model_dir,
            onnx_providers,
            runtime_dependencies=runtime_dependencies,
        )
        _loaded_configuration = requested_configuration


def unload_model() -> None:
    """
    プロセス全体の tsqyomi モデル参照を解除する。
    """

    global _loaded_configuration, _loaded_model

    with _lifecycle_lock:
        _loaded_model = None
        _loaded_configuration = None


def is_model_loaded() -> bool:
    """
    tsqyomi モデルがロード済みか返す。

    Returns:
        bool: モデルがロード済みなら True
    """

    with _lifecycle_lock:
        return _loaded_model is not None


def get_loaded_model() -> _TsqyomiModel:
    """開始済み推論が保持できるロード済みモデル参照を返す。

    Returns:
        _TsqyomiModel: 現在ロードされているモデル

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


def score_candidates(
    text: str,
    char_span: tuple[int, int],
    candidate_pronunciations: Sequence[str],
) -> list[TsqyomiCandidateScore]:
    """
    ロード済みの tsqyomi モデルで候補発音を採点する。

    Args:
        text (str): 入力本文
        char_span (tuple[int, int]): 対象語の半開区間
        candidate_pronunciations (Sequence[str]): 比較する2件以上の候補発音

    Returns:
        list[TsqyomiCandidateScore]: 入力順の候補採点結果

    Raises:
        TypeError: 候補発音が文字列のシーケンスでない場合
        ValueError: 対象範囲または候補発音が不正な場合
        RuntimeError: モデルが明示的にロードされていない場合
    """

    # `str` も Sequence なので、候補列として受理すると1文字ずつ別候補として採点されてしまう
    if isinstance(candidate_pronunciations, str) is True:
        raise TypeError("candidate_pronunciations must be a sequence of strings")

    # 呼び出し側の可変シーケンスが推論中に変更されないよう、検証前に tuple へ固定する
    raw_pronunciations = tuple(cast(Sequence[object], candidate_pronunciations))
    if any(isinstance(pronunciation, str) is False for pronunciation in raw_pronunciations):
        raise TypeError("candidate_pronunciations must be a sequence of strings")
    pronunciations = cast(tuple[str, ...], raw_pronunciations)
    if len(pronunciations) < 2:
        raise ValueError("candidate_pronunciations must contain at least two entries")
    if any(pronunciation == "" for pronunciation in pronunciations):
        raise ValueError("candidate_pronunciations must contain non-empty strings")
    return get_loaded_model().score_candidates(text, char_span, pronunciations)
