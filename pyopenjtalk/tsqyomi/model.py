"""tsqyomi v1 の配布資材取得と推論を管理する。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict, Union

import numpy as np
from pydantic import BaseModel

from .context import build_model_context


_MODEL_REPOSITORY = "tsukumijima/tsqyomi-models"
# モデル、トークナイザー、metadata の組み合わせを同一スナップショットへ固定する
_MODEL_REVISION = "ad4e0693dfd1821acf0a01b9886fc5ebe5b484af"
_MODEL_FILES = {
    "model": "v1/model.onnx",
    "tokenizer": "v1/tokenizer.json",
    "metadata": "v1/metadata.json",
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
    """推論時に参照する metadata だけを保持する。"""

    model_max_length: int
    pad_token_id: int
    cost_weight: float
    model_scored_surfaces: frozenset[str]
    # 最良候補との差がこの幅に収まる候補はコストを動かさず、辞書の判断へ委ねる
    ## 保留幅を持たない旧世代の配布 metadata では0として扱い、全候補をモデルのコスト差で並べる
    baseline_margin: float = 0.0

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> _TsqyomiMetadata:
        """
        配布 metadata から推論に必要な値を読み込む。

        Args:
            path (Path): metadata JSON のパス

        Returns:
            _TsqyomiMetadata: モデル設定
        """

        # 未定義の配布情報は Pydantic の標準動作で無視し、推論で使う4項目だけを検証する
        return cls.model_validate_json(path.read_bytes())


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
        """
        推論に必要な資材を1つのモデル参照へまとめる。

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

        Raises:
            ValueError: 候補発音が空の場合
        """

        # 空候補はバッチ長と最大ロジットを定義できないため、推論資材へ触れる前に拒否する
        if len(candidate_pronunciations) == 0:
            raise ValueError("candidate_pronunciations must not be empty")

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
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            # DirectML だけは同一セッションへの並行 Run() を許可しない
            if self._inference_lock is not None:
                with self._inference_lock:
                    batch_logits = self.session.run(["candidate_logits"], model_inputs)[0]
            else:
                batch_logits = self.session.run(["candidate_logits"], model_inputs)[0]
            logits.extend(float(logit) for logit in np.asarray(batch_logits).reshape(-1))

        if len(logits) != len(candidate_pronunciations):
            raise RuntimeError(
                "tsqyomi ONNX output size mismatch: "
                f"expected {len(candidate_pronunciations)} logits, got {len(logits)}"
            )

        maximum_logit = max(logits)
        return [
            {
                "pronunciation": pronunciation,
                "logit": logit,
                # 最良候補と拮抗する候補へコストを与えると、確信の低い読み替えが辞書の既定を押しのける
                "relative_cost": (
                    0.0
                    if maximum_logit - logit <= self.metadata.baseline_margin
                    else (maximum_logit - logit) * self.metadata.cost_weight
                ),
            }
            for pronunciation, logit in zip(candidate_pronunciations, logits)
        ]


_lifecycle_lock = Lock()
_loaded_model: _TsqyomiModel | None = None


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
) -> _TsqyomiModel:
    """
    検証済みの取得済み資材からモデルを構築する。

    Args:
        model_path (Path): ONNX モデルのパス
        tokenizer_path (Path): トークナイザー JSON のパス
        metadata_path (Path): metadata JSON のパス
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の実行プロバイダ順

    Returns:
        _TsqyomiModel: 構築したモデル

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
    metadata = _TsqyomiMetadata.load(metadata_path)
    # 固定リビジョンから個別に取得した各資材を、その実パスからロードする
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # 学習時の右側切り捨てが資材に残っていても、対象中央窓を作る前の系列長を正しく測る
    ## `build_model_context()` が512トークン以内へ縮めるため、推論時の自動切り捨ては使用しない
    tokenizer.no_truncation()
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=resolved_providers,
    )
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
    サーバーではリクエスト受付前に呼び出しを完了させ、ダウンロードと ONNX 初期化を起動処理内で済ませる。
    Hugging Face Hub の応答待ちは HF_HUB_ETAG_TIMEOUT と HF_HUB_DOWNLOAD_TIMEOUT で調整できる。

    Args:
        onnx_providers (Sequence[ONNXProvider] | None): ONNX Runtime の実行プロバイダ順
        cache_dir (str | Path | None): Hugging Face Hub のキャッシュディレクトリ

    Raises:
        ImportError: ONNX Runtime が導入されていない場合
        RuntimeError: 指定した実行プロバイダが利用できない場合
    """

    global _loaded_model

    with _lifecycle_lock:
        # 同じモデルを繰り返し取得しない
        if _loaded_model is not None:
            return

        # Hugging Face Hub は import が重いため、通常の pyopenjtalk import から分離する
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as ex:
            raise ImportError(
                "tsqyomi requires optional dependencies; install `pyopenjtalk-plus[tsqyomi]`"
            ) from ex

        # 固定した3資材を同じリビジョンから取得する
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


def get_loaded_model() -> _TsqyomiModel:
    """
    開始済み推論が保持できるロード済みモデル参照を返す。

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
    pronunciations = tuple(candidate_pronunciations)
    if len(pronunciations) < 2:
        raise ValueError("candidate_pronunciations must contain at least two entries")
    if any(pronunciation == "" for pronunciation in pronunciations):
        raise ValueError("candidate_pronunciations must contain non-empty strings")
    return get_loaded_model().score_candidates(text, char_span, pronunciations)
