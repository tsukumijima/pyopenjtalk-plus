# flake8: noqa

from collections.abc import Collection
from typing import Any

import numpy as np

class HTSEngine:
    def __init__(self, voice: bytes = b"htsvoice/mei_normal.htsvoice") -> None:
        """
        HTS 音声合成エンジンの Cython 実装。フルコンテキストラベルから波形を生成する。
        通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

        Args:
            voice (bytes): htsvoice ファイルのパス。デフォルトは mei_normal.htsvoice。
        """
        pass

    def load(self, voice: bytes) -> int:
        """
        htsvoice ファイルを読み込む。

        Args:
            voice (bytes): htsvoice ファイルのパス。

        Returns:
            int: 成功時 1、失敗時 0。
        """
        ...

    def get_sampling_frequency(self) -> int:
        """
        サンプリング周波数を取得する。

        Returns:
            int: サンプリング周波数 (Hz)。通常は 48000。
        """
        ...

    def get_fperiod(self) -> int:
        """
        フレーム周期を取得する。

        Returns:
            int: フレーム周期 (サンプル数)。
        """
        ...

    def set_speed(self, speed: float = 1.0) -> None:
        """
        話速を設定する。

        Args:
            speed (float): 話速倍率。1.0 が等倍。デフォルトは 1.0。
        """
        ...

    def add_half_tone(self, half_tone: float = 0.0) -> None:
        """
        基本周波数 (F0) に半音を追加する。

        Args:
            half_tone (float): 追加する半音数。0.0 が無変更。デフォルトは 0.0。
        """
        ...

    def synthesize(
        self, labels: Collection[str | bytes | bytearray]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        フルコンテキストラベルから音声波形を合成する。
        synthesize_from_strings() を呼び出し、生成された波形を返す。
        内部で refresh() が呼ばれるため、連続合成時は set_speed() / add_half_tone() を毎回設定する必要がある。

        Args:
            labels (Collection[str | bytes | bytearray]): フルコンテキストラベル文字列のコレクション。

        Returns:
            np.ndarray: 音声波形 (dtype: np.float64)。
        """
        ...

    def synthesize_from_strings(self, labels: Collection[str | bytes | bytearray]) -> None:
        """
        フルコンテキストラベル文字列から波形を合成する。低レベル API。
        波形は内部バッファに格納され、get_generated_speech() で取得する。
        失敗時は RuntimeError を送出する。

        Args:
            labels (Collection[str | bytes | bytearray]): フルコンテキストラベル文字列のコレクション。

        Raises:
            RuntimeError: 合成に失敗した場合。
        """
        ...

    def get_generated_speech(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        合成済み音声波形を取得する。
        synthesize_from_strings() 実行後に呼び出す。
        取得後は refresh() で内部バッファをクリアすること。

        Returns:
            np.ndarray: 音声波形 (dtype: np.float64)。
        """
        ...

    def get_fullcontext_label_format(self) -> str:
        """
        使用中のフルコンテキストラベルフォーマットを取得する。

        Returns:
            str: ラベルフォーマット名 (UTF-8 デコード済み)。
        """
        ...

    def refresh(self) -> None:
        """
        内部バッファをクリアする。
        synthesize_from_strings() 後に get_generated_speech() で波形を取得したら、
        次回合成前に refresh() を呼ぶ必要がある。
        """
        ...

    def clear(self) -> None:
        """
        ロード済みの htsvoice を解放し、エンジンを初期状態に戻す。
        """
        ...
