# coding: utf-8
# cython: boundscheck=True, wraparound=True
# cython: c_string_type=unicode, c_string_encoding=ascii
# cython: language_level=3
# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingParameterType=false
# pyright: reportMissingTypeArgument=false
# pyright: reportUnknownParameterType=false

from contextlib import contextmanager
from threading import RLock

import numpy as np

cimport numpy as np
np.import_array()

cimport cython
from libc.stdlib cimport malloc, free

from .htsengine cimport HTS_Engine
from .htsengine cimport (
    HTS_Engine_initialize, HTS_Engine_load, HTS_Engine_clear, HTS_Engine_refresh,
    HTS_Engine_get_sampling_frequency, HTS_Engine_get_fperiod,
    HTS_Engine_set_speed, HTS_Engine_add_half_tone,
    HTS_Engine_synthesize_from_strings,
    HTS_Engine_get_generated_speech, HTS_Engine_get_nsamples
)

def _generate_lock_manager():
    lock = RLock()

    @contextmanager
    def f():
        with lock:
            yield

    return f


cdef class HTSEngine:
    """
    HTS 音声合成エンジンの Cython 実装。フルコンテキストラベルから波形を生成する。
    通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

    Args:
        voice (bytes): htsvoice ファイルのパス。デフォルトは mei_normal.htsvoice。
    """
    cdef HTS_Engine* engine
    _lock_manager = _generate_lock_manager()

    def __cinit__(self, bytes voice=b"htsvoice/mei_normal.htsvoice"):
        self.engine = new HTS_Engine()

        HTS_Engine_initialize(self.engine)

        if self.load(voice) != 1:
            self.clear()
            raise RuntimeError("Failed to initialize HTS_Engine")

    @_lock_manager()
    def load(self, bytes voice):
        """
        htsvoice ファイルを読み込む。

        Args:
            voice (bytes): htsvoice ファイルのパス。

        Returns:
            int: 成功時 1、失敗時 0。
        """
        cdef char* voices = voice
        cdef char ret
        with nogil:
            ret = HTS_Engine_load(self.engine, &voices, 1)
        return ret

    @_lock_manager()
    def get_sampling_frequency(self):
        """
        サンプリング周波数を取得する。

        Returns:
            int: サンプリング周波数 (Hz)。通常は 48000。
        """
        return HTS_Engine_get_sampling_frequency(self.engine)

    @_lock_manager()
    def get_fperiod(self):
        """
        フレーム周期を取得する。

        Returns:
            int: フレーム周期 (サンプル数)。
        """
        return HTS_Engine_get_fperiod(self.engine)

    @_lock_manager()
    def set_speed(self, speed=1.0):
        """
        話速を設定する。

        Args:
            speed (float): 話速倍率。1.0 が等倍。デフォルトは 1.0。
        """
        HTS_Engine_set_speed(self.engine, speed)

    @_lock_manager()
    def add_half_tone(self, half_tone=0.0):
        """
        基本周波数 (F0) に半音を追加する。

        Args:
            half_tone (float): 追加する半音数。0.0 が無変更。デフォルトは 0.0。
        """
        HTS_Engine_add_half_tone(self.engine, half_tone)

    @_lock_manager()
    def synthesize(self, list labels):
        """
        フルコンテキストラベルから音声波形を合成する。
        synthesize_from_strings() を呼び出し、生成された波形を返す。
        内部で refresh() が呼ばれるため、連続合成時は set_speed() / add_half_tone() を毎回設定する必要がある。

        Args:
            labels (list[str]): フルコンテキストラベル文字列のリスト。

        Returns:
            np.ndarray: 音声波形 (dtype: np.float64)。
        """
        self.synthesize_from_strings(labels)
        x = self.get_generated_speech()
        self.refresh()
        return x

    @_lock_manager()
    def synthesize_from_strings(self, list labels):
        """
        フルコンテキストラベル文字列から波形を合成する。低レベル API。
        波形は内部バッファに格納され、get_generated_speech() で取得する。
        失敗時は RuntimeError を送出する。

        Args:
            labels (list[str]): フルコンテキストラベル文字列のリスト。

        Raises:
            RuntimeError: 合成に失敗した場合。
        """
        cdef size_t num_lines = len(labels)
        cdef char **lines = <char**> malloc((num_lines + 1) * sizeof(char*))
        for n in range(num_lines):
            lines[n] = <char*>labels[n]

        cdef char ret
        with nogil:
            ret = HTS_Engine_synthesize_from_strings(self.engine, lines, num_lines)
            free(lines)
        if ret != 1:
            raise RuntimeError("Failed to run synthesize_from_strings")

    @_lock_manager()
    def get_generated_speech(self):
        """
        合成済み音声波形を取得する。
        synthesize_from_strings() 実行後に呼び出す。
        取得後は refresh() で内部バッファをクリアすること。

        Returns:
            np.ndarray: 音声波形 (dtype: np.float64)。
        """
        cdef size_t nsamples = HTS_Engine_get_nsamples(self.engine)
        cdef np.ndarray speech = np.empty([nsamples], dtype=np.float64)
        cdef double[:] speech_view = speech
        cdef size_t index
        with (nogil, cython.boundscheck(False)):
            for index in range(nsamples):
                speech_view[index] = HTS_Engine_get_generated_speech(self.engine, index)
        return speech

    @_lock_manager()
    def get_fullcontext_label_format(self):
        """
        使用中のフルコンテキストラベルフォーマットを取得する。

        Returns:
            str: ラベルフォーマット名 (UTF-8 デコード済み)。
        """
        return (<bytes>HTS_Engine_get_fullcontext_label_format(self.engine)).decode("utf-8")

    @_lock_manager()
    def refresh(self):
        """
        内部バッファをクリアする。
        synthesize_from_strings() 後に get_generated_speech() で波形を取得したら、
        次回合成前に refresh() を呼ぶ必要がある。
        """
        HTS_Engine_refresh(self.engine)

    @_lock_manager()
    def clear(self):
        """
        ロード済みの htsvoice を解放し、エンジンを初期状態に戻す。
        """
        HTS_Engine_clear(self.engine)

    def __dealloc__(self):
        self.clear()
        del self.engine
