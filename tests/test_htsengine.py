"""HTS Engine の公開合成経路と既知の終了時不具合を検証する。"""

import subprocess
import sys

import numpy as np

import pyopenjtalk


def test_tts():
    x, sr = pyopenjtalk.tts("こんちゃっす")
    assert x.dtype == np.float64
    assert sr == 48000


def test_tts_speed():
    x, _ = pyopenjtalk.tts("こんちゃっす")

    x_fast, _ = pyopenjtalk.tts("こんちゃっす", speed=1.5)
    assert len(x) > len(x_fast)

    x_slow, _ = pyopenjtalk.tts("こんちゃっす", speed=0.5)
    assert len(x_slow) > len(x)


def test_tts_half_tone():
    x, _ = pyopenjtalk.tts("こんちゃっす")

    # +2
    x_high, _ = pyopenjtalk.tts("こんちゃっす", half_tone=2)
    # -2
    x_low, _ = pyopenjtalk.tts("こんちゃっす", half_tone=-2)

    # half_tone should not change durations
    assert len(x) == len(x_high) == len(x_low)


def test_htsengine():
    labels = pyopenjtalk.extract_fullcontext("こんちゃ")
    x, sr = pyopenjtalk.synthesize(labels)
    assert x.dtype == np.float64
    assert sr == 48000


def test_tts_engine_destruction_does_not_raise_at_interpreter_shutdown() -> None:
    """HTS エンジンを保持したプロセスが終了処理で未処理例外を出さないことを確認する。"""

    # デストラクタは Python の終了処理で初めて呼ばれるため、独立した子プロセスの標準エラーを検査する
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            'import pyopenjtalk; pyopenjtalk.tts("こんにちは。")',
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
