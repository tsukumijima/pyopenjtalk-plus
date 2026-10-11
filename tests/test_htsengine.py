"""HTS Engine の公開合成経路と既知の終了時不具合を検証する。"""

import subprocess
import sys

import numpy as np
import numpy.typing as npt
import pytest

import pyopenjtalk


def test_tts():
    x, sr = pyopenjtalk.tts("こんちゃっす")
    assert x.dtype == np.float64
    assert sr == 48000


@pytest.mark.parametrize("text", ["", " ", "。", "m(_ _)m"])
def test_tts_without_phonemes_returns_empty_waveform(text: str) -> None:
    """
    空文字・空白・句読点・無音の顔文字を tts() に渡してもプロセスが落ちず、float64 の空の波形とサンプリング周波数を返すことを確認する。
    空のラベルを HTS Engine へ渡すと NULL を参照して落ちるため、子プロセスで実際の合成経路を検査し、その後の通常の文の合成も確認する。
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np, pyopenjtalk as p; "
            "x, sr = p.tts(sys.argv[1]); "
            "assert x.shape == (0,) and x.dtype == np.float64 and sr == 48000; "
            "assert p.tts('こんにちは。')[0].size > 0",
            text,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr


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


def test_tts_propagates_use_vanilla(monkeypatch: pytest.MonkeyPatch) -> None:
    """tts() 関数に渡した use_vanilla オプションが、内部の extract_fullcontext() に正しく伝播することを確認する。"""

    captured_options: dict[str, object] = {}

    def capture_fullcontext(_text: str, **options: object) -> list[str]:
        # 音声合成を実行せず、tts() がラベル生成へ渡す公開オプションだけを観測する
        captured_options.update(options)
        return []

    def return_empty_audio(
        _labels: list[str],
        _speed: float,
        _half_tone: float,
    ) -> tuple[npt.NDArray[np.float64], int]:
        return np.array([], dtype=np.float64), 48000

    monkeypatch.setattr(pyopenjtalk, "extract_fullcontext", capture_fullcontext)
    monkeypatch.setattr(pyopenjtalk, "synthesize", return_empty_audio)

    pyopenjtalk.tts("三分の一", use_vanilla=True)

    assert captured_options["use_vanilla"] is True


def test_htsengine():
    labels = pyopenjtalk.extract_fullcontext("こんちゃ")
    x, sr = pyopenjtalk.synthesize(labels)
    assert x.dtype == np.float64
    assert sr == 48000


def test_htsengine_empty_labels_clear_previous_speech() -> None:
    """
    HTSEngine の低レベル API で通常の文を合成した直後に空のラベルを渡すと、前回の波形を残さず空にすることを確認する。
    公開 synthesize([]) も同じ float64 の空の波形を返し、続く通常の合成を妨げない。
    """

    engine = pyopenjtalk.HTSEngine(pyopenjtalk.DEFAULT_HTS_VOICE)
    labels = pyopenjtalk.extract_fullcontext("こんにちは。")
    engine.synthesize_from_strings(labels)
    assert engine.get_generated_speech().size > 0
    engine.synthesize_from_strings([])
    waveform = engine.get_generated_speech()
    assert waveform.shape == (0,)
    assert waveform.dtype == np.float64
    assert engine.synthesize([]).shape == (0,)
    assert engine.synthesize(labels).size > 0
    waveform, sr = pyopenjtalk.synthesize([])
    assert waveform.shape == (0,)
    assert waveform.dtype == np.float64
    assert sr == 48000


def test_tts_engine_destruction_does_not_raise_at_interpreter_shutdown() -> None:
    """HTS エンジンを保持したプロセスが終了処理で未処理例外を出さないことを確認する。"""

    # デストラクタは Python の終了処理で初めて呼ばれるため、独立した子プロセスの未処理例外を検査する
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            'import pyopenjtalk; pyopenjtalk.tts("こんにちは。")',
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Exception ignored in:" not in completed.stderr, completed.stderr
    assert "HTSEngine.__dealloc__" not in completed.stderr, completed.stderr
