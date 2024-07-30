# flake8: noqa

from collections.abc import Collection
from typing import Any

import numpy as np

class HTSEngine:
    def __init__(self, voice: bytes = b"htsvoice/mei_normal.htsvoice") -> None:
        """HTSEngine

        Args:
            voice (bytes): File path of htsvoice.
        """
        pass

    def load(self, voice: bytes) -> bytes: ...
    def get_sampling_frequency(self) -> int:
        """Get sampling frequency"""
        pass

    def get_fperiod(self) -> int:
        """Get frame period"""
        pass

    def set_speed(self, speed: float = 1.0) -> None:
        """Set speed

        Args:
            speed (float): speed
        """
        pass

    def add_half_tone(self, half_tone: float = 0.0) -> None:
        """Additional half tone in log-f0

        Args:
            half_tone (float): additional half tone
        """
        pass

    def synthesize(
        self, labels: Collection[str | bytes | bytearray]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Synthesize waveform from list of full-context labels

        Args:
            labels: full context labels

        Returns:
            np.ndarray: speech waveform
        """
        pass

    def synthesize_from_strings(self, labels: Collection[str | bytes | bytearray]) -> None:
        """Synthesize from strings"""
        pass

    def get_generated_speech(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Get generated speech"""
        pass

    def get_fullcontext_label_format(self) -> str:
        """Get full-context label format"""
        pass

    def refresh(self) -> None: ...
    def clear(self) -> None: ...
