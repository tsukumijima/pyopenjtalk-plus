from collections.abc import Collection
from typing import Any

import numpy as np

class HTSEngine:  # NOQA
    def __init__(self, voice: bytes = b"htsvoice/mei_normal.htsvoice") -> None:
        """HTSEngine

        Args:
            voice (bytes): File path of htsvoice.
        """
        pass
    def load(self, voice: bytes) -> bytes: ...  # NOQA
    def get_sampling_frequency(self) -> int:  # NOQA
        """Get sampling frequency"""
        pass
    def get_fperiod(self) -> int:  # NOQA
        """Get frame period"""
        pass
    def set_speed(self, speed: float = 1.0) -> None:  # NOQA
        """Set speed

        Args:
            speed (float): speed
        """
        pass
    def add_half_tone(self, half_tone: float = 0.0) -> None:  # NOQA
        """Additional half tone in log-f0

        Args:
            half_tone (float): additional half tone
        """
        pass
    def synthesize(  # NOQA
        self, labels: Collection[str | bytes | bytearray]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Synthesize waveform from list of full-context labels

        Args:
            labels: full context labels

        Returns:
            np.ndarray: speech waveform
        """
        pass
    def synthesize_from_strings(  # NOQA
        self, labels: Collection[str | bytes | bytearray]
    ) -> None:
        """Synthesize from strings"""
        pass
    def get_generated_speech(self) -> np.ndarray[Any, np.dtype[np.float64]]:  # NOQA
        """Get generated speech"""
        pass
    def get_fullcontext_label_format(self) -> str:  # NOQA
        """Get full-context label format"""
        pass
    def refresh(self) -> None: ...  # NOQA
    def clear(self) -> None: ...  # NOQA
