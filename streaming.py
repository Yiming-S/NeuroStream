"""
streaming.py — Generator-based trial delivery for streaming simulation.
"""

from typing import Generator, Optional, Tuple

import numpy as np


class StreamingSimulator:
    """
    Simulates a real-time EEG data stream by iterating through pre-loaded
    test-subject epochs one trial at a time.

    Each call to next_trial() yields:
      (epoch_array, true_label)
      epoch_array : (n_channels, n_times)
      true_label  : int  — ground-truth class index

    The generator pattern decouples the data source from the UI loop.
    """

    CLASS_NAMES = {0: "LEFT", 1: "RIGHT"}

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self._gen: Optional[Generator] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Restart the stream from the first trial."""
        self._gen = self._stream()

    def _stream(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        for i in range(len(self.X)):
            yield self.X[i], int(self.y[i])

    def next_trial(self) -> Optional[Tuple[np.ndarray, int]]:
        """Advance by one trial. Returns None when the stream is exhausted."""
        if self._gen is None:
            self.reset()
        try:
            return next(self._gen)
        except StopIteration:
            return None

    # ------------------------------------------------------------------
    @classmethod
    def label_name(cls, label: int) -> str:
        return cls.CLASS_NAMES.get(label, f"CLASS_{label}")
