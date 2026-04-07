"""
sources.py — Sample-level EEG data sources, ring buffer, and cross-thread
event types for the online architecture.

Threading model
───────────────
  ReplaySource.read_chunk()  is called from the acquisition worker thread.
  RingBuffer.write()         is called from the same worker thread.
  RingBuffer.read()          may be called from any thread (lock-protected).
  ProgressEvent / TrialResult are put into queue.Queue by the worker and
  consumed by the Tk main thread.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Cross-thread event types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProgressEvent:
    """Tentative prediction at a progressive sub-window."""
    trial_idx: int
    n_samples: int
    pred_label: int
    proba: np.ndarray           # shape (2,)
    epoch_slice: np.ndarray     # shape (n_channels, n_samples)


@dataclass
class TrialResult:
    """Final prediction for one completed trial."""
    trial_idx: int
    true_label: Optional[int]   # None when ground truth is unavailable
    pred_label: int
    proba: np.ndarray           # shape (2,)
    epoch: np.ndarray           # shape (n_channels, n_times)
    correct_so_far: int
    total_so_far: int
    conf_matrix_snapshot: List[List[int]]   # deep-copied 2×2


# ══════════════════════════════════════════════════════════════════════════════
# SampleSource (abstract)
# ══════════════════════════════════════════════════════════════════════════════

class SampleSource(ABC):
    """
    Minimal sample-level EEG source interface.

    Implementations must be non-blocking: read_chunk() returns immediately
    with whatever new data is available (or None).
    """

    @abstractmethod
    def start(self) -> None:
        """Begin acquisition / open connection."""

    @abstractmethod
    def read_chunk(self) -> Optional[np.ndarray]:
        """Return (n_channels, n_new_samples) or None if nothing new."""

    @abstractmethod
    def get_sfreq(self) -> float:
        """Sampling frequency in Hz."""

    @abstractmethod
    def get_n_channels(self) -> int:
        """Number of EEG channels."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition / close connection."""

    @abstractmethod
    def is_exhausted(self) -> bool:
        """True when the source has no more data. Always False for live."""


# ══════════════════════════════════════════════════════════════════════════════
# ReplaySource
# ══════════════════════════════════════════════════════════════════════════════

class ReplaySource(SampleSource):
    """
    Replays pre-loaded epoch data as a wall-clock-paced sample stream.

    Internally builds a continuous signal:
        trial_0 | gap(zeros) | trial_1 | gap | … | trial_N

    read_chunk() releases samples at real-time speed (1× sfreq).
    pause() / resume() freeze / thaw the wall-clock so pausing the UI
    does not cause a burst of buffered data on resume.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sfreq: float,
        gap_s: float = 4.0,
    ):
        """
        Parameters
        ----------
        X : (n_trials, n_channels, n_times)
        y : (n_trials,) ground-truth labels
        sfreq : sampling frequency in Hz
        gap_s : inter-trial interval filled with zeros (seconds)
        """
        self._y = y
        self._sfreq = sfreq
        self._n_channels = X.shape[1]

        # Build continuous stream + trial onset table
        gap = np.zeros((self._n_channels, int(gap_s * sfreq)))
        self._trial_onsets: List[int] = []
        segments: List[np.ndarray] = []
        pos = 0
        for i in range(len(X)):
            if i > 0:
                segments.append(gap)
                pos += gap.shape[1]
            self._trial_onsets.append(pos)
            segments.append(X[i])
            pos += X[i].shape[1]
        self._stream = np.concatenate(segments, axis=1)

        # Pacing state (reset in start())
        self._read_pos: int = 0
        self._start_time: float = 0.0
        self._total_paused: float = 0.0
        self._pause_start: float = 0.0
        self._started: bool = False

    # ── SampleSource interface ────────────────────────────────────────

    def start(self) -> None:
        self._read_pos = 0
        self._start_time = time.monotonic()
        self._total_paused = 0.0
        self._pause_start = 0.0
        self._started = True

    def read_chunk(self) -> Optional[np.ndarray]:
        if not self._started:
            return None
        elapsed = time.monotonic() - self._start_time - self._total_paused
        max_pos = min(int(elapsed * self._sfreq), self._stream.shape[1])
        if self._read_pos >= max_pos:
            return None
        chunk = self._stream[:, self._read_pos:max_pos]
        self._read_pos = max_pos
        return chunk

    def get_sfreq(self) -> float:
        return self._sfreq

    def get_n_channels(self) -> int:
        return self._n_channels

    def stop(self) -> None:
        self._started = False

    def is_exhausted(self) -> bool:
        return self._started and self._read_pos >= self._stream.shape[1]

    # ── Pause / resume (wall-clock freeze) ────────────────────────────

    def pause(self) -> None:
        self._pause_start = time.monotonic()

    def resume(self) -> None:
        self._total_paused += time.monotonic() - self._pause_start

    # ── Replay-specific (NOT on SampleSource) ─────────────────────────

    def get_trial_onsets(self) -> List[int]:
        """Absolute sample index of each trial onset in the stream."""
        return list(self._trial_onsets)

    def get_trial_labels(self) -> List[int]:
        """Ground-truth label for each trial."""
        return [int(lbl) for lbl in self._y]

    def get_n_trials(self) -> int:
        return len(self._y)


# ══════════════════════════════════════════════════════════════════════════════
# RingBuffer
# ══════════════════════════════════════════════════════════════════════════════

class RingBuffer:
    """
    Thread-safe circular buffer for multi-channel EEG.

    write() is called from the acquisition thread.
    read()  may be called from any thread.
    Both operations are protected by the same lock.
    """

    def __init__(self, n_channels: int, capacity_samples: int):
        self._buf = np.zeros((n_channels, capacity_samples))
        self._capacity = capacity_samples
        self._write_pos: int = 0
        self._lock = threading.Lock()

    @property
    def write_pos(self) -> int:
        return self._write_pos

    def write(self, chunk: np.ndarray) -> None:
        """Append chunk: (n_channels, n_new_samples)."""
        with self._lock:
            n = chunk.shape[1]
            for i in range(n):
                self._buf[:, (self._write_pos + i) % self._capacity] = chunk[:, i]
            self._write_pos += n

    def read(self, start: int, n_samples: int) -> Optional[np.ndarray]:
        """
        Read n_samples from absolute index *start*.
        Returns (n_channels, n_samples) or None if the requested range
        is not fully available (either not yet written or overwritten).
        """
        with self._lock:
            if start < 0 or n_samples <= 0:
                return None
            oldest = self._write_pos - self._capacity
            if start < oldest or start + n_samples > self._write_pos:
                return None
            out = np.empty((self._buf.shape[0], n_samples))
            for i in range(n_samples):
                out[:, i] = self._buf[:, (start + i) % self._capacity]
        return out

    def reset(self) -> None:
        with self._lock:
            self._buf[:] = 0.0
            self._write_pos = 0
