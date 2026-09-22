"""Fixed-size multi-channel ring buffer bridging the producer and the monitor.

One integer ``write_index`` is shared by every channel, so ECG and respiration
samples written in the same chunk always land on the same index.  That keeps the
two plots time-aligned and makes the CSV export a straight column dump.

Threading contract: the generator thread calls :meth:`write`, the GUI timer
calls :meth:`snapshot` / :meth:`chronological`.  Both take the same lock, and
the readers hand back copies, so the consumer never sees a half-written chunk.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Mapping, Sequence

import numpy as np

from .state import BUFFER_SIZE, SAMPLE_RATE

DEFAULT_CHANNELS = ("ecg", "resp", "pleth", "abp")


class RingBuffer:
    """Circular store of ``capacity`` samples per channel, backed by one 2-D array."""

    def __init__(
        self,
        capacity: int = BUFFER_SIZE,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        sample_rate: int = SAMPLE_RATE,
        fill: float = np.nan,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not channels:
            raise ValueError("at least one channel is required")

        self.capacity = int(capacity)
        self.channels = tuple(channels)
        self.sample_rate = int(sample_rate)
        self._fill = float(fill)
        self._index_of = {name: i for i, name in enumerate(self.channels)}

        self._lock = threading.Lock()
        self._data = np.full((len(self.channels), self.capacity), self._fill, dtype=np.float64)
        self._write_index = 0      # next slot to be written
        self._total_written = 0    # samples ever written (buffer is full once this hits capacity)

    # -- introspection --------------------------------------------------------
    @property
    def write_index(self) -> int:
        """Index of the next slot the producer will write (the sweep cursor)."""
        with self._lock:
            return self._write_index

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    @property
    def duration(self) -> float:
        """Buffer span in seconds."""
        return self.capacity / self.sample_rate

    def channel_index(self, name: str) -> int:
        try:
            return self._index_of[name]
        except KeyError:
            raise KeyError(f"unknown channel {name!r}; have {self.channels}") from None

    def time_axis(self) -> np.ndarray:
        """Fixed x-axis for the sweep display: 0 .. duration, one point per slot."""
        return np.arange(self.capacity, dtype=np.float64) / self.sample_rate

    # -- producer side --------------------------------------------------------
    def write(self, block: Mapping[str, np.ndarray] | np.ndarray) -> int:
        """Append one chunk and return the new ``write_index``.

        ``block`` is either a mapping of channel name -> 1-D array, or a 2-D
        array shaped ``(n_channels, n_samples)``.  Chunks longer than the buffer
        keep only their most recent ``capacity`` samples.
        """
        chunk = self._as_array(block)
        n = chunk.shape[1]
        if n == 0:
            return self.write_index
        if n > self.capacity:                 # only the tail can survive
            chunk = chunk[:, -self.capacity:]
            n = self.capacity

        with self._lock:
            start = self._write_index
            end = start + n
            if end <= self.capacity:
                self._data[:, start:end] = chunk
            else:                              # split across the wrap point
                head = self.capacity - start
                self._data[:, start:] = chunk[:, :head]
                self._data[:, : n - head] = chunk[:, head:]
            self._write_index = end % self.capacity
            self._total_written += n
            return self._write_index

    def _as_array(self, block: Mapping[str, np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(block, Mapping):
            missing = set(self.channels) - set(block)
            if missing:
                raise KeyError(f"write() is missing channel(s): {sorted(missing)}")
            columns = [np.asarray(block[name], dtype=np.float64).ravel() for name in self.channels]
            lengths = {c.size for c in columns}
            if len(lengths) != 1:
                raise ValueError(f"all channels must be the same length, got {lengths}")
            return np.vstack(columns)

        chunk = np.asarray(block, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.shape[0] != len(self.channels):
            raise ValueError(
                f"expected {len(self.channels)} channels, got array with shape {chunk.shape}"
            )
        return chunk

    # -- consumer side --------------------------------------------------------
    def snapshot(self) -> tuple[np.ndarray, int]:
        """Copy of the raw (unrotated) buffer plus the current write index.

        This is what the sweep renderer wants: the trace stays put on screen and
        only the cursor moves, exactly like a hospital monitor.
        """
        with self._lock:
            return self._data.copy(), self._write_index

    def chronological(self) -> np.ndarray:
        """Copy of the buffer reordered oldest -> newest (for export / analysis)."""
        with self._lock:
            if self._total_written < self.capacity:
                return self._data[:, : self._write_index].copy()
            return np.roll(self._data, -self._write_index, axis=1)

    def latest(self, n: int) -> np.ndarray:
        """The most recent ``n`` samples per channel, oldest first."""
        if n <= 0:
            return np.empty((len(self.channels), 0), dtype=np.float64)
        return self.chronological()[:, -n:]

    def clear(self) -> None:
        with self._lock:
            self._data.fill(self._fill)
            self._write_index = 0
            self._total_written = 0

    # -- persistence ----------------------------------------------------------
    def to_csv(self, path: str) -> int:
        """Dump the buffer oldest-first as ``time_s`` + one column per channel.

        Returns the number of sample rows written.
        """
        data = self.chronological()
        n = data.shape[1]
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", *self.channels])
            for row in np.column_stack([t, data.T]):
                writer.writerow([f"{v:.6f}" for v in row])
        return n

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self.capacity}, channels={self.channels}, "
            f"write_index={self.write_index}, total_written={self.total_written})"
        )
