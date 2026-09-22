"""Vital-sign numerics measured from the waveforms, the way a monitor does it.

Nothing here reads the simulation parameters.  Heart rate comes from a QRS
detector running over the ECG, blood pressure from the arterial trace, and
respiratory rate from the impedance signal - so V-Tach reads 160 whatever the
heart-rate slider says, AFib's rate wanders, a motion artifact can fool the
detector, and a flat line reads as a flat line.

The one exception is SpO2.  Saturation is a property of the blood's colour,
which a synthetic pleth does not have; the analyser reports the configured
target, but only while there is a pulsatile signal good enough to read it from.

Numpy only: scipy is a test dependency and is not shipped in the executable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# --- QRS detection (a Pan-Tompkins-style energy detector) --------------------
SMOOTH_S = 0.020             # 20 ms box filter: its first null sits exactly on 50 Hz
SLOPE_S = 0.008              # slope measured over 8 ms
INTEGRATE_S = 0.080          # energy integration window
REFRACTORY_S = 0.200         # no two QRS complexes closer than this
T_WAVE_WINDOW_S = 0.360      # a candidate this soon after a beat may be its T wave...
T_WAVE_RATIO = 0.5           # ...and is, unless it carries half that beat's energy
DETECT_FRACTION = 0.30       # of the strong-beat energy level
ENERGY_FLOOR = 1.5e-4        # absolute floor, so a flat line cannot "detect" noise

ECG_WINDOW_S = 10.0
MIN_WINDOW_S = 3.0           # below this there is not enough signal to report
HR_BEATS = 6                 # average the rate over this many recent intervals
ASYSTOLE_S = 4.0             # no QRS for this long -> rate 0
ASYSTOLE_MV = 0.25           # ... and the ECG this quiet -> it is asystole, not VF
REGULAR_CV = 0.15            # interval spread below which a rhythm is "regular"
PROMINENCE_MIN = 8.0         # an irregular rhythm must still have QRS standing out
MAX_CREDIBLE_BPM = 300

# --- pressure and pleth ------------------------------------------------------
PRESSURE_WINDOW_S = 4.0
PULSE_MIN_MMHG = 6.0         # pulse pressure below this is no pulse at all
PRESSURE_BEATS = 4           # sys/dia are the median of this many recent beats
PLETH_WINDOW_S = 4.0
PLETH_MIN_AMPLITUDE = 0.2    # normalised pleth swing needed to trust a saturation

# --- respiration -------------------------------------------------------------
RESP_WINDOW_S = 30.0         # respiratory rate needs a long look: 6/min is one breath per 10 s
APNOEA_S = 15.0              # no breath for this long is apnoea
APNOEA_AMPLITUDE = 0.12      # smoothed impedance swing below this is no breathing at all
BREATH_MIN_INTERVAL_S = 1.2
RR_BREATHS = 4


@dataclass(frozen=True, slots=True)
class Vitals:
    """One set of numerics.  ``None`` means "cannot be measured" - shown as ---."""

    hr: int | None = None
    spo2: int | None = None
    systolic: int | None = None
    diastolic: int | None = None
    mean_pressure: int | None = None
    resp_rate: int | None = None
    asystole: bool = False
    no_pulse: bool = False
    apnoea: bool = False
    chaotic: bool = False        # ECG activity with no countable rhythm (VF, torsades)
    window_s: float = 0.0


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Causal box filter, same length as the input."""
    if n <= 1 or x.size == 0:
        return x.astype(np.float64, copy=True)
    padded = np.concatenate((np.full(n - 1, x[0]), x))
    csum = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    return (csum[n:] - csum[:-n]) / n


def qrs_energy(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Smoothed slope energy: high on the steep edges of a QRS, low elsewhere."""
    smooth = _moving_average(ecg, max(1, round(SMOOTH_S * fs)))
    lag = max(1, round(SLOPE_S * fs))
    slope = np.zeros_like(smooth)
    slope[lag:] = smooth[lag:] - smooth[:-lag]
    return _moving_average(slope * slope, max(1, round(INTEGRATE_S * fs)))


def detect_qrs(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Sample indices of detected QRS complexes, oldest first.

    The threshold adapts to the strong beats in the window but never drops
    below an absolute floor.  A refractory period stops one complex - or a
    pacing spike and the complex it triggers - from counting twice, and a
    candidate arriving soon after a beat with much less energy is taken to be
    that beat's T wave, as in Pan & Tompkins (1985).
    """
    if ecg.size < fs:
        return np.empty(0, dtype=np.int64)
    energy = qrs_energy(ecg, fs)
    threshold = max(DETECT_FRACTION * float(np.percentile(energy, 99.0)), ENERGY_FLOOR)

    above = energy > threshold
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    refractory = round(REFRACTORY_S * fs)
    t_window = round(T_WAVE_WINDOW_S * fs)
    search = round(0.15 * fs)
    peaks: list[int] = []
    for start in edges:
        if peaks and start - peaks[-1] < refractory:
            continue
        stop = min(start + search, energy.size)
        peak = int(start + np.argmax(energy[start:stop]))
        if peaks and peak - peaks[-1] < t_window \
                and energy[peak] < T_WAVE_RATIO * energy[peaks[-1]]:
            continue
        peaks.append(peak)
    return np.asarray(peaks, dtype=np.int64)


def _prominence(energy: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    """How far the detected complexes stand out from the activity between them."""
    mask = np.ones(energy.size, dtype=bool)
    guard = round(0.12 * fs)
    for p in peaks:
        mask[max(0, p - guard):p + guard] = False
    between = float(np.percentile(energy[mask], 90)) if mask.any() else 0.0
    return float(np.median(energy[peaks])) / max(between, 1e-12)


def _heart_rate(ecg: np.ndarray, fs: int) -> tuple[int | None, bool, bool]:
    """(rate, asystole, chaotic) from the ECG alone."""
    peaks = detect_qrs(ecg, fs)
    recent = ecg[-round(ASYSTOLE_S * fs):]
    quiet = float(np.ptp(recent - _moving_average(recent, round(0.4 * fs)))) < ASYSTOLE_MV

    last_beat_age = (ecg.size - peaks[-1]) / fs if peaks.size else float("inf")
    if last_beat_age > ASYSTOLE_S:
        # No complexes lately: a quiet trace is asystole; a busy one is not a
        # rhythm anyone can count (VF looks like this).
        return (0, True, False) if quiet else (None, False, True)
    if peaks.size < 3:
        return None, False, False

    intervals = np.diff(peaks)
    spread = float(intervals.std() / intervals.mean())
    if spread >= REGULAR_CV:
        # Irregular is fine - AFib, Wenckebach and bigeminy all are - provided
        # the complexes are clearly there.  VF and torsades are irregular *and*
        # have nothing standing out from the fibrillation between "beats".
        if _prominence(qrs_energy(ecg, fs), peaks, fs) < PROMINENCE_MIN:
            return None, False, True

    rate = 60.0 * fs / float(intervals[-HR_BEATS:].mean())
    if rate > MAX_CREDIBLE_BPM:
        return None, False, True
    return int(round(rate)), False, False


def pressure_pulses(abp: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat-by-beat (peak index, preceding trough value) pairs.

    Systolic and diastolic have to come from the *same* beats.  Taking the
    window's highest peak and lowest trough independently reads nonsense in any
    transition - just after a successful shock it paired the first new systolic
    with the pressure floor left by the arrest, and showed 103/9.
    """
    half = round(0.15 * fs)
    padded = np.pad(abp, half, mode="edge")
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).max(axis=1)
    candidates = np.flatnonzero(abp >= local_max)
    peaks: list[int] = []
    troughs: list[float] = []
    previous = 0
    lookback = round(0.8 * fs)
    for c in candidates:
        if peaks and c - peaks[-1] < round(0.25 * fs):
            continue
        start = previous if previous < c else max(0, c - lookback)
        trough = float(abp[max(start, c - lookback):c + 1].min())
        if abp[c] - trough >= PULSE_MIN_MMHG:           # a pulse, not a plateau ripple
            peaks.append(int(c))
            troughs.append(trough)
        previous = int(c)
    return np.asarray(peaks, dtype=np.int64), np.asarray(troughs)


def _pressures(abp: np.ndarray, fs: int) -> tuple[int | None, int | None, int, bool]:
    window = abp[-round(PRESSURE_WINDOW_S * fs):]
    mean = int(round(float(window.mean())))
    peaks, troughs = pressure_pulses(window, fs)
    if peaks.size == 0:
        return None, None, mean, True
    # Median of the last few beats, so one odd beat (a PVC's weak pulse, say)
    # does not swing the displayed number.
    recent = slice(-PRESSURE_BEATS, None)
    return (int(round(float(np.median(window[peaks[recent]])))),
            int(round(float(np.median(troughs[recent])))),
            mean, False)


def breath_onsets(resp: np.ndarray, fs: int) -> np.ndarray:
    """Sample indices where inspiration crosses up through mid-level."""
    smooth = _moving_average(resp - _moving_average(resp, round(4.0 * fs)),
                             round(0.3 * fs))
    if float(np.ptp(smooth)) < APNOEA_AMPLITUDE:
        return np.empty(0, dtype=np.int64)
    level = 0.35 * float(np.ptp(smooth))
    rising = np.flatnonzero((smooth[1:] > level) & (smooth[:-1] <= level)) + 1
    breaths: list[int] = []
    for index in rising:
        if not breaths or index - breaths[-1] >= BREATH_MIN_INTERVAL_S * fs:
            breaths.append(int(index))
    return np.asarray(breaths, dtype=np.int64)


def _respiratory_rate(resp: np.ndarray, fs: int) -> tuple[int | None, bool]:
    breaths = breath_onsets(resp, fs)
    since_last = (resp.size - breaths[-1]) / fs if breaths.size else resp.size / fs
    if since_last > APNOEA_S:
        return 0, True
    if breaths.size < 2:
        return None, False
    interval = float(np.diff(breaths[-(RR_BREATHS + 1):]).mean())
    return int(round(60.0 * fs / interval)), False


def _finite_tail(data: Mapping[str, np.ndarray], name: str, seconds: float,
                 fs: int) -> np.ndarray:
    x = np.asarray(data.get(name, ()), dtype=np.float64)
    if x.size:
        x = x[np.isfinite(x)]
    return x[-round(seconds * fs):]


def analyze(data: Mapping[str, np.ndarray], fs: int,
            spo2_target: float | None = None) -> Vitals:
    """Measure every numeric from recent signal (oldest sample first).

    Each channel is read over its own window: 10 s of ECG, 4 s of pressure and
    pleth, and up to 30 s of respiration, because a slow breathing rate needs a
    long look to see even two breaths.
    """
    ecg = _finite_tail(data, "ecg", ECG_WINDOW_S, fs)
    seconds = ecg.size / fs
    if seconds < MIN_WINDOW_S:
        return Vitals(window_s=seconds)

    hr, asystole, chaotic = _heart_rate(ecg, fs)

    systolic = diastolic = mean = None
    no_pulse = False
    abp = _finite_tail(data, "abp", PRESSURE_WINDOW_S, fs)
    if abp.size >= PRESSURE_WINDOW_S * fs:
        systolic, diastolic, mean, no_pulse = _pressures(abp, fs)

    spo2 = None
    pleth = _finite_tail(data, "pleth", PLETH_WINDOW_S, fs)
    if spo2_target is not None and pleth.size >= PLETH_WINDOW_S * fs:
        if float(np.ptp(pleth)) >= PLETH_MIN_AMPLITUDE:
            spo2 = int(round(spo2_target))

    resp_rate, apnoea = None, False
    resp = _finite_tail(data, "resp", RESP_WINDOW_S, fs)
    if resp.size >= 8.0 * fs:
        resp_rate, apnoea = _respiratory_rate(resp, fs)

    return Vitals(hr=hr, spo2=spo2, systolic=systolic, diastolic=diastolic,
                  mean_pressure=mean, resp_rate=resp_rate, asystole=asystole,
                  no_pulse=no_pulse, apnoea=apnoea, chaotic=chaotic, window_s=seconds)


class VitalsAnalyzer:
    """Keeps enough history to measure from, fed incrementally from a ring buffer.

    The ring buffer holds the 10 s the monitor displays; respiration needs 30 s.
    Each :meth:`feed` pulls only the samples written since the last call, so the
    analyser keeps its own longer tail without re-reading the whole buffer.
    """

    def __init__(self, fs: int, history_s: float = RESP_WINDOW_S) -> None:
        self.fs = int(fs)
        self.capacity = round(history_s * fs)
        self._history: dict[str, np.ndarray] = {}
        self._seen = 0

    def reset(self) -> None:
        self._history.clear()
        self._seen = 0

    def feed(self, buffer) -> None:
        total = buffer.total_written
        fresh = min(total - self._seen, buffer.capacity)
        if fresh <= 0:
            return
        latest = buffer.latest(fresh)
        for index, name in enumerate(buffer.channels):
            joined = np.concatenate((self._history.get(name, np.empty(0)), latest[index]))
            self._history[name] = joined[-self.capacity:]
        self._seen = total

    def measure(self, spo2_target: float | None = None) -> Vitals:
        return analyze(self._history, self.fs, spo2_target)
