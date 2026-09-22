"""Synthetic ECG / respiration generation and the producer thread.

The math lives in :class:`WaveformEngine`, which is deliberately Qt-free so the
models can be unit-tested headlessly.  :class:`SignalGenerator` is a thin
``QThread`` that paces the engine and pushes chunks into the ring buffer.

What each condition *looks like* lives in :mod:`core.pathology`; this file is
about timing and signal assembly.  Signal chain per chunk (50 ms at 1 kHz):

    ventricular beats -> Windkessel circulation -> arterial pressure, pleth
    beats (5-Gaussian McSharry) or a baseline override (VF, asystole)
        -> atrial baseline (f-waves, flutter sawtooth)
        -> respiratory baseline sway
        -> baseline wander -> mains hum -> Gaussian sensor noise
        -> motion artifact (transient)

Everything is phase-continuous across chunk boundaries: beats are scheduled on
an absolute timeline, the fixed-frequency noises are evaluated from the absolute
sample counter, and the respiratory phase (whose frequency the user can change)
is carried in an accumulator.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .buffer import RingBuffer
from .hemodynamics import CPR_RATE, Circulation
from .pathology import (  # noqa: F401  (re-exported for callers and tests)
    AFIB_BEAT,
    AFIB_JITTER,
    BIGEMINY_COUPLING,
    COMPLETE_BLOCK_ESCAPE_BPM,
    ESCAPE_BEAT,
    FIRST_DEGREE_BEAT,
    MOBITZ_II_CONDUCTED,
    P,
    PACED_BEAT,
    PVC,
    P_ONLY,
    RESP_INSPIRATORY_FRACTION,
    RESP_SPECS,
    RHYTHMS,
    SHOCK_KINDS,
    THERAPY_CARDIOVERT,
    THERAPY_CPR,
    THERAPY_DEFIB,
    UNSYNCHRONISABLE,
    SINUS,
    STEMI_BEAT,
    T,
    TORSADES_TWIST_HZ,
    VFIB_AMPLITUDE,
    VTACH_BEAT,
    VTACH_RATE,
    WENCKEBACH_PR,
    BeatMorphology,
    RespSpec,
    RhythmSpec,
    fibrillatory_waves,
    flutter_waves,
    resp_envelope,
    skewed_respiration,
    therapy_for,
    with_pr,
)
from .state import (
    BASELINE_FREQUENCY,
    CHUNK_SAMPLES,
    MAINS_FREQUENCY,
    MOTION_DURATION_S,
    SAMPLE_RATE,
    SimulationState,
)

RR_NOMINAL = 0.8          # s; the RR the published theta/b values describe (75 bpm)
RATE_SCALING = True       # Bazett-style scaling of the slow waves with heart rate
SLOW_WAVES = [P, T]       # P and T shift/stretch with rate; the QRS does not

INFLUENCE_S = 0.65        # how far a beat's Gaussians can reach from its R-peak
LOOKAHEAD_S = 0.55        # schedule this far past the chunk so P waves are not clipped

# Respiratory coupling
RSA_DEPTH = 0.05                  # RR interval swing with respiration (+/- 5 %)
RESP_ECG_GAIN = 0.06              # mV of ECG baseline sway per unit respiration

# Ectopy
PVC_LEAD_S = 0.10         # schedule the PVC this far ahead so it renders in full
PVC_COUPLING = 0.62       # ectopic beat fires this fraction of an R-R after a sinus beat
MOTION_AMPLITUDE = 2.5    # mV; deliberately dwarfs the 1 mV R wave

BEAT_LOG_LIMIT = 4096     # bounded: the engine may run for hours

CHANNELS = ("ecg", "resp", "pleth", "abp")

# Chest compressions
CPR_ECG_MV = 0.8          # compression artifact: CPR can masquerade as a rhythm
CPR_RESP_GAIN = 0.25      # thoracic impedance change per compression
PLETH_RESP_GAIN = 0.05    # respiratory variation of the pleth baseline
PLETH_MOTION_GAIN = 0.5   # motion is the leading cause of bad SpO2 readings

# --- Electrical therapy ------------------------------------------------------
SHOCK_ARTIFACT_S = 1.2       # amplifier saturation and recovery after a shock
SHOCK_SPIKE_S = 0.02         # the discharge itself
SHOCK_SPIKE_MV = 6.0         # far beyond the display range: the trace blanks out
SHOCK_OFFSET_MV = 2.2        # post-discharge offset the amplifier decays out of
SHOCK_RECOVERY_TAU = 0.28    # s
POST_SHOCK_PAUSE_S = 1.1     # stunned myocardium before any rhythm resumes

# First-shock outcomes.  Real defibrillation is not certain, and a simulator
# that always converts teaches the wrong lesson - so the outcome is drawn, and
# reported explicitly rather than left to look like a bug.
DEFIB_SUCCESS = 0.75
CARDIOVERT_SUCCESS = 0.90
R_ON_T_RISK = 0.35           # unsynchronised shock landing on a T wave -> VF
ROSC_RHYTHM = "Sinus"

# Outcome codes, reported on the shock_delivered signal.
SHOCK_CONVERTED = "converted"
SHOCK_PERSISTS = "persists"
SHOCK_NO_EFFECT = "no_effect"
SHOCK_INDUCED_VF = "induced_vf"
SHOCK_NOT_DELIVERED = "not_delivered"

# Backwards-compatible alias; the shaping function now lives in core.pathology.
_skewed_respiration = skewed_respiration


@dataclass(slots=True)
class ScheduledBeat:
    """A beat placed on the absolute timeline with its parameters pre-scaled."""

    r_time: float
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    kind: str = "sinus"          # sinus | pvc | p | escape
    committed: bool = False      # handed to the circulation; its pulse is now fixed

    @property
    def ectopic(self) -> bool:
        return self.kind == "pvc"

    @property
    def conducts(self) -> bool:
        """True if this beat produces a ventricular complex."""
        return self.kind != "p"


class WaveformEngine:
    """Stateful, phase-continuous waveform source.  No Qt dependency."""

    def __init__(
        self,
        state: SimulationState,
        sample_rate: int = SAMPLE_RATE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.state = state
        self.fs = int(sample_rate)
        self.rng = rng if rng is not None else np.random.default_rng()

        self._n_emitted = 0            # absolute sample counter -> absolute time
        self._resp_u = 0.0             # respiratory cycle position in [0, 1)
        self._scheduled: list[ScheduledBeat] = []
        self._next_sinus_time = 0.05   # first beat, just far enough in to render
        self._next_escape_time = 0.05  # independent ventricular clock (AV dissociation)
        self._motion: np.ndarray | None = None
        self._motion_pos = 0
        self._shock: np.ndarray | None = None
        self._shock_pos = 0

        params = state.snapshot()
        self.circulation = Circulation(self.fs, params.systolic, params.diastolic)
        self._next_compression: float | None = None
        self._compressions: list[float] = []

        self._rhythm = "Sinus"
        self._beat_index = 0           # position within a repeating conduction cycle
        self._resp_spec: RespSpec = RESP_SPECS["Normal"]

        # Fixed random components of the fibrillation waveform, drawn on first
        # use so V-Fib is chaotic in shape but phase-continuous across chunks -
        # and so an engine that never fibrillates does not consume draws that
        # would shift every other rhythm's noise realisation.
        self._vfib: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

        # Diagnostics for tests and for the UI's rhythm readout.  Bounded, or a
        # long run leaks one tuple per beat forever.
        self.beat_log: deque[tuple[float, str]] = deque(maxlen=BEAT_LOG_LIMIT)

        # Set when a shock is delivered; the producer thread drains it and
        # re-emits it as a Qt signal so the UI can report what happened.
        self.pending_report: tuple[str, str, str] | None = None
        self.shock_log: deque[tuple[float, str, str, str]] = deque(maxlen=64)

    # -- clock ---------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        """Seconds of signal generated so far."""
        return self._n_emitted / self.fs

    def _resp_value_at(self, t: float, rate_hz: float) -> float:
        """Respiration amplitude at an absolute future time (for RSA lookahead)."""
        spec = self._resp_spec
        u = self._resp_u + rate_hz * (t - self.elapsed)
        wave = float(skewed_respiration(u, spec.inspiratory))
        return spec.amplitude * wave * float(resp_envelope(spec.envelope, t))

    def respiratory_rate_hz(self, params) -> float:
        """Breathing frequency, honouring any pattern that overrides the slider."""
        spec = RESP_SPECS.get(params.resp_pattern, RESP_SPECS["Normal"])
        bpm = spec.rate_bpm if spec.rate_bpm is not None else params.respiratory_rate
        return bpm / 60.0

    # -- beat construction ----------------------------------------------------
    def _make_beat(self, r_time: float, morph: BeatMorphology, rr: float,
                   kind: str = "sinus", gain: float = 1.0) -> ScheduledBeat:
        theta, b = morph.theta, morph.b
        if RATE_SCALING and morph.rate_scaled:
            # Bazett: the QT interval tracks sqrt(RR), so P and T migrate toward
            # the QRS as the rate climbs.  The QRS itself keeps its width.
            scale = float(np.clip(np.sqrt(rr / RR_NOMINAL), 0.6, 1.3))
            theta = theta.copy()
            b = b.copy()
            theta[SLOW_WAVES] *= scale
            b[SLOW_WAVES] *= scale
        a = morph.a if gain == 1.0 else morph.a * gain
        beat = ScheduledBeat(r_time, a, b, theta, kind)
        self.beat_log.append((r_time, kind))
        return beat

    # -- rhythm scheduling ----------------------------------------------------
    def _schedule_until(self, horizon: float, hr: float, resp_hz: float, rhythm: str) -> None:
        spec = RHYTHMS.get(rhythm, RHYTHMS["Sinus"])

        if rhythm != self._rhythm:
            # Restart the conduction cycle and pull both clocks back to now, so a
            # switch cannot back-date a burst of beats or leave a long dead gap.
            self._rhythm = rhythm
            self._beat_index = 0
            resume = max(self.elapsed + PVC_LEAD_S, min(self._next_sinus_time, horizon))
            self._next_sinus_time = resume
            self._next_escape_time = resume

        scheduler = spec.scheduler
        if scheduler == "none":                 # V-Fib / asystole: no beats at all
            self._next_sinus_time = horizon
            self._next_escape_time = horizon
        elif scheduler == "wenckebach":
            self._schedule_wenckebach(horizon, hr)
        elif scheduler == "mobitz2":
            self._schedule_mobitz2(horizon, hr)
        elif scheduler == "complete_block":
            self._schedule_complete_block(horizon, hr)
        elif scheduler == "bigeminy":
            self._schedule_bigeminy(horizon, hr)
        elif scheduler == "torsades":
            self._schedule_torsades(horizon)
        else:
            self._schedule_regular(horizon, hr, resp_hz, spec)

    def _schedule_regular(self, horizon: float, hr: float, resp_hz: float,
                          spec: RhythmSpec) -> None:
        while self._next_sinus_time < horizon:
            t_beat = self._next_sinus_time
            rate = spec.rate_bpm if spec.rate_bpm is not None else hr
            rr = 60.0 / rate
            if spec.jitter is not None:
                rr *= self.rng.uniform(*spec.jitter)
            if spec.rsa:
                # Respiratory sinus arrhythmia needs a functioning sinus node, so
                # it applies only to the rhythms that still have one.
                rr *= 1.0 - RSA_DEPTH * self._resp_value_at(t_beat, resp_hz)
            self._scheduled.append(self._make_beat(t_beat, spec.morphology, rr))
            self._next_sinus_time = t_beat + rr

    def _schedule_wenckebach(self, horizon: float, hr: float) -> None:
        """Mobitz I: the atrial rate is metronomic, the PR interval creeps out.

        Beats are placed by *atrial* time so the P-P interval stays constant and
        the R-R stretches - which is what makes Wenckebach recognisable.
        """
        cycle = len(WENCKEBACH_PR) + 1
        while self._next_sinus_time < horizon:
            t_p = self._next_sinus_time
            pp = 60.0 / hr
            step = self._beat_index % cycle
            if step < len(WENCKEBACH_PR):
                pr = WENCKEBACH_PR[step]
                self._scheduled.append(
                    self._make_beat(t_p + pr, with_pr(SINUS, pr), pp))
            else:
                self._scheduled.append(self._make_beat(t_p, P_ONLY, pp, kind="p"))
            self._beat_index += 1
            self._next_sinus_time = t_p + pp

    def _schedule_mobitz2(self, horizon: float, hr: float) -> None:
        """Mobitz II: PR never varies, then a QRS simply fails to appear."""
        cycle = MOBITZ_II_CONDUCTED + 1
        pr = 0.20
        while self._next_sinus_time < horizon:
            t_p = self._next_sinus_time
            pp = 60.0 / hr
            if self._beat_index % cycle < MOBITZ_II_CONDUCTED:
                self._scheduled.append(self._make_beat(t_p + pr, SINUS, pp))
            else:
                self._scheduled.append(self._make_beat(t_p, P_ONLY, pp, kind="p"))
            self._beat_index += 1
            self._next_sinus_time = t_p + pp

    def _schedule_complete_block(self, horizon: float, hr: float) -> None:
        """Third degree: atria and ventricles run on unrelated clocks."""
        pp = 60.0 / hr
        while self._next_sinus_time < horizon:
            t_p = self._next_sinus_time
            self._scheduled.append(self._make_beat(t_p, P_ONLY, pp, kind="p"))
            self._next_sinus_time = t_p + pp

        vv = 60.0 / COMPLETE_BLOCK_ESCAPE_BPM
        while self._next_escape_time < horizon:
            t_v = self._next_escape_time
            self._scheduled.append(self._make_beat(t_v, ESCAPE_BEAT, vv, kind="escape"))
            self._next_escape_time = t_v + vv

    def _schedule_bigeminy(self, horizon: float, hr: float) -> None:
        """Every sinus beat is trailed by a PVC and a compensatory pause."""
        while self._next_sinus_time < horizon:
            t_beat = self._next_sinus_time
            rr = 60.0 / hr
            if self._beat_index % 2 == 0:
                self._scheduled.append(self._make_beat(t_beat, SINUS, rr))
                self._next_sinus_time = t_beat + BIGEMINY_COUPLING * rr
            else:
                self._scheduled.append(self._make_beat(t_beat, PVC, rr, kind="pvc"))
                self._next_sinus_time = t_beat + (2.0 - BIGEMINY_COUPLING) * rr
            self._beat_index += 1

    def _schedule_torsades(self, horizon: float) -> None:
        """Polymorphic VT: the QRS axis rotates, so amplitude waxes and inverts."""
        rr = 60.0 / RHYTHMS["Torsades de Pointes"].rate_bpm
        while self._next_sinus_time < horizon:
            t_beat = self._next_sinus_time
            twist = float(np.cos(2.0 * np.pi * TORSADES_TWIST_HZ * t_beat))
            self._scheduled.append(
                self._make_beat(t_beat, VTACH_BEAT, rr, kind="escape", gain=twist))
            self._next_sinus_time = t_beat + rr

    # -- ectopy on demand -----------------------------------------------------
    def _inject_pvc(self, t0: float, hr: float) -> float:
        """Insert one ectopic beat plus a full compensatory pause.  Returns its time.

        The beat is coupled at a fixed fraction of the R-R interval after a
        preceding conducted beat, rather than dropped wherever the click landed:
        a PVC that happens to coincide with a scheduled beat would otherwise
        superimpose on it, which the ventricles' refractory period makes
        impossible.  If the click arrives too late to couple to the last beat,
        the ectopic beat follows the next one instead - at most a beat's delay,
        and it lands in a physiologically correct place.
        """
        rr = 60.0 / hr
        earliest = t0 + PVC_LEAD_S                 # cannot render into the past
        # Only ventricular complexes are valid coupling anchors; a bare P wave
        # from a blocked beat is not something a PVC can follow.
        conducted = sorted(b.r_time for b in self._scheduled
                           if b.conducts and not b.ectopic)

        anchor = next((t for t in conducted if t + PVC_COUPLING * rr >= earliest), None)
        if anchor is None:
            anchor = max(conducted) if conducted else earliest - PVC_COUPLING * rr
        pvc_t = max(earliest, anchor + PVC_COUPLING * rr)

        resume = anchor + 2.0 * rr                 # textbook full compensatory pause
        if resume <= pvc_t + 0.2 * rr:             # very fast rates: keep the pause visible
            resume = pvc_t + 1.2 * rr

        # Drop the beats the ectopic beat and its pause swallow.
        self._scheduled = [
            b for b in self._scheduled
            if b.ectopic or b.r_time <= anchor or b.r_time >= resume
        ]
        self._scheduled.append(self._make_beat(pvc_t, PVC, rr, kind="pvc"))
        self._scheduled.sort(key=lambda b: b.r_time)
        self._next_sinus_time = max(self._next_sinus_time, resume)
        return pvc_t

    # -- rendering ------------------------------------------------------------
    def _render_beats(self, t: np.ndarray) -> np.ndarray:
        """Sum every in-range beat's five Gaussians over the chunk's time axis."""
        ecg = np.zeros(t.size, dtype=np.float64)
        for beat in self._scheduled:
            if beat.r_time - INFLUENCE_S > t[-1] or beat.r_time + INFLUENCE_S < t[0]:
                continue
            x = (t - beat.r_time)[:, None] - beat.theta[None, :]
            ecg += (beat.a * np.exp(-(x * x) / (2.0 * beat.b * beat.b))).sum(axis=1)
        return ecg

    def _fibrillation(self, t: np.ndarray) -> np.ndarray:
        """Coarse disorganised activity: no complexes, no isoelectric line."""
        if self._vfib is None:
            freqs = self.rng.uniform(3.5, 7.5, 7)
            phases = self.rng.uniform(0.0, 2.0 * np.pi, 7)
            amps = self.rng.uniform(0.5, 1.0, 7)
            self._vfib = (freqs, phases, amps, float(amps.sum()))
        freqs, phases, amps, norm = self._vfib

        out = np.zeros(t.size, dtype=np.float64)
        for freq, phase, amp in zip(freqs, phases, amps):
            out += amp * np.sin(2.0 * np.pi * freq * t + phase)
        envelope = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.9 * t)
        return VFIB_AMPLITUDE * envelope * out / norm

    def _prune(self, t_start: float) -> None:
        cutoff = t_start - INFLUENCE_S
        self._scheduled = [b for b in self._scheduled if b.r_time >= cutoff]

    # -- electrical therapy ---------------------------------------------------
    def _build_shock_artifact(self) -> np.ndarray:
        """The discharge plus the amplifier's recovery from saturation.

        A real defibrillation blanks the trace: a huge transient far outside the
        display range, then a decaying offset while the front end recovers.
        """
        n = int(SHOCK_ARTIFACT_S * self.fs)
        t = np.arange(n, dtype=np.float64) / self.fs

        wave = SHOCK_OFFSET_MV * np.exp(-t / SHOCK_RECOVERY_TAU)
        spike = int(SHOCK_SPIKE_S * self.fs)
        # Biphasic discharge, as every modern defibrillator delivers.
        wave[:spike] += SHOCK_SPIKE_MV * np.sin(
            2.0 * np.pi * np.arange(spike, dtype=np.float64) / spike)
        return wave

    def _next_shock(self, n: int) -> np.ndarray:
        if self._shock is None:
            return np.zeros(n)
        take = self._shock[self._shock_pos:self._shock_pos + n]
        self._shock_pos += n
        if self._shock_pos >= self._shock.size:
            self._shock = None
            self._shock_pos = 0
        if take.size < n:
            take = np.concatenate([take, np.zeros(n - take.size)])
        return take

    def _shock_outcome(self, kind: str, rhythm: str) -> str:
        """Decide what one shock does to the rhythm it is delivered into.

        Encodes the distinctions that make this worth simulating at all:
        a synchronised shock cannot be timed without an R wave, asystole is not
        shockable, and an unsynchronised shock into an organised rhythm can land
        on the T wave and start VF.
        """
        indicated = therapy_for(rhythm).therapy

        if kind == THERAPY_CARDIOVERT and rhythm in UNSYNCHRONISABLE:
            return SHOCK_NOT_DELIVERED          # nothing to synchronise to
        if indicated == THERAPY_CPR:
            return SHOCK_NO_EFFECT              # asystole: CPR, not electricity

        if indicated == kind:                   # correct modality, correct rhythm
            threshold = DEFIB_SUCCESS if kind == THERAPY_DEFIB else CARDIOVERT_SUCCESS
            return SHOCK_CONVERTED if self.rng.random() < threshold else SHOCK_PERSISTS

        if kind == THERAPY_DEFIB:
            # Unsynchronised into an organised rhythm: R-on-T risk.
            if self.rng.random() < R_ON_T_RISK:
                return SHOCK_INDUCED_VF
            if indicated == THERAPY_CARDIOVERT:
                return (SHOCK_CONVERTED if self.rng.random() < CARDIOVERT_SUCCESS
                        else SHOCK_PERSISTS)
            return SHOCK_NO_EFFECT

        # Synchronised shock: safe, and still effective on a shockable-but-
        # organised rhythm such as monomorphic VT with a pulse.
        if indicated == THERAPY_DEFIB:
            return (SHOCK_CONVERTED if self.rng.random() < CARDIOVERT_SUCCESS
                    else SHOCK_PERSISTS)
        return SHOCK_NO_EFFECT

    def _deliver_shock(self, kind: str, t0: float) -> tuple[str, str]:
        """Apply one shock.  Returns ``(outcome, resulting rhythm)``."""
        rhythm = self.state.snapshot().cardiac_rhythm
        outcome = self._shock_outcome(kind, rhythm)

        if outcome == SHOCK_NOT_DELIVERED:
            return outcome, rhythm              # the defibrillator refuses to fire

        self._shock = self._build_shock_artifact()
        self._shock_pos = 0

        new_rhythm = rhythm
        if outcome == SHOCK_CONVERTED:
            new_rhythm = ROSC_RHYTHM
        elif outcome == SHOCK_INDUCED_VF:
            new_rhythm = "V-Fib"

        if new_rhythm != rhythm:
            self.state.update(cardiac_rhythm=new_rhythm)

        # Stunned myocardium: clear anything already scheduled past the shock and
        # hold off the next beat, so the new rhythm starts from a clean pause.
        resume = t0 + POST_SHOCK_PAUSE_S
        self._scheduled = [b for b in self._scheduled if b.r_time < t0]
        self._next_sinus_time = max(self._next_sinus_time, resume)
        self._next_escape_time = max(self._next_escape_time, resume)
        self._beat_index = 0
        # Adopt the new rhythm here rather than letting _schedule_until notice
        # it: its switch-over reset pulls the clocks back to the scheduling
        # horizon, which would erase the pause we just set.
        self._rhythm = new_rhythm
        return outcome, new_rhythm

    # -- mechanics ------------------------------------------------------------
    def _commit_beats(self, until: float, perfusion: float) -> None:
        """Hand each ventricular beat to the circulation once its R wave is due.

        Beats are committed lazily - only when the chunk reaches them - because
        a PVC or a shock can still delete a beat that is merely scheduled.  A
        beat committed early would leave a phantom pulse behind.
        """
        for beat in sorted(self._scheduled, key=lambda b: b.r_time):
            if beat.r_time >= until:
                break
            if beat.committed or not beat.conducts:
                continue
            beat.committed = True
            self.circulation.beat(beat.r_time, perfusion, ectopic=beat.ectopic)

    def _schedule_compressions(self, t: np.ndarray, active: bool) -> None:
        if not active:
            self._next_compression = None
        else:
            if self._next_compression is None:
                self._next_compression = float(t[0])
            period = 60.0 / CPR_RATE
            while self._next_compression <= t[-1]:
                self._compressions.append(self._next_compression)
                self.circulation.compress(self._next_compression)
                self._next_compression += period
        self._compressions = [c for c in self._compressions if c > t[0] - 0.6]

    def _compression_artifact(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What each compression does to the ECG and to the impedance signal."""
        ecg = np.zeros(t.size)
        resp = np.zeros(t.size)
        for c in self._compressions:
            tau = t - c
            ecg += CPR_ECG_MV * (np.exp(-0.5 * ((tau - 0.07) / 0.045) ** 2)
                                 - 0.35 * np.exp(-0.5 * ((tau - 0.22) / 0.07) ** 2))
            resp += CPR_RESP_GAIN * np.exp(-0.5 * ((tau - 0.10) / 0.06) ** 2)
        return ecg, resp

    # -- artifacts ------------------------------------------------------------
    def _build_motion(self) -> np.ndarray:
        """A detrended, Hann-enveloped random walk: violent, then back to baseline."""
        n = int(MOTION_DURATION_S * self.fs)
        walk = np.cumsum(self.rng.normal(0.0, 1.0, n))
        walk -= np.linspace(walk[0], walk[-1], n)     # ends where it started
        peak = np.max(np.abs(walk))
        if peak == 0.0:
            peak = 1.0
        return MOTION_AMPLITUDE * np.hanning(n) * (walk / peak)

    def _next_motion(self, n: int) -> np.ndarray:
        if self._motion is None:
            return np.zeros(n)
        take = self._motion[self._motion_pos:self._motion_pos + n]
        self._motion_pos += n
        if self._motion_pos >= self._motion.size:
            self._motion = None
            self._motion_pos = 0
        if take.size < n:
            take = np.concatenate([take, np.zeros(n - take.size)])
        return take

    # -- main entry point -----------------------------------------------------
    def next_chunk(self, n: int = CHUNK_SAMPLES) -> dict[str, np.ndarray]:
        """Produce the next ``n`` samples of every channel in :data:`CHANNELS`."""
        t0 = self.elapsed
        t = t0 + np.arange(n, dtype=np.float64) / self.fs

        # A shock can change the rhythm, so resolve it before reading the
        # parameters the rest of this chunk is built from.
        shock_kind = self.state.consume_shock()
        if shock_kind is not None:
            outcome, resulting = self._deliver_shock(shock_kind, t0)
            self.pending_report = (shock_kind, outcome, resulting)
            self.shock_log.append((t0, shock_kind, outcome, resulting))

        params = self.state.snapshot()
        rhythm = RHYTHMS.get(params.cardiac_rhythm, RHYTHMS["Sinus"])
        self._resp_spec = RESP_SPECS.get(params.resp_pattern, RESP_SPECS["Normal"])
        resp_hz = self.respiratory_rate_hz(params)
        # No-op unless the targets moved; recalibration takes a few milliseconds.
        self.circulation.calibrate(params.systolic, params.diastolic)

        # One-shot events, consumed exactly once each.
        if self.state.consume_pvc():
            self._inject_pvc(t0, params.heart_rate)
        if self.state.consume_motion():
            self._motion = self._build_motion()
            self._motion_pos = 0

        self._schedule_until(t[-1] + LOOKAHEAD_S, params.heart_rate, resp_hz,
                             params.cardiac_rhythm)
        self._commit_beats(t[-1] + 1.0 / self.fs, rhythm.perfusion)
        self._schedule_compressions(t, params.cpr_active)
        self._prune(t0)

        # Respiration channel, phase-continuous across chunks.
        spec = self._resp_spec
        u = self._resp_u + resp_hz * np.arange(n, dtype=np.float64) / self.fs
        resp = spec.amplitude * skewed_respiration(u, spec.inspiratory)
        if spec.envelope != "none":
            resp = resp * resp_envelope(spec.envelope, t)
        self._resp_u = float(np.mod(self._resp_u + resp_hz * n / self.fs, 1.0))

        # ECG: complexes (or a baseline override), then the noise stack.
        if rhythm.baseline == "vfib":
            ecg = self._fibrillation(t)
        elif rhythm.baseline == "flat":
            ecg = np.zeros(n, dtype=np.float64)
        else:
            ecg = self._render_beats(t)
            if rhythm.baseline == "afib":
                ecg += fibrillatory_waves(t)
            elif rhythm.baseline == "flutter":
                ecg += flutter_waves(t)

        if rhythm.baseline != "flat":
            # Asystole is the one rhythm that should read as a true flat line;
            # everywhere else the chest movement sways the ECG baseline.
            ecg += RESP_ECG_GAIN * resp

        if params.baseline_amplitude:
            ecg += params.baseline_amplitude * np.sin(2.0 * np.pi * BASELINE_FREQUENCY * t)
        if params.mains_amplitude:
            ecg += params.mains_amplitude * np.sin(2.0 * np.pi * MAINS_FREQUENCY * t)
        if params.gaussian_sigma:
            ecg += self.rng.normal(0.0, params.gaussian_sigma, n)

        motion = self._next_motion(n)
        ecg += motion
        ecg += self._next_shock(n)      # discharge artifact and amplifier recovery

        # Circulation: pressure and pleth are driven by the beats committed above.
        abp, pleth = self.circulation.render(t)
        # Breathing modulates the pleth only through the blood it moves, so the
        # variation scales with perfusion: a pulseless patient has a flat pleth.
        flow = max(rhythm.perfusion, 0.3 if params.cpr_active else 0.0)
        pleth = pleth + PLETH_RESP_GAIN * flow * resp + PLETH_MOTION_GAIN * motion

        if self._compressions:
            cpr_ecg, cpr_resp = self._compression_artifact(t)
            ecg += cpr_ecg
            resp = resp + cpr_resp

        # Movement shakes the chest belt too, just less than the ECG electrodes.
        resp = resp + 0.4 * motion
        if params.gaussian_sigma:
            resp = resp + self.rng.normal(0.0, 0.3 * params.gaussian_sigma, n)

        self._n_emitted += n
        return {"ecg": ecg, "resp": resp, "pleth": pleth, "abp": abp}


class SignalGenerator(QThread):
    """Producer thread: fills the ring buffer in real time.

    Pacing runs against a monotonic deadline rather than a bare fixed sleep, so
    the sample clock stays locked to the wall clock and the displayed rate
    matches the commanded BPM.  In practice the sleep lands near the nominal
    45 ms once generation cost is subtracted.
    """

    chunk_written = pyqtSignal(int)     # new write_index, emitted per chunk
    shock_delivered = pyqtSignal(str, str, str)   # (kind, outcome, resulting rhythm)

    MAX_LAG_S = 0.5                     # beyond this we stop trying to catch up

    def __init__(
        self,
        state: SimulationState,
        buffer: RingBuffer,
        chunk_samples: int = CHUNK_SAMPLES,
        rng: np.random.Generator | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.buffer = buffer
        self.chunk_samples = int(chunk_samples)
        self.engine = WaveformEngine(state, buffer.sample_rate, rng)

    def run(self) -> None:
        period = self.chunk_samples / self.buffer.sample_rate
        deadline = time.perf_counter()
        while not self.isInterruptionRequested():
            block = self.engine.next_chunk(self.chunk_samples)
            self.chunk_written.emit(self.buffer.write(block))

            report = self.engine.pending_report
            if report is not None:
                self.engine.pending_report = None
                self.shock_delivered.emit(*report)

            deadline += period
            slack = deadline - time.perf_counter()
            if slack > 0:
                self.msleep(int(slack * 1000))
            elif slack < -self.MAX_LAG_S:
                deadline = time.perf_counter()   # too far behind; resynchronise

    def stop(self, timeout_ms: int = 2000) -> bool:
        """Ask the thread to finish and wait for it.  Safe to call twice."""
        self.requestInterruption()
        return self.wait(timeout_ms)
