"""Arterial pressure and pulse oximetry, driven by the simulated heartbeats.

Pressure comes from a two-element Windkessel: the arterial tree as a compliance
``C`` draining through a peripheral resistance ``R`` towards venous pressure.
Every ventricular beat ejects a stroke volume into it as a half-sine flow pulse:

    C dP/dt = Q(t) - (P - Pv) / R

Nothing here is scripted per rhythm.  Hypotension in VT, the weak pulse after a
PVC, the stronger one after its compensatory pause, and the pressure decaying to
a flat line in VF all fall out of how much blood each beat actually moves - which
depends on how long the ventricle had to fill, and on how effectively the rhythm
contracts it.

The equation is solved exactly for piecewise-constant inflow, so it is stable at
any rate and needs no small-step integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Reference patient -------------------------------------------------------
REFERENCE_HR = 72.0          # bpm at which the target pressures are calibrated
SV_REF = 70.0                # ml, normal stroke volume
VENOUS_MMHG = 6.0            # pressure the arteries drain towards
MIN_PULSE_PRESSURE = 10.0    # mmHg; diastolic is held at least this far below systolic

# --- Timing ------------------------------------------------------------------
PEP_S = 0.07                 # pre-ejection period: R wave -> aortic valve opening
ART_TRANSIT_S = 0.04         # aortic root -> radial arterial line
PLETH_TRANSIT_S = 0.20       # aortic root -> fingertip
FILL_TAU = 0.22              # s; ventricular filling time constant
ISOVOLUMIC_S = 0.06          # relaxation before filling starts

# --- Beat quality ------------------------------------------------------------
PVC_EFFICIENCY = 0.5         # an ectopic beat contracts the ventricle out of sequence

# --- Waveform detail ---------------------------------------------------------
NOTCH_DIP = 3.5              # mmHg, dicrotic notch at aortic valve closure
NOTCH_REBOUND = 2.5          # mmHg, the dicrotic wave that follows it

# --- Chest compressions ------------------------------------------------------
CPR_RATE = 110.0             # compressions per minute (guideline 100-120)
CPR_STROKE_ML = 22.0         # forward flow per compression, ~30 % of normal
CPR_EJECTION_S = 0.22


@dataclass(slots=True)
class Ejection:
    """One pulse of flow into the aorta."""

    start: float             # s, aortic valve opening
    duration: float          # s
    volume: float            # ml
    compression: bool = False


def ejection_duration(rr: float) -> float:
    """Systolic ejection time shortens with rate, roughly as sqrt(RR)."""
    return float(np.clip(0.30 * np.sqrt(rr / 0.8), 0.17, 0.34))


def filling_fraction(rr: float) -> float:
    """How full the ventricle gets in the diastole this interval leaves it."""
    fill_time = max(rr - ejection_duration(rr) - ISOVOLUMIC_S, 0.0)
    return 1.0 - float(np.exp(-fill_time / FILL_TAU))


_REFERENCE_FILL = filling_fraction(60.0 / REFERENCE_HR)


def stroke_volume(rr: float, perfusion: float, ectopic: bool = False) -> float:
    """Frank-Starling: stroke volume follows the filling the preceding interval allowed."""
    volume = SV_REF * perfusion * filling_fraction(rr) / _REFERENCE_FILL
    return volume * (PVC_EFFICIENCY if ectopic else 1.0)


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def pleth_pulse(tau: np.ndarray) -> np.ndarray:
    """One fingertip volume pulse: a fast upstroke, slow runoff, dicrotic bump."""
    width = np.where(tau < 0.12, 0.05, 0.15)
    main = np.exp(-0.5 * ((tau - 0.12) / width) ** 2)
    return main + 0.22 * _gauss(tau, 0.40, 0.07)


def windkessel(u0: float, inflow: np.ndarray, resistance: float, compliance: float,
               dt: float) -> tuple[np.ndarray, float]:
    """Integrate ``C du/dt = q - u/R`` exactly, where ``u = P - Pv``.

    Returns the pressure excess at every input sample and the state after the
    last one.  Vectorised with a scaled cumulative sum, evaluated in blocks so
    the scale factor never overflows.
    """
    decay = float(np.exp(-dt / (resistance * compliance)))
    gain = resistance * (1.0 - decay)
    out = np.empty(inflow.size, dtype=np.float64)
    u = float(u0)
    block = 500
    for start in range(0, inflow.size, block):
        q = inflow[start:start + block]
        k = np.arange(q.size, dtype=np.float64)
        grow = decay ** -(k + 1.0)
        weighted = np.cumsum(q * grow)
        before = np.concatenate(([0.0], weighted[:-1]))
        out[start:start + q.size] = decay ** k * (u + gain * before)
        u = decay ** q.size * (u + gain * weighted[-1])
    return out, u


class Circulation:
    """Arterial pressure and plethysmograph for the beats it is told about.

    The engine calls :meth:`beat` once for each ventricular contraction, in time
    order, and :meth:`compress` for each chest compression during CPR; then
    :meth:`render` turns whatever flow is in flight into samples.
    """

    def __init__(self, sample_rate: int, systolic: float = 120.0,
                 diastolic: float = 80.0) -> None:
        self.fs = int(sample_rate)
        self.dt = 1.0 / self.fs
        self._ejections: list[Ejection] = []
        self._last_beat: float | None = None
        self.targets: tuple[float, float] | None = None
        self.resistance = 1.0
        self.compliance = 1.5
        self.map_target = 93.0
        self.calibrate(systolic, diastolic)
        self._u = self.map_target - VENOUS_MMHG      # start at mean pressure

    # -- calibration ------------------------------------------------------------
    def _reference_cycle(self, resistance: float, compliance: float,
                         start: float) -> tuple[float, float, float]:
        """Steady-state (sys, dia, mean) at the reference rate for given R and C."""
        rr = 60.0 / REFERENCE_HR
        beats = 12
        n = int(beats * rr * self.fs)
        t = np.arange(n, dtype=np.float64) * self.dt
        te = ejection_duration(rr)
        inflow = np.zeros(n)
        for k in range(beats):
            tau = t - k * rr
            live = (tau >= 0.0) & (tau < te)
            inflow[live] += SV_REF * np.pi / (2.0 * te) * np.sin(np.pi * tau[live] / te)
        u, _ = windkessel(start - VENOUS_MMHG, inflow, resistance, compliance, self.dt)
        last = u[-int(rr * self.fs):] + VENOUS_MMHG
        return float(last.max()), float(last.min()), float(last.mean())

    def calibrate(self, systolic: float, diastolic: float) -> None:
        """Choose R and C so the reference patient reads ``systolic/diastolic``.

        Resistance sets the pressure level (mean = Pv + cardiac output x R) and
        compliance the pulse pressure (roughly stroke volume / C), so each is
        corrected against the one quantity it mostly controls.  The level is
        matched at the systolic/diastolic midpoint rather than at the textbook
        ``dia + PP/3`` MAP estimate: this waveform's true mean is not exactly
        that, and targeting it leaves both readings a few mmHg off.
        """
        diastolic = min(diastolic, systolic - MIN_PULSE_PRESSURE)
        if self.targets == (systolic, diastolic):
            return
        midpoint = (systolic + diastolic) / 2.0
        pulse = systolic - diastolic
        output = SV_REF * REFERENCE_HR / 60.0

        resistance = (midpoint - VENOUS_MMHG) / output
        compliance = SV_REF / pulse
        mean_m = midpoint
        for _ in range(12):
            sys_m, dia_m, mean_m = self._reference_cycle(resistance, compliance, midpoint)
            level = (sys_m + dia_m) / 2.0
            resistance *= (midpoint - VENOUS_MMHG) / max(level - VENOUS_MMHG, 1e-3)
            compliance *= max(sys_m - dia_m, 1e-3) / pulse

        self.resistance, self.compliance = resistance, compliance
        self.targets = (systolic, diastolic)
        self.map_target = mean_m

    # -- events -----------------------------------------------------------------
    def beat(self, r_time: float, perfusion: float, ectopic: bool = False) -> float:
        """Register a ventricular contraction; returns its stroke volume in ml."""
        rr = (r_time - self._last_beat) if self._last_beat is not None \
            else 60.0 / REFERENCE_HR
        self._last_beat = r_time
        volume = stroke_volume(rr, perfusion, ectopic)
        if volume > 0.0:
            self._ejections.append(Ejection(r_time + PEP_S, ejection_duration(rr), volume))
        return volume

    def compress(self, t: float) -> None:
        """One chest compression: forward flow with no electrical event behind it."""
        self._ejections.append(Ejection(t, CPR_EJECTION_S, CPR_STROKE_ML, compression=True))

    @property
    def pressure(self) -> float:
        """Current arterial pressure, mmHg."""
        return self._u + VENOUS_MMHG

    # -- rendering --------------------------------------------------------------
    def render(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Arterial pressure (mmHg) and pleth (normalised) over the chunk ``t``."""
        n = t.size
        inflow = np.zeros(n)
        notch = np.zeros(n)
        pleth = np.zeros(n)
        first, last = float(t[0]), float(t[-1])

        for e in self._ejections:
            onset = e.start + ART_TRANSIT_S
            if onset <= last and onset + e.duration >= first:
                tau = t - onset
                live = (tau >= 0.0) & (tau < e.duration)
                inflow[live] += e.volume * np.pi / (2.0 * e.duration) * \
                    np.sin(np.pi * tau[live] / e.duration)

            scale = e.volume / SV_REF
            if not e.compression:
                closure = onset + e.duration
                if closure - 0.1 <= last and closure + 0.2 >= first:
                    notch += scale * (-NOTCH_DIP * _gauss(t, closure, 0.012)
                                      + NOTCH_REBOUND * _gauss(t, closure + 0.035, 0.03))

            finger = e.start + PLETH_TRANSIT_S
            if finger - 0.3 <= last and finger + 1.2 >= first:
                pleth += scale * pleth_pulse(t - finger)

        u, self._u = windkessel(self._u, inflow, self.resistance, self.compliance, self.dt)
        abp = VENOUS_MMHG + u + notch

        horizon = first - 1.5
        self._ejections = [e for e in self._ejections
                           if e.start + PLETH_TRANSIT_S + 1.2 >= horizon]
        return abp, pleth
