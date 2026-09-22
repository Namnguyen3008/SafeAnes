"""Thread-safe global simulation state shared between the UI and the generator.

The GUI thread writes parameters (slider moves, button clicks); the generator
QThread reads them once per chunk.  All access goes through a single lock, and
the producer pulls an immutable :class:`StateSnapshot` so that a slider moved
half-way through a chunk cannot tear the parameters it is working from.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Literal

from .pathology import (  # noqa: F401  (re-exported)
    CARDIAC_RHYTHMS,
    RESP_PATTERNS,
    SHOCK_KINDS,
)

# --- Acquisition constants ---------------------------------------------------
SAMPLE_RATE = 1000          # Hz
BUFFER_SECONDS = 10
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS   # 10,000 samples
CHUNK_MS = 50
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000   # 50 samples per producer tick

# --- Parameter limits (also used to configure the UI sliders) ----------------
HR_RANGE = (30.0, 200.0)        # bpm
RR_RANGE = (8.0, 30.0)          # breaths per minute
MAINS_RANGE = (0.0, 0.5)        # mV
BASELINE_RANGE = (0.0, 0.5)     # mV
GAUSSIAN_RANGE = (0.0, 0.2)     # mV (sigma)
SPO2_RANGE = (60.0, 100.0)      # % target saturation
SYSTOLIC_RANGE = (70.0, 220.0)  # mmHg, baseline the circulation is calibrated to
DIASTOLIC_RANGE = (30.0, 130.0) # mmHg

MAINS_FREQUENCY = 50.0          # Hz
BASELINE_FREQUENCY = 0.5        # Hz

MOTION_DURATION_S = 1.5         # how long an injected motion artifact lasts

# --- Pathology options (also used to populate the UI dropdowns) --------------
# The catalogue in core.pathology is the single source of truth; these names are
# re-exported here so the UI and the state validator agree by construction.
# `test_literals_match_the_catalogue` fails if the aliases below drift from it.
CardiacRhythm = Literal[
    "Sinus",
    "AFib", "Atrial Flutter", "SVT", "Junctional",
    "1st Degree AV Block", "Mobitz I (Wenckebach)", "Mobitz II", "3rd Degree AV Block",
    "PVC Bigeminy", "V-Tach", "Torsades de Pointes", "V-Fib",
    "Idioventricular", "Asystole",
    "STEMI", "Ischemia (ST Dep.)", "Hyperkalemia",
    "Paced",
]
RespPattern = Literal[
    "Normal", "Cheyne-Stokes", "Kussmaul", "Biot (Ataxic)", "Apnoea", "Agonal",
]

# Continuous parameters are clamped to these; categorical ones are checked
# against their allowed values instead.
_LIMITS = {
    "heart_rate": HR_RANGE,
    "respiratory_rate": RR_RANGE,
    "mains_amplitude": MAINS_RANGE,
    "baseline_amplitude": BASELINE_RANGE,
    "gaussian_sigma": GAUSSIAN_RANGE,
    "spo2_target": SPO2_RANGE,
    "systolic": SYSTOLIC_RANGE,
    "diastolic": DIASTOLIC_RANGE,
}
_CHOICES = {
    "cardiac_rhythm": CARDIAC_RHYTHMS,
    "resp_pattern": RESP_PATTERNS,
}
_FLAGS = ("alarm_enabled", "cpr_active")   # plain booleans: neither clamped nor enumerated


def _clamp(value: float, limits: tuple[float, float]) -> float:
    low, high = limits
    return low if value < low else high if value > high else value


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable copy of the tunable parameters, safe to read without a lock."""

    heart_rate: float = 72.0
    respiratory_rate: float = 15.0
    mains_amplitude: float = 0.0
    baseline_amplitude: float = 0.0
    gaussian_sigma: float = 0.0
    cardiac_rhythm: CardiacRhythm = "Sinus"
    resp_pattern: RespPattern = "Normal"
    alarm_enabled: bool = True
    spo2_target: float = 98.0
    systolic: float = 120.0
    diastolic: float = 80.0
    cpr_active: bool = False


class SimulationState:
    """Mutable, lock-guarded holder for :class:`StateSnapshot` plus one-shot flags.

    Continuous parameters are set with :meth:`update` and read with
    :meth:`snapshot`.  Event flags (motion artifact, PVC) are raised by the UI
    and consumed exactly once by the generator via the ``consume_*`` methods,
    which test-and-clear atomically so a single click yields a single event.
    """

    def __init__(self, **overrides: float) -> None:
        self._lock = threading.Lock()
        self._params = StateSnapshot()
        self._inject_motion = False
        self._inject_pvc = False
        self._pending_shock: str | None = None
        if overrides:
            self.update(**overrides)

    # -- continuous parameters ------------------------------------------------
    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return self._params

    def update(self, **kwargs) -> StateSnapshot:
        """Set one or more parameters.

        Numeric parameters are clamped to their published ranges; categorical
        ones (rhythm, breathing pattern) are validated against their allowed
        values, so a mistyped dropdown entry fails loudly instead of silently
        falling back to normal sinus rhythm.
        """
        unknown = set(kwargs) - set(StateSnapshot.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown state parameter(s): {sorted(unknown)}")

        values = {}
        for key, raw in kwargs.items():
            if key in _FLAGS:
                values[key] = bool(raw)
            elif key in _CHOICES:
                if raw not in _CHOICES[key]:
                    raise ValueError(
                        f"{key} must be one of {list(_CHOICES[key])}, got {raw!r}"
                    )
                values[key] = raw
            else:
                values[key] = _clamp(float(raw), _LIMITS[key])

        with self._lock:
            # frozen dataclass -> rebind rather than mutate, so any snapshot
            # already handed out to the producer stays internally consistent.
            self._params = replace(self._params, **values)
            return self._params

    # Convenience accessors used by the UI wiring.
    @property
    def heart_rate(self) -> float:
        return self.snapshot().heart_rate

    @heart_rate.setter
    def heart_rate(self, value: float) -> None:
        self.update(heart_rate=value)

    @property
    def respiratory_rate(self) -> float:
        return self.snapshot().respiratory_rate

    @respiratory_rate.setter
    def respiratory_rate(self, value: float) -> None:
        self.update(respiratory_rate=value)

    # -- one-shot events ------------------------------------------------------
    def trigger_motion(self) -> None:
        with self._lock:
            self._inject_motion = True

    def consume_motion(self) -> bool:
        """Return True at most once per :meth:`trigger_motion`, clearing the flag."""
        with self._lock:
            fired, self._inject_motion = self._inject_motion, False
            return fired

    def trigger_pvc(self) -> None:
        with self._lock:
            self._inject_pvc = True

    def consume_pvc(self) -> bool:
        """Return True at most once per :meth:`trigger_pvc`, clearing the flag."""
        with self._lock:
            fired, self._inject_pvc = self._inject_pvc, False
            return fired

    def trigger_shock(self, kind: str) -> None:
        """Request one shock.  ``kind`` is 'defibrillate' or 'cardiovert'."""
        if kind not in SHOCK_KINDS:
            raise ValueError(f"shock kind must be one of {list(SHOCK_KINDS)}, got {kind!r}")
        with self._lock:
            self._pending_shock = kind

    def consume_shock(self) -> str | None:
        """Return the pending shock kind at most once, clearing it."""
        with self._lock:
            kind, self._pending_shock = self._pending_shock, None
            return kind

    @property
    def pending_shock(self) -> str | None:
        """Peek at the queued shock without consuming it (diagnostics only)."""
        with self._lock:
            return self._pending_shock

    @property
    def inject_motion(self) -> bool:
        """Peek at the motion flag without consuming it (diagnostics only)."""
        with self._lock:
            return self._inject_motion

    @property
    def inject_pvc(self) -> bool:
        """Peek at the PVC flag without consuming it (diagnostics only)."""
        with self._lock:
            return self._inject_pvc

    def __repr__(self) -> str:
        p = self.snapshot()
        return (
            f"SimulationState(hr={p.heart_rate:.0f}, rr={p.respiratory_rate:.0f}, "
            f"mains={p.mains_amplitude:.3f}, baseline={p.baseline_amplitude:.3f}, "
            f"gauss={p.gaussian_sigma:.3f})"
        )
