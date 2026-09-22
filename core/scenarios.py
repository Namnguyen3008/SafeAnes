"""Scripted clinical scenarios: timed changes to the simulated patient.

A scenario is a list of steps, each applying a set of state changes (and
optionally one-shot events such as a PVC) at a time offset from its start.
Every scenario begins by resetting the patient to a known baseline, so it plays
out the same way whatever was on screen before - while leaving the signal
settings (noise, alarm toggle) as the operator had them.

The player is clock-agnostic: it is told the time on each :meth:`tick`, so the
UI drives it from a wall-clock timer and the tests drive it with plain numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

BASELINE: Mapping[str, Any] = MappingProxyType({
    "heart_rate": 72.0,
    "respiratory_rate": 15.0,
    "cardiac_rhythm": "Sinus",
    "resp_pattern": "Normal",
    "spo2_target": 98.0,
    "systolic": 120.0,
    "diastolic": 80.0,
    "cpr_active": False,
})

EVENTS = ("pvc", "motion")


@dataclass(frozen=True, slots=True)
class Step:
    at: float                                   # seconds after the scenario starts
    note: str
    changes: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    summary: str
    steps: tuple[Step, ...]
    tail_s: float = 20.0                        # how long to hold the final state

    @property
    def duration(self) -> float:
        return self.steps[-1].at + self.tail_s


def _s(at: float, note: str, events: tuple[str, ...] = (), **changes: Any) -> Step:
    return Step(at, note, dict(changes), events)


SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario(
        "Cardiac arrest: VT to VF",
        "Ventricular ectopy degenerates through VT into VF arrest. "
        "Start CPR and defibrillate.",
        (
            _s(0, "Stable, mildly tachycardic", heart_rate=92, systolic=128, diastolic=82,
               spo2_target=97),
            _s(10, "Frequent ventricular ectopy", cardiac_rhythm="PVC Bigeminy"),
            _s(22, "Sustained ventricular tachycardia - check for a pulse",
               cardiac_rhythm="V-Tach"),
            _s(36, "VF arrest - start CPR, defibrillate", cardiac_rhythm="V-Fib"),
        ),
    ),
    Scenario(
        "Progressive hypoxia",
        "Falling saturation drives a compensatory tachycardia and tachypnoea, "
        "then a pre-arrest bradycardia.",
        (
            _s(0, "Mild respiratory distress", heart_rate=84, respiratory_rate=18,
               spo2_target=96),
            _s(15, "Desaturating", heart_rate=98, respiratory_rate=24, spo2_target=91),
            _s(30, "Hypoxaemic, working hard to breathe", heart_rate=112,
               respiratory_rate=28, spo2_target=86),
            _s(45, "Severe hypoxaemia", heart_rate=124, respiratory_rate=30, spo2_target=80),
            _s(60, "Tiring - hypoxic bradycardia, pre-arrest", heart_rate=48,
               respiratory_rate=9, spo2_target=72, systolic=86, diastolic=50),
        ),
    ),
    Scenario(
        "AF with rapid ventricular response",
        "New atrial fibrillation with a fast rate and falling blood pressure. "
        "Synchronised cardioversion if unstable.",
        (
            _s(0, "Sinus rhythm", heart_rate=88, systolic=132, diastolic=84),
            _s(10, "New AF with rapid ventricular response", cardiac_rhythm="AFib",
               heart_rate=148, systolic=112, diastolic=72),
            _s(30, "Haemodynamic compromise - consider synchronised cardioversion",
               heart_rate=165, systolic=94, diastolic=62),
        ),
    ),
    Scenario(
        "Evolving heart block",
        "AV conduction deteriorates step by step to complete heart block.",
        (
            _s(0, "Sinus rhythm", heart_rate=76),
            _s(12, "PR prolonged - first-degree block", cardiac_rhythm="1st Degree AV Block"),
            _s(24, "Wenckebach - PR lengthens until a beat drops",
               cardiac_rhythm="Mobitz I (Wenckebach)"),
            _s(36, "Mobitz II - beats drop without warning", cardiac_rhythm="Mobitz II"),
            _s(48, "Complete heart block - prepare to pace",
               cardiac_rhythm="3rd Degree AV Block"),
        ),
    ),
    Scenario(
        "Hyperkalaemia",
        "Rising potassium: peaked T waves, then a wide slow escape rhythm, then VF.",
        (
            _s(0, "Sinus rhythm", heart_rate=80),
            _s(15, "Peaked T waves", cardiac_rhythm="Hyperkalemia", heart_rate=78),
            _s(35, "Wide, slow escape rhythm", cardiac_rhythm="Idioventricular"),
            _s(50, "Ventricular fibrillation", cardiac_rhythm="V-Fib"),
        ),
    ),
    Scenario(
        "Opioid overdose",
        "Respiratory depression progressing to apnoea, with hypoxic bradycardia.",
        (
            _s(0, "Drowsy", heart_rate=70, respiratory_rate=14, spo2_target=97),
            _s(15, "Respiratory depression", respiratory_rate=9, spo2_target=93),
            _s(30, "Agonal breathing", resp_pattern="Agonal", spo2_target=86, heart_rate=58),
            _s(45, "Apnoeic - ventilate", resp_pattern="Apnoea", spo2_target=78,
               heart_rate=48, systolic=96, diastolic=58),
            _s(60, "Hypoxic bradycardia", spo2_target=68, heart_rate=38),
        ),
    ),
)}


class ScenarioPlayer:
    """Applies a scenario's steps to the simulation state as time passes."""

    def __init__(self, state) -> None:
        self.state = state
        self.scenario: Scenario | None = None
        self._started = 0.0
        self._next = 0

    @property
    def running(self) -> bool:
        return self.scenario is not None

    def start(self, name: str, now: float) -> list[Step]:
        """Reset the patient to baseline, then apply every step already due."""
        self.scenario = SCENARIOS[name]
        self._started = now
        self._next = 0
        self.state.update(**BASELINE)
        return self.tick(now)

    def stop(self) -> None:
        self.scenario = None
        self._next = 0

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self._started) if self.running else 0.0

    def progress(self, now: float) -> float:
        """Fraction of the scenario played, 0..1."""
        if not self.running:
            return 0.0
        return min(1.0, self.elapsed(now) / self.scenario.duration)

    @property
    def next_step(self) -> Step | None:
        if not self.running or self._next >= len(self.scenario.steps):
            return None
        return self.scenario.steps[self._next]

    def tick(self, now: float) -> list[Step]:
        """Apply every step whose time has come; returns them, oldest first.

        When the final step has been held for the scenario's tail, the player
        stops by itself.  The patient is left where the scenario put them.
        """
        if not self.running:
            return []
        applied: list[Step] = []
        elapsed = self.elapsed(now)
        steps = self.scenario.steps
        while self._next < len(steps) and steps[self._next].at <= elapsed:
            step = steps[self._next]
            if step.changes:
                self.state.update(**step.changes)
            for event in step.events:
                if event == "pvc":
                    self.state.trigger_pvc()
                elif event == "motion":
                    self.state.trigger_motion()
            applied.append(step)
            self._next += 1
        if elapsed >= self.scenario.duration:
            self.stop()
        return applied
