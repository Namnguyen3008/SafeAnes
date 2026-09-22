"""Physiological alarms: measured numerics checked against configurable limits.

These complement the rhythm alarms in :mod:`core.pathology` (which know that VF
needs a shock) with the ones a bedside monitor raises on the numbers themselves:
a heart rate out of range, a falling saturation, no arterial pulse, apnoea.

Every alarm names the parameter it belongs to, so the display can flag the right
numeric tile - and carries a text message, because alarm state must never be
conveyed by colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vitals import Vitals

HIGH = "high"          # act now: red, flashing
MEDIUM = "medium"      # attend soon: amber, steady

# Beyond these the alarm escalates from MEDIUM to HIGH regardless of the limits.
CRITICAL_HR_LOW = 40
CRITICAL_HR_HIGH = 160
CRITICAL_SPO2 = 85
CRITICAL_MAP = 50


@dataclass(frozen=True, slots=True)
class AlarmLimits:
    hr_low: int = 50
    hr_high: int = 120
    spo2_low: int = 90
    map_low: int = 65
    rr_low: int = 8
    rr_high: int = 30


# Adjustable range of each limit, for the UI.
LIMIT_RANGES: dict[str, tuple[int, int]] = {
    "hr_low": (25, 100),
    "hr_high": (80, 220),
    "spo2_low": (70, 99),
    "map_low": (40, 90),
    "rr_low": (3, 15),
    "rr_high": (15, 45),
}


@dataclass(frozen=True, slots=True)
class Alarm:
    code: str            # stable identifier, e.g. "hr_high"
    priority: str        # HIGH or MEDIUM
    parameter: str       # the tile it belongs to: hr | spo2 | abp | resp
    message: str

    @property
    def is_high(self) -> bool:
        return self.priority == HIGH


def evaluate(v: Vitals, limits: AlarmLimits = AlarmLimits()) -> list[Alarm]:
    """Every alarm the numerics currently justify, highest priority first."""
    alarms: list[Alarm] = []

    def raise_(code: str, priority: str, parameter: str, message: str) -> None:
        alarms.append(Alarm(code, priority, parameter, message))

    if v.window_s == 0.0 or (v.hr is None and not v.asystole and not v.chaotic
                             and v.mean_pressure is None):
        return alarms                         # not enough signal yet to judge anything

    # -- heart rate --------------------------------------------------------------
    if v.asystole:
        raise_("asystole", HIGH, "hr", "ASYSTOLE")
    elif v.chaotic:
        raise_("hr_unreadable", HIGH, "hr", "HR NOT COUNTABLE")
    elif v.hr is not None:
        if v.hr < limits.hr_low:
            priority = HIGH if v.hr < CRITICAL_HR_LOW else MEDIUM
            raise_("hr_low", priority, "hr", f"HR LOW {v.hr}")
        elif v.hr > limits.hr_high:
            priority = HIGH if v.hr > CRITICAL_HR_HIGH else MEDIUM
            raise_("hr_high", priority, "hr", f"HR HIGH {v.hr}")

    # -- circulation -------------------------------------------------------------
    if v.no_pulse:
        raise_("no_pulse", HIGH, "abp", "NO ARTERIAL PULSE")
    elif v.mean_pressure is not None and v.mean_pressure < limits.map_low:
        priority = HIGH if v.mean_pressure < CRITICAL_MAP else MEDIUM
        raise_("map_low", priority, "abp", f"MAP LOW {v.mean_pressure}")

    # -- oxygenation -------------------------------------------------------------
    if v.spo2 is None:
        if v.mean_pressure is not None:
            raise_("spo2_no_signal", MEDIUM, "spo2", "SpO2 NO PULSE")
    elif v.spo2 < limits.spo2_low:
        priority = HIGH if v.spo2 < CRITICAL_SPO2 else MEDIUM
        raise_("spo2_low", priority, "spo2", f"SpO2 LOW {v.spo2}")

    # -- breathing ---------------------------------------------------------------
    if v.apnoea:
        raise_("apnoea", HIGH, "resp", "APNOEA")
    elif v.resp_rate is not None:
        if v.resp_rate < limits.rr_low:
            raise_("rr_low", MEDIUM, "resp", f"RR LOW {v.resp_rate}")
        elif v.resp_rate > limits.rr_high:
            raise_("rr_high", MEDIUM, "resp", f"RR HIGH {v.resp_rate}")

    alarms.sort(key=lambda a: 0 if a.is_high else 1)
    return alarms


class AlarmTracker:
    """Debounces alarms the way a monitor does, so they do not flap.

    A condition must persist for ``onset_s`` before it is raised, and stay
    absent for ``clear_s`` before it clears.  Without this, a signal sitting on
    a threshold - or VF under CPR, where the rate meter alternates between
    "not countable" and counting compression artifacts - raises and clears an
    alarm every update, which buries the event log and trains people to ignore
    alarms.
    """

    def __init__(self, onset_s: float = 1.0, clear_s: float = 3.0) -> None:
        self.onset_s = onset_s
        self.clear_s = clear_s
        self._pending: dict[str, float] = {}               # code -> first seen
        self._active: dict[str, tuple[Alarm, float]] = {}  # code -> (alarm, last seen)

    def reset(self) -> None:
        self._pending.clear()
        self._active.clear()

    @property
    def active(self) -> list[Alarm]:
        alarms = [alarm for alarm, _ in self._active.values()]
        return sorted(alarms, key=lambda a: 0 if a.is_high else 1)

    def update(self, alarms: list[Alarm], now: float
               ) -> tuple[list[Alarm], list[Alarm], list[Alarm]]:
        """Feed the latest evaluation.  Returns (active, newly raised, cleared)."""
        raised: list[Alarm] = []
        present = {a.code: a for a in alarms}

        for code, alarm in present.items():
            if code in self._active:
                self._active[code] = (alarm, now)         # keep the message current
                continue
            first = self._pending.setdefault(code, now)
            if now - first >= self.onset_s:
                self._active[code] = (alarm, now)
                self._pending.pop(code, None)
                raised.append(alarm)
        for code in list(self._pending):
            if code not in present:
                del self._pending[code]                    # a blip, never raised

        cleared = [alarm for code, (alarm, seen) in self._active.items()
                   if code not in present and now - seen >= self.clear_s]
        for alarm in cleared:
            del self._active[alarm.code]
        return self.active, raised, cleared


def by_parameter(alarms: list[Alarm]) -> dict[str, Alarm]:
    """The most urgent alarm for each tile."""
    worst: dict[str, Alarm] = {}
    for alarm in alarms:                      # already highest-priority first
        worst.setdefault(alarm.parameter, alarm)
    return worst
