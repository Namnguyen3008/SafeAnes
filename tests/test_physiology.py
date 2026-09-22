"""Circulation, measured vitals, alarm limits, CPR and scenarios.

The circulation is checked by what it produces - pressures, pulse timing, the
weak pulse after a PVC - rather than by its parameters, and the vitals by
comparing what the analyser measures against the ground truth the engine logs.

Run directly (``python tests/test_physiology.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alarms import HIGH, MEDIUM, AlarmLimits, by_parameter, evaluate   # noqa: E402
from core.buffer import RingBuffer                                           # noqa: E402
from core.generator import CHANNELS, WaveformEngine                          # noqa: E402
from core.hemodynamics import (                                              # noqa: E402
    CPR_RATE,
    PVC_EFFICIENCY,
    VENOUS_MMHG,
    Circulation,
    stroke_volume,
    windkessel,
)
from core.pathology import RHYTHMS                                           # noqa: E402
from core.scenarios import BASELINE, SCENARIOS, ScenarioPlayer               # noqa: E402
from core.state import CARDIAC_RHYTHMS, SimulationState, StateSnapshot       # noqa: E402
from core.vitals import Vitals, VitalsAnalyzer, analyze, detect_qrs          # noqa: E402

FS = 1000


def simulate(seconds: float, seed: int = 3, engine: WaveformEngine | None = None,
             **params):
    """Run the engine and return (engine, {channel: samples})."""
    engine = engine or WaveformEngine(SimulationState(**params), FS,
                                      np.random.default_rng(seed))
    blocks = [engine.next_chunk(50) for _ in range(int(seconds * 20))]
    return engine, {k: np.concatenate([b[k] for b in blocks]) for k in CHANNELS}


def logged_rate(engine: WaveformEngine) -> float:
    """Ground-truth ventricular rate over the last few beats, from the engine."""
    times = sorted(t for t, kind in engine.beat_log
                   if kind != "p" and engine.elapsed - 10.0 < t <= engine.elapsed)
    return 60.0 / float(np.diff(times[-7:]).mean())


def pulse_peaks(abp: np.ndarray) -> np.ndarray:
    """Systolic peaks: local maxima that stand well above the preceding trough."""
    idx = []
    for i in range(300, abp.size - 300):
        window = abp[i - 250:i + 250]
        if abp[i] == window.max() and abp[i] - window.min() > 4.0:
            if not idx or i - idx[-1] > 200:
                idx.append(i)
    return np.asarray(idx)


# --- the circulation ----------------------------------------------------------
def test_windkessel_integration_is_exact():
    q = np.random.default_rng(0).uniform(0, 300, 2345)
    fast, end = windkessel(80.0, q, 1.1, 1.7, 1.0 / FS)
    a = np.exp(-(1.0 / FS) / (1.1 * 1.7))
    c = 1.1 * (1 - a)
    u, slow = 80.0, []
    for x in q:
        slow.append(u)
        u = a * u + c * x
    assert np.abs(np.array(slow) - fast).max() < 1e-9
    assert abs(u - end) < 1e-9


def test_calibration_hits_the_target_pressures():
    for systolic, diastolic in ((120, 80), (90, 55), (160, 95), (200, 120)):
        _, d = simulate(15.0, systolic=systolic, diastolic=diastolic)
        v = analyze(d, FS, 98)
        assert abs(v.systolic - systolic) <= 3, f"{systolic}/{diastolic}: sys {v.systolic}"
        assert abs(v.diastolic - diastolic) <= 3, f"{systolic}/{diastolic}: dia {v.diastolic}"


def test_changing_the_target_recalibrates_live():
    engine, _ = simulate(8.0)
    engine.state.update(systolic=160, diastolic=100)
    _, d = simulate(15.0, engine=engine)
    v = analyze(d, FS, 98)
    assert abs(v.systolic - 160) <= 4 and abs(v.diastolic - 100) <= 4


def test_frank_starling_shorter_filling_means_a_weaker_beat():
    rates = (40, 60, 80, 120, 160, 200)
    volumes = [stroke_volume(60.0 / bpm, 1.0) for bpm in rates]
    assert volumes == sorted(volumes, reverse=True), volumes
    assert stroke_volume(0.8, 1.0, ectopic=True) == PVC_EFFICIENCY * stroke_volume(0.8, 1.0)


def test_diastolic_calibration_never_inverts_the_pulse():
    c = Circulation(FS, 100, 110)            # diastolic above systolic: clamp, don't explode
    assert c.targets[1] < c.targets[0]


def test_arrest_rhythms_have_no_pulse_and_pressure_falls_to_venous():
    for rhythm in ("V-Fib", "Asystole"):
        _, d = simulate(20.0, cardiac_rhythm=rhythm)
        v = analyze(d, FS, 98)
        assert v.no_pulse, f"{rhythm} has a pulse"
        assert v.mean_pressure < VENOUS_MMHG + 3, f"{rhythm}: MAP {v.mean_pressure}"
        assert np.ptp(d["pleth"][-4000:]) < 0.1, f"{rhythm}: pleth still pulsatile"


def test_systolic_and_diastolic_come_from_the_same_beats():
    """After a flat stretch (arrest, say), the reading must describe the new
    beats, not pair their peaks with the floor the arrest left behind."""
    _, pulsing = simulate(10.0)
    flat = np.full(2500, 12.0)
    abp = np.concatenate((flat, pulsing["abp"][-1800:]))   # 2.5 s flat, then ~2 beats
    v = analyze({"ecg": np.zeros(abp.size), "abp": abp}, FS, 98)
    assert v.systolic is not None
    assert v.diastolic > 60, f"diastolic {v.diastolic} still reflects the flat stretch"


def test_ventricular_tachycardia_is_hypotensive():
    _, d = simulate(15.0, cardiac_rhythm="V-Tach")
    v = analyze(d, FS, 98)
    assert v.mean_pressure < 50, f"VT MAP {v.mean_pressure}"


def test_complete_heart_block_widens_the_pulse_pressure():
    _, sinus = simulate(15.0)
    _, block = simulate(15.0, cardiac_rhythm="3rd Degree AV Block")
    s, b = analyze(sinus, FS, 98), analyze(block, FS, 98)
    assert (b.systolic - b.diastolic) > (s.systolic - s.diastolic)
    assert b.mean_pressure < s.mean_pressure - 15, "slow escape rhythm should be hypotensive"


def test_a_pvc_gives_a_weak_pulse_and_the_next_beat_a_strong_one():
    engine, before = simulate(6.0, heart_rate=60)
    engine.state.trigger_pvc()
    _, after_ = simulate(6.0, engine=engine)
    start = 0.0                                    # one record, both halves
    pvc_t = next(t for t, kind in engine.beat_log if kind == "pvc")

    abp = np.concatenate((before["abp"], after_["abp"]))
    peaks = pulse_peaks(abp)
    times = peaks / FS + start
    heights = abp[peaks] - np.array([abp[max(0, p - 300):p].min() for p in peaks])
    after = np.flatnonzero(times > pvc_t)[:2]
    normal = np.median(heights[times < pvc_t - 0.5])
    weak, strong = heights[after[0]], heights[after[1]]
    assert weak < 0.8 * normal, f"PVC pulse {weak:.1f} vs normal {normal:.1f}"
    assert strong > normal, f"post-pause beat {strong:.1f} not potentiated vs {normal:.1f}"


def test_dropped_beats_produce_no_pulse():
    engine, d = simulate(30.0, cardiac_rhythm="Mobitz II", heart_rate=80)
    conducted = sum(1 for t, kind in engine.beat_log
                    if kind == "sinus" and engine.elapsed - 20.0 < t <= engine.elapsed - 0.5)
    peaks = pulse_peaks(d["abp"][-20000:])
    assert abs(peaks.size - conducted) <= 1, f"{peaks.size} pulses for {conducted} beats"


def test_the_pulse_follows_the_heartbeat():
    """Every pleth pulse trails its QRS by the same transit delay."""
    _, d = simulate(15.0, heart_rate=72)
    qrs = detect_qrs(d["ecg"], FS)
    pleth = d["pleth"]
    lags = []
    for q in qrs[2:-2]:
        segment = pleth[q:q + 600]
        lags.append(int(np.argmax(segment)))
    lags = np.array(lags)
    assert 250 < np.median(lags) < 450, f"pulse lag {np.median(lags)} ms"
    assert lags.std() < 40, "pleth is not locked to the QRS"


# --- CPR ----------------------------------------------------------------------
def test_cpr_generates_pulses_in_cardiac_arrest():
    engine, off = simulate(10.0, cardiac_rhythm="V-Fib")
    engine.state.update(cpr_active=True)
    _, on = simulate(15.0, engine=engine)
    before, during = analyze(off, FS, 98), analyze(on, FS, 98)
    assert before.no_pulse and not during.no_pulse
    assert during.mean_pressure > before.mean_pressure + 12

    peaks = pulse_peaks(on["abp"][-8000:])
    rate = 60.0 * FS / np.diff(peaks).mean()
    assert abs(rate - CPR_RATE) < 6, f"compression rate {rate:.0f}"


def test_cpr_artifact_fools_the_rate_meter():
    """The reason rhythm checks need compressions paused: the artifact counts."""
    engine, _ = simulate(4.0, cardiac_rhythm="Asystole")
    engine.state.update(cpr_active=True)
    _, d = simulate(12.0, engine=engine)
    v = analyze(d, FS, 98)
    assert v.hr is not None and abs(v.hr - CPR_RATE) < 8, f"HR during CPR {v.hr}"


def test_cpr_stops_when_switched_off():
    engine, _ = simulate(4.0, cardiac_rhythm="Asystole", cpr_active=True)
    engine.state.update(cpr_active=False)
    _, d = simulate(12.0, engine=engine)
    assert analyze(d, FS, 98).no_pulse


# --- measured vitals ----------------------------------------------------------
COUNTABLE = [r for r in CARDIAC_RHYTHMS
             if r not in ("V-Fib", "Torsades de Pointes", "Asystole")]


def test_heart_rate_is_measured_for_every_organised_rhythm():
    for rhythm in COUNTABLE:
        engine, d = simulate(12.0, cardiac_rhythm=rhythm, heart_rate=72)
        truth = logged_rate(engine)
        hr = analyze(d, FS, 98).hr
        assert hr is not None, f"{rhythm}: no rate"
        assert abs(hr - truth) <= 3, f"{rhythm}: measured {hr}, truth {truth:.0f}"


def test_heart_rate_ignores_the_slider_when_the_rhythm_sets_it():
    _, d = simulate(12.0, cardiac_rhythm="V-Tach", heart_rate=50)
    assert analyze(d, FS, 98).hr == 160


def test_fibrillation_cannot_be_counted():
    for rhythm in ("V-Fib", "Torsades de Pointes"):
        _, d = simulate(12.0, cardiac_rhythm=rhythm)
        v = analyze(d, FS, 98)
        assert v.hr is None and v.chaotic, f"{rhythm}: hr={v.hr} chaotic={v.chaotic}"


def test_asystole_reads_zero():
    _, d = simulate(12.0, cardiac_rhythm="Asystole")
    v = analyze(d, FS, 98)
    assert v.hr == 0 and v.asystole


def test_rate_meter_rejects_mains_wander_and_noise():
    for noise in (dict(mains_amplitude=0.5), dict(baseline_amplitude=0.5),
                  dict(gaussian_sigma=0.1)):
        _, d = simulate(12.0, heart_rate=72, **noise)
        assert abs(analyze(d, FS, 98).hr - 72) <= 2, noise


def test_rate_tracks_across_the_whole_slider_range():
    for bpm in (30, 60, 100, 150, 200):
        engine, d = simulate(14.0, heart_rate=bpm)
        assert abs(analyze(d, FS, 98).hr - logged_rate(engine)) <= 3, bpm


def test_spo2_needs_a_pulse_to_read():
    _, perfusing = simulate(10.0, spo2_target=91)
    _, arrest = simulate(10.0, cardiac_rhythm="V-Fib", spo2_target=91)
    assert analyze(perfusing, FS, 91).spo2 == 91
    assert analyze(arrest, FS, 91).spo2 is None


def test_respiratory_rate_including_slow_agonal_breathing():
    for pattern, rr, expected in (("Normal", 8, 8), ("Normal", 20, 20),
                                  ("Kussmaul", 15, 28), ("Agonal", 15, 6)):
        _, d = simulate(45.0, resp_pattern=pattern, respiratory_rate=rr)
        v = analyze(d, FS, 98)
        assert v.resp_rate is not None and abs(v.resp_rate - expected) <= 1, \
            f"{pattern}@{rr}: {v.resp_rate}"


def test_apnoea_is_detected():
    _, d = simulate(30.0, resp_pattern="Apnoea")
    v = analyze(d, FS, 98)
    assert v.apnoea and v.resp_rate == 0


def test_not_enough_signal_reports_nothing():
    _, d = simulate(1.5)
    v = analyze(d, FS, 98)
    assert v.hr is None and v.systolic is None and v.resp_rate is None
    assert evaluate(v) == []


def test_incremental_analyser_matches_a_one_shot_analysis():
    state = SimulationState(heart_rate=88)
    engine = WaveformEngine(state, FS, np.random.default_rng(1))
    buffer = RingBuffer()
    analyzer = VitalsAnalyzer(FS)
    blocks = []
    for i in range(700):                              # 35 s, fed every half second
        block = engine.next_chunk(50)
        blocks.append(block)
        buffer.write(block)
        if i % 10 == 9:
            analyzer.feed(buffer)
    whole = {k: np.concatenate([b[k] for b in blocks]) for k in CHANNELS}
    assert analyzer.measure(98) == analyze(whole, FS, 98)


# --- alarms -------------------------------------------------------------------
def test_a_healthy_patient_raises_nothing():
    _, d = simulate(35.0)
    assert evaluate(analyze(d, FS, 98)) == []


def test_limits_raise_and_escalate():
    base = Vitals(hr=72, spo2=98, systolic=120, diastolic=80, mean_pressure=93,
                  resp_rate=15, window_s=10.0)
    codes = lambda v: {a.code: a.priority for a in evaluate(v)}   # noqa: E731
    from dataclasses import replace
    assert codes(replace(base, hr=130)) == {"hr_high": MEDIUM}
    assert codes(replace(base, hr=170)) == {"hr_high": HIGH}
    assert codes(replace(base, hr=45)) == {"hr_low": MEDIUM}
    assert codes(replace(base, hr=35)) == {"hr_low": HIGH}
    assert codes(replace(base, spo2=88)) == {"spo2_low": MEDIUM}
    assert codes(replace(base, spo2=80)) == {"spo2_low": HIGH}
    assert codes(replace(base, mean_pressure=45)) == {"map_low": HIGH}
    assert codes(replace(base, resp_rate=40)) == {"rr_high": MEDIUM}
    # Limits are configurable.
    assert evaluate(replace(base, hr=130), AlarmLimits(hr_high=140)) == []


def test_arrest_raises_high_priority_alarms_on_the_right_tiles():
    _, d = simulate(15.0, cardiac_rhythm="V-Fib")
    alarms = evaluate(analyze(d, FS, 98))
    worst = by_parameter(alarms)
    assert worst["hr"].code == "hr_unreadable" and worst["hr"].is_high
    assert worst["abp"].code == "no_pulse" and worst["abp"].is_high
    assert "spo2" in worst
    assert alarms[0].is_high, "alarms must be ordered highest priority first"


def test_every_alarm_carries_text_not_just_a_colour():
    for rhythm in ("V-Fib", "Asystole", "V-Tach", "3rd Degree AV Block"):
        _, d = simulate(35.0, cardiac_rhythm=rhythm, resp_pattern="Apnoea")
        for alarm in evaluate(analyze(d, FS, 98)):
            assert alarm.message.strip(), alarm


# --- scenarios ----------------------------------------------------------------
def test_every_scenario_step_is_a_valid_state_change():
    for scenario in SCENARIOS.values():
        state = SimulationState()
        times = [s.at for s in scenario.steps]
        assert times == sorted(times) and times[0] == 0, scenario.name
        for step in scenario.steps:
            state.update(**step.changes)                # raises on a bad key or value
            assert step.note, f"{scenario.name}: step at {step.at}s has no note"


def test_player_resets_to_baseline_and_applies_steps_on_time():
    state = SimulationState(cardiac_rhythm="V-Fib", gaussian_sigma=0.05)
    player = ScenarioPlayer(state)
    applied = player.start("Evolving heart block", now=100.0)
    assert [s.at for s in applied] == [0]
    assert state.snapshot().cardiac_rhythm == "Sinus"      # baseline, then step 0
    assert state.snapshot().gaussian_sigma == 0.05, "signal settings must survive"

    assert player.tick(111.0) == []
    assert [s.at for s in player.tick(125.0)] == [12, 24]  # catches up in order
    assert state.snapshot().cardiac_rhythm == "Mobitz I (Wenckebach)"
    assert player.next_step.at == 36


def test_player_stops_by_itself_and_leaves_the_patient_there():
    state = SimulationState()
    player = ScenarioPlayer(state)
    scenario = SCENARIOS["Evolving heart block"]
    player.start(scenario.name, now=0.0)
    player.tick(scenario.duration + 1.0)
    assert not player.running
    assert state.snapshot().cardiac_rhythm == "3rd Degree AV Block"
    assert player.progress(0.0) == 0.0


def test_cardiac_arrest_scenario_end_to_end():
    """Scenario -> state -> engine -> analyser: the patient actually arrests."""
    state = SimulationState()
    engine = WaveformEngine(state, FS, np.random.default_rng(4))
    player = ScenarioPlayer(state)
    player.start("Cardiac arrest: VT to VF", now=0.0)
    blocks = []
    for i in range(int(50 * 20)):                          # 50 s of signal
        player.tick(i * 0.05)
        blocks.append(engine.next_chunk(50))
    d = {k: np.concatenate([b[k] for b in blocks]) for k in CHANNELS}
    v = analyze(d, FS, state.snapshot().spo2_target)
    assert state.snapshot().cardiac_rhythm == "V-Fib"
    assert v.no_pulse and v.chaotic and v.spo2 is None


def test_baseline_only_touches_patient_fields():
    assert set(BASELINE) <= set(StateSnapshot.__dataclass_fields__)
    assert not {"mains_amplitude", "baseline_amplitude", "gaussian_sigma",
                "alarm_enabled"} & set(BASELINE)


def test_every_rhythm_still_perfuses_or_not_as_its_table_says():
    """Rhythms with zero perfusion must be pulseless; the rest must not be."""
    for rhythm, spec in RHYTHMS.items():
        _, d = simulate(12.0, cardiac_rhythm=rhythm)
        v = analyze(d, FS, 98)
        if spec.perfusion == 0.0:
            assert v.no_pulse, f"{rhythm} should be pulseless"
        elif spec.perfusion >= 0.6:
            assert not v.no_pulse, f"{rhythm} should have a pulse"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
