"""Catalogue of beat morphologies, cardiac rhythms and breathing patterns.

Kept separate from :mod:`core.generator` so the engine stays about *timing and
signal assembly* while this file stays about *what each condition looks like*.
Adding a rhythm should mean editing this table, not the engine.

Every beat is five Gaussians in the order [P, Q, R, S, T]:

    z(t) = sum_i a_i * exp( -(t - theta_i)^2 / (2 b_i^2) )

``a`` in mV, ``b`` (width) and ``theta`` (offset from the R peak) in seconds.
Rhythms that need real scheduling logic - dropped beats, AV dissociation,
alternating ectopy - name a ``scheduler``; the engine implements those.
Rhythms whose signal is not beat-shaped at all name a ``baseline`` override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Wave order throughout is [P, Q, R, S, T].
P, Q, R, S, T = range(5)


@dataclass(frozen=True, slots=True)
class BeatMorphology:
    """One beat expressed as five Gaussians: amplitude, width and phase offset."""

    a: np.ndarray        # mV
    b: np.ndarray        # s (standard deviation)
    theta: np.ndarray    # s, relative to the R peak
    rate_scaled: bool = True

    @staticmethod
    def of(a, b, theta, rate_scaled: bool = True) -> "BeatMorphology":
        return BeatMorphology(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
            rate_scaled,
        )


def with_pr(morph: BeatMorphology, pr_seconds: float) -> BeatMorphology:
    """Same beat with the P wave moved to a given PR interval."""
    theta = morph.theta.copy()
    theta[P] = -abs(pr_seconds)
    return BeatMorphology(morph.a, morph.b, theta, morph.rate_scaled)


# --- Beat morphologies ------------------------------------------------------

# Normal sinus beat - the standard parameter set.
SINUS = BeatMorphology.of(
    a=[0.15, -0.15, 1.00, -0.15, 0.30],
    b=[0.04, 0.01, 0.01, 0.01, 0.04],
    theta=[-0.20, -0.05, 0.00, 0.05, 0.30],
)

# Premature ventricular contraction: no P wave (ventricular origin, so the
# atria are not depolarised by this beat), a broad high-amplitude QRS, and a
# T wave discordant with the QRS - the textbook morphology.
PVC = BeatMorphology.of(
    a=[0.00, -0.20, 1.80, -0.35, -0.55],
    b=[0.04, 0.025, 0.040, 0.045, 0.090],
    theta=[-0.20, -0.06, 0.00, 0.08, 0.30],
    rate_scaled=False,     # ectopic beats are wide by definition; do not rate-scale
)

# AFib: Absent P-wave (index 0 is 0.0). The baseline f-waves are added later.
AFIB_BEAT = BeatMorphology.of(
    a=[0.00, -0.15, 1.00, -0.15, 0.30],
    b=[0.04, 0.01, 0.01, 0.01, 0.04],
    theta=[-0.20, -0.05, 0.00, 0.05, 0.30],
)

# Flutter conducts an ordinary ventricular complex; the atrial activity is the
# sawtooth added to the baseline, so the beat itself carries no P wave.
FLUTTER_BEAT = AFIB_BEAT

# STEMI (Anteroseptal Ischemia): The S-wave is elevated above baseline and
# merges directly into a widened, tall T-wave (the classic "tombstone").
STEMI_BEAT = BeatMorphology.of(
    a=[0.15, -0.15, 1.00, 0.30, 0.45],       # Elevated S (0.30), tall T (0.45)
    b=[0.04, 0.01, 0.01, 0.04, 0.06],        # Wider ST segment integration
    theta=[-0.20, -0.05, 0.00, 0.06, 0.18],  # T-wave shifted closer to QRS
)

# Subendocardial ischaemia: a depressed ST plateau and an inverted T wave -
# the mirror image of the STEMI pattern above.
ISCHEMIA_BEAT = BeatMorphology.of(
    a=[0.15, -0.15, 1.00, -0.28, -0.25],
    b=[0.04, 0.01, 0.01, 0.05, 0.05],
    theta=[-0.20, -0.05, 0.00, 0.07, 0.30],
)

# Hyperkalaemia: tall narrow "peaked" T waves, a widened QRS and a P wave that
# flattens out as the potassium climbs.  The R wave is widened (~42 ms FWHM
# against ~24 ms in sinus) but kept steeper than the T: at b=0.025 the peaked T
# carried more slope energy than the QRS, and a rate meter counted it as a beat.
HYPERKALEMIA_BEAT = BeatMorphology.of(
    a=[0.03, -0.15, 1.00, -0.15, 0.85],
    b=[0.04, 0.020, 0.018, 0.025, 0.028],
    theta=[-0.20, -0.06, 0.00, 0.06, 0.26],
)

# Ventricular Tachycardia: Extremely wide QRS complexes, no P or T waves.
VTACH_BEAT = BeatMorphology.of(
    a=[0.00, 0.00, 1.20, -1.00, 0.00],
    b=[0.01, 0.01, 0.06, 0.07, 0.01],
    theta=[-0.20, -0.05, 0.00, 0.10, 0.30],
    rate_scaled=False,
)

# Supraventricular tachycardia: narrow complexes, P waves lost in the preceding
# T wave at these rates.
SVT_BEAT = BeatMorphology.of(
    a=[0.00, -0.15, 1.00, -0.15, 0.25],
    b=[0.04, 0.01, 0.01, 0.01, 0.035],
    theta=[-0.20, -0.05, 0.00, 0.05, 0.28],
)

# Junctional escape: the impulse starts at the AV node, so the atria depolarise
# backwards - a small inverted P sitting right against the QRS.
JUNCTIONAL_BEAT = BeatMorphology.of(
    a=[-0.08, -0.15, 1.00, -0.15, 0.30],
    b=[0.030, 0.01, 0.01, 0.01, 0.04],
    theta=[-0.10, -0.05, 0.00, 0.05, 0.30],
)

# Ventricular escape focus: wide, slow, no atrial activity of its own.  Used by
# complete heart block and, at a slower rate, by idioventricular rhythm.
ESCAPE_BEAT = BeatMorphology.of(
    a=[0.00, -0.10, 1.00, -0.45, 0.35],
    b=[0.04, 0.020, 0.045, 0.050, 0.090],
    theta=[-0.20, -0.05, 0.00, 0.09, 0.32],
    rate_scaled=False,
)
IDIOVENTRICULAR_BEAT = ESCAPE_BEAT

# Ventricular pacing: a high-voltage stimulus artifact (carried in the P slot)
# followed by a wide, LBBB-like paced complex.
#
# A real stimulus is 0.5-2 ms.  At a 10 s sweep across ~1400 px one pixel is
# about 7 ms, so a true-width spike is sub-pixel and simply never draws.  Real
# monitors have the same problem and solve it the same way - "pace enhancement",
# rendering the marker wider than life.  b[P] below is that accommodation, not a
# claim about physiology.
PACING_SPIKE_WIDTH = 0.008    # s; ~3 px at the monitor's sweep speed

PACED_BEAT = BeatMorphology.of(
    a=[0.55, 0.00, 1.00, -0.40, 0.30],
    b=[PACING_SPIKE_WIDTH, 0.010, 0.035, 0.045, 0.080],
    theta=[-0.060, -0.05, 0.00, 0.08, 0.32],
    rate_scaled=False,     # the spike must not be stretched by the rate scaler
)

# An atrial depolarisation with nothing conducted after it - the "dropped beat"
# of second-degree block, and the independent P train of third-degree block.
# theta is all zero so the P lands on its own scheduled time.
P_ONLY = BeatMorphology.of(
    a=[0.15, 0.00, 0.00, 0.00, 0.00],
    b=[0.04, 0.01, 0.01, 0.01, 0.04],
    theta=[0.00, 0.00, 0.00, 0.00, 0.00],
    rate_scaled=False,
)

FIRST_DEGREE_BEAT = with_pr(SINUS, 0.32)     # PR > 200 ms, every beat conducted


# --- Rhythm table -----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RhythmSpec:
    """How one rhythm is generated.

    ``rate_bpm`` of None means "follow the heart-rate slider"; anything else
    overrides it, because the rate is a property of the arrhythmia rather than
    something the operator sets.
    """

    morphology: BeatMorphology
    rate_bpm: float | None = None
    jitter: tuple[float, float] | None = None    # multiplicative R-R jitter
    rsa: bool = False                            # needs an intact sinus node
    scheduler: str = "regular"
    baseline: str = ""                           # extra or replacement baseline
    note: str = ""
    # Mechanical effectiveness of each ventricular beat, 0..1.  Scales stroke
    # volume, so it is what makes VF pulseless and VT hypotensive.
    perfusion: float = 1.0


AFIB_JITTER = (0.7, 1.3)          # "irregularly irregular" R-R multiplier
VTACH_RATE = 160.0                # bpm; V-Tach overrides the heart rate slider
TORSADES_RATE = 240.0
TORSADES_TWIST_HZ = 0.55          # how fast the axis rotates about the baseline
COMPLETE_BLOCK_ESCAPE_BPM = 38.0
# PR lengthens by a *decreasing* increment (+120 ms, +60 ms), so the R-R
# interval shortens across the cycle before the dropped beat - the classic
# grouped-beating sign.  A constant increment would give a flat R-R.
WENCKEBACH_PR = (0.16, 0.28, 0.34)
MOBITZ_II_CONDUCTED = 3           # conduct 3, drop the 4th
BIGEMINY_COUPLING = 0.55          # the PVC lands this far into the R-R

FLUTTER_RATE_HZ = 5.0             # 300 atrial waves per minute
FLUTTER_AMPLITUDE = 0.16
VFIB_AMPLITUDE = 0.42

RHYTHMS: dict[str, RhythmSpec] = {
    "Sinus": RhythmSpec(SINUS, rsa=True, note="Normal sinus rhythm"),

    # -- atrial ------------------------------------------------------------
    "AFib": RhythmSpec(
        AFIB_BEAT, jitter=AFIB_JITTER, baseline="afib",
        note="No P waves, fibrillatory baseline, irregularly irregular", perfusion=0.85),
    "Atrial Flutter": RhythmSpec(
        FLUTTER_BEAT, rate_bpm=150.0, baseline="flutter",
        note="Sawtooth F waves at 300/min, 2:1 conduction", perfusion=0.85),
    "SVT": RhythmSpec(
        SVT_BEAT, rate_bpm=180.0,
        note="Narrow-complex tachycardia, P waves not visible", perfusion=0.85),
    "Junctional": RhythmSpec(
        JUNCTIONAL_BEAT, rate_bpm=48.0,
        note="AV nodal escape with a retrograde P wave", perfusion=0.85),

    # -- conduction blocks --------------------------------------------------
    "1st Degree AV Block": RhythmSpec(
        FIRST_DEGREE_BEAT, rsa=True,
        note="PR fixed at 320 ms; every beat still conducts"),
    "Mobitz I (Wenckebach)": RhythmSpec(
        SINUS, scheduler="wenckebach",
        note="PR lengthens beat by beat until a QRS is dropped"),
    "Mobitz II": RhythmSpec(
        SINUS, scheduler="mobitz2",
        note="Constant PR, then a QRS drops without warning"),
    "3rd Degree AV Block": RhythmSpec(
        ESCAPE_BEAT, scheduler="complete_block",
        note="Complete AV dissociation: independent P and QRS trains", perfusion=0.85),

    # -- ventricular --------------------------------------------------------
    "PVC Bigeminy": RhythmSpec(
        SINUS, scheduler="bigeminy",
        note="Every sinus beat followed by a PVC"),
    "V-Tach": RhythmSpec(
        VTACH_BEAT, rate_bpm=VTACH_RATE,
        note="Broad monomorphic complexes at a fixed fast rate", perfusion=0.35),
    "Torsades de Pointes": RhythmSpec(
        VTACH_BEAT, rate_bpm=TORSADES_RATE, scheduler="torsades",
        note="Polymorphic VT twisting about the isoelectric line", perfusion=0.15),
    "V-Fib": RhythmSpec(
        SINUS, scheduler="none", baseline="vfib",
        note="Chaotic disorganised activity, no identifiable complexes", perfusion=0.0),
    "Idioventricular": RhythmSpec(
        IDIOVENTRICULAR_BEAT, rate_bpm=34.0,
        note="Slow wide ventricular escape rhythm", perfusion=0.6),
    "Asystole": RhythmSpec(
        SINUS, scheduler="none", baseline="flat",
        note="No electrical activity", perfusion=0.0),

    # -- ischaemia and metabolic -------------------------------------------
    "STEMI": RhythmSpec(
        STEMI_BEAT, rsa=True,
        note="ST elevation merging into a hyperacute T wave", perfusion=0.75),
    "Ischemia (ST Dep.)": RhythmSpec(
        ISCHEMIA_BEAT, rsa=True,
        note="ST depression with T wave inversion", perfusion=0.9),
    "Hyperkalemia": RhythmSpec(
        HYPERKALEMIA_BEAT, rsa=True,
        note="Peaked T waves, widened QRS, flattened P", perfusion=0.9),

    # -- device -------------------------------------------------------------
    "Paced": RhythmSpec(
        PACED_BEAT, rate_bpm=70.0,
        note="Pacing spike followed by a wide paced complex", perfusion=0.9),
}

CARDIAC_RHYTHMS = tuple(RHYTHMS)


# --- Baseline overrides -----------------------------------------------------
def fibrillatory_waves(t: np.ndarray) -> np.ndarray:
    """Chaotic 5-8 Hz undulation typical of fibrillating atria."""
    return (0.03 * np.sin(2.0 * np.pi * 5.5 * t)
            + 0.02 * np.sin(2.0 * np.pi * 7.1 * t + 0.5))


def flutter_waves(t: np.ndarray) -> np.ndarray:
    """Sawtooth atrial activity at 300/min - the flutter 'picket fence'."""
    phase = np.mod(t * FLUTTER_RATE_HZ, 1.0)
    return FLUTTER_AMPLITUDE * (2.0 * phase - 1.0)


# --- Breathing patterns -----------------------------------------------------
RESP_INSPIRATORY_FRACTION = 0.4    # I:E ratio of 1:1.5
CHEYNE_STOKES_PERIOD_S = 60.0
BIOT_CYCLE_S = 18.0
BIOT_RAMP_S = 0.6


@dataclass(frozen=True, slots=True)
class RespSpec:
    rate_bpm: float | None = None      # None -> follow the respiratory-rate slider
    amplitude: float = 1.0
    inspiratory: float = RESP_INSPIRATORY_FRACTION
    envelope: str = "none"
    note: str = ""


RESP_SPECS: dict[str, RespSpec] = {
    "Normal": RespSpec(note="Regular tidal breathing"),
    "Cheyne-Stokes": RespSpec(
        envelope="cheyne",
        note="Crescendo-decrescendo then apnoea, 60 s cycle"),
    "Kussmaul": RespSpec(
        rate_bpm=28.0, amplitude=1.35,
        note="Deep, rapid, laboured breathing"),
    "Biot (Ataxic)": RespSpec(
        envelope="biot",
        note="Irregular clusters of breaths separated by pauses"),
    "Apnoea": RespSpec(
        envelope="apnoea",
        note="Complete cessation of respiratory effort"),
    "Agonal": RespSpec(
        rate_bpm=6.0, amplitude=1.2, inspiratory=0.12,
        note="Slow, infrequent gasps"),
}

RESP_PATTERNS = tuple(RESP_SPECS)


def skewed_respiration(u, inspiratory: float = RESP_INSPIRATORY_FRACTION):
    """Asymmetric breathing waveform over cycle position ``u`` in [0, 1).

    The cycle is phase-warped so the rise (inspiration) occupies ``inspiratory``
    of the period and the fall (expiration) the rest.  Both branches are built
    from ``-cos``, whose slope is zero at the junctions, so the joins are smooth.
    """
    u = np.mod(u, 1.0)
    warped = np.where(u < inspiratory,
                      0.5 * u / inspiratory,
                      0.5 + 0.5 * (u - inspiratory) / (1.0 - inspiratory))
    return -np.cos(2.0 * np.pi * warped)


def resp_envelope(kind: str, t):
    """Slow amplitude envelope applied on top of the breathing waveform."""
    t = np.asarray(t, dtype=np.float64)
    if kind == "cheyne":
        # A sine clipped at zero: half the cycle waxes and wanes, half is apnoeic.
        return np.maximum(0.0, np.sin(2.0 * np.pi * np.mod(t / CHEYNE_STOKES_PERIOD_S, 1.0)))
    if kind == "apnoea":
        return np.zeros_like(t)
    if kind == "biot":
        # Clusters of breathing with irregular length and depth, then a pause.
        # Derived from the cluster index so it is deterministic and continuous
        # across chunk boundaries rather than re-rolled every frame.
        cycle = np.floor(t / BIOT_CYCLE_S).astype(np.int64)
        elapsed = np.mod(t, BIOT_CYCLE_S)
        duty = 0.30 + 0.45 * (np.mod(cycle * 2654435761, 1000) / 1000.0)
        depth = 0.55 + 0.45 * (np.mod(cycle * 40503, 997) / 997.0)
        # Trapezoid rather than a step, so clusters start and stop smoothly.
        on = np.clip(np.minimum(elapsed / BIOT_RAMP_S,
                                (duty * BIOT_CYCLE_S - elapsed) / BIOT_RAMP_S), 0.0, 1.0)
        return depth * on
    return np.ones_like(t)


# --- Therapy indications ----------------------------------------------------
# Which rhythms call for electrical therapy, and which kind.  Kept as one table
# rather than a field on each rhythm so the whole policy can be reviewed at once.
#
# The distinctions that matter clinically, and that this table encodes:
#   * Defibrillation is UNSYNCHRONISED - for pulseless VF/VT, where there is no
#     organised R wave to synchronise to.
#   * Cardioversion is SYNCHRONISED to the R wave, for organised but unstable
#     rhythms.  Delivering it unsynchronised risks landing the shock on the T
#     wave and inducing VF.
#   * Asystole is NOT shockable.  Shocking it is a classic error; it needs CPR.
#   * Bradyarrhythmias need pacing, not a shock.

THERAPY_DEFIB = "defibrillate"
THERAPY_CARDIOVERT = "cardiovert"
THERAPY_CPR = "cpr"
THERAPY_PACE = "pace"

SHOCK_KINDS = (THERAPY_DEFIB, THERAPY_CARDIOVERT)

URGENCY_LETHAL = "lethal"      # arrest rhythm: act now
URGENCY_URGENT = "urgent"      # unstable, but perfusing


@dataclass(frozen=True, slots=True)
class TherapySpec:
    therapy: str = ""          # "" = no electrical therapy indicated
    urgency: str = ""
    alarm: str = ""            # banner headline
    advice: str = ""           # one line of what the operator should do


NO_THERAPY = TherapySpec()

THERAPIES: dict[str, TherapySpec] = {
    "V-Fib": TherapySpec(
        THERAPY_DEFIB, URGENCY_LETHAL, "VENTRICULAR FIBRILLATION",
        "Defibrillate immediately (unsynchronised) and continue CPR."),
    "V-Tach": TherapySpec(
        THERAPY_DEFIB, URGENCY_LETHAL, "VENTRICULAR TACHYCARDIA",
        "Pulseless: defibrillate. With a pulse: synchronised cardioversion."),
    "Torsades de Pointes": TherapySpec(
        THERAPY_DEFIB, URGENCY_LETHAL, "TORSADES DE POINTES",
        "Defibrillate if pulseless; magnesium and correct the QT otherwise."),
    "Asystole": TherapySpec(
        THERAPY_CPR, URGENCY_LETHAL, "ASYSTOLE - NOT SHOCKABLE",
        "CPR and adrenaline. A shock will not help; do not defibrillate."),

    "AFib": TherapySpec(
        THERAPY_CARDIOVERT, URGENCY_URGENT, "ATRIAL FIBRILLATION",
        "Synchronised cardioversion if unstable; rate control if stable."),
    "Atrial Flutter": TherapySpec(
        THERAPY_CARDIOVERT, URGENCY_URGENT, "ATRIAL FLUTTER",
        "Synchronised cardioversion if unstable."),
    "SVT": TherapySpec(
        THERAPY_CARDIOVERT, URGENCY_URGENT, "SUPRAVENTRICULAR TACHYCARDIA",
        "Vagal manoeuvres, then adenosine; cardiovert if unstable."),

    "3rd Degree AV Block": TherapySpec(
        THERAPY_PACE, URGENCY_URGENT, "COMPLETE HEART BLOCK",
        "Transcutaneous pacing; atropine is unlikely to help."),
    "Idioventricular": TherapySpec(
        THERAPY_PACE, URGENCY_URGENT, "IDIOVENTRICULAR RHYTHM",
        "Transcutaneous pacing; check for a pulse (may be PEA)."),
    "Mobitz II": TherapySpec(
        THERAPY_PACE, URGENCY_URGENT, "MOBITZ II AV BLOCK",
        "High risk of progression to complete block; prepare to pace."),
}

# No organised R wave, so a synchronised shock cannot be timed at all.
UNSYNCHRONISABLE = ("V-Fib", "Asystole")


def therapy_for(rhythm: str) -> TherapySpec:
    return THERAPIES.get(rhythm, NO_THERAPY)
