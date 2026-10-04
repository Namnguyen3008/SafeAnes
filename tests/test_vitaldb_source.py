"""Contract tests for VitalDB track mapping and local replay caching."""

from __future__ import annotations

import gzip
import importlib
import importlib.util
from io import BytesIO
from typing import ClassVar

import numpy as np

from core.data_sources import PatientTimeline, SignalStatus, TimelineSignal


def _require_source():
    spec = importlib.util.find_spec("core.vitaldb_source")
    assert spec is not None, "core.vitaldb_source must define the VitalDB source contract"
    return importlib.import_module("core.vitaldb_source")


def test_uc04_and_uc05_track_mappings_match_the_model_contracts():
    source = _require_source()

    assert source.UC04_NUMERIC_FEATURES == (
        "MAP", "SBP", "DBP", "HR", "SPO2", "RR", "ETCO2"
    )
    assert source.UC05_NUMERIC_FEATURES == (
        "SPO2", "ETCO2", "RR_CO2", "TV", "MV", "PIP", "PEEP", "PPLAT",
        "MAWP", "FIO2", "COMPLIANCE", "VENT_LEAK", "SET_FIO2", "SET_TV",
        "SET_PIP", "SET_RR", "HR", "MAC",
    )
    assert source.UC04_WAVEFORMS["ecg"].track_candidates == ("SNUADC/ECG_II",)
    assert source.UC05_WAVEFORMS["ppg"].track_candidates == ("SNUADC/PLETH",)
    assert source.UC05_WAVEFORMS["ppg"].sample_rate_hz == 500.0
    assert source.UC05_WAVEFORMS["flow"].track_candidates == ()
    assert source.UC05_WAVEFORMS["resp"].track_candidates == ()
    assert source.UC04_NUMERIC_TRACKS["ETCO2"] == ("Solar8000/ETCO2",)
    assert source.UC05_NUMERIC_TRACKS["ETCO2"] == ("Primus/ETCO2", "Solar8000/ETCO2")
    assert source.UC05_NUMERIC_ALIASES["ETCO2"] == "ETCO2_UC05"


def test_local_cache_round_trips_signals_status_and_public_case_metadata(tmp_path):
    source = _require_source()
    timeline = PatientTimeline(
        case_id=17,
        duration_sec=3.0,
        source_label="VitalDB Real Replay",
        waveforms={
            "ecg": TimelineSignal(
                np.array([1.0, np.nan, 3.0]), 1.0, SignalStatus.AVAILABLE,
                track_name="SNUADC/ECG_II", units="mV",
            ),
            "flow": TimelineSignal(
                None, 62.5, SignalStatus.NOT_SUPPORTED,
                reason="The UC05 training track map does not define this channel.",
            ),
        },
        numerics={
            "HR": TimelineSignal(
                np.array([60.0, 61.0, 62.0]), 1.0, SignalStatus.AVAILABLE,
                track_name="Solar8000/HR", units="bpm",
            )
        },
        static_features={"age": 50.0, "sex_male": 1.0},
    )
    cache = source.LocalTimelineCache(tmp_path)

    cache.save(timeline)
    restored = cache.load(17)

    assert restored is not None
    assert restored.case_id == 17
    assert restored.duration_sec == 3.0
    assert restored.source_label == "VitalDB Real Replay"
    assert restored.static_features == {"age": 50.0, "sex_male": 1.0}
    assert restored.waveforms["ecg"].status is SignalStatus.AVAILABLE
    assert restored.waveforms["flow"].status is SignalStatus.NOT_SUPPORTED
    np.testing.assert_allclose(restored.waveforms["ecg"].values, [1.0, np.nan, 3.0], equal_nan=True)
    np.testing.assert_allclose(restored.numerics["HR"].values, [60.0, 61.0, 62.0])



def test_cached_timeline_backfills_anesthesia_bounds_from_vitaldb_metadata(tmp_path):
    source = _require_source()
    timeline = PatientTimeline(
        case_id=17,
        duration_sec=900.0,
        source_label="VitalDB Real Replay",
        numerics={"ETCO2_UC05": TimelineSignal(np.full(900, 35.0), 1.0, SignalStatus.AVAILABLE)},
        static_features={"age": 50.0},
    )
    cache = source.LocalTimelineCache(tmp_path)
    cache.save(timeline)
    metadata = [{"caseid": "17", "caseend": "900", "anestart": "300", "aneend": "840"}]
    loader = source.VitalDBSource(cache=cache, metadata_loader=lambda: metadata)

    restored = loader.load_timeline(17)

    assert restored.static_features["anestart_sec"] == 300.0
    assert restored.static_features["aneend_sec"] == 840.0
    cached_again = cache.load(17)
    assert cached_again.static_features["anestart_sec"] == 300.0
    assert cached_again.static_features["aneend_sec"] == 840.0


def test_case_metadata_uses_public_api_when_vitaldb_has_no_configured_base_url(monkeypatch):
    source = _require_source()
    import vitaldb

    monkeypatch.setattr(vitaldb.api, "API_URL", None)
    monkeypatch.setattr(
        source,
        "urlopen",
        lambda url, timeout: BytesIO(b"caseid,age,caseend,sex\n1,77,11542,M\n"),
    )

    rows = source.VitalDBSource._read_case_metadata()

    assert rows == [{"caseid": "1", "age": "77", "caseend": "11542", "sex": "M"}]


def test_case_metadata_decodes_the_gzipped_vitaldb_response(monkeypatch):
    source = _require_source()
    import vitaldb

    class Response(BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Encoding": "gzip"}

    monkeypatch.setattr(vitaldb.api, "API_URL", None)
    monkeypatch.setattr(
        source,
        "urlopen",
        lambda url, timeout: Response(gzip.compress(b"caseid,age,caseend,sex\n1,77,11542,M\n")),
    )

    rows = source.VitalDBSource._read_case_metadata()

    assert rows[0]["caseid"] == "1"
    assert rows[0]["age"] == "77"


def test_loader_preserves_model_specific_etco2_track_preference(tmp_path):
    source = _require_source()

    class FakeReader:
        def __init__(self, case_id, track_names):
            self.track_names = set(track_names)

        def get_track_names(self):
            return sorted(self.track_names)

        def to_numpy(self, names, interval):
            rows = max(1, int(600 / interval))
            values = {
                "Primus/ETCO2": 41.0,
                "Solar8000/ETCO2": 33.0,
            }
            return np.column_stack([
                np.full(rows, values.get(name, 1.0), dtype=np.float32)
                for name in names
            ])

    fake_vitaldb = type("FakeVitalDB", (), {"VitalFile": FakeReader})
    timeline = source.VitalDBSource(
        cache=source.LocalTimelineCache(tmp_path),
        vitaldb_module=fake_vitaldb,
        metadata_loader=lambda: [{"caseid": "1", "age": "77", "caseend": "600", "sex": "M"}],
    ).load_timeline(1, use_cache=False)

    assert timeline.numerics["ETCO2"].track_name == "Solar8000/ETCO2"
    assert timeline.numerics["ETCO2_UC05"].track_name == "Primus/ETCO2"
    assert timeline.numerics["ETCO2"].values[0] == 33.0
    assert timeline.numerics["ETCO2_UC05"].values[0] == 41.0
