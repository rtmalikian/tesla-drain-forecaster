"""Config parsing: vehicles, 12-hour times, alert threshold, cold
threshold, baseline window, AI mode."""
import pytest

from tesla_drain_forecaster.config import (ConfigError, format_12h_time,
                                           load_settings, parse_12h_time)


def test_defaults_load():
    s = load_settings({})
    assert s.server_port == 8650
    assert s.alert_threshold_pct == 4.0
    assert s.baseline_nights == 14
    assert s.cold_threshold_f == 40.0
    assert s.ai_mode == "auto"
    assert s.model_id == "tesla-drain-forecaster"


def test_vehicles_parsed():
    s = load_settings({"TDF_VEHICLES": "Model Y=VINY,Model 3=VIN3"})
    assert [(v.name, v.vin) for v in s.vehicles] == [("Model Y", "VINY"),
                                                     ("Model 3", "VIN3")]


def test_bad_thresholds_rejected():
    with pytest.raises(ConfigError):
        load_settings({"TDF_ALERT_THRESHOLD_PCT": "20"})
    with pytest.raises(ConfigError):
        load_settings({"TDF_ALERT_THRESHOLD_PCT": "abc"})
    with pytest.raises(ConfigError):
        load_settings({"TDF_BASELINE_NIGHTS": "2"})
    with pytest.raises(ConfigError):
        load_settings({"TDF_COLD_THRESHOLD_F": "100"})


def test_bad_region_and_ai_mode_rejected():
    with pytest.raises(ConfigError):
        load_settings({"TESLA_REGION": "xx"})
    with pytest.raises(ConfigError):
        load_settings({"TDF_AI_MODE": "sometimes"})


def test_bad_embedding_dim_rejected():
    with pytest.raises(ConfigError):
        load_settings({"TDF_EMBEDDING_DIM": "2"})


def test_12h_times():
    assert parse_12h_time("4:00 PM") == 16 * 60
    assert parse_12h_time("12:00 AM") == 0
    assert parse_12h_time("12:30 PM") == 12 * 60 + 30
    assert format_12h_time(16 * 60) == "4:00 PM"
    assert format_12h_time(0) == "12:00 AM"
    with pytest.raises(ConfigError):
        parse_12h_time("16:00")
    with pytest.raises(ConfigError):
        parse_12h_time("13:00 PM")
