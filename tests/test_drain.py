"""Drain learning, prediction, culprit ranking, similar nights,
validation, and CSV export."""
from datetime import datetime, timedelta, timezone

import pytest

from tesla_drain_forecaster.drain import (DrainError, NightRecord,
                                          drain_report, explain_night,
                                          export_csv, learn_model,
                                          similar_nights)


def _n(day, evening_soc, morning_soc, sentry=False, low_f=None, cop=False,
       plugged=False, update=False, charged=False, car="Model Y"):
    evening = datetime(2026, 10, day, 22, 0, tzinfo=timezone.utc)
    return NightRecord(car, evening, evening + timedelta(hours=9),
                       evening_soc, morning_soc, sentry_on=sentry,
                       low_temp_f=low_f, cop_on=cop, plugged_in=plugged,
                       software_update=update, charged_overnight=charged,
                       location_name="Home")


def _quiet(day, loss=1.0):
    return _n(day, 70.0, 70.0 - loss)


def test_loss_math_and_validation():
    n = _n(3, 70.0, 68.0)
    assert n.loss_pct() == pytest.approx(2.0)
    assert n.is_drain_night()
    with pytest.raises(DrainError):
        _n(3, 120.0, 68.0).validate()
    with pytest.raises(DrainError):
        bad = _n(3, 70.0, 68.0)
        bad.morning = bad.evening
        bad.validate()
    assert not _n(3, 70.0, 68.0, charged=True).is_drain_night()


def test_baseline_is_median_of_quiet_nights():
    nights = [_quiet(1, 1.0), _quiet(2, 1.2), _quiet(3, 1.1),
              _n(4, 70.0, 67.0, sentry=True)]  # 3.0% sentry night
    model = learn_model(nights, "Model Y", window=90)
    assert model.baseline_pct == pytest.approx(1.1)
    assert model.clean_nights == 3


def test_sentry_delta_learned_when_both_groups_big_enough():
    nights = [_quiet(1, 1.0), _quiet(2, 1.2), _quiet(3, 1.1),
              _n(4, 70.0, 67.0, sentry=True),
              _n(5, 70.0, 66.8, sentry=True),
              _n(6, 70.0, 67.2, sentry=True)]
    model = learn_model(nights, "Model Y", window=90)
    delta, learned = model.deltas["Sentry Mode on"]
    assert learned
    assert delta == pytest.approx(2.0, abs=0.3)


def test_default_delta_used_when_group_too_small():
    nights = [_quiet(1, 1.0), _quiet(2, 1.2),
              _n(3, 70.0, 67.0, sentry=True)]  # only one sentry night
    model = learn_model(nights, "Model Y", window=90)
    delta, learned = model.deltas["Sentry Mode on"]
    assert not learned
    assert delta == 1.5


def test_charged_nights_excluded_from_learning():
    nights = [_quiet(1, 1.0), _quiet(2, 1.2),
              _n(3, 70.0, 75.0, charged=True)]  # gained charge: not drain
    model = learn_model(nights, "Model Y", window=90)
    assert model.nights_used == 2
    assert model.baseline_pct == pytest.approx(1.1)


def test_prediction_clamps_and_sums_parts():
    nights = [_quiet(1, 1.0), _quiet(2, 1.2), _quiet(3, 1.1)]
    model = learn_model(nights, "Model Y", window=90)
    pred = model.predict({"Sentry Mode on": True, "cold night": True,
                          "Cabin Overheat Protection on": False,
                          "software update": False,
                          "plugged in (not charging)": False})
    assert pred.predicted_pct == pytest.approx(1.1 + 1.5 + 1.0)
    quiet = model.predict({})
    assert quiet.predicted_pct == pytest.approx(1.1)
    # No nights at all: default baseline, still a sane forecast.
    empty = learn_model([], "Model Y")
    assert empty.baseline_pct == 1.0
    assert empty.predict({}) .predicted_pct == 1.0


def test_cold_threshold_boundary():
    # Exactly at the threshold is NOT a cold night; one degree below is.
    assert not _n(1, 70.0, 69.0, low_f=40.0).factors(40.0)["cold night"]
    assert _n(1, 70.0, 69.0, low_f=39.0).factors(40.0)["cold night"]


def test_explain_night_ranks_sentry_first():
    nights = [_quiet(1, 1.0), _quiet(2, 1.2), _quiet(3, 1.1),
              _n(4, 70.0, 66.0, sentry=True, low_f=35.0)]
    model = learn_model(nights, "Model Y", window=90)
    text = explain_night(nights[-1], model)
    assert "Sentry Mode on" in text
    assert text.index("Sentry Mode on") < text.index("cold night")
    assert "4.0%" in text


def test_similar_nights_prefers_matching_conditions():
    target = _n(10, 70.0, 66.0, sentry=True)
    nights = [target,
              _n(3, 70.0, 66.2, sentry=True),   # same shape
              _quiet(4, 1.0),                    # quiet night
              _n(5, 70.0, 66.1, sentry=True)]   # same shape
    sims = similar_nights(target, nights)
    assert len(sims) == 3
    assert all(s[0] is not target for s in sims)
    assert sims[0][0].sentry_on  # closest match is a sentry night


def test_report_and_csv_export(tmp_path):
    nights = [_quiet(1, 1.0), _quiet(2, 1.2),
              _n(3, 70.0, 65.0, sentry=True)]
    report = drain_report(nights, alert_threshold_pct=4.0)
    assert "Model Y" in report
    assert "Worst night" in report
    out = tmp_path / "nights.csv"
    assert export_csv(nights, str(out)) == 3
    rows = out.read_text().splitlines()
    assert rows[0].startswith("night,car,place")
    assert len(rows) == 4
