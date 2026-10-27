"""Simulated October nights for tests, demos, and the GitHub Pages POC.
No API calls, no real cars. One pretend Model Y, 21 nights, Oct 5–25
2026. The pattern is deliberately learnable:

- quiet nights lose about 1.0–1.2% (the baseline)
- Sentry Mode on adds ~2 points, a cold night (35°F) ~1.5,
  Cabin Overheat Protection ~0.5, a software-update night ~2,
  being plugged in (not charging) shaves ~0.3
- worst night: Oct 21 — Sentry on AND a cold night, 4.6% lost

Core invariant the e2e test asserts: the learned baseline is ~1.1%,
the Sentry delta is learned positive, and tonight's forecast
(Sentry on, 38°F, plugged in) beats the quiet-night baseline and
crosses the 4% alert threshold.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .drain import NightRecord, drain_report, learn_model

DEMO_MONTH = "2026-10"


def _night(day: int, evening_soc: float, sentry: bool, low_f: float,
           cop: bool, plugged: bool, update: bool) -> NightRecord:
    loss = 1.0
    if sentry:
        loss += 2.0
    if low_f < 40:
        loss += 1.5
    if cop:
        loss += 0.5
    if update:
        loss += 2.0
    if plugged:
        loss -= 0.3
    loss += 0.1 * ((day - 5) % 3)
    return NightRecord(
        "Model Y",
        datetime(2026, 10, day, 22, 0, tzinfo=timezone.utc),
        datetime(2026, 10, day + 1, 7, 0, tzinfo=timezone.utc),
        evening_soc, round(evening_soc - loss, 1),
        sentry_on=sentry, low_temp_f=low_f, cop_on=cop,
        plugged_in=plugged, software_update=update,
        location_name="Home")


def demo_nights_raw() -> list[NightRecord]:
    nights = []
    for i in range(21):
        day = 5 + i
        nights.append(_night(
            day,
            evening_soc=68.0 + (i % 5),
            sentry=(i % 4 == 0),
            low_f=35.0 if i % 5 == 1 else 55.0,
            cop=(i % 7 == 3),
            plugged=(i % 3 == 0),
            update=(i == 10)))
    return nights


def run_demo_month(settings) -> list[NightRecord]:
    """Demo nights need no pricing — returns them as-is."""
    return demo_nights_raw()


def tonight_conditions() -> dict:
    """The demo's tonight: Sentry on, cold (38°F), plugged in."""
    return {
        "Sentry Mode on": True,
        "cold night": True,
        "Cabin Overheat Protection on": False,
        "software update": False,
        "plugged in (not charging)": True,
    }


def timeline_lines(settings) -> list[str]:
    """Human-readable demo timeline for `tdrain simulate`."""
    nights = run_demo_month(settings)
    model = learn_model(nights, "Model Y", settings.cold_threshold_f,
                        settings.baseline_nights)
    forecast = model.predict(tonight_conditions())
    lines = ["Tesla Drain Forecaster — demo nights (pretend car, pretend nights)"]
    for n in nights:
        factors = [name for name, on in n.factors(settings.cold_threshold_f).items()
                   if on]
        factor_s = f" ({', '.join(factors)})" if factors else " (quiet night)"
        lines.append(
            f"{n.evening.strftime('%a %b %d, %I:%M %p')} → "
            f"{n.morning.strftime('%I:%M %p')} — {n.vehicle} at "
            f"{n.location_name}: {n.evening_soc:.0f}% → "
            f"{n.morning_soc:.1f}%, lost {n.loss_pct():.1f}%{factor_s}")
    lines.append("")
    lines.append(drain_report(nights, settings.cold_threshold_f,
                              settings.baseline_nights,
                              settings.alert_threshold_pct))
    lines.append("")
    lines.append("Tonight's forecast (Sentry on, overnight low 38°F, plugged in):")
    lines.append("  " + forecast.summary_line())
    if forecast.predicted_pct >= settings.alert_threshold_pct:
        lines.append(
            f"  Above your {settings.alert_threshold_pct:.0f}% alert threshold — "
            f"switching Sentry off tonight would save roughly "
            f"{model.deltas['Sentry Mode on'][0]:.1f} points.")
    lines.append("")
    lines.append("Why the forecast is trustworthy: the baseline comes from")
    lines.append("the car's own quiet nights, and every factor delta is either")
    lines.append("learned from nights with vs without that factor, or labelled")
    lines.append("as a default. No factor is ever guessed silently.")
    lines.append("")
    lines.append("Expected Fleet API cost: this app only reads the car's")
    lines.append("last-known state with no-wake reads — two reads a night, no")
    lines.append("wake-ups, no commands, so the real bill stays near zero.")
    lines.append("The night record lives on your computer, not in Tesla's cloud.")
    return lines
