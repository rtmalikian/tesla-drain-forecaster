"""Overnight phantom-drain learning: pure decision logic, no network,
no clock except the one callers pass in. Every function here is
unit-testable.

Method (the one owners reason about by hand, automated)
--------------------------------------------------------
A "night" is one parked evening-to-morning pair: battery % at bedtime
minus battery % in the morning, on a night the car did NOT charge.
Nights when the car charged are not drain nights and are excluded
from every statistic — charging would hide the loss.

Baseline
  The median loss of "clean" nights (no factor switched on) over the
  recent window. If there are no clean nights, the median of all drain
  nights. If there are no nights at all, 1.0% (the typical overnight
  loss owners report for a healthy parked Tesla).

Factor deltas
  For each factor — Sentry Mode on, a cold night (overnight low below
  the owner's cold threshold, default 40°F), Cabin Overheat Protection
  left on, a software-update night, plugged in but not charging — the
  delta is the median loss of nights with the factor minus the median
  loss of nights without it, when both groups have at least 2 nights.
  Otherwise a transparent default is used (Sentry +1.5, cold +1.0,
  Cabin Overheat Protection +0.5, software update +1.0, plugged in
  −0.3). Every prediction prints its breakdown, so the number is
  never a black box.

Prediction
  Tonight's predicted loss = baseline + the deltas of tonight's
  conditions, clamped to 0–15%. A prediction at or above the owner's
  alert threshold (default 4%) is flagged "worth a look".

Culprit ranking
  For any recorded night, the factors are ranked by their delta, so
  "why did I lose 6% Tuesday?" gets "Sentry was on (+2.0), it was a
  cold night (+1.5), baseline 1.0 — that explains 4.5 of the 6 points;
  the rest is normal night-to-night variation."

Similar nights
  Drain-day vectors ("sentry on, cold night, plugged in, loss 4%")
  are embedded with the same deterministic local embeddings the
  /v1/embeddings endpoint serves, and ranked by cosine similarity —
  "nights like last Tuesday" without any AI backend.

This app never wakes a car to measure drain: evening and morning
readings use no-wake vehicle data (the car reports its last-known
state while asleep). Wakes are the priciest Fleet API call, so a
sleeping car is simply skipped and read later.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

MIN_GROUP_NIGHTS = 2      # nights per side before a delta is "learned"
DEFAULT_BASELINE_PCT = 1.0
COLD_DEFAULT_F = 40.0
MAX_PREDICTED_PCT = 15.0

# factor -> (attribute on NightRecord, default delta in points)
FACTOR_DEFAULTS = {
    "Sentry Mode on": 1.5,
    "cold night": 1.0,
    "Cabin Overheat Protection on": 0.5,
    "software update": 1.0,
    "plugged in (not charging)": -0.3,
}


class DrainError(Exception):
    """Human-readable drain problem (bad night, bad file)."""


@dataclass
class NightRecord:
    vehicle: str
    evening: datetime
    morning: datetime
    evening_soc: float     # percent, 0-100
    morning_soc: float     # percent, 0-100
    sentry_on: bool = False
    low_temp_f: float | None = None   # overnight low, Fahrenheit
    cop_on: bool = False              # Cabin Overheat Protection left on
    plugged_in: bool = False          # plugged in but NOT charging
    software_update: bool = False
    charged_overnight: bool = False   # car charged: not a drain night
    location_name: str = ""

    def validate(self) -> None:
        if not self.vehicle.strip():
            raise DrainError("A night record needs a car name.")
        if self.morning <= self.evening:
            raise DrainError(
                "The morning reading must come after the evening reading — "
                "check the night's times.")
        for label, soc in (("evening", self.evening_soc),
                           ("morning", self.morning_soc)):
            if not (0 <= soc <= 100):
                raise DrainError(
                    f"The {label} battery percentage must be between 0 and 100, "
                    f"got {soc}.")

    def loss_pct(self) -> float:
        """Points lost overnight (negative if the car gained charge)."""
        return self.evening_soc - self.morning_soc

    def is_drain_night(self) -> bool:
        return not self.charged_overnight

    def factors(self, cold_threshold_f: float = COLD_DEFAULT_F) -> dict:
        """Factor name -> on/off for this night."""
        return {
            "Sentry Mode on": self.sentry_on,
            "cold night": (self.low_temp_f is not None
                           and self.low_temp_f < cold_threshold_f),
            "Cabin Overheat Protection on": self.cop_on,
            "software update": self.software_update,
            "plugged in (not charging)": self.plugged_in,
        }

    def vector_text(self, cold_threshold_f: float = COLD_DEFAULT_F) -> str:
        on = [name for name, flag in self.factors(cold_threshold_f).items()
              if flag]
        loss_bucket = int(round(self.loss_pct()))
        return ("night " + (" ".join(on) if on else "quiet") +
                f" loss {loss_bucket}pct " +
                (self.location_name or "parked"))

    # -- JSONL store ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "vehicle": self.vehicle,
            "evening": self.evening.isoformat(),
            "morning": self.morning.isoformat(),
            "evening_soc": self.evening_soc,
            "morning_soc": self.morning_soc,
            "sentry_on": self.sentry_on,
            "low_temp_f": self.low_temp_f,
            "cop_on": self.cop_on,
            "plugged_in": self.plugged_in,
            "software_update": self.software_update,
            "charged_overnight": self.charged_overnight,
            "location_name": self.location_name,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "NightRecord":
        return cls(
            vehicle=row["vehicle"],
            evening=datetime.fromisoformat(row["evening"]),
            morning=datetime.fromisoformat(row["morning"]),
            evening_soc=float(row["evening_soc"]),
            morning_soc=float(row["morning_soc"]),
            sentry_on=bool(row.get("sentry_on", False)),
            low_temp_f=(None if row.get("low_temp_f") is None
                        else float(row["low_temp_f"])),
            cop_on=bool(row.get("cop_on", False)),
            plugged_in=bool(row.get("plugged_in", False)),
            software_update=bool(row.get("software_update", False)),
            charged_overnight=bool(row.get("charged_overnight", False)),
            location_name=row.get("location_name", ""),
        )


def load_nights(path: str) -> list[NightRecord]:
    p = Path(path).expanduser()
    if not p.exists():
        return []
    nights = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            nights.append(NightRecord.from_dict(json.loads(line)))
    return nights


def append_night(path: str, night: NightRecord) -> None:
    night.validate()
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(night.to_dict()) + "\n")


def export_csv(nights: list[NightRecord], out_path: str) -> int:
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["night", "car", "place", "evening_soc_pct",
                    "morning_soc_pct", "loss_pct", "sentry_on",
                    "low_temp_f", "cop_on", "plugged_in",
                    "software_update", "charged_overnight"])
        for n in nights:
            w.writerow([n.evening.strftime("%Y-%m-%d"), n.vehicle,
                        n.location_name, n.evening_soc, n.morning_soc,
                        round(n.loss_pct(), 1), int(n.sentry_on),
                        "" if n.low_temp_f is None else n.low_temp_f,
                        int(n.cop_on), int(n.plugged_in),
                        int(n.software_update),
                        int(n.charged_overnight)])
    return len(nights)


# -- learning -----------------------------------------------------------

def _drain_nights(nights: list[NightRecord],
                  vehicle: str | None = None) -> list[NightRecord]:
    out = [n for n in nights if n.is_drain_night()]
    if vehicle:
        out = [n for n in out if n.vehicle == vehicle]
    return out


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


@dataclass
class DrainModel:
    vehicle: str
    nights_used: int
    baseline_pct: float
    deltas: dict          # factor name -> (delta, learned: bool)
    clean_nights: int

    def predict(self, conditions: dict) -> "Prediction":
        total = self.baseline_pct
        parts = [("baseline (quiet nights)", self.baseline_pct)]
        for name, on in conditions.items():
            if not on:
                continue
            delta, _learned = self.deltas.get(name, (FACTOR_DEFAULTS[name], False))
            total += delta
            parts.append((name, delta))
        predicted = max(0.0, min(MAX_PREDICTED_PCT, total))
        return Prediction(self.vehicle, predicted, parts)


def learn_model(nights: list[NightRecord], vehicle: str,
                cold_threshold_f: float = COLD_DEFAULT_F,
                window: int = 14) -> DrainModel:
    """Learn one car's model from its most recent drain nights."""
    drain = sorted(_drain_nights(nights, vehicle),
                   key=lambda n: n.evening)[-window:] if window else \
        _drain_nights(nights, vehicle)
    if not drain:
        return DrainModel(vehicle, 0, DEFAULT_BASELINE_PCT,
                          {name: (delta, False)
                           for name, delta in FACTOR_DEFAULTS.items()}, 0)
    clean = [n for n in drain
             if not any(n.factors(cold_threshold_f).values())]
    baseline = _median_or_none([n.loss_pct() for n in clean])
    if baseline is None:
        baseline = _median_or_none([n.loss_pct() for n in drain])
    deltas = {}
    for name, default in FACTOR_DEFAULTS.items():
        with_factor = [n.loss_pct() for n in drain
                       if n.factors(cold_threshold_f)[name]]
        without = [n.loss_pct() for n in drain
                   if not n.factors(cold_threshold_f)[name]]
        if (len(with_factor) >= MIN_GROUP_NIGHTS
                and len(without) >= MIN_GROUP_NIGHTS):
            learned = statistics.median(with_factor) - statistics.median(without)
            learned = max(-2.0, min(6.0, learned))
            deltas[name] = (learned, True)
        else:
            deltas[name] = (default, False)
    return DrainModel(vehicle, len(drain), baseline, deltas, len(clean))


@dataclass
class Prediction:
    vehicle: str
    predicted_pct: float
    parts: list  # (label, points) in order

    def summary_line(self) -> str:
        bits = " + ".join(f"{label} {pts:+.1f}"
                          for label, pts in self.parts)
        total = sum(pts for _, pts in self.parts)
        note = ""
        if total < 0:
            note = " (floored at 0% — a parked car can't gain charge)"
        elif total > MAX_PREDICTED_PCT:
            note = f" (capped at {MAX_PREDICTED_PCT:.0f}%)"
        return (f"{self.vehicle}: predicted overnight loss "
                f"{self.predicted_pct:.1f}% ({bits}){note}.")

    def explain_line(self, actual: float) -> str:
        residual = actual - self.predicted_pct
        if abs(residual) <= 0.5:
            verdict = "about what the model expected"
        elif residual > 0:
            verdict = f"{residual:.1f} points higher than expected"
        else:
            verdict = f"{-residual:.1f} points lower than expected"
        bits = ", ".join(f"{label} {pts:+.1f}" for label, pts in self.parts)
        return (f"Actual loss {actual:.1f}% vs predicted "
                f"{self.predicted_pct:.1f}% — {verdict} "
                f"({bits}).")


def explain_night(night: NightRecord, model: DrainModel,
                  cold_threshold_f: float = COLD_DEFAULT_F) -> str:
    """Plain-English answer to 'why did I lose X% that night?'"""
    prediction = model.predict(night.factors(cold_threshold_f))
    factors = night.factors(cold_threshold_f)
    culprits = sorted(
        ((name, delta) for name, (delta, _learned)
         in model.deltas.items()
         if factors[name] and delta > 0),
        key=lambda item: -item[1])
    date_s = night.evening.strftime("%A %B %d")
    if culprits:
        culprit_s = "; ".join(f"{name} (+{delta:.1f})"
                                for name, delta in culprits)
        lead = (f"On {date_s} the {night.vehicle} lost "
                f"{night.loss_pct():.1f}% overnight. Likely culprits, "
                f"biggest first: {culprit_s}. ")
    elif any(factors.values()):
        on_s = ", ".join(name for name, on in factors.items() if on)
        lead = (f"On {date_s} the {night.vehicle} lost "
                f"{night.loss_pct():.1f}% overnight with only {on_s} "
                f"switched on — no factor that adds drain. ")
    else:
        lead = (f"On {date_s} the {night.vehicle} lost "
                f"{night.loss_pct():.1f}% overnight with no drain factor "
                f"switched on — a quiet night. ")
    return lead + prediction.explain_line(night.loss_pct())


def similar_nights(target: NightRecord, nights: list[NightRecord],
                   cold_threshold_f: float = COLD_DEFAULT_F,
                   limit: int = 3) -> list[tuple[NightRecord, float]]:
    """Most similar drain nights to `target`, by drain-day vector cosine."""
    from .embeddings import embed_text
    target_vec = embed_text(target.vector_text(cold_threshold_f))
    scored = []
    for n in nights:
        if n is target or not n.is_drain_night():
            continue
        if n.evening == target.evening and n.vehicle == target.vehicle:
            continue
        vec = embed_text(n.vector_text(cold_threshold_f))
        dot = sum(a * b for a, b in zip(target_vec, vec))
        norm = (math.sqrt(sum(a * a for a in target_vec)) *
                math.sqrt(sum(b * b for b in vec)))
        scored.append((n, dot / norm if norm else 0.0))
    scored.sort(key=lambda item: -item[1])
    return scored[:limit]


# -- reports / AI brief -------------------------------------------------

def drain_report(nights: list[NightRecord],
                 cold_threshold_f: float = COLD_DEFAULT_F,
                 window: int = 14,
                 alert_threshold_pct: float = 4.0) -> str:
    drain = _drain_nights(nights)
    if not drain:
        return "No overnight records yet."
    lines = []
    for car in sorted({n.vehicle for n in drain}):
        model = learn_model(nights, car, cold_threshold_f, window)
        car_nights = _drain_nights(nights, car)
        losses = [n.loss_pct() for n in car_nights]
        worst = max(car_nights, key=lambda n: n.loss_pct())
        learned_s = ", ".join(
            f"{name} {delta:+.1f}{' (learned)' if learned else ''}"
            for name, (delta, learned) in model.deltas.items())
        lines.append(
            f"{car}: {len(car_nights)} nights recorded, average loss "
            f"{statistics.mean(losses):.1f}%, baseline (quiet nights) "
            f"{model.baseline_pct:.1f}%.")
        lines.append(f"  Learned factors: {learned_s}.")
        lines.append(
            f"  Worst night: {worst.evening.strftime('%a %b %d')} — "
            f"{worst.loss_pct():.1f}% lost "
            f"({worst.evening_soc:.0f}% → {worst.morning_soc:.0f}%).")
        tonight = model.predict({})  # quiet-night forecast as the floor
        lines.append(f"  Quiet-night forecast: {tonight.predicted_pct:.1f}%.")
        if model.baseline_pct >= alert_threshold_pct:
            lines.append(
                f"  Flag: even quiet nights lose {model.baseline_pct:.1f}% — "
                f"above your {alert_threshold_pct:.0f}% alert threshold. "
                f"Check Sentry and Cabin Overheat Protection before bed.")
    return "\n".join(lines)


def drain_brief(nights: list[NightRecord],
                cold_threshold_f: float = COLD_DEFAULT_F,
                window: int = 14,
                alert_threshold_pct: float = 4.0) -> str:
    """Grounding context for the LLM: computed stats only, never guesses."""
    if not nights:
        return "No overnight records yet."
    return drain_report(nights, cold_threshold_f, window, alert_threshold_pct)
