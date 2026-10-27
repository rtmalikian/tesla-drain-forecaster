"""Transport-independent service: takes OpenAI-shaped request dicts,
returns OpenAI-shaped response dicts. The HTTP server (server.py) is a
thin wrapper; tests drive this class directly.

Flow for a user message:
  1. Rules engine answers drain questions from the learned model
     (no AI needed — "predict tonight's loss", "why did I lose 6%
     Tuesday?", "what's my worst night?").
  2. If rules don't match and ai_mode is auto|always → LLM backend with
     the drain brief as context. The model explains; it never invents
     a loss figure — stats come from the night math.
  3. ai_mode=never (or LLM failure) → capability-list answer.

This app never wakes or commands a car: it reads recorded nights and
forecasts from them. LLM-emitted text is never executed as a command.
"""
from __future__ import annotations

import itertools
import re
import time

from .drain import (drain_brief, drain_report, explain_night, learn_model,
                    similar_nights)
from .embeddings import embed_text
from .llm import LlmBackend

CAPABILITY_BLURB = (
    "I learn your Tesla's overnight drain pattern and predict tonight's "
    "loss: baseline from your quiet nights, plus what Sentry Mode, cold "
    "nights, Cabin Overheat Protection, software updates, and being "
    "plugged in add. Try: \"predict tonight's loss\", \"why did I lose "
    "6% on Tuesday?\", \"what's my worst night?\" or \"show me nights "
    "like last Tuesday\". "
    "Connect an LLM backend (TDF_AI_BASE_URL) and I can explain in "
    "free-form words too. I never wake or control a car — I only read "
    "and forecast overnight drain."
)

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday")


class DrainForecasterService:
    def __init__(self, settings, nights_provider=None, llm_factory=None):
        self.settings = settings
        self.nights_provider = nights_provider or (lambda: [])
        self.llm_factory = llm_factory
        self._ids = itertools.count(1)

    # -- plumbing ------------------------------------------------------
    def _llm(self):
        if self.llm_factory is not None:
            return self.llm_factory()
        if self.settings.ai_mode == "never":
            return None
        if not self.settings.ai_base_url:
            return None
        return LlmBackend(self.settings.ai_base_url, self.settings.ai_api_key,
                          self.settings.ai_model, self.settings.ai_timeout_seconds)

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}-{int(time.time())}-{next(self._ids)}"

    # -- OpenAI surface ------------------------------------------------
    def models_list(self) -> dict:
        return {"object": "list", "data": [{
            "id": self.settings.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tesla-drain-forecaster",
        }]}

    def chat_completions(self, body: dict) -> dict:
        messages = body.get("messages") or []
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = str(m.get("content") or "")
                break
        nights = self.nights_provider()
        text = self._rules_answer(user_text, nights)
        if text is None:
            text = self._llm_answer(user_text, nights)
        return {
            "id": self._next_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.settings.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def embeddings(self, body: dict) -> dict:
        raw = body.get("input", "")
        texts = raw if isinstance(raw, list) else [raw]
        data = []
        for i, t in enumerate(texts):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embed_text(str(t), self.settings.embedding_dim),
            })
        return {
            "object": "list",
            "data": data,
            "model": self.settings.model_id,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    # -- internals -----------------------------------------------------
    def _car_nights(self, q: str, nights: list):
        for name in {n.vehicle for n in nights}:
            if name.lower() in q:
                return [n for n in nights if n.vehicle == name]
        return nights

    def _rules_answer(self, user_text: str, nights: list) -> str | None:
        q = user_text.lower()
        if not nights:
            return None
        car_nights = self._car_nights(q, nights)
        drain = [n for n in car_nights if n.is_drain_night()]
        if not drain:
            return None
        cars = sorted({n.vehicle for n in drain})
        cold = self.settings.cold_threshold_f
        window = self.settings.baseline_nights
        alert = self.settings.alert_threshold_pct

        if "predict" in q or "forecast" in q or "tonight" in q:
            # Assume tonight looks like the most recent night — stated
            # plainly, so the owner can correct it.
            latest = max(drain, key=lambda n: n.evening)
            model = learn_model(nights, latest.vehicle, cold, window)
            conditions = latest.factors(cold)
            pred = model.predict(conditions)
            assumed = ", ".join(name for name, on in conditions.items()
                                if on) or "a quiet night (nothing on)"
            flag = (f" That's above your {alert:.0f}% alert threshold — "
                    f"worth switching Sentry off tonight."
                    if pred.predicted_pct >= alert else "")
            return (f"Tonight's forecast for {latest.vehicle}: "
                    f"{pred.summary_line()} Assuming tonight looks like "
                    f"your most recent night ({assumed}).{flag}")

        if "why" in q and ("lose" in q or "loss" in q or "drain" in q):
            target_day = None
            for day in WEEKDAYS:
                if day in q:
                    target_day = WEEKDAYS.index(day)
                    break
            candidates = drain
            if target_day is not None:
                candidates = [n for n in drain
                              if n.evening.weekday() == target_day]
            if not candidates:
                return None
            night = max(candidates, key=lambda n: n.evening)
            model = learn_model(nights, night.vehicle, cold, window)
            return explain_night(night, model, cold)

        if "similar" in q or "nights like" in q:
            latest = max(drain, key=lambda n: n.evening)
            sims = similar_nights(latest, nights, cold)
            if not sims:
                return "No other nights recorded yet to compare with."
            lines = [f"Nights most like {latest.evening.strftime('%a %b %d')} "
                     f"({latest.loss_pct():.1f}% lost):"]
            for n, score in sims:
                lines.append(
                    f"  • {n.evening.strftime('%a %b %d')}: "
                    f"{n.loss_pct():.1f}% lost "
                    f"(similarity {score:.2f})")
            return "\n".join(lines)

        if "worst" in q:
            worst = max(drain, key=lambda n: n.loss_pct())
            model = learn_model(nights, worst.vehicle, cold, window)
            return (f"Worst night: {worst.evening.strftime('%a %b %d')} — "
                    f"{worst.loss_pct():.1f}% lost "
                    f"({worst.evening_soc:.0f}% → {worst.morning_soc:.0f}%). "
                    + explain_night(worst, model, cold))

        if "average" in q or "typical" in q or "usual" in q:
            lines = []
            for car in cars:
                car_drain = [n for n in drain if n.vehicle == car]
                avg = sum(n.loss_pct() for n in car_drain) / len(car_drain)
                model = learn_model(nights, car, cold, window)
                lines.append(
                    f"{car}: average overnight loss {avg:.1f}% over "
                    f"{len(car_drain)} nights; quiet-night baseline "
                    f"{model.baseline_pct:.1f}%.")
            return "\n".join(lines)

        if "report" in q or "summary" in q or "how is" in q or "how's" in q:
            return drain_report(car_nights, cold, window, alert)

        if "sentry" in q:
            lines = []
            for car in cars:
                model = learn_model(nights, car, cold, window)
                delta, learned = model.deltas["Sentry Mode on"]
                lines.append(
                    f"{car}: Sentry Mode adds about {delta:+.1f} points "
                    f"per night{' (learned from your nights)' if learned else ''}.")
            return "\n".join(lines)
        return None

    def _llm_answer(self, user_text: str, nights: list) -> str:
        llm = self._llm()
        if llm is None:
            return CAPABILITY_BLURB
        try:
            return llm.chat(drain_brief(nights, self.settings.cold_threshold_f,
                                        self.settings.baseline_nights,
                                        self.settings.alert_threshold_pct),
                            user_text)
        except Exception:
            return CAPABILITY_BLURB + " (The AI backend didn't respond just now.)"
