"""End-to-end: the demo nights' learned baseline is ~1.1%, the Sentry
delta is learned positive, tonight's forecast beats the quiet-night
baseline and crosses the 4% alert threshold, and the HTTP server
answers a drain question live."""
import threading
from http.server import ThreadingHTTPServer

import pytest
import requests

from tesla_drain_forecaster.config import load_settings
from tesla_drain_forecaster.drain import learn_model
from tesla_drain_forecaster.server import make_handler
from tesla_drain_forecaster.service import DrainForecasterService
from tesla_drain_forecaster.simulator import (run_demo_month, timeline_lines,
                                              tonight_conditions)


def _settings():
    return load_settings({"TDF_VEHICLES": "Model Y=VINY",
                          "TDF_AI_MODE": "never"})


def test_demo_nights_baseline_and_sentry_delta():
    nights = run_demo_month(_settings())
    assert len(nights) == 21
    model = learn_model(nights, "Model Y", 40.0, 14)
    assert model.baseline_pct == pytest.approx(1.1)
    delta, learned = model.deltas["Sentry Mode on"]
    assert learned
    assert delta > 0


def test_demo_tonight_forecast_beats_baseline_and_alerts():
    settings = _settings()
    nights = run_demo_month(settings)
    model = learn_model(nights, "Model Y", 40.0, 14)
    forecast = model.predict(tonight_conditions())
    assert forecast.predicted_pct > model.baseline_pct
    assert forecast.predicted_pct >= settings.alert_threshold_pct
    lines = "\n".join(timeline_lines(settings))
    assert "alert threshold" in lines
    assert "4.6%" in lines  # the worst night, Oct 21


def test_http_server_answers_live():
    settings = _settings()
    nights = run_demo_month(settings)
    service = DrainForecasterService(settings, nights_provider=lambda: nights)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        models = requests.get(f"http://127.0.0.1:{port}/v1/models",
                              timeout=5).json()
        assert models["data"][0]["id"] == "tesla-drain-forecaster"
        chat = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "tesla-drain-forecaster",
                  "messages": [{"role": "user",
                                "content": "predict tonight's loss"}]},
            timeout=5).json()
        assert "predicted overnight loss" in \
            chat["choices"][0]["message"]["content"]
        emb = requests.post(
            f"http://127.0.0.1:{port}/v1/embeddings",
            json={"input": "sentry night"}, timeout=5).json()
        assert len(emb["data"][0]["embedding"]) == 384
    finally:
        httpd.shutdown()
        httpd.server_close()
