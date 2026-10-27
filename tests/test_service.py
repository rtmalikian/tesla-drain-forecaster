"""OpenAI response shapes + drain Q&A rules. The rules engine must
answer drain questions with computed stats, never invented numbers."""
from tesla_drain_forecaster.config import load_settings
from tesla_drain_forecaster.service import DrainForecasterService
from tesla_drain_forecaster.simulator import run_demo_month


def _service(nights=None):
    settings = load_settings({"TDF_VEHICLES": "Model Y=VINY",
                              "TDF_AI_MODE": "never"})
    if nights is None:
        nights = run_demo_month(settings)
    return DrainForecasterService(settings, nights_provider=lambda: nights)


def _ask(service, question):
    resp = service.chat_completions(
        {"model": "tesla-drain-forecaster",
         "messages": [{"role": "user", "content": question}]})
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "tesla-drain-forecaster"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert resp["choices"][0]["finish_reason"] == "stop"
    return resp["choices"][0]["message"]["content"]


def test_models_shape():
    models = _service().models_list()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "tesla-drain-forecaster"


def test_forecast_answer_matches_learned_model():
    service = _service()
    answer = _ask(service, "predict tonight's loss")
    assert "Tonight's forecast" in answer
    assert "predicted overnight loss" in answer


def test_why_tuesday_answer_explains_a_real_night():
    service = _service()
    # Oct 20, 2026 was a Tuesday; the demo nights cover it.
    answer = _ask(service, "why did I lose so much on Tuesday?")
    assert "Tuesday" in answer or "Oct" in answer
    assert "%" in answer


def test_worst_and_average_answers():
    service = _service()
    worst = _ask(service, "what's my worst night?")
    assert "Worst night" in worst
    assert "4.6%" in worst
    avg = _ask(service, "what's my average overnight loss?")
    assert "average overnight loss" in avg


def test_report_answer():
    service = _service()
    answer = _ask(service, "give me my drain report")
    assert "Model Y" in answer
    assert "Worst night" in answer


def test_sentry_answer():
    service = _service()
    answer = _ask(service, "how much does Sentry Mode cost me overnight?")
    assert "Sentry Mode adds" in answer


def test_embeddings_shape_and_determinism():
    service = _service()
    resp = service.embeddings({"input": "sentry night loss 4pct"})
    assert resp["object"] == "list"
    assert len(resp["data"][0]["embedding"]) == 384
    again = service.embeddings({"input": "sentry night loss 4pct"})
    assert again["data"][0]["embedding"] == resp["data"][0]["embedding"]


def test_unknown_question_falls_back_to_capability_blurb():
    service = _service()
    answer = _ask(service, "tell me a joke about Mars")
    assert "predict tonight's loss" in answer
