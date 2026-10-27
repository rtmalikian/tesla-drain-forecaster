"""The setup wizard must produce a config the app can load — no developer
knowledge required from the (scripted) user."""
import os

from tesla_drain_forecaster.config import load_settings
from tesla_drain_forecaster.setup import run_setup


def _scripted(answers):
    it = iter(answers)

    def input_fn(prompt=""):
        return next(it)
    return input_fn


def test_wizard_defaults_write_loadable_config(tmp_path):
    env_path = tmp_path / ".env"
    # CID, secret, car, VIN, done, sentry n, cop n, plugged n,
    # threshold default, AI choice default, port default, server key blank
    answers = ["CID123", "SECRET456", "Model Y", "VINY", "",
               "n", "n", "n", "", "2", "", ""]
    run_setup(input_fn=_scripted(answers), out=lambda *_: None,
              env_path=str(env_path))
    text = env_path.read_text()
    assert "TDF_VEHICLES=Model Y=VINY" in text
    assert "TDF_ALERT_THRESHOLD_PCT=4" in text
    assert "TDF_AI_MODE=auto" in text
    # The written file must load through the real config parser.
    env = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v
    settings = load_settings(env)
    assert settings.alert_threshold_pct == 4.0
    assert settings.baseline_nights == 14
    assert settings.vehicles[0].name == "Model Y"
    assert oct(os.stat(env_path).st_mode & 0o777) == "0o600"


def test_wizard_custom_threshold_and_no_ai(tmp_path):
    env_path = tmp_path / ".env"
    answers = ["CID123", "SECRET456", "Model 3", "VIN3", "",
               "y", "n", "y", "3", "1", "", ""]
    run_setup(input_fn=_scripted(answers), out=lambda *_: None,
              env_path=str(env_path))
    text = env_path.read_text()
    assert "TDF_ALERT_THRESHOLD_PCT=3" in text
    assert "TDF_AI_MODE=never" in text
    assert "sentry=on" in text


def test_wizard_keeps_existing_config_when_declined(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TDF_VEHICLES=Keep=VINKEEP\n")
    run_setup(input_fn=_scripted(["n"]), out=lambda *_: None,
              env_path=str(env_path))
    assert env_path.read_text() == "TDF_VEHICLES=Keep=VINKEEP\n"
