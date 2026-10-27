"""Guided first-run setup: plain-language questions, writes .env itself.

Built for Tesla owners, not developers — defaults everywhere, numbered
choices, drain factors in plain words ("Sentry Mode watches the car
with its cameras all night — that costs battery"). Testable: input
and output are injectable.
"""
from __future__ import annotations

import os
from pathlib import Path


def _ask(input_fn, out, prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    out(f"{prompt}{suffix}")
    answer = input_fn("> ").strip()
    return answer or default


def _ask_required(input_fn, out, prompt: str, help_text: str) -> str:
    while True:
        out(help_text)
        answer = input_fn(f"{prompt} ").strip()
        if answer:
            return answer
        out("That one's required — without it the app can't do its job.")


def _ask_yes_no(input_fn, out, prompt: str, default: str = "n") -> bool:
    answer = _ask(input_fn, out, prompt + " (y/n)", default)
    return answer.lower() in ("y", "yes")


def run_setup(input_fn=input, out=print, env_path: str = ".env") -> Path:
    out("")
    out("Welcome to Tesla Drain Forecaster setup — I'll ask a few questions and")
    out("write the settings file for you. Press Enter to accept a [default].")
    out("")

    path = Path(env_path)
    if path.exists():
        again = _ask(input_fn, out,
                     "A settings file (.env) already exists. Replace it? (y/n)", "n")
        if again.lower() not in ("y", "yes"):
            out("Keeping your existing settings. Nothing changed.")
            return path

    out("STEP 1 of 5 — Connect to Tesla")
    out("Create a free developer app at developer.tesla.com (sign in, then")
    out("'Create App'). Tesla will show you a Client ID and Client Secret.")
    client_id = _ask_required(input_fn, out, "Paste your Client ID:",
                              "First, the Client ID:")
    client_secret = _ask_required(
        input_fn, out, "Paste your Client Secret:",
        "Now the Client Secret (it stays on this computer):")

    out("")
    out("STEP 2 of 5 — Your cars")
    out("For each car: a nickname and its VIN. Find the VIN in your Tesla")
    out("app under Controls → Software (or on the car's touchscreen).")
    vehicles: list[tuple[str, str]] = []
    while True:
        n = len(vehicles) + 1
        name = _ask(input_fn, out,
                    f"Car {n} nickname (e.g. Model Y) — or press Enter if done:",
                    "")
        if not name:
            if vehicles:
                break
            out("Add at least one car to continue.")
            continue
        vin = _ask_required(input_fn, out, f"VIN for {name}:",
                            f"Now the VIN for {name}:")
        vehicles.append((name, vin))

    out("")
    out("STEP 3 of 5 — Your usual overnight setup")
    out("These are just your starting guesses — the app learns the real")
    out("numbers from your recorded nights and corrects them.")
    out("Sentry Mode watches the car with its cameras all night. Do you")
    out("usually leave it on at home?")
    sentry = _ask_yes_no(input_fn, out, "Sentry Mode usually on at home?", "n")
    out("Cabin Overheat Protection keeps the cabin cool on hot days, even")
    out("parked. Do you usually leave it on?")
    cop = _ask_yes_no(input_fn, out, "Cabin Overheat Protection usually on?", "n")
    out("Do you usually plug in overnight (even when not charging)?")
    plugged = _ask_yes_no(input_fn, out, "Usually plugged in overnight?", "n")

    out("")
    out("STEP 4 of 5 — When should I flag a bad night?")
    out("A forecast at or above this loss gets flagged 'worth a look'.")
    out("Most parked Teslas lose 1–2% overnight; 4% is a good flag line.")
    threshold = _ask(input_fn, out,
                     "Flag forecasts at or above this % loss:", "4")

    out("")
    out("STEP 5 of 5 — The AI coach (optional) and the forecast server")
    out("  1) Built-in answers only — no AI, nothing leaves this computer")
    out("  2) Ollama on this computer — free local AI (default)")
    out("  3) Another AI service (OpenAI, Groq, LM Studio…) — you'll paste its details")
    choice = _ask(input_fn, out, "Pick 1, 2 or 3:", "2")
    ai_mode, ai_url, ai_key, ai_model = "auto", "http://localhost:11434/v1", "", "llama3.1"
    if choice == "1":
        ai_mode = "never"
    elif choice == "3":
        ai_url = _ask_required(input_fn, out, "Service address (base URL):",
                               "Paste the service's API address:")
        ai_key = _ask(input_fn, out, "API key for that service (stored only here):")
        ai_model = _ask(input_fn, out, "Model name:", "gpt-4o-mini")
    port = _ask(input_fn, out, "Port for the forecast address:", "8650")
    server_key = _ask(input_fn, out,
                      "Optional password for the forecast address (blank = none):")

    lines = [
        "# Written by `tdrain setup` — edit by re-running setup any time.",
        f"TESLA_CLIENT_ID={client_id}",
        f"TESLA_CLIENT_SECRET={client_secret}",
        "TESLA_REGION=na",
        "TESLA_REDIRECT_URI=http://localhost:8080/callback",
        "TDF_VEHICLES=" + ",".join(f"{name}={vin}" for name, vin in vehicles),
        "TDF_SERVER_HOST=127.0.0.1",
        f"TDF_SERVER_PORT={port}",
        f"TDF_SERVER_API_KEY={server_key}",
        "TDF_MODEL_ID=tesla-drain-forecaster",
        f"TDF_AI_BASE_URL={ai_url}",
        f"TDF_AI_API_KEY={ai_key}",
        f"TDF_AI_MODEL={ai_model}",
        f"TDF_AI_MODE={ai_mode}",
        "TDF_AI_TIMEOUT_SECONDS=20",
        "TDF_EMBEDDING_DIM=384",
        "TDF_TOKEN_FILE=~/.tdf/tokens.json",
        f"TDF_ALERT_THRESHOLD_PCT={threshold}",
        "TDF_BASELINE_NIGHTS=14",
        "TDF_COLD_THRESHOLD_F=40",
        "TDF_DATA_FILE=~/.tdf/drain.jsonl",
        "TDF_DRY_RUN=0",
        f"# Usual overnight setup (starting guesses; learned from nights):",
        f"# sentry={'on' if sentry else 'off'} cop={'on' if cop else 'off'} "
        f"plugged={'yes' if plugged else 'no'}",
    ]
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o600)

    out("")
    out("All set — settings saved to .env (only readable by you).")
    out("Next steps:")
    out("  1) tdrain auth       — sign in to Tesla once (opens a link)")
    out("  2) tdrain simulate   — see a demo month with pretend nights")
    out("  3) tdrain record     — log last night (evening % → morning %)")
    out("  4) tdrain predict    — tonight's forecast before bed")
    out("  5) tdrain serve      — start the drain-forecast Q&A API")
    return path
