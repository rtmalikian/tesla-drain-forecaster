"""CLI: tdrain setup | auth | serve | ask | record | predict | report | export | preview | simulate

Safety posture: recording a night only writes to a file on your own
computer — nothing is sent anywhere, and no car is ever woken or
commanded. `record --dry-run` and `preview` show the forecast before
anything is stored.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from .config import ConfigError, load_settings, parse_12h_time

__version__ = "0.1.0"


def _setup_logging(verbose: bool) -> None:
    import logging
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")


def _client(settings):
    from .tesla_client import TeslaClient
    return TeslaClient(settings.client_id, settings.client_secret,
                       region=settings.region, redirect_uri=settings.redirect_uri,
                       token_file=settings.token_file)


def cmd_auth(args, settings) -> int:
    client = _client(settings)
    print("Open this URL in your browser and authorize:")
    print(client.authorize_url())
    print("Virtual-key pairing: https://developer.tesla.com/docs/fleet-api#pairing")
    return 0


def cmd_setup(args, settings) -> int:
    from .setup import run_setup
    run_setup()
    return 0


def _nights_for_api(settings):
    """Real nights if they exist, else the demo nights (labelled as such)
    so the API is testable before the first real night."""
    from .drain import load_nights
    from .simulator import run_demo_month
    real = load_nights(settings.data_file)
    if real:
        return real, False
    return run_demo_month(settings), True


def cmd_serve(args, settings) -> int:
    from .server import serve
    from .service import DrainForecasterService
    nights, is_demo = _nights_for_api(settings)
    service = DrainForecasterService(settings, nights_provider=lambda: nights)
    print("Night data goes ONLY to your configured LLM backend "
          f"(TDF_AI_BASE_URL={settings.ai_base_url or 'none'}); "
          "the rules engine answers drain questions with no backend at all.")
    if is_demo:
        print("(Serving the demo nights — record a real night and the")
        print(" API will answer from your nights instead.)")
    print("This app only reads and forecasts overnight drain — it never "
          "wakes or commands a car.")
    serve(service)
    return 0


def cmd_ask(args, settings) -> int:
    from .service import DrainForecasterService
    nights, is_demo = _nights_for_api(settings)
    service = DrainForecasterService(settings, nights_provider=lambda: nights)
    if is_demo:
        print("(Demo nights — record a real night to ask about your car.)")
    body = {"model": settings.model_id,
            "messages": [{"role": "user", "content": args.question}]}
    resp = service.chat_completions(body)
    print(resp["choices"][0]["message"]["content"])
    return 0


def _pick_car(args_car, settings, nights=None):
    cars = [v.name for v in settings.vehicles]
    if not cars and nights:
        # No cars configured yet (e.g. previewing against demo nights):
        # pick from the cars actually in the record instead of prompting.
        cars = sorted({n.vehicle for n in nights})
    if args_car:
        return args_car
    if len(cars) == 1:
        print(f"Car: {cars[0]} (your only car — run setup to add more)")
        return cars[0]
    if cars:
        print("Which car?")
        for i, name in enumerate(cars, 1):
            print(f"  {i}) {name}")
        pick = input("> ").strip()
        try:
            return cars[int(pick) - 1]
        except (ValueError, IndexError):
            print("That wasn't one of the numbered cars — try again.")
            return None
    return input("Car nickname (e.g. Model Y): ").strip() or "Model Y"


def _ask(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}\n> ").strip()
    return answer or default


def _ask_yes_no(prompt, default="n"):
    return _ask(prompt + " (y/n)", default).lower() in ("y", "yes")


def _parse_night_args(args, settings):
    """Build a NightRecord from CLI args, or ask interactively for
    anything missing. Returns a night or None on bad input."""
    from .drain import NightRecord
    car = _pick_car(args.car, settings)
    if car is None:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_s = args.date or _ask("Night of (date you went to bed, like 2026-10-26)",
                               today)
    evening_s = args.evening or _ask("Bedtime (like 10:00 PM)", "10:00 PM")
    morning_s = args.morning or _ask("Morning reading time (like 7:00 AM)",
                                     "7:00 AM")
    try:
        day = datetime.strptime(date_s, "%Y-%m-%d")
    except ValueError:
        print(f"Couldn't read that date: {date_s!r}. Write it like 2026-10-26.")
        return None
    try:
        evening_min = parse_12h_time(evening_s)
        morning_min = parse_12h_time(morning_s)
    except ConfigError as exc:
        print(f"Couldn't read that time: {exc}")
        return None
    evening = day.replace(hour=evening_min // 60, minute=evening_min % 60,
                          tzinfo=timezone.utc)
    morning = day.replace(hour=morning_min // 60, minute=morning_min % 60,
                          tzinfo=timezone.utc)
    if morning <= evening:  # morning is the next day
        morning += timedelta(days=1)
    try:
        evening_soc = float(args.evening_soc if args.evening_soc is not None
                            else _ask("Battery % at bedtime:", "70"))
        morning_soc = float(args.morning_soc if args.morning_soc is not None
                            else _ask("Battery % in the morning:", "68"))
    except ValueError:
        print("Those need to be numbers — battery % like 70.")
        return None
    sentry = args.sentry if args.sentry is not None else \
        _ask_yes_no("Was Sentry Mode on overnight?", "n")
    low_temp = args.low_temp
    if low_temp is None:
        raw = _ask("Overnight low temperature in °F (blank if unknown):", "")
        low_temp = float(raw) if raw else None
    cop = args.cop if args.cop is not None else \
        _ask_yes_no("Was Cabin Overheat Protection on?", "n")
    plugged = args.plugged if args.plugged is not None else \
        _ask_yes_no("Was the car plugged in (but not charging)?", "n")
    update = args.update if args.update is not None else \
        _ask_yes_no("Did a software update install overnight?", "n")
    charged = args.charged if args.charged is not None else \
        _ask_yes_no("Did the car charge overnight? (if yes, this isn't a drain night)",
                    "n")
    place = args.place if args.place is not None else _ask(
        "Where was the car parked (blank is fine):", "")
    night = NightRecord(car, evening, morning, evening_soc, morning_soc,
                        sentry_on=sentry, low_temp_f=low_temp, cop_on=cop,
                        plugged_in=plugged, software_update=update,
                        charged_overnight=charged, location_name=place)
    try:
        night.validate()
    except Exception as exc:
        print(f"That night doesn't add up: {exc}")
        return None
    return night


def cmd_record(args, settings) -> int:
    from .drain import append_night
    night = _parse_night_args(args, settings)
    if night is None:
        return 2
    print("")
    print("Here's what I would record:")
    print(f"  {night.vehicle} at {night.location_name or 'parked'}, "
          f"{night.evening.strftime('%b %d, %I:%M %p')} → "
          f"{night.morning.strftime('%I:%M %p')}, "
          f"{night.evening_soc:.0f}% → {night.morning_soc:.0f}% "
          f"(lost {night.loss_pct():.1f}%)")
    if night.charged_overnight:
        print("  (Charged overnight — stored, but excluded from drain stats.)")
    if args.dry_run or settings.dry_run:
        print("Dry run — nothing was saved. Re-run without --dry-run to save it.")
        return 0
    if not args.yes:
        answer = input("Save this night? (y/n) ").strip().lower()
        if answer not in ("y", "yes"):
            print("Not saved.")
            return 0
    append_night(settings.data_file, night)
    print(f"Saved to {settings.data_file}.")
    return 0


def _conditions_from_flags(args, settings):
    return {
        "Sentry Mode on": bool(args.sentry),
        "cold night": (args.low_temp is not None
                       and args.low_temp < settings.cold_threshold_f),
        "Cabin Overheat Protection on": bool(args.cop),
        "software update": bool(args.update),
        "plugged in (not charging)": bool(args.plugged),
    }


def cmd_predict(args, settings) -> int:
    from .drain import learn_model, load_nights
    nights = load_nights(settings.data_file)
    if not nights:
        print("No nights recorded yet — nothing to learn from.")
        print("Record a night with `tdrain record`, or see a demo forecast")
        print("with `tdrain simulate`.")
        return 0
    car = _pick_car(args.car, settings, nights)
    if car is None:
        return 2
    model = learn_model(nights, car, settings.cold_threshold_f,
                        settings.baseline_nights)
    if model.nights_used == 0:
        print(f"No drain nights recorded for {car} yet.")
        return 0
    conditions = _conditions_from_flags(args, settings)
    pred = model.predict(conditions)
    print(f"Tonight's forecast for {car}:")
    print("  " + pred.summary_line())
    if pred.predicted_pct >= settings.alert_threshold_pct:
        print(f"  Above your {settings.alert_threshold_pct:.0f}% alert "
              f"threshold — worth switching Sentry off tonight.")
    print(f"  (Learned from {model.nights_used} recent nights; "
          f"quiet-night baseline {model.baseline_pct:.1f}%.)")
    return 0


def cmd_report(args, settings) -> int:
    from .drain import drain_report, load_nights
    nights = load_nights(settings.data_file)
    if not nights:
        print("No nights recorded yet — nothing to report.")
        print("Record a night with `tdrain record`, or see a demo month")
        print("with `tdrain simulate`.")
        return 0
    if args.car:
        nights = [n for n in nights if n.vehicle.lower() == args.car.lower()]
    print(drain_report(nights, settings.cold_threshold_f,
                       settings.baseline_nights,
                       settings.alert_threshold_pct))
    return 0


def cmd_export(args, settings) -> int:
    from .drain import export_csv, load_nights
    nights = load_nights(settings.data_file)
    if not nights:
        print("No nights recorded yet — nothing to export.")
        return 0
    out = args.out or "drain-nights.csv"
    n = export_csv(nights, out)
    print(f"Wrote {n} nights to {out} (date, car, place, SoC swing, loss %, "
          "Sentry, low temp, Cabin Overheat Protection, plugged, update).")
    return 0


def cmd_preview(args, settings) -> int:
    """What tonight would lose under hypothetical conditions — saves nothing."""
    from .drain import learn_model, load_nights
    nights = load_nights(settings.data_file)
    if not nights:
        from .simulator import run_demo_month
        nights = run_demo_month(settings)
        print("(Preview against the demo nights — record real nights to "
              "preview against your own car.)")
    car = _pick_car(args.car, settings, nights)
    if car is None:
        return 2
    model = learn_model(nights, car, settings.cold_threshold_f,
                        settings.baseline_nights)
    conditions = _conditions_from_flags(args, settings)
    pred = model.predict(conditions)
    print("Preview (nothing saved):")
    print("  " + pred.summary_line())
    if not any(conditions.values()):
        print("  That's a quiet night — try --sentry or --low-temp 35 to "
              "see what they add.")
    return 0


def cmd_simulate(_args, settings) -> int:
    from .simulator import timeline_lines
    for line in timeline_lines(settings):
        print(line)
    return 0


def _bool_flag(parser, name, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"),
                       action="store_true", default=None, help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"),
                       action="store_false", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tdrain",
        description="A free overnight phantom-drain forecaster for your "
                    "Teslas: learn each car's drain pattern and predict "
                    "tonight's loss before bed.")
    p.add_argument("--version", action="version", version=f"tdrain {__version__}")
    p.add_argument("--env-file", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="Guided first-run setup (plain-language questions)")
    sub.add_parser("auth", help="Authorize with your Tesla account (one-time)")
    sub.add_parser("serve", help="Run the OpenAI-compatible drain-forecast API server")
    ask_p = sub.add_parser("ask", help="One-shot drain question (your nights, or demo nights)")
    ask_p.add_argument("question", help="e.g. 'predict tonight's loss'")
    rec_p = sub.add_parser("record", help="Log last night (asks what it needs)")
    rec_p.add_argument("--car", default=None)
    rec_p.add_argument("--date", default=None, help="YYYY-MM-DD (bedtime date)")
    rec_p.add_argument("--evening", default=None, help="e.g. 10:00 PM")
    rec_p.add_argument("--morning", default=None, help="e.g. 7:00 AM")
    rec_p.add_argument("--evening-soc", type=float, default=None)
    rec_p.add_argument("--morning-soc", type=float, default=None)
    rec_p.add_argument("--low-temp", type=float, default=None,
                       help="Overnight low in °F")
    rec_p.add_argument("--place", default=None)
    rec_p.add_argument("--dry-run", action="store_true",
                       help="Show what would be recorded, save nothing")
    rec_p.add_argument("--yes", action="store_true",
                       help="Save without asking for confirmation")
    _bool_flag(rec_p, "sentry", "Sentry Mode was on")
    _bool_flag(rec_p, "cop", "Cabin Overheat Protection was on")
    _bool_flag(rec_p, "plugged", "Car was plugged in (not charging)")
    _bool_flag(rec_p, "update", "A software update installed overnight")
    _bool_flag(rec_p, "charged", "Car charged overnight")
    pre_p = sub.add_parser("predict", help="Tonight's forecast from your nights")
    pre_p.add_argument("--car", default=None)
    pre_p.add_argument("--low-temp", type=float, default=None)
    _bool_flag(pre_p, "sentry", "Sentry Mode will be on tonight")
    _bool_flag(pre_p, "cop", "Cabin Overheat Protection will be on")
    _bool_flag(pre_p, "plugged", "Car will be plugged in (not charging)")
    _bool_flag(pre_p, "update", "A software update is pending tonight")
    rep_p = sub.add_parser("report", help="Overnight drain report")
    rep_p.add_argument("--car", default=None)
    exp_p = sub.add_parser("export", help="Export nights to CSV")
    exp_p.add_argument("--out", default=None)
    prv_p = sub.add_parser("preview", help="What tonight would lose (saves nothing)")
    prv_p.add_argument("--car", default=None)
    prv_p.add_argument("--low-temp", type=float, default=None)
    _bool_flag(prv_p, "sentry", "Sentry Mode on")
    _bool_flag(prv_p, "cop", "Cabin Overheat Protection on")
    _bool_flag(prv_p, "plugged", "Plugged in (not charging)")
    _bool_flag(prv_p, "update", "Software update tonight")
    sub.add_parser("simulate", help="Demo nights timeline (no API calls)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    env = dict(os.environ)
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
        env = dict(os.environ)
    else:
        from dotenv import load_dotenv
        load_dotenv(override=False)
        env = dict(os.environ)
    try:
        settings = load_settings(env)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    handlers = {"setup": cmd_setup, "auth": cmd_auth, "serve": cmd_serve,
                "ask": cmd_ask, "record": cmd_record, "predict": cmd_predict,
                "report": cmd_report, "export": cmd_export,
                "preview": cmd_preview, "simulate": cmd_simulate}
    return handlers[args.command](args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
