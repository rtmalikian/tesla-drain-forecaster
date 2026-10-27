# Tesla Drain Forecaster — Free Overnight Phantom Drain Predictor for Tesla Model 3, Model Y, Model S, Model X & Cybertruck

> **▶ Try the live demo: https://rtmalikian.github.io/tesla-drain-forecaster/** (interactive, runs on pretend data)

**Free & open source overnight phantom-drain prediction via the Tesla Fleet API: learn each car's overnight drain pattern from recorded nights and predict tonight's loss before bed — so you know whether to switch Sentry Mode off, and why last Tuesday cost you 6% — no paid app, no subscription, no waking your car.**

## Why?

Tesla owners ask the same questions, constantly, with no free answer:

- "Why did my car lose 6% overnight?" — the Tesla app shows the loss, never the reason.
- "Should I leave Sentry Mode on tonight?" — nobody tells you what it costs *your* car, on *your* driveway.
- "Is this drain normal?" — drain-detective tools report drain after the fact; nobody predicts tonight's loss before bed.

Every parked Tesla loses some charge overnight — Sentry Mode's cameras, Cabin Overheat Protection, cold weather, a software update. Drain Forecaster learns *your* car's pattern: the quiet-night baseline, plus what each factor adds, learned from your own nights (not averages from the internet). Then it forecasts tonight, with the breakdown printed next to the number.

## How it works

```
Oct 05  68% → 65.3%, lost 2.7%  (Sentry Mode on, plugged in)
Oct 07  70% → 68.8%, lost 1.2%  (quiet night)
Oct 09  72% → 68.9%, lost 3.1%  (Sentry Mode on)
Oct 13  71% → 67.8%, lost 3.2%  (Sentry Mode on)
Oct 21  69% → 64.4%, lost 4.6%  (Sentry Mode on, cold night) ← worst night
...
Baseline (quiet nights): 1.1%
Learned factors: Sentry Mode on +2.1, cold night +2.5,
  Cabin Overheat Protection on +1.5, plugged in −1.5
Tonight (Sentry on, 38°F, plugged in): predicted loss 4.2% — above the 4% alert threshold
```

**The baseline.** The median loss of your "quiet" nights (no factor switched on) over the recent window. If there are no quiet nights, the median of all drain nights. No nights at all? 1.0%, the typical overnight loss owners report for a healthy parked Tesla — and the app says so.

**The factor deltas.** For each factor — Sentry Mode on, a cold night (overnight low below your threshold, default 40°F), Cabin Overheat Protection left on, a software-update night, plugged in but not charging — the delta is the median loss of nights *with* the factor minus the median of nights *without* it, when both groups have at least 2 nights. Otherwise a transparent default is used (Sentry +1.5, cold +1.0, Cabin Overheat Protection +0.5, software update +1.0, plugged in −0.3). Every prediction prints its breakdown, so the number is never a black box.

**The forecast.** Tonight's predicted loss = baseline + tonight's factor deltas, clamped to 0–15%. A forecast at or above your alert threshold (default 4%) is flagged "worth a look" — and tells you what switching Sentry off would save.

**Why did I lose 6% Tuesday?** The app finds that Tuesday's night and ranks its factors biggest-first, with the residual stated honestly ("about what the model expected" vs "1.4 points higher than expected"). Similar nights are found by drain-day vector similarity — "nights like last Tuesday" without any AI backend.

**Expected API cost:** this app only reads the car's last-known state with no-wake reads — two reads a night, no wake-ups (wakes are the priciest Fleet API call, so a sleeping car is simply skipped and read later), no commands, so the real bill stays near zero. The night record lives on your computer, not in Tesla's cloud.

## Quickstart (for owners, not developers)

No AI needed: everything below works with no LLM backend at all. The AI coach is an optional extra.

```bash
git clone https://github.com/rtmalikian/tesla-drain-forecaster.git
cd tesla-drain-forecaster
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
tdrain setup        # plain-language questions — writes .env for you
tdrain simulate     # see demo nights with pretend data (no car needed)
tdrain record       # log last night: bedtime % → morning % (it asks what it needs)
tdrain predict      # tonight's forecast before bed
tdrain serve        # start the drain-forecast API on http://127.0.0.1:8650
```

What you'll see when it works: `tdrain simulate` prints 21 demo nights, the 1.1% baseline, the learned factor deltas, the worst night (Oct 21, 4.6%), and tonight's forecast (4.2%, above the alert threshold). `tdrain record` asks for the car (numbered list), the night, times in "10:00 PM" words, battery %, and yes/no questions for Sentry, Cabin Overheat Protection, plugged in, and software update — then shows a preview and asks "Save this night?" before writing anything.

### Commands

| Command | What it does |
|---|---|
| `tdrain setup` | Guided setup wizard (writes `.env` itself) |
| `tdrain auth` | One-time Tesla authorization (opens a link) |
| `tdrain serve` | Run the OpenAI-compatible forecast API server |
| `tdrain ask "…"` | One-shot drain question (your nights, or the demo nights) |
| `tdrain record` | Log last night (asks what it needs; `--dry-run` previews) |
| `tdrain predict` | Tonight's forecast (`--sentry --low-temp 35 --plugged`) |
| `tdrain report` | Overnight drain report (`--car "Model Y"` filters) |
| `tdrain export` | Export nights to CSV |
| `tdrain preview --sentry --low-temp 35` | What tonight would lose (saves nothing) |
| `tdrain simulate` | Demo nights timeline (no API calls) |

### Preview before action

`record` always shows what it would save and asks for confirmation (or use `--dry-run` to preview without saving). `preview` computes a hypothetical night's forecast without storing anything. The app never wakes or commands a car — it only reads recorded nights and forecasts from them.

## AI endpoints

Drain Forecaster embeds OpenAI-compatible endpoints so any AI client (ChatGPT-style apps, Ollama clients, LangChain, Open WebUI) can ask about your overnight drain:

```bash
# What models are available?
curl http://127.0.0.1:8650/v1/models

# Predict tonight (rules engine answers — no LLM needed)
curl http://127.0.0.1:8650/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "tesla-drain-forecaster",
  "messages": [{"role": "user", "content": "predict tonight'"'"'s loss"}]
}'

# Why did Tuesday cost so much?
curl http://127.0.0.1:8650/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "tesla-drain-forecaster",
  "messages": [{"role": "user", "content": "why did I lose so much on Tuesday?"}]
}'

# Embeddings over drain-day vectors (local deterministic 384-dim vectors by default)
curl http://127.0.0.1:8650/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "tesla-drain-forecaster", "input": "sentry night cold loss 4pct"
}'
```

Streaming is not supported (`stream: true` is ignored). The rules engine answers drain questions from the learned model with no backend at all; with a backend configured (`TDF_AI_MODE=auto`, default Ollama at `http://localhost:11434/v1`), the LLM explains the stats in free-form words — it never invents a loss figure, and it never wakes or commands a car. Vehicle data is sent only to the backend you configure, nowhere else. Optional bearer auth: set `TDF_SERVER_API_KEY` and clients must send `Authorization: Bearer <key>`.

## Tesla developer setup

1. Create a free app at [developer.tesla.com](https://developer.tesla.com) and note the Client ID and Secret.
2. Run `tdrain setup` and paste them when asked (they stay in `.env` on your computer).
3. Run `tdrain auth` and follow the link to authorize; pair the virtual key per [Tesla's pairing docs](https://developer.tesla.com/docs/fleet-api#pairing).
4. Record nights with `tdrain record` — or let the Fleet API feed them in as the integration grows.

## `.env` reference

| Variable | Default | Meaning |
|---|---|---|
| `TESLA_CLIENT_ID` / `TESLA_CLIENT_SECRET` | — | Your Tesla developer app credentials |
| `TESLA_REGION` | `na` | `na`, `eu`, or `cn` |
| `TDF_VEHICLES` | — | `Model Y=VIN,Model 3=VIN2` (picked by number in setup) |
| `TDF_SERVER_HOST` / `TDF_SERVER_PORT` | `127.0.0.1` / `8650` | The forecast API address |
| `TDF_SERVER_API_KEY` | (blank) | Optional bearer token for `/v1/*` |
| `TDF_AI_BASE_URL` | `http://localhost:11434/v1` | Pluggable LLM backend (Ollama default) |
| `TDF_AI_API_KEY` / `TDF_AI_MODEL` | — / `llama3.1` | Backend credentials and model |
| `TDF_AI_MODE` | `auto` | `auto` (rules first) / `always` / `never` |
| `TDF_ALERT_THRESHOLD_PCT` | `4` | Flag forecasts at or above this % loss |
| `TDF_BASELINE_NIGHTS` | `14` | Recent drain nights the model learns from |
| `TDF_COLD_THRESHOLD_F` | `40` | Overnight low below this °F = a cold night |
| `TDF_DATA_FILE` | `~/.tdf/drain.jsonl` | Your night record (local JSONL) |
| `TDF_DRY_RUN` | `0` | `1` = preview instead of saving |

## FAQ

**How accurate is the forecast?**
It's your own car's pattern, not an internet average: the median of your quiet nights plus learned factor deltas. With 14+ nights including a few quiet ones and a few Sentry nights, the forecast is usually within about a point. The app always prints the breakdown, so you can see exactly what it's assuming.

**Why did my car lose more than predicted?**
The explanation names the residual honestly. A big unexplained residual usually means a factor the app doesn't track yet (a 12V battery issue, a stuck update download, extreme cold below the threshold) — the model improves as more nights accumulate.

**Do I need an AI backend?**
No. The rules engine answers "predict tonight's loss", "why did I lose 6% Tuesday?", "what's my worst night?", and "show me nights like last Tuesday" with no backend at all. The LLM only adds free-form explanations on top of the same computed stats.

**Does it wake my car?**
No. Evening and morning readings use no-wake vehicle data (the car's last-known state while asleep). A sleeping car is skipped and read later — wakes are the priciest Fleet API call, and this app never issues one.

**What if I charge overnight?**
Charged nights aren't drain nights: they're stored but excluded from every statistic, because charging would hide the loss. `tdrain record` asks "Did the car charge overnight?" and handles it.

**What if I only have a few nights?**
The model works from night one (defaults fill in for factors it hasn't seen enough of), and every delta is labelled learned or default. More nights = more learned deltas = a sharper forecast.

**Is my data private?**
Yes. Nights live in a JSONL file on your computer. Nothing is uploaded except to the AI backend you explicitly configure (local Ollama by default, which never leaves your machine).

## Keywords

Tesla Fleet API, free open source, Model 3, Model Y, Model S, Model X, Cybertruck, phantom drain, vampire drain, overnight battery loss, Sentry Mode battery drain, Cabin Overheat Protection, Tesla battery drain overnight, phantom drain predictor, EV battery drain.

## License

MIT — Raphael Malikian. Built as part of the Tesla AI app series: one free, open-source Tesla tool per day, each with OpenAI-compatible endpoints and a deterministic offline engine.
