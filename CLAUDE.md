# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Umaplay is a Windows automation bot for *Umamusume: Pretty Derby*. It drives the game via a capture→perceive→decide→act loop: a controller captures the game window (Steam/Scrcpy/BlueStacks/ADB), YOLO + OCR perceive the screen, scenario-specific policy logic (URA / Unity Cup) decides the next action, and the controller executes clicks/gestures. A FastAPI backend (`server/`) serves a React web UI (`web/`) for configuration.

**Read `docs/ai/SYSTEM_OVERVIEW.md` first for the full architecture map** — it is actively maintained and more detailed than this file. Also check `docs/ai/SOPs/` before making structural changes; there is a written SOP for common tasks:
- `docs/ai/SOPs/adding-new-scenario.md` — onboarding a new training scenario end-to-end (runtime + config schema + web UI).
- `docs/ai/SOPs/waiter-usage-and-integration.md` — how the shared `Waiter` (`core/utils/waiter.py`) drives detection-based clicks/polling; read before adding new automation flows.
- `docs/ai/SOPs/sop-config-back-front.md` — how config flows between `web/` and `core/settings.py`.
- `docs/ai/SOPs/sop-presets-tab-groups.md`, `docs/ai/SOPs/towards-custom-training-policy-graph.md` — narrower, read when touching those areas.
- `docs/ai/policies/<scenario>/` — Mermaid flowcharts + notes describing the actual lobby/training/scoring decision logic per scenario (`ura`, `unity_cup`). These are the source of truth for *why* the bot makes a given decision — read them before changing training/lobby policy.

## Commands

### Python backend
```bash
conda activate env_uma          # project targets Python 3.10
python main.py                  # run bot + FastAPI config server (opens http://127.0.0.1:8000)
python main.py --port 8080      # override server port

# Standalone remote inference server (offloads YOLO/OCR/OpenCV to a stronger host)
uvicorn server.main_inference:app --host 0.0.0.0 --port 8001
```
- Hotkeys once running: **F2** start/stop main career loop, **F7** Team Trials, **F8** Daily Races, **F9** Roulette/Prize Derby. Game must be on the career lobby screen.

### Tests
```bash
pytest                                    # run full suite (config lives in pytest.ini)
pytest tests/test_turns.py                # single file
pytest tests/test_turns.py::test_name     # single test
pytest tests/core/actions/test_training_policy.py -k some_case
```
- `tests/conftest.py` inserts the repo root onto `sys.path`; no package install needed.
- `tests/data/` holds fixture screenshots (events, lobby stats, roulette, shop) used by perception/analyzer tests.

### Lint / type-check
```bash
ruff check .              # Python lint (config: ruff.toml; excludes dev_play.ipynb, runs, debug)
```
- `pyrightconfig.json` configures Pyright at `basic` mode for Python 3.10 if you need type-checking.

### Web UI (`web/`)
```bash
cd web
npm install
npm run dev       # Vite dev server on :5173, proxies /config, /api/skills, /api/races to :8000
npm run build     # tsc -b && vite build -> web/dist, served by FastAPI at "/"
npm run lint       # eslint .
```
Backend must be running (`python main.py`) for the dev server's proxied endpoints to work.

### Data / catalog pipeline
```bash
python build_catalog.py       # regenerate compressed event catalog from datasets/in_game/events.json
python datasets/scrape_events.py --html-file events_full_html.txt --support-defaults "Name-Rarity-Attr" --out supports_events.json --debug
```
Full reproducible flow (scrape GameTora → merge → rebuild catalog → rebuild web assets) is documented in `README.dev.md`.

## Architecture essentials

- **Core loop**: `main.py` builds a controller + OCR/YOLO engines from `Settings`, then runs `AgentScenario` (`core/agent_scenario.py`, abstract base) via `AgentURA` (`core/actions/ura/agent.py`) or `AgentUnityCup` (`core/actions/unity_cup/agent.py`), selected by `Settings.ACTIVE_SCENARIO`. `BotState`/`NavState` in `main.py` own the worker threads and reload config fresh on every Start so UI edits take effect without a restart.
- **Scenario routing**: `core/scenarios/registry.py` maps scenario keys (`ura`, `unity_cup`, aliases) to policy callables. Adding a scenario means adding a `core/scenarios/<name>.py` module *and* a `core/actions/<name>/` package (agent, lobby, training_check, training_policy) — see the SOP above, don't wire scenario logic ad hoc into the shared agent.
- **Perception stack** (`core/perception/`): `yolo/` (local + remote detector interfaces), `ocr/` (PaddleOCR local/remote), `analyzers/` (screen classification, mood/energy/hint/badge reading, template matching for trainee/support portraits), `extractors/` (stats/goals/energy parsing), `classifiers/` (e.g. Unity Cup spirit classifier). Local vs. remote is a runtime toggle (`Settings.USE_EXTERNAL_PROCESSOR`) — remote offloads to `server/main_inference.py` for thin clients without Torch/OpenCV (`requirements_client_only.txt`).
- **Waiter** (`core/utils/waiter.py`): the shared synchronization primitive — nearly all clicking/polling in `core/actions/` goes through `Waiter.click_when()`/`seen()`/`try_click_once()`. Read the SOP before adding new click flows instead of hand-rolling polling loops.
- **Action flows** (`core/actions/`): one module per game surface — `lobby.py`, `training_policy.py`/`training_check.py` (scoring/decision), `race.py`/`daily_race.py`, `team_trials.py`, `roulette.py`, `skills.py`, `events.py`/`EventFlow`, `claw.py`. Scenario-specific variants of lobby/training live under `core/actions/ura/` and `core/actions/unity_cup/`.
- **Controllers** (`core/controllers/`): `IController` (`base.py`) abstracts capture+input for Steam, Scrcpy/Android, BlueStacks, and ADB. `main.py:make_controller_from_settings()` is the single place that picks one based on `Settings.MODE`.
- **Settings** (`core/settings.py`): the runtime config surface. `Settings.apply_config(cfg)` maps `prefs/config.json` (written by the web UI's `POST /config`) into class attributes consumed everywhere; `Settings.extract_runtime_preset()` pulls the active preset's skill list/race plan/style. When adding a config field, wire it through here as well as `web/src/models/config.schema.ts` (see `web/README.md`'s "Add a new field" walkthrough).
- **Persistence** (`prefs/`, `datasets/in_game/`): `prefs/config.json` is the live config (seeded from `config.sample.json`); `core/utils/skill_memory.py` and `core/utils/pal_memory.py` persist per-run skill purchases and PAL chain state; `datasets/in_game/{skills,races,events}.json` back the web UI pickers and race scheduling and are served via `server/main.py`'s `/api/*` routes.
- **Web UI** (`web/`): React + TS + Vite + MUI + Zustand + Zod + React Query. State lives in `web/src/store/configStore.ts`, schema/types are derived from `web/src/models/config.schema.ts`. Scenario-specific "Bot Strategy" panels use a registry pattern under `web/src/components/presets/strategy/` — see that folder's own `README.md` before adding a new scenario's strategy UI.

## Working conventions

- **Notebook-first prototyping**: when explicitly asked to prototype in Jupyter (`dev_nav.ipynb`, `dev_play.ipynb`), do not touch any `.py` file until the user asks for migration. Keep prototype code under a `## PROTOTYPE` section, one cell per target module, each cell starting with a comment naming its destination file (e.g. `# core/actions/roulette.py`). Copy the current file contents into the notebook before editing there. Wait for explicit approval before writing changes into `.py` files, even if the prototype looks finished — the user does the copy-paste themselves.
- Windows-only project (paths, hotkeys, controllers assume Windows); dev shell is PowerShell/conda, not POSIX.
- Debug artifacts land under `debug/<agent>/<tag>/`; `main.py` auto-prunes them past 250MB per agent via `collect_data_training.py`.
