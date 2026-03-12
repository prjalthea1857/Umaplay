# tests/test_telemetry.py
"""Unit tests for the TelemetryLogger – no game / OCR dependency needed."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Patch Settings before importing TelemetryLogger
_tmp_dir = tempfile.mkdtemp()

@pytest.fixture(autouse=True)
def _telemetry_settings(tmp_path):
    """Enable telemetry and redirect output into a temp directory."""
    with patch("core.settings.Settings.ENABLE_TELEMETRY", True), \
         patch("core.settings.Settings.TELEMETRY_DIR", tmp_path):
        yield tmp_path


def _read_events(log_dir: Path):
    """Read all JSONL events from the first .jsonl file found under *log_dir*."""
    jsonl_files = list(log_dir.rglob("*.jsonl"))
    assert jsonl_files, f"No .jsonl files found under {log_dir}"
    lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


# ---- log_skill_acquired ----

def test_log_skill_acquired_writes_jsonl(tmp_path):
    from core.utils.telemetry import TelemetryLogger
    tl = TelemetryLogger(run_id="test_skill", trainee_name="TestUma")

    tl.log_skill_acquired(
        skill_name="Concentration",
        grade="○",
        cost_clicks=1,
        turn=5,
    )

    events = _read_events(tmp_path)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "skill_acquired"
    assert e["data"]["skill_name"] == "Concentration"
    assert e["data"]["grade"] == "○"
    assert e["data"]["cost_clicks"] == 1
    assert e["data"]["turn"] == "5"


# ---- log_race_result ----

def test_log_race_result_writes_jsonl(tmp_path):
    from core.utils.telemetry import TelemetryLogger
    tl = TelemetryLogger(run_id="test_race", trainee_name="TestUma")

    tl.log_race_result(turn=10, race_name="Tokyo Yushun", won=True, retried=False)

    events = _read_events(tmp_path)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "race_result"
    assert e["data"]["race_name"] == "Tokyo Yushun"
    assert e["data"]["won"] is True
    assert e["data"]["retried"] is False


def test_log_race_result_loss_retried(tmp_path):
    from core.utils.telemetry import TelemetryLogger
    tl = TelemetryLogger(run_id="test_race_loss", trainee_name="TestUma")

    tl.log_race_result(turn=12, race_name="Satsuki Sho", won=False, retried=True)

    events = _read_events(tmp_path)
    e = events[0]
    assert e["data"]["won"] is False
    assert e["data"]["retried"] is True


# ---- log_career_end (enriched) ----

def test_log_career_end_with_evaluation(tmp_path):
    from core.utils.telemetry import TelemetryLogger
    tl = TelemetryLogger(run_id="test_end", trainee_name="TestUma")

    tl.log_career_end(
        final_stats={"SPD": 900, "STA": 800},
        final_skill_pts=1200,
        final_evaluation=14500,
        final_rank="S",
    )

    events = _read_events(tmp_path)
    e = events[0]
    assert e["type"] == "career_end"
    assert e["data"]["final_evaluation"] == "14500"
    assert e["data"]["final_rank"] == "S"


def test_log_career_end_backwards_compat(tmp_path):
    from core.utils.telemetry import TelemetryLogger
    tl = TelemetryLogger(run_id="test_compat", trainee_name="TestUma")

    # Old call signature – no evaluation / rank
    tl.log_career_end(
        final_stats={"SPD": 700},
        final_skill_pts=500,
    )

    events = _read_events(tmp_path)
    e = events[0]
    assert e["data"]["final_evaluation"] is None
    assert e["data"]["final_rank"] is None
    assert e["data"]["final_skill_pts"] == "500"


# ---- disabled telemetry ----

def test_disabled_telemetry_no_file(tmp_path):
    with patch("core.settings.Settings.ENABLE_TELEMETRY", False):
        from core.utils.telemetry import TelemetryLogger
        tl = TelemetryLogger(run_id="test_disabled", trainee_name="TestUma")
        tl.log_skill_acquired(skill_name="Focus", grade=None, cost_clicks=1)
        tl.log_race_result(turn=1, won=True)
        tl.log_career_end(final_stats={}, final_skill_pts=0)

    jsonl_files = list(tmp_path.rglob("*.jsonl"))
    assert len(jsonl_files) == 0, "No files should be created when telemetry is disabled"
