import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.settings import Settings
from core.utils.logger import logger_uma

class TelemetryLogger:
    """
    Logs structured data to a JSON lines file for calibration and analysis.
    """

    def __init__(self, run_id: Optional[str] = None, trainee_name: Optional[str] = None):
        self.enabled = Settings.ENABLE_TELEMETRY
        self.log_file: Optional[Path] = None
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.trainee_name = trainee_name

        if self.enabled:
            try:
                import re
                
                safe_trainee = "Unknown_Trainee"
                if self.trainee_name and str(self.trainee_name).strip():
                    safe_trainee = re.sub(r'[\\/*?:"<>|]', "", str(self.trainee_name).strip())
                
                date_str = time.strftime("%Y-%m-%d")
                match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{6})$", self.run_id)
                if match:
                    date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

                self.log_dir = Settings.TELEMETRY_DIR / safe_trainee / date_str
                self.log_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"telemetry_{self.run_id}.jsonl"
                self.log_file = self.log_dir / filename
                logger_uma.info(f"[telemetry] Initialized run: {self.run_id} at {self.log_file}")
            except Exception as e:
                logger_uma.error(f"[telemetry] Failed to initialize telemetry logger: {e}")
                self.enabled = False

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self.enabled or not self.log_file:
            return

        payload = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "trainee_name": self.trainee_name,
            "type": event_type,
            "data": data,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger_uma.debug(f"[telemetry] Failed to write to {self.log_file}: {e}")

    def log_turn_state(
        self,
        turn: Any,
        energy: Any,
        skill_pts: Any,
        stats: Any,
        mood: Any,
        is_summer: Any,
        infirmary_on: Any,
        goal: Any,
    ) -> None:
        """Log state at the start of a turn."""
        self._log_event(
            "turn_state",
            {
                "turn": str(turn) if turn is not None else None,
                "energy": str(energy) if energy is not None else None,
                "skill_pts": str(skill_pts) if skill_pts is not None else None,
                "stats": stats,
                "mood": str(mood) if mood is not None else None,
                "is_summer": bool(is_summer),
                "infirmary_on": bool(infirmary_on),
                "goal": str(goal) if goal is not None else None,
            },
        )

    def log_training_action(
        self, turn: Any, action_type: str, action_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the result of the training policy decision."""
        details = action_details or {}
        self._log_event(
            "training_action",
            {
                "turn": str(turn) if turn is not None else None,
                "action": action_type,
                **details,
            },
        )

    def log_event_choice(
        self,
        event_name: Optional[str],
        choices_available: int,
        choice_made_index: int,
        choice_text: Optional[str] = None,
        strategy_used: Optional[str] = None,
    ) -> None:
        """Log random event choice decisions."""
        self._log_event(
            "event_choice",
            {
                "event_name": event_name,
                "choices_available": choices_available,
                "choice_made_index": choice_made_index,
                "choice_text": choice_text,
                "strategy_used": strategy_used,
            },
        )

    def log_race_start(
        self,
        turn: Any,
        race_name: Optional[str],
        prioritize_g1: bool,
        is_goal: bool,
        select_style: Optional[str] = None,
    ) -> None:
        """Log when entering a race."""
        self._log_event(
            "race_start",
            {
                "turn": str(turn) if turn is not None else None,
                "race_name": race_name,
                "prioritize_g1": prioritize_g1,
                "is_goal": is_goal,
                "select_style": select_style,
            },
        )

    def log_race_result(
        self,
        turn: Any,
        race_name: Optional[str] = None,
        won: bool = True,
        retried: bool = False,
    ) -> None:
        """Log the outcome of a race after completion."""
        self._log_event(
            "race_result",
            {
                "turn": str(turn) if turn is not None else None,
                "race_name": race_name,
                "won": won,
                "retried": retried,
            },
        )

    def log_skill_acquired(
        self,
        skill_name: str,
        grade: Optional[str] = None,
        cost_clicks: int = 1,
        turn: Any = None,
    ) -> None:
        """Log a skill purchase."""
        self._log_event(
            "skill_acquired",
            {
                "skill_name": skill_name,
                "grade": grade,
                "cost_clicks": cost_clicks,
                "turn": str(turn) if turn is not None else None,
            },
        )

    # ------------------------------------------------------------------
    # Stat grade helpers
    # ------------------------------------------------------------------

    _STAT_GRADE_THRESHOLDS = [
        (1200, "SS+"),
        (1100, "SS"),
        (1000, "S"),
        (900,  "A+"),
        (800,  "A"),
        (700,  "B+"),
        (600,  "B"),
        (500,  "C+"),
        (400,  "C"),
        (300,  "D"),
        (0,    "G"),
    ]

    @classmethod
    def _stat_grade(cls, value: float) -> str:
        """Return the grade letter for a single stat value."""
        for threshold, grade in cls._STAT_GRADE_THRESHOLDS:
            if value >= threshold:
                return grade
        return "G"

    @classmethod
    def _compute_rank(cls, stats: Dict[str, Any]) -> Optional[str]:
        """
        Derive a simple pass/fail rank from the five final stats.

        S criteria (all must hold):
          - top-2 stats are SS or SS+  (>= 1100)
          - 3rd stat is S or above     (>= 1000)
          - 4th and 5th stats are >= 300
        """
        if not stats:
            return None
        try:
            values = sorted(
                (float(v) for v in stats.values() if v is not None),
                reverse=True,
            )
        except (TypeError, ValueError):
            return None

        if len(values) < 5:
            return None

        if values[0] >= 1100 and values[1] >= 1100 and values[2] >= 1000 and values[3] >= 300 and values[4] >= 300:
            return "S"
        return None

    def log_career_end(
        self,
        final_stats: Any,
        final_skill_pts: Any,
        final_evaluation: Any = None,
        final_rank: Optional[str] = None,
    ) -> None:
        """Log the end of the career run.

        If *final_rank* is not supplied it is automatically derived from
        *final_stats* using the S-rank stat thresholds.
        """
        if final_rank is None and isinstance(final_stats, dict):
            final_rank = self._compute_rank(final_stats)

        self._log_event(
            "career_end",
            {
                "final_stats": final_stats,
                "final_skill_pts": str(final_skill_pts) if final_skill_pts is not None else None,
                "final_evaluation": str(final_evaluation) if final_evaluation is not None else None,
                "final_rank": final_rank,
                "stat_grades": {
                    k: self._stat_grade(float(v))
                    for k, v in (final_stats or {}).items()
                    if v is not None
                },
            },
        )
